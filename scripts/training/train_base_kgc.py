#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

try:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.sampling import BasicNegativeSampler, BernoulliNegativeSampler, PseudoTypedNegativeSampler
except Exception as e:
    raise SystemExit("This script requires PyKEEN. Install with: pip install pykeen") from e


# ============================================================
# Helpers
# ============================================================

def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def perf_now() -> float:
    return time.perf_counter()


def format_seconds(sec: float | None) -> str | None:
    if sec is None:
        return None
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, type):
        return obj.__name__
    if callable(obj) and hasattr(obj, "__name__"):
        return obj.__name__
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return repr(obj)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    tmp.replace(path)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def flat_metrics(metric_results: Any) -> dict[str, float]:
    if metric_results is None:
        return {}
    try:
        raw = metric_results.to_flat_dict()
    except Exception:
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            pass
    return out


def pick_metric(flat: dict[str, float], names: list[str]) -> Optional[float]:
    for name in names:
        if name in flat:
            return flat[name]
    return None


def summarize_eval(flat: dict[str, float]) -> dict[str, Optional[float]]:
    return {
        "mrr": pick_metric(flat, ["both.realistic.inverse_harmonic_mean_rank", "inverse_harmonic_mean_rank", "mrr"]),
        "hits@1": pick_metric(flat, ["both.realistic.hits_at_1", "hits_at_1"]),
        "hits@3": pick_metric(flat, ["both.realistic.hits_at_3", "hits_at_3"]),
        "hits@10": pick_metric(flat, ["both.realistic.hits_at_10", "hits_at_10"]),
        "mean_rank": pick_metric(flat, ["both.realistic.arithmetic_mean_rank", "arithmetic_mean_rank"]),
    }


def model_kwargs_for(model: str, embedding_dim: int) -> dict[str, Any]:
    m = model.lower()
    if m == "transe":
        return {"embedding_dim": embedding_dim, "scoring_fct_norm": 1}
    if m == "pairre":
        return {"embedding_dim": embedding_dim}
    return {"embedding_dim": embedding_dim}


# ============================================================
# Dataset + optional schema sidecars
# ============================================================

@dataclass
class DatasetStats:
    num_entities: int
    num_relations: int
    train_size: int
    valid_size: int
    test_size: int
    has_entity_types: bool
    has_relation_constraints: bool
    has_disjoint_pairs: bool


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(processed_dir: Path, create_inverse_triples: bool) -> tuple[TriplesFactory, TriplesFactory, TriplesFactory, DatasetStats, dict[str, Any]]:
    entity_to_id = load_json(processed_dir / "entity2id.json")
    relation_to_id = load_json(processed_dir / "relation2id.json")

    train_tf = TriplesFactory.from_path(
        path=processed_dir / "train.tsv",
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=create_inverse_triples,
    )
    valid_tf = TriplesFactory.from_path(
        path=processed_dir / "valid.tsv",
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=create_inverse_triples,
    )
    test_tf = TriplesFactory.from_path(
        path=processed_dir / "test.tsv",
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=create_inverse_triples,
    )

    sidecars = {
        "entity_types_path": str(processed_dir / "entity_types.json"),
        "relation_constraints_path": str(processed_dir / "relation_constraints.json"),
        "disjoint_pairs_path": str(processed_dir / "disjoint_pairs.json"),
        "relation_split_stats_path": str(processed_dir / "relation_split_stats.json"),
        "file_stats_path": str(processed_dir / "file_stats.json"),
        "has_entity_types": (processed_dir / "entity_types.json").exists(),
        "has_relation_constraints": (processed_dir / "relation_constraints.json").exists(),
        "has_disjoint_pairs": (processed_dir / "disjoint_pairs.json").exists(),
    }

    stats = DatasetStats(
        num_entities=len(entity_to_id),
        num_relations=len(relation_to_id),
        train_size=train_tf.num_triples,
        valid_size=valid_tf.num_triples,
        test_size=test_tf.num_triples,
        has_entity_types=sidecars["has_entity_types"],
        has_relation_constraints=sidecars["has_relation_constraints"],
        has_disjoint_pairs=sidecars["has_disjoint_pairs"],
    )
    return train_tf, valid_tf, test_tf, stats, sidecars


# ============================================================
# Heuristics
# ============================================================

def choose_portfolio(stats: DatasetStats, portfolio: str) -> list[dict[str, Any]]:
    # A compact, robust portfolio for typed / schema-rich KGs.
    candidates = [
        {
            "name": "complex_pseudotyped",
            "model": "ComplEx",
            "embedding_dim": 400 if stats.num_entities < 200_000 else 256,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": 1024 if stats.train_size < 5_000_000 else 512,
            "num_negs": 32,
            "epochs": 200,
            "create_inverse_triples": True,
            "negative_sampler_cls": PseudoTypedNegativeSampler,
            "negative_sampler_kwargs": {},
            "regularizer": "lp",
            "regularizer_kwargs": {"p": 2.0, "weight": 1e-5},
            "loss": None,
            "loss_kwargs": None,
        },
        {
            "name": "rotate_bernoulli_nssa",
            "model": "RotatE",
            "embedding_dim": 512 if stats.num_entities < 200_000 else 384,
            "optimizer": "Adam",
            "learning_rate": 5e-4,
            "batch_size": 512,
            "num_negs": 64,
            "epochs": 250,
            "create_inverse_triples": True,
            "negative_sampler_cls": BernoulliNegativeSampler,
            "negative_sampler_kwargs": {
                "filtered": True,
                "filterer": "bloom",
                "filterer_kwargs": {"error_rate": 1e-4},
            },
            "regularizer": None,
            "regularizer_kwargs": None,
            "loss": "nssa",
            "loss_kwargs": {"margin": 9.0, "adversarial_temperature": 1.0},
        },
        {
            "name": "pairre_pseudotyped",
            "model": "PairRE",
            "embedding_dim": 300 if stats.num_entities < 200_000 else 200,
            "optimizer": "Adam",
            "learning_rate": 7e-4,
            "batch_size": 768,
            "num_negs": 64,
            "epochs": 250,
            "create_inverse_triples": True,
            "negative_sampler_cls": PseudoTypedNegativeSampler,
            "negative_sampler_kwargs": {},
            "regularizer": None,
            "regularizer_kwargs": None,
            "loss": None,
            "loss_kwargs": None,
        },
        {
            "name": "transe_bernoulli_margin",
            "model": "TransE",
            "embedding_dim": 400 if stats.num_entities < 200_000 else 256,
            "optimizer": "Adagrad",
            "learning_rate": 0.1,
            "batch_size": 2048 if stats.train_size < 5_000_000 else 1024,
            "num_negs": 64,
            "epochs": 250,
            "create_inverse_triples": True,
            "negative_sampler_cls": BernoulliNegativeSampler,
            "negative_sampler_kwargs": {
                "filtered": True,
                "filterer": "bloom",
                "filterer_kwargs": {"error_rate": 1e-4},
            },
            "regularizer": None,
            "regularizer_kwargs": None,
            "loss": "marginranking",
            "loss_kwargs": {"margin": 1.0},
        },
    ]

    if portfolio == "small":
        return [candidates[0], candidates[1]]
    if portfolio == "medium":
        return [candidates[0], candidates[1], candidates[2]]
    if portfolio == "large":
        return [candidates[0], candidates[1], candidates[2], candidates[3]]
    return candidates


def evaluate(
    *,
    model: Any,
    mapped_triples: torch.Tensor,
    additional_filter_triples: list[torch.Tensor],
    filtered: bool,
    batch_size: Optional[int],
    slice_size: Optional[int],
    use_tqdm: bool,
) -> dict[str, Any]:
    evaluator = RankBasedEvaluator(filtered=filtered)
    started = perf_now()
    metric_results = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples,
        additional_filter_triples=additional_filter_triples,
        batch_size=batch_size,
        slice_size=slice_size,
        use_tqdm=use_tqdm,
        do_time_consuming_checks=batch_size is None,
    )
    elapsed = perf_now() - started
    flat = flat_metrics(metric_results)
    return {
        "elapsed_seconds": round(elapsed, 6),
        "elapsed_human": format_seconds(elapsed),
        "flat_metrics": flat,
        **summarize_eval(flat),
    }


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General schema-aware KGC portfolio trainer for processed Eurostat/DrugBank/DBpedia-style datasets.")
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--portfolio", type=str, default="medium", choices=["small", "medium", "large"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--eval-slice-size", type=int, default=None)
    p.add_argument("--filtered-eval", action="store_true")
    p.add_argument("--stop-frequency", type=int, default=10)
    p.add_argument("--stop-patience", type=int, default=3)
    p.add_argument("--stop-relative-delta", type=float, default=0.001)
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_tqdm = not args.no_progress
    set_all_seeds(args.seed)

    train_tf0, valid_tf0, test_tf0, stats0, sidecars0 = load_dataset(args.processed_dir, create_inverse_triples=True)
    save_json(args.output_dir / "dataset_stats.json", asdict(stats0))
    save_json(args.output_dir / "dataset_sidecars.json", sidecars0)

    portfolio = choose_portfolio(stats0, args.portfolio)
    if args.max_runs is not None:
        portfolio = portfolio[: args.max_runs]

    all_runs: list[dict[str, Any]] = []
    best_run: Optional[dict[str, Any]] = None
    best_result = None

    started_all = perf_now()

    for i, cfg in enumerate(portfolio, start=1):
        run_name = f"{i:02d}_{cfg['name']}"
        run_dir = args.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== [{i}/{len(portfolio)}] {run_name} ===", flush=True)
        train_tf, valid_tf, test_tf, stats, sidecars = load_dataset(
            args.processed_dir,
            create_inverse_triples=bool(cfg["create_inverse_triples"]),
        )

        started = perf_now()
        try:
            pipe_kwargs = dict(
                training=train_tf,
                validation=valid_tf,
                testing=test_tf,
                model=cfg["model"],
                model_kwargs=model_kwargs_for(cfg["model"], cfg["embedding_dim"]),
                optimizer=cfg["optimizer"],
                optimizer_kwargs={"lr": cfg["learning_rate"]},
                training_loop="sLCWA",
                training_kwargs={
                    "num_epochs": cfg["epochs"],
                    "batch_size": cfg["batch_size"],
                },
                stopper="early",
                stopper_kwargs={
                    "frequency": args.stop_frequency,
                    "patience": args.stop_patience,
                    "relative_delta": args.stop_relative_delta,
                },
                evaluator="RankBasedEvaluator",
                evaluator_kwargs={"filtered": args.filtered_eval},
                negative_sampler=cfg["negative_sampler_cls"],
                negative_sampler_kwargs={
                    **cfg["negative_sampler_kwargs"],
                    "num_negs_per_pos": cfg["num_negs"],
                },
                random_seed=args.seed,
                device=args.device,
            )
            if cfg["loss"] is not None:
                pipe_kwargs["loss"] = cfg["loss"]
                pipe_kwargs["loss_kwargs"] = cfg["loss_kwargs"]
            if cfg["regularizer"] is not None:
                pipe_kwargs["regularizer"] = cfg["regularizer"]
                pipe_kwargs["regularizer_kwargs"] = cfg["regularizer_kwargs"]

            result = pipeline(**pipe_kwargs)

            valid_report = evaluate(
                model=result.model,
                mapped_triples=valid_tf.mapped_triples,
                additional_filter_triples=[train_tf.mapped_triples] if args.filtered_eval else [],
                filtered=args.filtered_eval,
                batch_size=args.eval_batch_size,
                slice_size=args.eval_slice_size,
                use_tqdm=use_tqdm,
            )
            test_report = evaluate(
                model=result.model,
                mapped_triples=test_tf.mapped_triples,
                additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples] if args.filtered_eval else [],
                filtered=args.filtered_eval,
                batch_size=args.eval_batch_size,
                slice_size=args.eval_slice_size,
                use_tqdm=use_tqdm,
            )

            elapsed = perf_now() - started
            losses = [float(x) for x in (getattr(result, "losses", None) or [])]
            serializable_cfg = {
                **cfg,
                "negative_sampler_cls": getattr(cfg["negative_sampler_cls"], "__name__", str(cfg["negative_sampler_cls"])),
            }
            run_summary = {
                "status": "ok",
                "run_name": run_name,
                "dataset_name": args.dataset_name or args.processed_dir.name,
                "processed_dir": str(args.processed_dir),
                "config": serializable_cfg,
                "dataset_stats": asdict(stats),
                "dataset_sidecars": sidecars,
                "filtered_eval": args.filtered_eval,
                "seed": args.seed,
                "device": args.device,
                "validation": valid_report,
                "test": test_report,
                "training_losses": losses,
                "train_elapsed_seconds": round(elapsed, 6),
                "train_elapsed_human": format_seconds(elapsed),
                "stopped_epoch_guess": len(losses) if losses else None,
                "saved_at_utc": now_iso_utc(),
            }
            save_json(run_dir / "summary.json", run_summary)
            result.save_to_directory(run_dir / "pykeen_artifacts")
            all_runs.append(run_summary)

            print(
                f"VALID MRR={valid_report['mrr']:.6f} | VALID H@10={valid_report['hits@10']:.6f} | "
                f"TEST MRR={test_report['mrr']:.6f} | TEST H@10={test_report['hits@10']:.6f}",
                flush=True,
            )

            if best_run is None or (valid_report["mrr"] is not None and valid_report["mrr"] > (best_run["validation"]["mrr"] or -1.0)):
                best_run = run_summary
                best_result = result

        except Exception as e:
            elapsed = perf_now() - started
            serializable_cfg = {
                **cfg,
                "negative_sampler_cls": getattr(cfg["negative_sampler_cls"], "__name__", str(cfg["negative_sampler_cls"])),
            }
            failed = {
                "status": "failed",
                "run_name": run_name,
                "dataset_name": args.dataset_name or args.processed_dir.name,
                "processed_dir": str(args.processed_dir),
                "config": serializable_cfg,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "train_elapsed_seconds": round(elapsed, 6),
                "train_elapsed_human": format_seconds(elapsed),
                "saved_at_utc": now_iso_utc(),
            }
            save_json(run_dir / "summary.json", failed)
            all_runs.append(failed)
            print(f"FAILED: {failed['error']}", flush=True)
        finally:
            maybe_clear_cuda_cache()

    leaderboard = []
    for r in all_runs:
        if r["status"] != "ok":
            continue
        leaderboard.append(
            {
                "run_name": r["run_name"],
                "model": r["config"]["model"],
                "embedding_dim": r["config"]["embedding_dim"],
                "negative_sampler": getattr(r["config"]["negative_sampler_cls"], "__name__", str(r["config"]["negative_sampler_cls"])),
                "validation_mrr": r["validation"]["mrr"],
                "validation_hits@10": r["validation"]["hits@10"],
                "test_mrr": r["test"]["mrr"],
                "test_hits@10": r["test"]["hits@10"],
            }
        )
    leaderboard.sort(key=lambda x: x["validation_mrr"], reverse=True)

    overview = {
        "dataset_name": args.dataset_name or args.processed_dir.name,
        "processed_dir": str(args.processed_dir),
        "output_dir": str(args.output_dir),
        "portfolio": args.portfolio,
        "seed": args.seed,
        "device": args.device,
        "filtered_eval": args.filtered_eval,
        "started_at_utc": now_iso_utc(),
        "total_elapsed_seconds": round(perf_now() - started_all, 6),
        "total_elapsed_human": format_seconds(perf_now() - started_all),
        "dataset_stats": asdict(stats0),
        "dataset_sidecars": sidecars0,
        "leaderboard": leaderboard,
        "best_run": best_run,
    }
    save_json(args.output_dir / "leaderboard.json", overview)

    if best_run is None or best_result is None:
        raise SystemExit("No successful run completed.")

    best_dir = args.output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_result.save_to_directory(best_dir / "pykeen_artifacts")
    save_json(best_dir / "best_summary.json", best_run)

    print("\n=== BEST RUN ===", flush=True)
    print(json.dumps({
        "run_name": best_run["run_name"],
        "validation_mrr": best_run["validation"]["mrr"],
        "validation_hits@10": best_run["validation"]["hits@10"],
        "test_mrr": best_run["test"]["mrr"],
        "test_hits@10": best_run["test"]["hits@10"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
