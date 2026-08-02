#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import random
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Optional

import numpy as np
import torch

try:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.sampling import BernoulliNegativeSampler, PseudoTypedNegativeSampler
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def remove_path_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path)


def save_pykeen_artifacts_safely(result: Any, out_dir: Path) -> None:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    remove_path_if_exists(out_dir)

    if getattr(result, "model", None) is not None:
        try:
            result.model = result.model.cpu()
        except Exception:
            pass

    maybe_clear_cuda_cache()
    gc.collect()
    result.save_to_directory(out_dir)
    maybe_clear_cuda_cache()
    gc.collect()


def copy_best_artifacts(best_run_dir: Path, best_dir: Path, best_run: dict[str, Any]) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)

    src_artifacts = best_run_dir / "pykeen_artifacts"
    src_checkpoint = best_run_dir / "base_model_checkpoint.pt"

    if not src_artifacts.exists():
        raise FileNotFoundError(f"Missing best run artifacts dir: {src_artifacts}")
    if not src_checkpoint.exists():
        raise FileNotFoundError(f"Missing best run checkpoint: {src_checkpoint}")

    dst_artifacts = best_dir / "pykeen_artifacts"
    dst_checkpoint = best_dir / "base_model_checkpoint.pt"

    remove_path_if_exists(dst_artifacts)
    if dst_checkpoint.exists():
        dst_checkpoint.unlink()

    shutil.copytree(src_artifacts, dst_artifacts)
    shutil.copy2(src_checkpoint, dst_checkpoint)
    save_json(best_dir / "best_summary.json", best_run)


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


def parse_device_index(device_str: str) -> Optional[int]:
    if not device_str.startswith("cuda"):
        return None
    if ":" not in device_str:
        return 0
    return int(device_str.split(":")[1])


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cuda error: out of memory" in text
        or "cublas_status_alloc_failed" in text
    )


def write_status(run_dir: Path, **payload: Any) -> None:
    status = {
        "timestamp_utc": now_iso_utc(),
        **payload,
    }
    save_json(run_dir / "status.json", status)


# ============================================================
# Dataset
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
    size_profile: str


def infer_size_profile(num_entities: int, train_size: int) -> str:
    return "large"


def load_dataset(
    processed_dir: Path,
    create_inverse_triples: bool,
) -> tuple[TriplesFactory, TriplesFactory, TriplesFactory, DatasetStats, dict[str, Any]]:
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
        size_profile=infer_size_profile(len(entity_to_id), train_tf.num_triples),
    )
    return train_tf, valid_tf, test_tf, stats, sidecars


# ============================================================
# Portfolio
# ============================================================

def choose_portfolio(stats: DatasetStats, portfolio: str) -> list[dict[str, Any]]:
    """
    Portfolio logic:
      - xlarge: very conservative
      - large : realistic DBpedia-safe budget
      - small : richer / smaller-dataset-oriented portfolio
    """

    if stats.size_profile == "xlarge":
        candidates = [
            {
                "name": "complex_pseudotyped_xlarge",
                "model": "ComplEx",
                "embedding_dim": 128,
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "batch_size": 512,
                "num_negs": 4,
                "epochs": 12,
                "create_inverse_triples": True,
                "negative_sampler": "pseudo_typed",
                "negative_sampler_kwargs": {},
                "regularizer": "lp",
                "regularizer_kwargs": {"p": 2.0, "weight": 1e-5},
                "loss": None,
                "loss_kwargs": None,
                "use_early_stopper": True,
            },
            {
                "name": "transe_bernoulli_margin_xlarge",
                "model": "TransE",
                "embedding_dim": 128,
                "optimizer": "Adagrad",
                "learning_rate": 0.05,
                "batch_size": 1024,
                "num_negs": 4,
                "epochs": 10,
                "create_inverse_triples": True,
                "negative_sampler": "bernoulli",
                "negative_sampler_kwargs": {
                    "filtered": True,
                    "filterer": "bloom",
                    "filterer_kwargs": {"error_rate": 1e-4},
                },
                "regularizer": None,
                "regularizer_kwargs": None,
                "loss": "marginranking",
                "loss_kwargs": {"margin": 1.0},
                "use_early_stopper": True,
            },
        ]
        if portfolio == "small":
            return candidates[:1]
        if portfolio == "medium":
            return candidates[:2]
        return candidates[:2]

    if stats.size_profile == "large":
        candidates = [
            {
                "name": "complex_pseudotyped_large",
                "model": "ComplEx",
                "embedding_dim": 160,
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "batch_size": 256,
                "num_negs": 8,
                "epochs": 20,
                "create_inverse_triples": True,
                "negative_sampler": "pseudo_typed",
                "negative_sampler_kwargs": {},
                "regularizer": "lp",
                "regularizer_kwargs": {"p": 2.0, "weight": 1e-5},
                "loss": None,
                "loss_kwargs": None,
                "use_early_stopper": False,
            },
            {
                "name": "pairre_pseudotyped_large",
                "model": "PairRE",
                "embedding_dim": 128,
                "optimizer": "Adam",
                "learning_rate": 7e-4,
                "batch_size": 256,
                "num_negs": 8,
                "epochs": 20,
                "create_inverse_triples": True,
                "negative_sampler": "pseudo_typed",
                "negative_sampler_kwargs": {},
                "regularizer": None,
                "regularizer_kwargs": None,
                "loss": None,
                "loss_kwargs": None,
                "use_early_stopper": False,
            },
            {
                "name": "transe_bernoulli_margin_large",
                "model": "TransE",
                "embedding_dim": 160,
                "optimizer": "Adagrad",
                "learning_rate": 0.05,
                "batch_size": 512,
                "num_negs": 8,
                "epochs": 16,
                "create_inverse_triples": True,
                "negative_sampler": "bernoulli",
                "negative_sampler_kwargs": {
                    "filtered": True,
                    "filterer": "bloom",
                    "filterer_kwargs": {"error_rate": 1e-4},
                },
                "regularizer": None,
                "regularizer_kwargs": None,
                "loss": "marginranking",
                "loss_kwargs": {"margin": 1.0},
                "use_early_stopper": False,
            },
        ]
        if portfolio == "small":
            return candidates[:1]
        if portfolio == "medium":
            return candidates[:2]
        return candidates

    candidates = [
        {
            "name": "complex_pseudotyped",
            "model": "ComplEx",
            "embedding_dim": 256,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": 512,
            "num_negs": 16,
            "epochs": 80,
            "create_inverse_triples": True,
            "negative_sampler": "pseudo_typed",
            "negative_sampler_kwargs": {},
            "regularizer": "lp",
            "regularizer_kwargs": {"p": 2.0, "weight": 1e-5},
            "loss": None,
            "loss_kwargs": None,
            "use_early_stopper": True,
        },
        {
            "name": "pairre_pseudotyped",
            "model": "PairRE",
            "embedding_dim": 200,
            "optimizer": "Adam",
            "learning_rate": 7e-4,
            "batch_size": 512,
            "num_negs": 16,
            "epochs": 80,
            "create_inverse_triples": True,
            "negative_sampler": "pseudo_typed",
            "negative_sampler_kwargs": {},
            "regularizer": None,
            "regularizer_kwargs": None,
            "loss": None,
            "loss_kwargs": None,
            "use_early_stopper": True,
        },
        {
            "name": "transe_bernoulli_margin",
            "model": "TransE",
            "embedding_dim": 256,
            "optimizer": "Adagrad",
            "learning_rate": 0.05,
            "batch_size": 1024,
            "num_negs": 16,
            "epochs": 60,
            "create_inverse_triples": True,
            "negative_sampler": "bernoulli",
            "negative_sampler_kwargs": {
                "filtered": True,
                "filterer": "bloom",
                "filterer_kwargs": {"error_rate": 1e-4},
            },
            "regularizer": None,
            "regularizer_kwargs": None,
            "loss": "marginranking",
            "loss_kwargs": {"margin": 1.0},
            "use_early_stopper": True,
        },
    ]

    if portfolio == "small":
        return candidates[:1]
    if portfolio == "medium":
        return candidates[:2]
    return candidates


def resolve_negative_sampler(name: str):
    name = name.lower().strip()
    if name == "pseudo_typed":
        return PseudoTypedNegativeSampler
    if name == "bernoulli":
        return BernoulliNegativeSampler
    raise ValueError(f"Unsupported negative sampler: {name}")


# ============================================================
# Evaluation
# ============================================================

def evaluate_once(
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


def batch_candidates(initial: Optional[int], device_str: str) -> list[Optional[int]]:
    if initial is not None and initial > 0:
        vals = []
        b = initial
        while b >= 1:
            vals.append(b)
            if b == 1:
                break
            b = max(1, b // 2)
        return vals

    if device_str.startswith("cuda"):
        return [64, 32, 16, 8, 4, 2, 1]
    return [128, 64, 32, 16, 8, 4, 2, 1]


def evaluate_with_fallback(
    *,
    model: Any,
    mapped_triples: torch.Tensor,
    additional_filter_triples: list[torch.Tensor],
    filtered: bool,
    batch_size: Optional[int],
    slice_size: Optional[int],
    preferred_device: str,
    fallback_to_cpu: bool,
    use_tqdm: bool,
) -> dict[str, Any]:
    tried: list[dict[str, Any]] = []

    device_plan = [preferred_device]
    if preferred_device.startswith("cuda") and fallback_to_cpu:
        device_plan.append("cpu")

    last_exc: Optional[BaseException] = None

    for dev in device_plan:
        try:
            model = model.to(torch.device(dev))
            if dev.startswith("cuda"):
                idx = parse_device_index(dev)
                if idx is not None:
                    torch.cuda.set_device(idx)
            maybe_clear_cuda_cache()
            gc.collect()
        except Exception as e:
            last_exc = e
            tried.append({"device": dev, "batch_size": None, "error": f"{type(e).__name__}: {e}"})
            continue

        for bsz in batch_candidates(batch_size, dev):
            try:
                report = evaluate_once(
                    model=model,
                    mapped_triples=mapped_triples,
                    additional_filter_triples=additional_filter_triples,
                    filtered=filtered,
                    batch_size=bsz,
                    slice_size=slice_size,
                    use_tqdm=use_tqdm,
                )
                report["eval_device"] = dev
                report["eval_batch_size_used"] = bsz
                report["eval_slice_size_used"] = slice_size
                report["fallback_attempts"] = tried
                return report
            except RuntimeError as e:
                last_exc = e
                tried.append({"device": dev, "batch_size": bsz, "error": f"{type(e).__name__}: {e}"})
                maybe_clear_cuda_cache()
                gc.collect()
                if is_cuda_oom(e):
                    continue
                raise
            except Exception as e:
                last_exc = e
                tried.append({"device": dev, "batch_size": bsz, "error": f"{type(e).__name__}: {e}"})
                maybe_clear_cuda_cache()
                gc.collect()
                raise

    if last_exc is None:
        raise RuntimeError("Evaluation failed for unknown reason.")
    raise last_exc


# ============================================================
# Worker
# ============================================================

def run_one_candidate(
    *,
    cfg: dict[str, Any],
    run_idx: int,
    total_runs: int,
    args_dict: dict[str, Any],
    device_str: str,
) -> None:
    args_output_dir = Path(args_dict["output_dir"])
    processed_dir = Path(args_dict["processed_dir"])
    dataset_name = args_dict["dataset_name"] or processed_dir.name
    filtered_eval = bool(args_dict["filtered_eval"])
    eval_batch_size = args_dict["eval_batch_size"]
    eval_slice_size = args_dict["eval_slice_size"]
    use_tqdm = not bool(args_dict["no_progress"])
    allow_cpu_eval_fallback = bool(args_dict["allow_cpu_eval_fallback"])

    run_name = f"{run_idx:02d}_{cfg['name']}"
    run_dir = args_output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "training_loop.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name

    print(f"\n=== [{run_idx}/{total_runs}] {run_name} on {device_str} ===", flush=True)

    write_status(
        run_dir,
        status="starting",
        run_name=run_name,
        device=device_str,
        dataset_name=dataset_name,
        config=cfg,
        checkpoint_path=str(checkpoint_path),
        checkpoint_exists=checkpoint_path.exists(),
    )

    save_json(
        run_dir / "run_config.json",
        {
            "run_name": run_name,
            "dataset_name": dataset_name,
            "processed_dir": str(processed_dir),
            "device": device_str,
            "config": cfg,
            "seed": int(args_dict["seed"]),
            "filtered_eval": filtered_eval,
            "eval_batch_size": eval_batch_size,
            "eval_slice_size": eval_slice_size,
            "checkpoint_frequency_minutes": args_dict["checkpoint_frequency_minutes"],
            "checkpoint_on_failure": bool(args_dict["checkpoint_on_failure"]),
            "saved_at_utc": now_iso_utc(),
        },
    )

    idx = parse_device_index(device_str)
    if idx is not None and torch.cuda.is_available():
        torch.cuda.set_device(idx)

    set_all_seeds(int(args_dict["seed"]))

    train_tf, valid_tf, test_tf, stats, sidecars = load_dataset(
        processed_dir,
        create_inverse_triples=bool(cfg["create_inverse_triples"]),
    )

    started = perf_now()

    try:
        negative_sampler_cls = resolve_negative_sampler(cfg["negative_sampler"])

        write_status(
            run_dir,
            status="training_preparing",
            run_name=run_name,
            device=device_str,
            dataset_name=dataset_name,
            dataset_stats=asdict(stats),
            config=cfg,
            checkpoint_exists=checkpoint_path.exists(),
        )

        pipe_kwargs: dict[str, Any] = dict(
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
                "checkpoint_name": checkpoint_name,
                "checkpoint_directory": checkpoint_dir,
                "checkpoint_frequency": args_dict["checkpoint_frequency_minutes"],
                "checkpoint_on_failure": bool(args_dict["checkpoint_on_failure"]),
            },
            evaluator="RankBasedEvaluator",
            evaluator_kwargs={"filtered": filtered_eval},
            evaluation_kwargs={
                **({"batch_size": eval_batch_size} if eval_batch_size is not None else {}),
                **({"slice_size": eval_slice_size} if eval_slice_size is not None else {}),
                "use_tqdm": use_tqdm,
            },
            evaluation_fallback=True,
            use_testing_data=False,
            negative_sampler=negative_sampler_cls,
            negative_sampler_kwargs={
                **cfg["negative_sampler_kwargs"],
                "num_negs_per_pos": cfg["num_negs"],
            },
            random_seed=int(args_dict["seed"]),
            device=device_str,
            use_tqdm=use_tqdm,
        )

        if cfg.get("use_early_stopper", True):
            pipe_kwargs["stopper"] = "early"
            pipe_kwargs["stopper_kwargs"] = {
                "frequency": args_dict["stop_frequency"],
                "patience": args_dict["stop_patience"],
                "relative_delta": args_dict["stop_relative_delta"],
            }

        if cfg["loss"] is not None:
            pipe_kwargs["loss"] = cfg["loss"]
            pipe_kwargs["loss_kwargs"] = cfg["loss_kwargs"]

        if cfg["regularizer"] is not None:
            pipe_kwargs["regularizer"] = cfg["regularizer"]
            pipe_kwargs["regularizer_kwargs"] = cfg["regularizer_kwargs"]

        write_status(
            run_dir,
            status="training_running",
            run_name=run_name,
            device=device_str,
            dataset_name=dataset_name,
            config=cfg,
            checkpoint_exists=checkpoint_path.exists(),
        )

        result = pipeline(**pipe_kwargs)

        losses = [float(x) for x in (getattr(result, "losses", None) or [])]

        write_status(
            run_dir,
            status="training_finished",
            run_name=run_name,
            device=device_str,
            dataset_name=dataset_name,
            losses_count=len(losses),
            checkpoint_exists=checkpoint_path.exists(),
        )

        serializable_cfg = {**cfg}

        checkpoint_payload = {
            "model_state_dict": result.model.state_dict(),
            "model_name": cfg["model"],
            "model_kwargs": model_kwargs_for(cfg["model"], cfg["embedding_dim"]),
            "entity_to_id": dict(train_tf.entity_to_id),
            "relation_to_id": dict(train_tf.relation_to_id),
            "config": serializable_cfg,
            "create_inverse_triples": bool(cfg["create_inverse_triples"]),
            "dataset_name": dataset_name,
            "processed_dir": str(processed_dir),
            "dataset_stats": asdict(stats),
            "saved_at_utc": now_iso_utc(),
            "pykeen_training_checkpoint_path": str(checkpoint_path),
        }
        torch.save(checkpoint_payload, run_dir / "base_model_checkpoint.pt")

        write_status(
            run_dir,
            status="validation_running",
            run_name=run_name,
            device=device_str,
            dataset_name=dataset_name,
        )

        valid_report = evaluate_with_fallback(
            model=result.model,
            mapped_triples=valid_tf.mapped_triples,
            additional_filter_triples=[train_tf.mapped_triples] if filtered_eval else [],
            filtered=filtered_eval,
            batch_size=eval_batch_size,
            slice_size=eval_slice_size,
            preferred_device=device_str,
            fallback_to_cpu=allow_cpu_eval_fallback,
            use_tqdm=use_tqdm,
        )
        maybe_clear_cuda_cache()
        gc.collect()

        write_status(
            run_dir,
            status="test_running",
            run_name=run_name,
            device=device_str,
            dataset_name=dataset_name,
            validation=valid_report,
        )

        test_report = evaluate_with_fallback(
            model=result.model,
            mapped_triples=test_tf.mapped_triples,
            additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples] if filtered_eval else [],
            filtered=filtered_eval,
            batch_size=eval_batch_size,
            slice_size=eval_slice_size,
            preferred_device=device_str,
            fallback_to_cpu=allow_cpu_eval_fallback,
            use_tqdm=use_tqdm,
        )
        maybe_clear_cuda_cache()
        gc.collect()

        elapsed = perf_now() - started

        run_summary = {
            "status": "ok",
            "run_name": run_name,
            "dataset_name": dataset_name,
            "processed_dir": str(processed_dir),
            "config": serializable_cfg,
            "dataset_stats": asdict(stats),
            "dataset_sidecars": sidecars,
            "filtered_eval": filtered_eval,
            "seed": int(args_dict["seed"]),
            "device": device_str,
            "validation": valid_report,
            "test": test_report,
            "training_losses": losses,
            "train_elapsed_seconds": round(elapsed, 6),
            "train_elapsed_human": format_seconds(elapsed),
            "stopped_epoch_guess": len(losses) if losses else None,
            "pykeen_training_checkpoint_path": str(checkpoint_path),
            "checkpoint_exists": checkpoint_path.exists(),
            "saved_at_utc": now_iso_utc(),
        }

        save_json(run_dir / "summary.json", run_summary)
        save_pykeen_artifacts_safely(result, run_dir / "pykeen_artifacts")

        write_status(
            run_dir,
            status="finished_ok",
            run_name=run_name,
            device=device_str,
            dataset_name=dataset_name,
            validation=valid_report,
            test=test_report,
            elapsed_human=format_seconds(elapsed),
        )

        print(
            f"[OK] {run_name} on {device_str} | "
            f"VALID MRR={valid_report['mrr']:.6f} | VALID H@10={valid_report['hits@10']:.6f} | "
            f"TEST MRR={test_report['mrr']:.6f} | TEST H@10={test_report['hits@10']:.6f}",
            flush=True,
        )

    except Exception as e:
        elapsed = perf_now() - started
        failed = {
            "status": "failed",
            "run_name": run_name,
            "dataset_name": dataset_name,
            "processed_dir": str(processed_dir),
            "config": cfg,
            "device": device_str,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "train_elapsed_seconds": round(elapsed, 6),
            "train_elapsed_human": format_seconds(elapsed),
            "pykeen_training_checkpoint_path": str(checkpoint_path),
            "checkpoint_exists": checkpoint_path.exists(),
            "saved_at_utc": now_iso_utc(),
        }
        save_json(run_dir / "summary.json", failed)
        write_status(
            run_dir,
            status="failed",
            run_name=run_name,
            device=device_str,
            dataset_name=dataset_name,
            error=failed["error"],
            train_elapsed_human=failed["train_elapsed_human"],
            checkpoint_exists=checkpoint_path.exists(),
        )
        print(f"[FAILED] {run_name} on {device_str}: {failed['error']}", flush=True)
    finally:
        maybe_clear_cuda_cache()
        gc.collect()


def worker_loop(task_queue: mp.Queue, args_dict: dict[str, Any], device_str: str, total_runs: int) -> None:
    if device_str.startswith("cuda"):
        idx = parse_device_index(device_str)
        if idx is not None and torch.cuda.is_available():
            torch.cuda.set_device(idx)

    while True:
        try:
            item = task_queue.get(timeout=3)
        except Empty:
            break

        if item is None:
            break

        run_idx, cfg = item
        run_one_candidate(
            cfg=cfg,
            run_idx=run_idx,
            total_runs=total_runs,
            args_dict=args_dict,
            device_str=device_str,
        )


# ============================================================
# Merge
# ============================================================

def merge_outputs(
    *,
    output_dir: Path,
    dataset_name: str,
    processed_dir: Path,
    portfolio_name: str,
    seed: int,
    devices: list[str],
    filtered_eval: bool,
    stats0: DatasetStats,
    sidecars0: dict[str, Any],
    portfolio: list[dict[str, Any]],
    started_at_utc: str,
    started_wall: float,
) -> None:
    all_runs: list[dict[str, Any]] = []
    best_run: Optional[dict[str, Any]] = None
    best_run_dir: Optional[Path] = None

    for i, cfg in enumerate(portfolio, start=1):
        run_name = f"{i:02d}_{cfg['name']}"
        summary_path = output_dir / run_name / "summary.json"

        if not summary_path.exists():
            failed = {
                "status": "failed",
                "run_name": run_name,
                "dataset_name": dataset_name,
                "processed_dir": str(processed_dir),
                "config": cfg,
                "error": "Missing summary.json (worker crash or external interruption).",
                "saved_at_utc": now_iso_utc(),
            }
            all_runs.append(failed)
            save_json(output_dir / run_name / "summary.json", failed)
            continue

        r = load_json(summary_path)
        all_runs.append(r)

        if r.get("status") != "ok":
            continue

        cur_mrr = r.get("validation", {}).get("mrr")
        best_mrr = None if best_run is None else best_run.get("validation", {}).get("mrr")

        if best_run is None or (
            cur_mrr is not None and (best_mrr is None or cur_mrr > best_mrr)
        ):
            best_run = r
            best_run_dir = output_dir / run_name

    leaderboard = []
    for r in all_runs:
        if r.get("status") != "ok":
            continue
        leaderboard.append(
            {
                "run_name": r["run_name"],
                "model": r["config"]["model"],
                "embedding_dim": r["config"]["embedding_dim"],
                "negative_sampler": str(r["config"]["negative_sampler"]),
                "validation_mrr": r["validation"]["mrr"],
                "validation_hits@10": r["validation"]["hits@10"],
                "test_mrr": r["test"]["mrr"],
                "test_hits@10": r["test"]["hits@10"],
                "device": r.get("device"),
                "stopped_epoch_guess": r.get("stopped_epoch_guess"),
                "checkpoint_exists": r.get("checkpoint_exists"),
            }
        )

    leaderboard.sort(
        key=lambda x: x["validation_mrr"] if x["validation_mrr"] is not None else float("-inf"),
        reverse=True,
    )

    total_elapsed = perf_now() - started_wall
    overview = {
        "dataset_name": dataset_name,
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "portfolio": portfolio_name,
        "seed": seed,
        "devices": devices,
        "filtered_eval": filtered_eval,
        "started_at_utc": started_at_utc,
        "finished_at_utc": now_iso_utc(),
        "total_elapsed_seconds": round(total_elapsed, 6),
        "total_elapsed_human": format_seconds(total_elapsed),
        "dataset_stats": asdict(stats0),
        "dataset_sidecars": sidecars0,
        "leaderboard": leaderboard,
        "best_run": best_run,
    }
    save_json(output_dir / "leaderboard.json", overview)

    if best_run is None or best_run_dir is None:
        raise SystemExit("No successful run completed.")

    best_dir = output_dir / "best_model"
    copy_best_artifacts(best_run_dir, best_dir, best_run)

    print("\n=== BEST RUN ===", flush=True)
    print(
        json.dumps(
            {
                "run_name": best_run["run_name"],
                "validation_mrr": best_run["validation"]["mrr"],
                "validation_hits@10": best_run["validation"]["hits@10"],
                "test_mrr": best_run["test"]["mrr"],
                "test_hits@10": best_run["test"]["hits@10"],
                "device": best_run.get("device"),
                "stopped_epoch_guess": best_run.get("stopped_epoch_guess"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-GPU schema-aware KGC portfolio trainer with resumeable PyKEEN checkpoints."
    )
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--portfolio", type=str, default="small", choices=["small", "medium", "large"])
    p.add_argument("--devices", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    # evaluation
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--eval-slice-size", type=int, default=None)
    p.add_argument("--filtered-eval", action="store_true")
    p.add_argument("--allow-cpu-eval-fallback", action="store_true")

    # stopper
    p.add_argument("--stop-frequency", type=int, default=8)
    p.add_argument("--stop-patience", type=int, default=2)
    p.add_argument("--stop-relative-delta", type=float, default=0.002)

    # checkpoint / incremental training
    p.add_argument("--checkpoint-frequency-minutes", type=int, default=30)
    p.add_argument("--checkpoint-on-failure", action="store_true")

    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started_wall = perf_now()
    started_at_utc = now_iso_utc()

    train_tf0, valid_tf0, test_tf0, stats0, sidecars0 = load_dataset(
        args.processed_dir,
        create_inverse_triples=True,
    )
    del train_tf0, valid_tf0, test_tf0
    gc.collect()

    save_json(args.output_dir / "dataset_stats.json", asdict(stats0))
    save_json(args.output_dir / "dataset_sidecars.json", sidecars0)

    portfolio = choose_portfolio(stats0, args.portfolio)
    if args.max_runs is not None:
        portfolio = portfolio[: args.max_runs]

    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    if not devices:
        raise SystemExit("No devices provided.")
    if any(d.startswith("cuda") for d in devices) and not torch.cuda.is_available():
        raise SystemExit("CUDA devices requested but torch.cuda.is_available() is False.")

    save_json(
        args.output_dir / "run_plan.json",
        {
            "dataset_name": args.dataset_name or args.processed_dir.name,
            "processed_dir": str(args.processed_dir),
            "output_dir": str(args.output_dir),
            "stats": asdict(stats0),
            "portfolio_name": args.portfolio,
            "portfolio_size": len(portfolio),
            "devices": devices,
            "seed": args.seed,
            "filtered_eval": args.filtered_eval,
            "eval_batch_size": args.eval_batch_size,
            "eval_slice_size": args.eval_slice_size,
            "allow_cpu_eval_fallback": args.allow_cpu_eval_fallback,
            "stop_frequency": args.stop_frequency,
            "stop_patience": args.stop_patience,
            "stop_relative_delta": args.stop_relative_delta,
            "checkpoint_frequency_minutes": args.checkpoint_frequency_minutes,
            "checkpoint_on_failure": args.checkpoint_on_failure,
            "started_at_utc": started_at_utc,
            "portfolio": portfolio,
        },
    )

    task_queue: mp.Queue = mp.Queue()
    for i, cfg in enumerate(portfolio, start=1):
        task_queue.put((i, cfg))
    for _ in devices:
        task_queue.put(None)

    args_dict = {
        "processed_dir": str(args.processed_dir),
        "output_dir": str(args.output_dir),
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "filtered_eval": args.filtered_eval,
        "eval_batch_size": args.eval_batch_size,
        "eval_slice_size": args.eval_slice_size,
        "stop_frequency": args.stop_frequency,
        "stop_patience": args.stop_patience,
        "stop_relative_delta": args.stop_relative_delta,
        "allow_cpu_eval_fallback": args.allow_cpu_eval_fallback,
        "checkpoint_frequency_minutes": args.checkpoint_frequency_minutes,
        "checkpoint_on_failure": args.checkpoint_on_failure,
        "no_progress": args.no_progress,
    }

    procs: list[mp.Process] = []
    total_runs = len(portfolio)

    for device_str in devices:
        p = mp.Process(
            target=worker_loop,
            args=(task_queue, args_dict, device_str, total_runs),
            daemon=False,
        )
        p.start()
        procs.append(p)

    bad_exit = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            bad_exit = True

    merge_outputs(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name or args.processed_dir.name,
        processed_dir=args.processed_dir,
        portfolio_name=args.portfolio,
        seed=args.seed,
        devices=devices,
        filtered_eval=args.filtered_eval,
        stats0=stats0,
        sidecars0=sidecars0,
        portfolio=portfolio,
        started_at_utc=started_at_utc,
        started_wall=started_wall,
    )

    if bad_exit:
        raise SystemExit("At least one worker exited abnormally. Check per-run summary.json files and job logs.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()