#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_exp12_quota_matched_baselines as base


METHODS = ["base", "optq", "minswap", "hardval"]

METRICS = [
    "hit10",
    "mrr10",
    "viol_at_k",
    "adm_at_k",
    "unknown_at_k",
    "pres_at_k",
    "shift_at_k",
]


def eval_target_metrics(topk: list[int], target_id: int) -> dict[str, float]:
    if target_id in topk:
        rank = topk.index(target_id) + 1
        return {
            "hit10": 1.0,
            "mrr10": 1.0 / rank,
        }
    return {
        "hit10": 0.0,
        "mrr10": 0.0,
    }


def init_collectors(scopes: list[str]) -> dict[str, dict[str, dict[str, list[float]]]]:
    return {
        scope: {
            method: {metric: [] for metric in METRICS}
            for method in METHODS
        }
        for scope in scopes
    }


def bootstrap_ci(values, *, rng: np.random.Generator, samples: int) -> tuple[float, float | None, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return float("nan"), None, None

    mean = float(np.mean(arr))

    if samples <= 0 or n < 2:
        return mean, None, None

    boot = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(arr[idx]))

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def add_method_values(
    collector: dict[str, list[float]],
    *,
    topk: list[int],
    target_id: int,
    semantic_metrics: dict[str, float],
) -> None:
    target_metrics = eval_target_metrics(topk, target_id)

    collector["hit10"].append(target_metrics["hit10"])
    collector["mrr10"].append(target_metrics["mrr10"])
    collector["viol_at_k"].append(float(semantic_metrics["viol_at_k"]))
    collector["adm_at_k"].append(float(semantic_metrics["adm_at_k"]))
    collector["unknown_at_k"].append(float(semantic_metrics["unknown_at_k"]))
    collector["pres_at_k"].append(float(semantic_metrics["pres_at_k"]))
    collector["shift_at_k"].append(float(semantic_metrics["shift_at_k"]))


def run_exp19(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = base.Logger(output_dir / "run.log")

    base.save_json_atomic(output_dir / "config.json", vars(args))
    base.set_all_seeds(args.seed)

    started_total = base.perf_now()

    logger.log("[STEP 1/8] Resolving checkpoint and loading payload")
    checkpoint_path = base.resolve_checkpoint_path(args.run_dir)
    checkpoint_payload = base.load_checkpoint_payload(checkpoint_path)
    create_inverse_triples = bool(checkpoint_payload.get("create_inverse_triples", True))
    logger.log(
        f"[STEP 1/8] Done | checkpoint={checkpoint_path} "
        f"create_inverse_triples={create_inverse_triples}"
    )

    logger.log("[STEP 2/8] Loading dataset")
    bundle = base.load_dataset(args.processed_dir, create_inverse_triples=create_inverse_triples)
    split_tf = {"train": bundle.train_tf, "valid": bundle.valid_tf, "test": bundle.test_tf}[args.split]
    logger.log(
        f"[STEP 2/8] Done | entities={len(bundle.entity_to_id)} "
        f"relations={len(bundle.relation_to_id)} triples={split_tf.num_triples}"
    )

    logger.log("[STEP 3/8] Loading ontology")
    ontology = base.load_ontology_bundle(args.processed_dir, bundle.entity_to_id, bundle.relation_to_id)
    use_disjoint = bool(args.use_disjoint and ontology.has_disjoint_pairs)
    logger.log(
        f"[STEP 3/8] Done | has_entity_types={ontology.has_entity_types} "
        f"has_relation_constraints={ontology.has_relation_constraints} "
        f"has_disjoint_pairs={ontology.has_disjoint_pairs} use_disjoint={use_disjoint}"
    )

    logger.log("[STEP 4/8] Loading frozen model")
    model = base.load_model_from_payload(
        payload=checkpoint_payload,
        train_tf=bundle.train_tf,
        device=args.device,
        logger=logger,
    )
    logger.log("[STEP 4/8] Done")

    logger.log("[STEP 5/8] Building query list")
    allowed_query_ids = base.load_allowed_query_ids(args.query_id_file)
    queries = base.build_queries(
        mapped_triples=split_tf.mapped_triples,
        id_to_relation=bundle.id_to_relation,
        split_name=args.split,
        mode=args.mode,
        max_queries=args.max_queries,
        allowed_query_ids=allowed_query_ids,
    )
    total_queries_all = len(queries)
    if total_queries_all == 0:
        raise RuntimeError("No queries were built. Check split/mode/max_queries/query-id-file.")

    logger.log(f"[STEP 5/8] Done | total_candidate_queries={total_queries_all}")

    if args.quota <= 0 or args.quota > args.top_k:
        raise ValueError(f"quota must be in [1, top_k], got quota={args.quota}, top_k={args.top_k}")

    scopes = list(args.summary_scopes)
    collectors = init_collectors(scopes)

    counts = {
        scope: {
            "num_queries": 0,
            "num_feasible": 0,
            "target_in_topm": 0,
        }
        for scope in scopes
    }

    progress_json_path = output_dir / "progress.json"
    ci_csv_path = output_dir / "bootstrap_ci.csv"
    summary_json_path = output_dir / "summary.json"

    logger.log(
        f"[STEP 6/8] Running per-query evaluation | "
        f"quota={args.quota} top_m={args.top_m} scopes={scopes}"
    )

    processed = 0
    effective_top_m_global = None
    step_started = base.perf_now()

    pbar = tqdm(
        range(0, total_queries_all, args.query_batch_size),
        desc="EXP-19 batches",
        unit="batch",
    )

    for batch_start in pbar:
        batch_end = min(batch_start + args.query_batch_size, total_queries_all)
        batch_queries = queries[batch_start:batch_end]

        scores = base.score_batch(model, batch_queries, args.device)
        num_candidates = scores.shape[1]
        effective_top_m = min(args.top_m, num_candidates)

        if effective_top_m < args.top_k:
            raise RuntimeError(
                f"effective_top_m={effective_top_m} is smaller than top_k={args.top_k}"
            )

        if effective_top_m_global is None:
            effective_top_m_global = effective_top_m
            logger.log(
                f"[STEP 6/8] Using effective_top_m={effective_top_m_global} "
                f"(requested top_m={args.top_m}, num_candidates={num_candidates})"
            )

        topm_scores, topm_indices = torch.topk(
            scores,
            k=effective_top_m,
            dim=1,
            largest=True,
            sorted=True,
        )

        for row_idx, qitem in enumerate(batch_queries):
            cand_ids = topm_indices[row_idx].detach().cpu().tolist()
            cand_scores = topm_scores[row_idx].detach().cpu().tolist()

            target_id = qitem.tail_id if qitem.mode == "tail" else qitem.head_id
            target_in_topm = target_id in set(cand_ids)

            statuses = []
            energies = []

            for cand_id in cand_ids:
                if qitem.mode == "tail":
                    head_id, rel_id, tail_id = qitem.head_id, qitem.rel_id, cand_id
                else:
                    head_id, rel_id, tail_id = cand_id, qitem.rel_id, qitem.tail_id

                st = base.evaluate_candidate_semantics(
                    head_id=head_id,
                    rel_id=rel_id,
                    tail_id=tail_id,
                    ontology=ontology,
                    check_policy=args.check_policy,
                    use_domain=args.use_domain,
                    use_range=args.use_range,
                    use_disjoint=use_disjoint,
                    unknown_penalty=args.unknown_penalty,
                    binary_like=args.binary_like,
                )
                statuses.append(st)
                energies.append(st.energy)

            status_by_id = {cid: st for cid, st in zip(cand_ids, statuses)}
            checkable_by_id = {cid: st.checkable for cid, st in status_by_id.items()}
            admissible_by_id = {cid: st.admissible for cid, st in status_by_id.items()}
            violated_by_id = {cid: st.violated for cid, st in status_by_id.items()}
            unknown_by_id = {cid: st.unknown for cid, st in status_by_id.items()}

            base_topk = cand_ids[:args.top_k]
            base_rank_map = {cid: i + 1 for i, cid in enumerate(cand_ids)}

            base_viol_present = any(violated_by_id.get(cid, False) for cid in base_topk)
            admissible_outside_topk = sum(
                1 for cid in cand_ids[args.top_k:] if admissible_by_id.get(cid, False)
            )
            blind_strict = base_viol_present and (admissible_outside_topk >= 1)

            active_scopes = []
            for scope in scopes:
                if scope == "full":
                    active_scopes.append(scope)
                elif scope == "blind_strict" and blind_strict:
                    active_scopes.append(scope)

            if not active_scopes:
                processed += 1
                continue

            feasible, lambda_star, tau_q = base.compute_lambda_star_quota(
                base_scores=cand_scores,
                energies=energies,
                k=args.top_k,
                q=args.quota,
            )

            # Deployment-style fallback: if the target quota is infeasible,
            # controlled methods leave the frozen ranking unchanged.
            method_topks: dict[str, list[int]] = {
                "base": base_topk,
                "optq": base_topk,
                "minswap": base_topk,
                "hardval": base_topk,
            }

            if feasible:
                optq_ranked = base.stable_rerank_with_lambda(
                    cand_ids=cand_ids,
                    base_scores=cand_scores,
                    energies=energies,
                    lam=lambda_star,
                )
                method_topks["optq"] = optq_ranked[:args.top_k]

                method_topks["minswap"] = base.rerank_quota_fill_minswap_to_k(
                    cand_ids=cand_ids,
                    admissible_by_id=admissible_by_id,
                    q=args.quota,
                    k=args.top_k,
                )

                method_topks["hardval"] = base.rerank_hard_validation_fill_to_k(
                    cand_ids=cand_ids,
                    checkable_by_id=checkable_by_id,
                    violated_by_id=violated_by_id,
                    k=args.top_k,
                )

            for scope in active_scopes:
                counts[scope]["num_queries"] += 1
                if feasible:
                    counts[scope]["num_feasible"] += 1
                if target_in_topm:
                    counts[scope]["target_in_topm"] += 1

                for method in METHODS:
                    topk = method_topks[method]

                    semantic_metrics = base.evaluate_returned_topk(
                        returned_topk=topk,
                        checkable_by_id=checkable_by_id,
                        admissible_by_id=admissible_by_id,
                        violated_by_id=violated_by_id,
                        unknown_by_id=unknown_by_id,
                        base_topk=base_topk,
                        base_rank_map=base_rank_map,
                        k=args.top_k,
                    )

                    add_method_values(
                        collectors[scope][method],
                        topk=topk,
                        target_id=target_id,
                        semantic_metrics=semantic_metrics,
                    )

            processed += 1

        elapsed = base.perf_now() - step_started
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total_queries_all - processed)
        eta_sec = remaining / rate if rate > 0 else None

        base.save_json_atomic(progress_json_path, {
            "status": "running",
            "updated_at_utc": base.now_iso_utc(),
            "processed_queries": processed,
            "total_queries": total_queries_all,
            "progress_fraction": processed / total_queries_all if total_queries_all else 0.0,
            "elapsed_seconds": elapsed,
            "elapsed_human": base.format_seconds(elapsed),
            "eta_seconds": eta_sec,
            "eta_human": base.format_seconds(eta_sec),
            "queries_per_second": rate,
            "effective_top_m": effective_top_m,
            "quota": args.quota,
        })

        pbar.set_postfix({"kept": processed, "q": args.quota})

        batch_number = (batch_start // args.query_batch_size) + 1
        if (batch_number % args.log_every) == 0 or batch_end == total_queries_all:
            logger.log(
                f"[STEP 6/8] kept={processed} elapsed={base.format_seconds(elapsed)} "
                f"eta={base.format_seconds(eta_sec)}"
            )

        del scores, topm_scores, topm_indices
        base.maybe_clear_cuda_cache()

    logger.log("[STEP 6/8] Done")
    logger.log("[STEP 7/8] Computing bootstrap CIs")

    rng = np.random.default_rng(args.bootstrap_seed)

    rows_out = []

    for scope in scopes:
        nq = counts[scope]["num_queries"]
        nf = counts[scope]["num_feasible"]
        target_n = counts[scope]["target_in_topm"]

        if nq == 0:
            continue

        meta = {
            "dataset_name": args.dataset_name,
            "scope": scope,
            "quota": args.quota,
            "num_queries": nq,
            "num_feasible": nf,
            "feasible_rate": nf / nq if nq else "",
            "target_in_topm_rate": target_n / nq if nq else "",
            "bootstrap_samples": args.bootstrap_samples,
        }

        for method in METHODS:
            for metric in METRICS:
                values = collectors[scope][method][metric]
                mean, lo, hi = bootstrap_ci(
                    values,
                    rng=rng,
                    samples=args.bootstrap_samples,
                )

                rows_out.append({
                    **meta,
                    "method": method,
                    "kind": "absolute",
                    "metric": metric,
                    "mean": mean,
                    "ci_low": "" if lo is None else lo,
                    "ci_high": "" if hi is None else hi,
                })

        for method in ["optq", "minswap", "hardval"]:
            for metric in METRICS:
                method_values = np.asarray(collectors[scope][method][metric], dtype=np.float64)
                base_values = np.asarray(collectors[scope]["base"][metric], dtype=np.float64)
                delta_values = method_values - base_values

                mean, lo, hi = bootstrap_ci(
                    delta_values,
                    rng=rng,
                    samples=args.bootstrap_samples,
                )

                rows_out.append({
                    **meta,
                    "method": method,
                    "kind": "delta_vs_base",
                    "metric": f"delta_{metric}",
                    "mean": mean,
                    "ci_low": "" if lo is None else lo,
                    "ci_high": "" if hi is None else hi,
                })

    with ci_csv_path.open("w", newline="", encoding="utf-8") as f:
        cols = [
            "dataset_name",
            "scope",
            "quota",
            "method",
            "kind",
            "metric",
            "num_queries",
            "num_feasible",
            "feasible_rate",
            "target_in_topm_rate",
            "bootstrap_samples",
            "mean",
            "ci_low",
            "ci_high",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows_out)

    total_elapsed = base.perf_now() - started_total

    base.save_json_atomic(summary_json_path, {
        "status": "done",
        "updated_at_utc": base.now_iso_utc(),
        "dataset_name": args.dataset_name,
        "quota": args.quota,
        "summary_scopes": scopes,
        "bootstrap_samples": args.bootstrap_samples,
        "elapsed_seconds": total_elapsed,
        "elapsed_human": base.format_seconds(total_elapsed),
        "artifacts": {
            "bootstrap_ci_csv": str(ci_csv_path),
            "progress_json": str(progress_json_path),
            "run_log": str(output_dir / "run.log"),
        },
    })

    logger.log("[STEP 8/8] Done")
    logger.log(
        f"[DONE] quota={args.quota} processed_queries={processed} "
        f"bootstrap_samples={args.bootstrap_samples} "
        f"elapsed={base.format_seconds(total_elapsed)}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="EXP-19: Bootstrap confidence intervals for semantic-control deltas."
    )

    ap.add_argument("--processed-dir", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--dataset-name", default="dataset")
    ap.add_argument("--split", choices=["train", "valid", "test"], default="test")
    ap.add_argument("--mode", choices=["tail", "head", "all"], default="all")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--top-m", type=int, default=5000)
    ap.add_argument("--quota", type=int, default=5)
    ap.add_argument("--summary-scopes", choices=["full", "blind_strict"], nargs="+", default=["full", "blind_strict"])
    ap.add_argument("--query-batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--check-policy", choices=["available_any", "available_all"], default="available_any")
    ap.add_argument("--use-domain", action="store_true")
    ap.add_argument("--use-range", action="store_true")
    ap.add_argument("--use-disjoint", action="store_true")
    ap.add_argument("--unknown-penalty", type=float, default=1.0)
    ap.add_argument("--binary-like", action="store_true")
    ap.add_argument("--query-id-file", type=Path, default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)

    return ap.parse_args()


def main() -> None:
    run_exp19(parse_args())


if __name__ == "__main__":
    main()
