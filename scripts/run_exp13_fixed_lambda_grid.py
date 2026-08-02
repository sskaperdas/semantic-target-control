#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_exp12_quota_matched_baselines as base


def mean_or_blank(total: float, n: int):
    return (total / n) if n > 0 else ""


def lambda_token(lam: float) -> str:
    s = f"{lam:g}"
    return s.replace("-", "m").replace(".", "p")


def init_agg() -> dict[str, Any]:
    return {
        "num_queries": 0,
        "num_feasible": 0,
        "metric_rows": 0,
        "quota_success": 0,
        "viol_sum": 0.0,
        "adm_sum": 0.0,
        "cov_sum": 0.0,
        "unknown_sum": 0.0,
        "pres_sum": 0.0,
        "shift_sum": 0.0,
        "adm_count_sum": 0.0,
        "lambda_star_sum": 0.0,
        "lambda_star_values": [],
    }


def add_metrics(
    agg: dict[str, Any],
    *,
    metrics: dict[str, float],
    adm_count: int,
    quota: int,
    lambda_star: float | None = None,
) -> None:
    agg["metric_rows"] += 1
    agg["viol_sum"] += metrics["viol_at_k"]
    agg["adm_sum"] += metrics["adm_at_k"]
    agg["cov_sum"] += metrics["cov_at_k"]
    agg["unknown_sum"] += metrics["unknown_at_k"]
    agg["pres_sum"] += metrics["pres_at_k"]
    agg["shift_sum"] += metrics["shift_at_k"]
    agg["adm_count_sum"] += adm_count
    if adm_count >= quota:
        agg["quota_success"] += 1
    if lambda_star is not None:
        agg["lambda_star_sum"] += lambda_star
        agg["lambda_star_values"].append(lambda_star)


def percentile(values: list[float], q: float):
    if not values:
        return ""
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def run_exp13(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = base.Logger(output_dir / "run.log")

    config = vars(args).copy()
    config["lambdas"] = list(args.lambdas)
    base.save_json_atomic(output_dir / "config.json", config)
    base.set_all_seeds(args.seed)

    started_total = base.perf_now()

    logger.log("[STEP 1/7] Resolving checkpoint and loading payload")
    checkpoint_path = base.resolve_checkpoint_path(args.run_dir)
    checkpoint_payload = base.load_checkpoint_payload(checkpoint_path)
    create_inverse_triples = bool(checkpoint_payload.get("create_inverse_triples", True))
    logger.log(
        f"[STEP 1/7] Done | checkpoint={checkpoint_path} "
        f"create_inverse_triples={create_inverse_triples}"
    )

    logger.log("[STEP 2/7] Loading dataset")
    bundle = base.load_dataset(args.processed_dir, create_inverse_triples=create_inverse_triples)
    split_tf = {"train": bundle.train_tf, "valid": bundle.valid_tf, "test": bundle.test_tf}[args.split]
    logger.log(
        f"[STEP 2/7] Done | entities={len(bundle.entity_to_id)} "
        f"relations={len(bundle.relation_to_id)} triples={split_tf.num_triples}"
    )

    logger.log("[STEP 3/7] Loading ontology")
    ontology = base.load_ontology_bundle(args.processed_dir, bundle.entity_to_id, bundle.relation_to_id)
    use_disjoint = bool(args.use_disjoint and ontology.has_disjoint_pairs)
    logger.log(
        f"[STEP 3/7] Done | has_entity_types={ontology.has_entity_types} "
        f"has_relation_constraints={ontology.has_relation_constraints} "
        f"has_disjoint_pairs={ontology.has_disjoint_pairs} use_disjoint={use_disjoint}"
    )

    logger.log("[STEP 4/7] Loading frozen model")
    model = base.load_model_from_payload(
        payload=checkpoint_payload,
        train_tf=bundle.train_tf,
        device=args.device,
        logger=logger,
    )
    logger.log("[STEP 4/7] Done")

    logger.log("[STEP 5/7] Building query list")
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
    logger.log(f"[STEP 5/7] Done | total_candidate_queries={total_queries_all}")

    if args.quota <= 0 or args.quota > args.top_k:
        raise ValueError(f"quota must be in [1, top_k], got quota={args.quota}, top_k={args.top_k}")

    lambdas = sorted(set(float(x) for x in args.lambdas))
    scopes = list(args.summary_scopes)

    summary_csv_path = output_dir / "fixed_lambda_summary.csv"
    progress_json_path = output_dir / "progress.json"
    summary_json_path = output_dir / "summary.json"

    # key: (scope, method, lambda_label)
    aggs: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(init_agg)

    logger.log(
        f"[STEP 6/7] Running EXP-13 fixed-lambda grid | "
        f"quota={args.quota} lambdas={lambdas} scopes={scopes}"
    )

    processed = 0
    effective_top_m_global = None
    step_started = base.perf_now()

    pbar = tqdm(
        range(0, total_queries_all, args.query_batch_size),
        desc="EXP-13 batches",
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
                f"[STEP 6/7] Using effective_top_m={effective_top_m_global} "
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
            admissible_in_topm = sum(1 for cid in cand_ids if admissible_by_id.get(cid, False))
            admissible_outside_topk = sum(
                1 for cid in cand_ids[args.top_k:] if admissible_by_id.get(cid, False)
            )

            is_blind = base_viol_present and (admissible_in_topm >= 1)
            blind_strict = base_viol_present and (admissible_outside_topk >= 1)

            active_scopes = []
            for scope in scopes:
                if scope == "full":
                    active_scopes.append(scope)
                elif scope == "blind" and is_blind:
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

            method_keys = [("optq", "adaptive")]
            method_keys.extend(("fixed_lambda", lambda_token(lam)) for lam in lambdas)

            for scope in active_scopes:
                for method, lam_label in method_keys:
                    agg = aggs[(scope, method, lam_label)]
                    agg["num_queries"] += 1
                    if feasible:
                        agg["num_feasible"] += 1

            if feasible:
                optq_ranked = base.stable_rerank_with_lambda(
                    cand_ids=cand_ids,
                    base_scores=cand_scores,
                    energies=energies,
                    lam=lambda_star,
                )
                optq_topk = optq_ranked[:args.top_k]
                optq_metrics = base.evaluate_returned_topk(
                    returned_topk=optq_topk,
                    checkable_by_id=checkable_by_id,
                    admissible_by_id=admissible_by_id,
                    violated_by_id=violated_by_id,
                    unknown_by_id=unknown_by_id,
                    base_topk=base_topk,
                    base_rank_map=base_rank_map,
                    k=args.top_k,
                )
                optq_adm_count = base.count_admissible_in_topk(
                    returned_topk=optq_topk,
                    admissible_by_id=admissible_by_id,
                )

                for scope in active_scopes:
                    add_metrics(
                        aggs[(scope, "optq", "adaptive")],
                        metrics=optq_metrics,
                        adm_count=optq_adm_count,
                        quota=args.quota,
                        lambda_star=lambda_star,
                    )

                for lam in lambdas:
                    ranked = base.stable_rerank_with_lambda(
                        cand_ids=cand_ids,
                        base_scores=cand_scores,
                        energies=energies,
                        lam=lam,
                    )
                    topk = ranked[:args.top_k]
                    metrics = base.evaluate_returned_topk(
                        returned_topk=topk,
                        checkable_by_id=checkable_by_id,
                        admissible_by_id=admissible_by_id,
                        violated_by_id=violated_by_id,
                        unknown_by_id=unknown_by_id,
                        base_topk=base_topk,
                        base_rank_map=base_rank_map,
                        k=args.top_k,
                    )
                    adm_count = base.count_admissible_in_topk(
                        returned_topk=topk,
                        admissible_by_id=admissible_by_id,
                    )

                    for scope in active_scopes:
                        add_metrics(
                            aggs[(scope, "fixed_lambda", lambda_token(lam))],
                            metrics=metrics,
                            adm_count=adm_count,
                            quota=args.quota,
                            lambda_star=None,
                        )

            processed += 1

        elapsed = base.perf_now() - step_started
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total_queries_all - processed)
        eta_sec = remaining / rate if rate > 0 else None

        progress_payload = {
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
        }
        base.save_json_atomic(progress_json_path, progress_payload)

        pbar.set_postfix({"kept": processed, "q": args.quota})

        batch_number = (batch_start // args.query_batch_size) + 1
        if (batch_number % args.log_every) == 0 or batch_end == total_queries_all:
            logger.log(
                f"[STEP 6/7] kept={processed} elapsed={base.format_seconds(elapsed)} "
                f"eta={base.format_seconds(eta_sec)}"
            )

        del scores, topm_scores, topm_indices
        base.maybe_clear_cuda_cache()

    logger.log("[STEP 6/7] Done")
    logger.log("[STEP 7/7] Writing summaries")

    out_cols = [
        "dataset_name",
        "scope",
        "quota",
        "method",
        "lambda_label",
        "lambda_value",
        "num_queries",
        "num_feasible",
        "feasible_rate",
        "metric_rows",
        "quota_success_rate_feasible",
        "viol_at_k",
        "adm_at_k",
        "cov_at_k",
        "unknown_at_k",
        "pres_at_k",
        "shift_at_k",
        "adm_count_topk",
        "lambda_star_mean",
        "lambda_star_p50",
        "lambda_star_p90",
    ]

    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_cols)

        for scope in scopes:
            # optq first
            ordered_keys = [(scope, "optq", "adaptive")]
            ordered_keys.extend((scope, "fixed_lambda", lambda_token(lam)) for lam in lambdas)

            for key in ordered_keys:
                agg = aggs[key]
                nq = agg["num_queries"]
                nf = agg["num_feasible"]
                mr = agg["metric_rows"]
                method, lam_label = key[1], key[2]
                lam_value = "adaptive" if method == "optq" else lam_label.replace("p", ".")

                writer.writerow([
                    args.dataset_name,
                    scope,
                    args.quota,
                    method,
                    lam_label,
                    lam_value,
                    int(nq),
                    int(nf),
                    (nf / nq) if nq > 0 else "",
                    int(mr),
                    (agg["quota_success"] / nf) if nf > 0 else "",
                    mean_or_blank(agg["viol_sum"], mr),
                    mean_or_blank(agg["adm_sum"], mr),
                    mean_or_blank(agg["cov_sum"], mr),
                    mean_or_blank(agg["unknown_sum"], mr),
                    mean_or_blank(agg["pres_sum"], mr),
                    mean_or_blank(agg["shift_sum"], mr),
                    mean_or_blank(agg["adm_count_sum"], mr),
                    mean_or_blank(agg["lambda_star_sum"], len(agg["lambda_star_values"])),
                    percentile(agg["lambda_star_values"], 0.5),
                    percentile(agg["lambda_star_values"], 0.9),
                ])

    total_elapsed = base.perf_now() - started_total
    summary = {
        "status": "done",
        "updated_at_utc": base.now_iso_utc(),
        "dataset_name": args.dataset_name,
        "quota": args.quota,
        "lambdas": lambdas,
        "summary_scopes": scopes,
        "elapsed_seconds": total_elapsed,
        "elapsed_human": base.format_seconds(total_elapsed),
        "artifacts": {
            "summary_csv": str(summary_csv_path),
            "progress_json": str(progress_json_path),
            "run_log": str(output_dir / "run.log"),
        },
    }
    base.save_json_atomic(summary_json_path, summary)

    logger.log("[STEP 7/7] Done")
    logger.log(
        f"[DONE] quota={args.quota} lambdas={lambdas} "
        f"processed_queries={processed} elapsed={base.format_seconds(total_elapsed)}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="EXP-13: Fixed global lambda grid versus query-specific OptQ."
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
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
    ap.add_argument("--summary-scopes", choices=["full", "blind", "blind_strict"], nargs="+", default=["full", "blind_strict"])
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

    return ap.parse_args()


def main() -> None:
    run_exp13(parse_args())


if __name__ == "__main__":
    main()
