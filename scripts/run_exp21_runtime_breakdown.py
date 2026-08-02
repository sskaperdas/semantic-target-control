#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_exp12_quota_matched_baselines as base


def run_exp21(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = base.Logger(output_dir / "run.log")

    base.save_json_atomic(output_dir / "config.json", vars(args))
    base.set_all_seeds(args.seed)

    total_started = perf_counter()

    logger.log("[STEP 1/6] Loading checkpoint")
    checkpoint_path = base.resolve_checkpoint_path(args.run_dir)
    checkpoint_payload = base.load_checkpoint_payload(checkpoint_path)
    create_inverse_triples = bool(checkpoint_payload.get("create_inverse_triples", True))
    logger.log(f"[STEP 1/6] Done | checkpoint={checkpoint_path}")

    logger.log("[STEP 2/6] Loading dataset")
    bundle = base.load_dataset(args.processed_dir, create_inverse_triples=create_inverse_triples)
    split_tf = {"train": bundle.train_tf, "valid": bundle.valid_tf, "test": bundle.test_tf}[args.split]
    logger.log(
        f"[STEP 2/6] Done | entities={len(bundle.entity_to_id)} "
        f"relations={len(bundle.relation_to_id)} triples={split_tf.num_triples}"
    )

    logger.log("[STEP 3/6] Loading ontology")
    ontology = base.load_ontology_bundle(args.processed_dir, bundle.entity_to_id, bundle.relation_to_id)
    use_disjoint = bool(args.use_disjoint and ontology.has_disjoint_pairs)
    logger.log(
        f"[STEP 3/6] Done | has_relation_constraints={ontology.has_relation_constraints} "
        f"use_disjoint={use_disjoint}"
    )

    logger.log("[STEP 4/6] Loading frozen model")
    model = base.load_model_from_payload(
        payload=checkpoint_payload,
        train_tf=bundle.train_tf,
        device=args.device,
        logger=logger,
    )
    logger.log("[STEP 4/6] Done")

    logger.log("[STEP 5/6] Building queries")
    allowed_query_ids = base.load_allowed_query_ids(args.query_id_file)
    queries = base.build_queries(
        mapped_triples=split_tf.mapped_triples,
        id_to_relation=bundle.id_to_relation,
        split_name=args.split,
        mode=args.mode,
        max_queries=args.max_queries,
        allowed_query_ids=allowed_query_ids,
    )

    if not queries:
        raise RuntimeError("No queries built.")

    logger.log(f"[STEP 5/6] Done | queries={len(queries)}")

    score_topm_seconds = 0.0
    semantic_eval_seconds = 0.0
    optq_control_seconds = 0.0
    cleanup_seconds = 0.0

    num_queries = 0
    num_feasible = 0
    effective_top_m_global = None

    logger.log("[STEP 6/6] Timing scoring / semantics / OptQ control")

    pbar = tqdm(
        range(0, len(queries), args.query_batch_size),
        desc="EXP-21 batches",
        unit="batch",
    )

    for batch_start in pbar:
        batch_end = min(batch_start + args.query_batch_size, len(queries))
        batch_queries = queries[batch_start:batch_end]

        t0 = perf_counter()
        scores = base.score_batch(model, batch_queries, args.device)
        num_candidates = scores.shape[1]
        effective_top_m = min(args.top_m, num_candidates)
        if effective_top_m < args.top_k:
            raise RuntimeError(f"effective_top_m={effective_top_m} < top_k={args.top_k}")

        topm_scores, topm_indices = torch.topk(
            scores,
            k=effective_top_m,
            dim=1,
            largest=True,
            sorted=True,
        )
        score_topm_seconds += perf_counter() - t0

        if effective_top_m_global is None:
            effective_top_m_global = effective_top_m
            logger.log(
                f"[STEP 6/6] effective_top_m={effective_top_m_global} "
                f"num_candidates={num_candidates}"
            )

        for row_idx, qitem in enumerate(batch_queries):
            cand_ids = topm_indices[row_idx].detach().cpu().tolist()
            cand_scores = topm_scores[row_idx].detach().cpu().tolist()

            t_sem = perf_counter()
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
                energies.append(st.energy)
            semantic_eval_seconds += perf_counter() - t_sem

            t_ctrl = perf_counter()
            feasible, lambda_star, _tau_q = base.compute_lambda_star_quota(
                base_scores=cand_scores,
                energies=energies,
                k=args.top_k,
                q=args.quota,
            )
            if feasible:
                _ranked = base.stable_rerank_with_lambda(
                    cand_ids=cand_ids,
                    base_scores=cand_scores,
                    energies=energies,
                    lam=lambda_star,
                )
                num_feasible += 1
            optq_control_seconds += perf_counter() - t_ctrl

            num_queries += 1

        t_clean = perf_counter()
        del scores, topm_scores, topm_indices
        base.maybe_clear_cuda_cache()
        cleanup_seconds += perf_counter() - t_clean

        pbar.set_postfix({"queries": num_queries, "feas": num_feasible})

        batch_number = (batch_start // args.query_batch_size) + 1
        if (batch_number % args.log_every) == 0 or batch_end == len(queries):
            logger.log(
                f"[STEP 6/6] queries={num_queries} feasible={num_feasible} "
                f"score={score_topm_seconds:.2f}s sem={semantic_eval_seconds:.2f}s "
                f"ctrl={optq_control_seconds:.2f}s"
            )

    total_seconds = perf_counter() - total_started
    measured_seconds = score_topm_seconds + semantic_eval_seconds + optq_control_seconds + cleanup_seconds
    other_seconds = max(0.0, total_seconds - measured_seconds)

    total_candidates = num_queries * int(effective_top_m_global)

    row = {
        "dataset_name": args.dataset_name,
        "num_queries": num_queries,
        "num_feasible": num_feasible,
        "feasible_rate": num_feasible / num_queries if num_queries else "",
        "top_m": effective_top_m_global,
        "top_k": args.top_k,
        "quota": args.quota,
        "total_candidates_evaluated": total_candidates,
        "total_seconds": total_seconds,
        "score_topm_seconds": score_topm_seconds,
        "semantic_eval_seconds": semantic_eval_seconds,
        "optq_control_seconds": optq_control_seconds,
        "cleanup_seconds": cleanup_seconds,
        "other_seconds": other_seconds,
        "queries_per_second": num_queries / total_seconds if total_seconds > 0 else "",
        "candidates_per_second": total_candidates / total_seconds if total_seconds > 0 else "",
        "score_topm_pct": score_topm_seconds / total_seconds if total_seconds > 0 else "",
        "semantic_eval_pct": semantic_eval_seconds / total_seconds if total_seconds > 0 else "",
        "optq_control_pct": optq_control_seconds / total_seconds if total_seconds > 0 else "",
    }

    out_csv = output_dir / "runtime_breakdown.csv"
    cols = list(row.keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerow(row)

    base.save_json_atomic(output_dir / "summary.json", row)

    logger.log(f"[DONE] wrote {out_csv}")
    logger.log(
        f"[DONE] total={total_seconds:.2f}s score={score_topm_seconds:.2f}s "
        f"semantic={semantic_eval_seconds:.2f}s optq={optq_control_seconds:.2f}s"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="EXP-21: Runtime decomposition for scoring, semantic evaluation, and OptQ control."
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
    run_exp21(parse_args())


if __name__ == "__main__":
    main()
