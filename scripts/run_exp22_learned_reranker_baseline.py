from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

warnings.filterwarnings('ignore', message='X does not have valid feature names.*')

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_SKLEARN = True
except Exception:
    HistGradientBoostingClassifier = None
    HAS_SKLEARN = False

from run_exp12_quota_matched_baselines import (
    Logger,
    now_iso_utc,
    format_seconds,
    set_all_seeds,
    maybe_clear_cuda_cache,
    load_dataset,
    load_ontology_bundle,
    resolve_checkpoint_path,
    load_checkpoint_payload,
    load_model_from_payload,
    load_allowed_query_ids,
    build_queries,
    score_batch,
    evaluate_candidate_semantics,
    compute_lambda_star_quota,
    stable_rerank_with_lambda,
    evaluate_returned_topk,
    count_admissible_in_topk,
    save_json_atomic,
)

FEATURE_NAMES = [
    "base_score",
    "score_z",
    "score_gap_top",
    "score_gap_prev",
    "reciprocal_rank",
    "log_rank",
    "rank_pct",
    "is_base_topk",
    "semantic_energy",
    "is_checkable",
    "is_admissible",
    "is_violated",
    "is_unknown",
    "mode_tail",
    "mode_head",
    "rel_id",
]


def perf_now() -> float:
    return time.perf_counter()


def safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def sigmoid_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def split_tf(bundle: Any, split: str) -> Any:
    if split == "train":
        return bundle.train_tf
    if split == "valid":
        return bundle.valid_tf
    if split == "test":
        return bundle.test_tf
    raise ValueError(f"Unsupported split: {split}")


def build_split_queries(
    *,
    bundle: Any,
    split: str,
    mode: str,
    max_queries: Optional[int],
    allowed_query_ids: Optional[set[str]],
) -> list[Any]:
    tf = split_tf(bundle, split)
    return build_queries(
        mapped_triples=tf.mapped_triples,
        id_to_relation=bundle.id_to_relation,
        split_name=split,
        mode=mode,
        max_queries=max_queries,
        allowed_query_ids=allowed_query_ids,
    )


def status_for_candidate(
    *,
    qitem: Any,
    cand_id: int,
    ontology: Any,
    args: argparse.Namespace,
) -> Any:
    if qitem.mode == "tail":
        head_id, rel_id, tail_id = qitem.head_id, qitem.rel_id, cand_id
    else:
        head_id, rel_id, tail_id = cand_id, qitem.rel_id, qitem.tail_id

    return evaluate_candidate_semantics(
        head_id=head_id,
        rel_id=rel_id,
        tail_id=tail_id,
        ontology=ontology,
        check_policy=args.check_policy,
        use_domain=args.use_domain,
        use_range=args.use_range,
        use_disjoint=args.use_disjoint,
        unknown_penalty=args.unknown_penalty,
        binary_like=args.binary_like,
    )


def make_features(
    *,
    qitem: Any,
    cand_score: float,
    rank0: int,
    effective_top_m: int,
    top_score: float,
    prev_score: Optional[float],
    score_mean: float,
    score_std: float,
    status: Any,
    top_k: int,
) -> list[float]:
    rank1 = rank0 + 1
    score_z = 0.0 if score_std <= 1e-12 else (cand_score - score_mean) / score_std
    score_gap_top = top_score - cand_score
    score_gap_prev = 0.0 if prev_score is None else prev_score - cand_score

    return [
        float(cand_score),
        float(score_z),
        float(score_gap_top),
        float(score_gap_prev),
        float(1.0 / rank1),
        float(math.log1p(rank1)),
        float(rank1 / max(1, effective_top_m)),
        float(rank1 <= top_k),
        float(status.energy),
        float(status.checkable),
        float(status.admissible),
        float(status.violated),
        float(status.unknown),
        float(qitem.mode == "tail"),
        float(qitem.mode == "head"),
        float(qitem.rel_id),
    ]


def label_from_status(status: Any, label_policy: str) -> int:
    if label_policy == "admissible":
        return int(bool(status.admissible))
    if label_policy == "nonviolating_checkable":
        return int(bool(status.checkable and not status.violated))
    raise ValueError(f"Unsupported label_policy: {label_policy}")


def choose_train_indices(
    *,
    effective_top_m: int,
    top_k: int,
    sample_per_query: int,
    rng: random.Random,
) -> list[int]:
    if sample_per_query <= 0 or sample_per_query >= effective_top_m:
        return list(range(effective_top_m))

    chosen = set(range(min(top_k, effective_top_m)))

    for frac in [0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95]:
        idx = int(round(frac * (effective_top_m - 1)))
        if 0 <= idx < effective_top_m:
            chosen.add(idx)

    remaining = sample_per_query - len(chosen)
    if remaining > 0:
        pool = [i for i in range(effective_top_m) if i not in chosen]
        if remaining >= len(pool):
            chosen.update(pool)
        else:
            chosen.update(rng.sample(pool, remaining))

    return sorted(chosen)


def collect_training_matrix(
    *,
    model: Any,
    bundle: Any,
    ontology: Any,
    queries: list[Any],
    args: argparse.Namespace,
    logger: Logger,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    logger.log(f"[TRAIN] Collecting candidate-level samples from split={split_name} queries={len(queries)}")

    rng = random.Random(args.seed + 1009)
    xs: list[list[float]] = []
    ys: list[int] = []

    batch_starts = list(range(0, len(queries), args.query_batch_size))

    for batch_start in tqdm(batch_starts, desc=f"collect-{split_name}"):
        batch_queries = queries[batch_start: batch_start + args.query_batch_size]

        scores = score_batch(model=model, queries=batch_queries, device=args.device)
        effective_top_m = min(args.top_m, int(scores.shape[1]))
        if effective_top_m < args.top_k:
            raise ValueError(f"effective_top_m={effective_top_m} < top_k={args.top_k}")

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

            arr = np.asarray(cand_scores, dtype=np.float64)
            score_mean = float(arr.mean())
            score_std = float(arr.std())
            top_score = float(cand_scores[0])

            sample_indices = choose_train_indices(
                effective_top_m=effective_top_m,
                top_k=args.top_k,
                sample_per_query=args.candidate_sample_per_query,
                rng=rng,
            )

            for rank0 in sample_indices:
                cand_id = int(cand_ids[rank0])
                cand_score = float(cand_scores[rank0])
                prev_score = None if rank0 == 0 else float(cand_scores[rank0 - 1])

                st = status_for_candidate(
                    qitem=qitem,
                    cand_id=cand_id,
                    ontology=ontology,
                    args=args,
                )

                xs.append(
                    make_features(
                        qitem=qitem,
                        cand_score=cand_score,
                        rank0=rank0,
                        effective_top_m=effective_top_m,
                        top_score=top_score,
                        prev_score=prev_score,
                        score_mean=score_mean,
                        score_std=score_std,
                        status=st,
                        top_k=args.top_k,
                    )
                )
                ys.append(label_from_status(st, args.label_policy))

        del scores, topm_scores, topm_indices
        maybe_clear_cuda_cache()

    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int32)

    logger.log(f"[TRAIN] Collected X={X.shape} y_pos={int(y.sum())} y_neg={int((1-y).sum())}")

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"Training labels have only one class. positives={int(y.sum())}, n={len(y)}"
        )

    return X, y


def fit_learner(X: np.ndarray, y: np.ndarray, args: argparse.Namespace, logger: Logger) -> tuple[Any, str]:
    n = len(y)
    pos = int(y.sum())
    neg = int(n - pos)

    w_pos = n / max(1, 2 * pos)
    w_neg = n / max(1, 2 * neg)
    sample_weight = np.where(y == 1, w_pos, w_neg).astype(np.float32)

    if args.learner in ["auto", "lightgbm"] and HAS_LIGHTGBM:
        clf = LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbose=-1,
        )
        kind = "lightgbm"
    elif args.learner in ["auto", "sklearn_hgb"] and HAS_SKLEARN:
        clf = HistGradientBoostingClassifier(
            max_iter=args.n_estimators,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.num_leaves,
            random_state=args.seed,
        )
        kind = "sklearn_hgb"
    else:
        raise RuntimeError(
            "No supported learner available. Install lightgbm or sklearn, or set --learner accordingly."
        )

    logger.log(f"[TRAIN] Fitting learner={kind} n={n} pos={pos} neg={neg}")
    clf.fit(X, y, sample_weight=sample_weight)
    return clf, kind


def learner_signal(clf: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        classes = list(getattr(clf, "classes_", [0, 1]))
        if 1 in classes:
            idx = classes.index(1)
        else:
            idx = min(1, proba.shape[1] - 1)
        p = proba[:, idx]
        return sigmoid_logit(np.asarray(p, dtype=np.float64))

    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(X), dtype=np.float64)

    pred = np.asarray(clf.predict(X), dtype=np.float64)
    return pred


def factual_metrics(qitem: Any, returned_topk: list[int]) -> dict[str, float]:
    target = int(qitem.target_entity_id)
    if target in returned_topk:
        rank = returned_topk.index(target) + 1
    else:
        rank = None

    return {
        "hit1": float(rank is not None and rank <= 1),
        "hit3": float(rank is not None and rank <= 3),
        "hit10": float(rank is not None and rank <= 10),
        "mrr10": 0.0 if rank is None or rank > 10 else float(1.0 / rank),
        "target_rank_if_hit": float(rank) if rank is not None and rank <= 10 else math.nan,
    }


def new_agg() -> defaultdict[str, float]:
    return defaultdict(float)


def update_agg(
    *,
    agg: defaultdict[str, float],
    qitem: Any,
    returned_topk: list[int],
    base_topk: list[int],
    base_rank_map: dict[int, int],
    checkable_by_id: dict[int, bool],
    admissible_by_id: dict[int, bool],
    violated_by_id: dict[int, bool],
    unknown_by_id: dict[int, bool],
    feasible: bool,
    target_in_topm: bool,
    base_hit10: bool,
    top_k: int,
    quota: int,
) -> None:
    sem = evaluate_returned_topk(
        returned_topk=returned_topk,
        checkable_by_id=checkable_by_id,
        admissible_by_id=admissible_by_id,
        violated_by_id=violated_by_id,
        unknown_by_id=unknown_by_id,
        base_topk=base_topk,
        base_rank_map=base_rank_map,
        k=top_k,
    )
    fac = factual_metrics(qitem, returned_topk)

    agg["num_queries"] += 1
    agg["num_feasible"] += float(feasible)
    agg["target_in_topm"] += float(target_in_topm)

    if feasible:
        agg["quota_success_feasible_num"] += float(
            count_admissible_in_topk(
                returned_topk=returned_topk,
                admissible_by_id=admissible_by_id,
            ) >= quota
        )
        agg["quota_success_feasible_den"] += 1

    for key in ["hit1", "hit3", "hit10", "mrr10"]:
        agg[key] += fac[key]

    if not math.isnan(fac["target_rank_if_hit"]):
        agg["target_rank_sum_if_hit"] += fac["target_rank_if_hit"]
        agg["target_rank_count_if_hit"] += 1

    if base_hit10:
        agg["retention_den"] += 1
        agg["retention_num"] += fac["hit10"]

    for key in ["viol_at_k", "adm_at_k", "unknown_at_k", "pres_at_k", "shift_at_k"]:
        agg[key] += sem[key]


def scope_allowed(scope: str, *, is_blind: bool, blind_strict: bool) -> bool:
    if scope == "full":
        return True
    if scope == "blind":
        return is_blind
    if scope == "blind_strict":
        return blind_strict
    raise ValueError(f"Unsupported scope: {scope}")


def summarize_aggs(
    *,
    aggs: dict[tuple[str, str, Optional[float]], defaultdict[str, float]],
    dataset_name: str,
    quota: int,
    top_m: int,
    top_k: int,
    learner_kind: str,
) -> pd.DataFrame:
    rows = []
    for (scope, method, alpha), agg in sorted(aggs.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]))):
        n = int(agg["num_queries"])
        if n <= 0:
            continue

        row = {
            "dataset_name": dataset_name,
            "scope": scope,
            "quota": quota,
            "top_m": top_m,
            "top_k": top_k,
            "method": method,
            "learner_kind": learner_kind if method == "learned" else "",
            "alpha": alpha if alpha is not None else "",
            "num_queries": n,
            "num_feasible": int(agg["num_feasible"]),
            "feasible_rate": safe_div(agg["num_feasible"], n),
            "target_in_topm_rate": safe_div(agg["target_in_topm"], n),
            "quota_success_rate_feasible": safe_div(
                agg["quota_success_feasible_num"],
                agg["quota_success_feasible_den"],
            ),
            "hit1": safe_div(agg["hit1"], n),
            "hit3": safe_div(agg["hit3"], n),
            "hit10": safe_div(agg["hit10"], n),
            "mrr10": safe_div(agg["mrr10"], n),
            "mean_target_rank_if_hit": safe_div(
                agg["target_rank_sum_if_hit"],
                agg["target_rank_count_if_hit"],
            ),
            "target_retention_given_base_hit10": safe_div(
                agg["retention_num"],
                agg["retention_den"],
            ),
            "viol_at_k": safe_div(agg["viol_at_k"], n),
            "adm_at_k": safe_div(agg["adm_at_k"], n),
            "unknown_at_k": safe_div(agg["unknown_at_k"], n),
            "pres_at_k": safe_div(agg["pres_at_k"], n),
            "shift_at_k": safe_div(agg["shift_at_k"], n),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_split(
    *,
    model: Any,
    learner: Any,
    learner_kind: str,
    bundle: Any,
    ontology: Any,
    queries: list[Any],
    args: argparse.Namespace,
    alpha_values: list[float],
    split_name: str,
    logger: Logger,
) -> pd.DataFrame:
    logger.log(
        f"[EVAL] split={split_name} queries={len(queries)} alphas={alpha_values} scopes={args.summary_scopes}"
    )

    aggs: dict[tuple[str, str, Optional[float]], defaultdict[str, float]] = {}

    def get_agg(scope: str, method: str, alpha: Optional[float]) -> defaultdict[str, float]:
        key = (scope, method, alpha)
        if key not in aggs:
            aggs[key] = new_agg()
        return aggs[key]

    batch_starts = list(range(0, len(queries), args.query_batch_size))

    for batch_start in tqdm(batch_starts, desc=f"eval-{split_name}"):
        batch_queries = queries[batch_start: batch_start + args.query_batch_size]
        scores = score_batch(model=model, queries=batch_queries, device=args.device)

        effective_top_m = min(args.top_m, int(scores.shape[1]))
        if effective_top_m < args.top_k:
            raise ValueError(f"effective_top_m={effective_top_m} < top_k={args.top_k}")

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

            arr = np.asarray(cand_scores, dtype=np.float64)
            score_mean = float(arr.mean())
            score_std = float(arr.std())
            top_score = float(cand_scores[0])

            statuses = []
            energies = []
            features = []

            for rank0, cand_id in enumerate(cand_ids):
                st = status_for_candidate(
                    qitem=qitem,
                    cand_id=int(cand_id),
                    ontology=ontology,
                    args=args,
                )
                statuses.append(st)
                energies.append(float(st.energy))

                prev_score = None if rank0 == 0 else float(cand_scores[rank0 - 1])
                features.append(
                    make_features(
                        qitem=qitem,
                        cand_score=float(cand_scores[rank0]),
                        rank0=rank0,
                        effective_top_m=effective_top_m,
                        top_score=top_score,
                        prev_score=prev_score,
                        score_mean=score_mean,
                        score_std=score_std,
                        status=st,
                        top_k=args.top_k,
                    )
                )

            status_by_id = {cid: st for cid, st in zip(cand_ids, statuses)}
            checkable_by_id = {cid: st.checkable for cid, st in status_by_id.items()}
            admissible_by_id = {cid: st.admissible for cid, st in status_by_id.items()}
            violated_by_id = {cid: st.violated for cid, st in status_by_id.items()}
            unknown_by_id = {cid: st.unknown for cid, st in status_by_id.items()}

            base_topk = cand_ids[: args.top_k]
            base_rank_map = {cid: i + 1 for i, cid in enumerate(cand_ids)}

            base_viol_present = any(violated_by_id.get(cid, False) for cid in base_topk)
            admissible_in_topm = sum(1 for cid in cand_ids if admissible_by_id.get(cid, False))
            admissible_outside_topk = sum(
                1 for cid in cand_ids[args.top_k:] if admissible_by_id.get(cid, False)
            )

            is_blind = base_viol_present and (admissible_in_topm >= 1)
            blind_strict = base_viol_present and (admissible_outside_topk >= 1)

            feasible, lambda_star, _ = compute_lambda_star_quota(
                base_scores=cand_scores,
                energies=energies,
                k=args.top_k,
                q=args.quota,
            )

            if feasible:
                optq_ranked = stable_rerank_with_lambda(
                    cand_ids=cand_ids,
                    base_scores=cand_scores,
                    energies=energies,
                    lam=float(lambda_star),
                )
                optq_topk = optq_ranked[: args.top_k]
            else:
                optq_topk = list(base_topk)

            target_in_topm = int(qitem.target_entity_id) in set(cand_ids)
            base_hit10 = int(qitem.target_entity_id) in set(base_topk)

            Xq = np.asarray(features, dtype=np.float32)
            signal = learner_signal(learner, Xq)
            base_score_z = Xq[:, FEATURE_NAMES.index("score_z")]

            learned_topk_by_alpha: dict[float, list[int]] = {}
            for alpha in alpha_values:
                final_score = base_score_z + float(alpha) * signal
                order = sorted(range(len(cand_ids)), key=lambda i: (-float(final_score[i]), i))
                learned_topk_by_alpha[float(alpha)] = [cand_ids[i] for i in order[: args.top_k]]

            for scope in args.summary_scopes:
                if not scope_allowed(scope, is_blind=is_blind, blind_strict=blind_strict):
                    continue

                update_agg(
                    agg=get_agg(scope, "base", None),
                    qitem=qitem,
                    returned_topk=base_topk,
                    base_topk=base_topk,
                    base_rank_map=base_rank_map,
                    checkable_by_id=checkable_by_id,
                    admissible_by_id=admissible_by_id,
                    violated_by_id=violated_by_id,
                    unknown_by_id=unknown_by_id,
                    feasible=feasible,
                    target_in_topm=target_in_topm,
                    base_hit10=base_hit10,
                    top_k=args.top_k,
                    quota=args.quota,
                )

                update_agg(
                    agg=get_agg(scope, "optq", None),
                    qitem=qitem,
                    returned_topk=optq_topk,
                    base_topk=base_topk,
                    base_rank_map=base_rank_map,
                    checkable_by_id=checkable_by_id,
                    admissible_by_id=admissible_by_id,
                    violated_by_id=violated_by_id,
                    unknown_by_id=unknown_by_id,
                    feasible=feasible,
                    target_in_topm=target_in_topm,
                    base_hit10=base_hit10,
                    top_k=args.top_k,
                    quota=args.quota,
                )

                for alpha, learned_topk in learned_topk_by_alpha.items():
                    update_agg(
                        agg=get_agg(scope, "learned", float(alpha)),
                        qitem=qitem,
                        returned_topk=learned_topk,
                        base_topk=base_topk,
                        base_rank_map=base_rank_map,
                        checkable_by_id=checkable_by_id,
                        admissible_by_id=admissible_by_id,
                        violated_by_id=violated_by_id,
                        unknown_by_id=unknown_by_id,
                        feasible=feasible,
                        target_in_topm=target_in_topm,
                        base_hit10=base_hit10,
                        top_k=args.top_k,
                        quota=args.quota,
                    )

        del scores, topm_scores, topm_indices
        maybe_clear_cuda_cache()

    return summarize_aggs(
        aggs=aggs,
        dataset_name=args.dataset_name,
        quota=args.quota,
        top_m=args.top_m,
        top_k=args.top_k,
        learner_kind=learner_kind,
    )


def select_alpha(validation_summary: pd.DataFrame, args: argparse.Namespace, logger: Logger) -> float:
    sub = validation_summary[
        (validation_summary["method"] == "learned")
        & (validation_summary["scope"] == args.selection_scope)
    ].copy()

    if sub.empty:
        logger.log(
            f"[SELECT] No rows for selection_scope={args.selection_scope}; falling back to full."
        )
        sub = validation_summary[
            (validation_summary["method"] == "learned")
            & (validation_summary["scope"] == "full")
        ].copy()

    if sub.empty:
        raise RuntimeError("No learned validation rows available for alpha selection.")

    safe = sub[sub["quota_success_rate_feasible"] >= args.selection_quota_success_threshold].copy()

    if not safe.empty:
        chosen = safe.sort_values(
            ["pres_at_k", "hit10", "mrr10", "shift_at_k"],
            ascending=[False, False, False, True],
        ).iloc[0]
        logger.log(
            "[SELECT] Chose alpha from quota-safe candidates "
            f"threshold={args.selection_quota_success_threshold}"
        )
    else:
        chosen = sub.sort_values(
            ["quota_success_rate_feasible", "pres_at_k", "hit10", "mrr10"],
            ascending=[False, False, False, False],
        ).iloc[0]
        logger.log("[SELECT] No quota-safe alpha; chose best quota-success/preservation tradeoff.")

    alpha = float(chosen["alpha"])
    logger.log(
        f"[SELECT] alpha={alpha} scope={chosen['scope']} "
        f"quota_success={chosen['quota_success_rate_feasible']} "
        f"pres={chosen['pres_at_k']} hit10={chosen['hit10']}"
    )
    return alpha


def write_feature_importance(learner: Any, out_path: Path) -> None:
    if hasattr(learner, "feature_importances_"):
        vals = list(getattr(learner, "feature_importances_"))
        df = pd.DataFrame({"feature": FEATURE_NAMES, "importance": vals})
        df = df.sort_values("importance", ascending=False)
        df.to_csv(out_path, index=False)
    else:
        pd.DataFrame({"feature": FEATURE_NAMES, "importance": [math.nan] * len(FEATURE_NAMES)}).to_csv(
            out_path, index=False
        )


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(output_dir / "run.log")
    t0 = perf_now()

    set_all_seeds(args.seed)

    logger.log("[EXP22] Learned reranker baseline")
    logger.log(f"[ARGS] {vars(args)}")

    checkpoint_path = resolve_checkpoint_path(args.run_dir)
    payload = load_checkpoint_payload(checkpoint_path)
    create_inverse_triples = bool(payload.get("create_inverse_triples", True))

    bundle = load_dataset(args.processed_dir, create_inverse_triples=create_inverse_triples)
    ontology = load_ontology_bundle(args.processed_dir, bundle.entity_to_id, bundle.relation_to_id)

    if args.dataset_name is None:
        args.dataset_name = args.processed_dir.name

    model = load_model_from_payload(
        payload=payload,
        train_tf=bundle.train_tf,
        device=args.device,
        logger=logger,
    )

    train_queries = build_split_queries(
        bundle=bundle,
        split=args.train_split,
        mode=args.mode,
        max_queries=args.train_max_queries,
        allowed_query_ids=None,
    )
    valid_queries = build_split_queries(
        bundle=bundle,
        split=args.valid_split,
        mode=args.mode,
        max_queries=args.valid_max_queries,
        allowed_query_ids=None,
    )

    eval_allowed = load_allowed_query_ids(args.query_id_file)
    eval_queries = build_split_queries(
        bundle=bundle,
        split=args.eval_split,
        mode=args.mode,
        max_queries=args.eval_max_queries,
        allowed_query_ids=eval_allowed,
    )

    if not train_queries:
        raise RuntimeError("No train queries built.")
    if not valid_queries:
        raise RuntimeError("No validation queries built.")
    if not eval_queries:
        raise RuntimeError("No eval queries built.")

    X_train, y_train = collect_training_matrix(
        model=model,
        bundle=bundle,
        ontology=ontology,
        queries=train_queries,
        args=args,
        logger=logger,
        split_name=args.train_split,
    )

    learner, learner_kind = fit_learner(X_train, y_train, args, logger)
    write_feature_importance(learner, output_dir / "feature_importance.csv")

    validation_summary = evaluate_split(
        model=model,
        learner=learner,
        learner_kind=learner_kind,
        bundle=bundle,
        ontology=ontology,
        queries=valid_queries,
        args=args,
        alpha_values=args.alphas,
        split_name=args.valid_split,
        logger=logger,
    )
    validation_summary.to_csv(output_dir / "validation_alpha_summary.csv", index=False)

    selected_alpha = select_alpha(validation_summary, args, logger)

    eval_alpha_values = args.alphas if args.eval_all_alphas else [selected_alpha]

    test_summary = evaluate_split(
        model=model,
        learner=learner,
        learner_kind=learner_kind,
        bundle=bundle,
        ontology=ontology,
        queries=eval_queries,
        args=args,
        alpha_values=eval_alpha_values,
        split_name=args.eval_split,
        logger=logger,
    )
    test_summary.to_csv(output_dir / "learned_reranker_summary.csv", index=False)

    elapsed = perf_now() - t0

    summary = {
        "status": "ok",
        "updated_at_utc": now_iso_utc(),
        "experiment_name": "exp22_learned_reranker_baseline",
        "dataset_name": args.dataset_name,
        "learner_kind": learner_kind,
        "selected_alpha": selected_alpha,
        "feature_names": FEATURE_NAMES,
        "train_split": args.train_split,
        "valid_split": args.valid_split,
        "eval_split": args.eval_split,
        "train_queries": len(train_queries),
        "valid_queries": len(valid_queries),
        "eval_queries": len(eval_queries),
        "train_samples": int(len(y_train)),
        "train_positive_rate": float(np.mean(y_train)),
        "top_m": args.top_m,
        "top_k": args.top_k,
        "quota": args.quota,
        "elapsed_seconds": elapsed,
        "elapsed_human": format_seconds(elapsed),
        "artifacts": {
            "validation_alpha_summary": str(output_dir / "validation_alpha_summary.csv"),
            "learned_reranker_summary": str(output_dir / "learned_reranker_summary.csv"),
            "feature_importance": str(output_dir / "feature_importance.csv"),
            "run_log": str(output_dir / "run.log"),
        },
    }
    save_json_atomic(output_dir / "summary.json", summary)

    logger.log(f"[DONE] elapsed={format_seconds(elapsed)} selected_alpha={selected_alpha}")
    if hasattr(logger, 'close'):
        logger.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-name", type=str, default=None)

    p.add_argument("--train-split", type=str, default="train", choices=["train", "valid", "test"])
    p.add_argument("--valid-split", type=str, default="valid", choices=["train", "valid", "test"])
    p.add_argument("--eval-split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--mode", type=str, default="all", choices=["tail", "head", "all"])

    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--top-m", type=int, default=5000)
    p.add_argument("--quota", type=int, default=5)

    p.add_argument("--query-batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train-max-queries", type=int, default=3000)
    p.add_argument("--valid-max-queries", type=int, default=1000)
    p.add_argument("--eval-max-queries", type=int, default=None)
    p.add_argument("--query-id-file", type=Path, default=None)

    p.add_argument("--candidate-sample-per-query", type=int, default=128)

    p.add_argument("--check-policy", type=str, default="available_any", choices=["available_any", "available_all"])
    p.add_argument("--use-domain", action="store_true")
    p.add_argument("--use-range", action="store_true")
    p.add_argument("--use-disjoint", action="store_true")
    p.add_argument("--unknown-penalty", type=float, default=1.0)
    p.add_argument("--binary-like", action="store_true")

    p.add_argument("--summary-scopes", type=str, nargs="+", default=["full", "blind_strict"], choices=["full", "blind", "blind_strict"])

    p.add_argument("--learner", type=str, default="auto", choices=["auto", "lightgbm", "sklearn_hgb"])
    p.add_argument("--label-policy", type=str, default="admissible", choices=["admissible", "nonviolating_checkable"])

    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    p.add_argument("--selection-scope", type=str, default="blind_strict", choices=["full", "blind", "blind_strict"])
    p.add_argument("--selection-quota-success-threshold", type=float, default=0.999)
    p.add_argument("--eval-all-alphas", action="store_true", help="Evaluate all alpha values on eval split, not only selected alpha.")

    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--n-jobs", type=int, default=8)

    args = p.parse_args()

    if args.quota < 1 or args.quota > args.top_k:
        raise ValueError(f"quota must satisfy 1 <= quota <= top_k, got quota={args.quota}, top_k={args.top_k}")

    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
