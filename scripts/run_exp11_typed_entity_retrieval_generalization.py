#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import pykeen
    from pykeen.models import model_resolver
    from pykeen.triples import TriplesFactory
except Exception as e:
    raise SystemExit(
        "This script requires PyKEEN. Install with: pip install pykeen"
    ) from e


# ============================================================
# Helpers
# ============================================================

def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def perf_now() -> float:
    return time.perf_counter()


def format_seconds(sec: float | None) -> str:
    if sec is None or math.isinf(sec) or math.isnan(sec):
        return "unknown"
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, q))


def safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return repr(obj)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, data: Any, retries: int = 10, sleep_sec: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)

    last_err = None
    for _ in range(retries):
        try:
            tmp.replace(path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(sleep_sec)
    raise last_err


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


def get_env_metadata(device_str: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "pykeen_version": getattr(pykeen, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "device_requested": device_str,
        "platform": sys.platform,
        "pid": os.getpid(),
    }

    if torch.cuda.is_available():
        try:
            requested = torch.device(device_str)
            idx = requested.index
            if idx is None:
                idx = torch.cuda.current_device()
            info["cuda_device_index"] = idx
            info["cuda_device_name"] = torch.cuda.get_device_name(idx)
            info["cuda_device_capability"] = list(torch.cuda.get_device_capability(idx))
        except Exception as e:
            info["cuda_metadata_error"] = f"{type(e).__name__}: {e}"

    return info


class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        line = f"[{now_iso_utc()}] {msg}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ============================================================
# Data loading
# ============================================================

@dataclass
class DatasetBundle:
    train_tf: TriplesFactory
    valid_tf: TriplesFactory
    test_tf: TriplesFactory
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    id_to_entity: dict[int, str]
    id_to_relation: dict[int, str]


def load_dataset(processed_dir: Path, create_inverse_triples: bool) -> DatasetBundle:
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

    return DatasetBundle(
        train_tf=train_tf,
        valid_tf=valid_tf,
        test_tf=test_tf,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        id_to_entity={v: k for k, v in entity_to_id.items()},
        id_to_relation={v: k for k, v in relation_to_id.items()},
    )


# ============================================================
# Ontology
# ============================================================

@dataclass
class OntologyBundle:
    entity_types: dict[int, set[str]]
    relation_domains: dict[int, set[str]]
    relation_ranges: dict[int, set[str]]
    has_entity_types: bool
    has_relation_constraints: bool


def _normalize_type_value(x: Any) -> str:
    return str(x).strip()


def _extract_type_set(v: Any) -> set[str]:
    if v is None:
        return set()
    if isinstance(v, list):
        return {_normalize_type_value(x) for x in v if x is not None}
    if isinstance(v, dict):
        for key in ("types", "classes", "values", "type_ids", "type_names"):
            if key in v and isinstance(v[key], list):
                return {_normalize_type_value(x) for x in v[key] if x is not None}
        return set()
    return {_normalize_type_value(v)}


def load_entity_types(path: Path, entity_to_id: dict[str, int]) -> dict[int, set[str]]:
    if not path.exists():
        return {}

    raw = load_json(path)
    out: dict[int, set[str]] = {}

    if isinstance(raw, dict):
        for k, v in raw.items():
            ent_id: Optional[int] = None
            if k in entity_to_id:
                ent_id = entity_to_id[k]
            else:
                try:
                    maybe_id = int(k)
                    if maybe_id in entity_to_id.values():
                        ent_id = maybe_id
                except Exception:
                    pass
            if ent_id is not None:
                out[ent_id] = _extract_type_set(v)

    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            ent = item.get("entity") or item.get("entity_id") or item.get("id") or item.get("name")
            if ent is None:
                continue
            ent_id: Optional[int] = None
            if isinstance(ent, str) and ent in entity_to_id:
                ent_id = entity_to_id[ent]
            else:
                try:
                    maybe_id = int(ent)
                    if maybe_id in entity_to_id.values():
                        ent_id = maybe_id
                except Exception:
                    pass
            if ent_id is not None:
                out[ent_id] = _extract_type_set(item)

    return out


def load_relation_constraints(path: Path, relation_to_id: dict[str, int]) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    if not path.exists():
        return {}, {}

    raw = load_json(path)
    domains: dict[int, set[str]] = {}
    ranges: dict[int, set[str]] = {}

    if isinstance(raw, dict):
        for k, v in raw.items():
            rel_id: Optional[int] = None
            if k in relation_to_id:
                rel_id = relation_to_id[k]
            else:
                try:
                    maybe_id = int(k)
                    if maybe_id in relation_to_id.values():
                        rel_id = maybe_id
                except Exception:
                    pass
            if rel_id is None or not isinstance(v, dict):
                continue
            domains[rel_id] = _extract_type_set(v.get("domain"))
            ranges[rel_id] = _extract_type_set(v.get("range"))

    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            rel = item.get("relation") or item.get("relation_id") or item.get("id") or item.get("name")
            if rel is None:
                continue
            rel_id: Optional[int] = None
            if isinstance(rel, str) and rel in relation_to_id:
                rel_id = relation_to_id[rel]
            else:
                try:
                    maybe_id = int(rel)
                    if maybe_id in relation_to_id.values():
                        rel_id = maybe_id
                except Exception:
                    pass
            if rel_id is None:
                continue
            domains[rel_id] = _extract_type_set(item.get("domain"))
            ranges[rel_id] = _extract_type_set(item.get("range"))

    return domains, ranges


def load_ontology_bundle(processed_dir: Path, entity_to_id: dict[str, int], relation_to_id: dict[str, int]) -> OntologyBundle:
    entity_types_path = processed_dir / "entity_types.json"
    relation_constraints_path = processed_dir / "relation_constraints.json"

    entity_types = load_entity_types(entity_types_path, entity_to_id)
    relation_domains, relation_ranges = load_relation_constraints(relation_constraints_path, relation_to_id)

    return OntologyBundle(
        entity_types=entity_types,
        relation_domains=relation_domains,
        relation_ranges=relation_ranges,
        has_entity_types=entity_types_path.exists(),
        has_relation_constraints=relation_constraints_path.exists(),
    )


# ============================================================
# Model loading
# ============================================================

def resolve_checkpoint_path(run_dir: Path) -> Path:
    if run_dir.is_file():
        return run_dir
    ckpt = run_dir / "base_model_checkpoint.pt"
    if ckpt.exists():
        return ckpt
    raise FileNotFoundError(f"Could not find checkpoint at: {ckpt}")


def load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_model_from_payload(
    payload: dict[str, Any],
    train_tf: TriplesFactory,
    device: str,
    logger: Logger,
):
    model_name = payload["model_name"]
    model_kwargs = payload["model_kwargs"]
    state_dict = payload["model_state_dict"]

    logger.log(f"[MODEL] Rebuilding model_name={model_name} on device={device}")
    model = model_resolver.make(
        model_name,
        triples_factory=train_tf,
        **model_kwargs,
    )
    model.load_state_dict(state_dict)
    model = model.to(torch.device(device))
    model.eval()
    return model


# ============================================================
# Query / retrieval semantics
# ============================================================

@dataclass
class QueryItem:
    query_id: str
    row_index: int
    mode: str
    head_id: int
    rel_id: int
    tail_id: int
    relation_name: str
    target_entity_id: int


@dataclass
class CandidateTypeStatus:
    checkable: bool
    admissible: bool
    violating: bool
    unknown: bool
    energy: float
    expected_type_count: int
    candidate_type_count: int


def load_allowed_query_ids(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None

    if not path.exists():
        raise FileNotFoundError(f"Query id file does not exist: {path}")

    allowed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            qid = line.strip()
            if qid:
                allowed.add(qid)

    if not allowed:
        raise ValueError(f"Query id file is empty: {path}")

    return allowed


def build_queries(
    mapped_triples: torch.Tensor,
    id_to_relation: dict[int, str],
    split_name: str,
    mode: str,
    max_queries: Optional[int],
    allowed_query_ids: Optional[set[str]],
) -> list[QueryItem]:
    queries: list[QueryItem] = []

    for row_index, row in enumerate(mapped_triples.tolist()):
        h, r, t = int(row[0]), int(row[1]), int(row[2])
        rel_name = id_to_relation.get(r, str(r))

        if mode in ("tail", "all"):
            qid = f"{split_name}_tail_{row_index:09d}"
            if allowed_query_ids is None or qid in allowed_query_ids:
                queries.append(
                    QueryItem(
                        query_id=qid,
                        row_index=row_index,
                        mode="tail",
                        head_id=h,
                        rel_id=r,
                        tail_id=t,
                        relation_name=rel_name,
                        target_entity_id=t,
                    )
                )

        if mode in ("head", "all"):
            qid = f"{split_name}_head_{row_index:09d}"
            if allowed_query_ids is None or qid in allowed_query_ids:
                queries.append(
                    QueryItem(
                        query_id=qid,
                        row_index=row_index,
                        mode="head",
                        head_id=h,
                        rel_id=r,
                        tail_id=t,
                        relation_name=rel_name,
                        target_entity_id=h,
                    )
                )

        if max_queries is not None and len(queries) >= max_queries:
            queries = queries[:max_queries]
            break

    return queries


def score_batch(model: Any, queries: list[QueryItem], device: str) -> torch.Tensor:
    if not queries:
        raise ValueError("score_batch() received empty batch")

    valid_modes = {"head", "tail"}
    modes = {q.mode for q in queries}
    unsupported = modes - valid_modes
    if unsupported:
        raise ValueError(f"Unsupported query modes in batch: {sorted(unsupported)}")

    scores_out: Optional[torch.Tensor] = None

    with torch.no_grad():
        tail_positions = [i for i, q in enumerate(queries) if q.mode == "tail"]
        if tail_positions:
            tail_queries = [queries[i] for i in tail_positions]
            hr_batch = torch.tensor(
                [[q.head_id, q.rel_id] for q in tail_queries],
                dtype=torch.long,
                device=device,
            )
            tail_scores = model.score_t(hr_batch)

            if tail_scores.ndim != 2:
                raise RuntimeError(
                    f"Expected 2D tail score tensor, got shape={tuple(tail_scores.shape)}"
                )

            if scores_out is None:
                scores_out = torch.empty(
                    (len(queries), tail_scores.shape[1]),
                    dtype=tail_scores.dtype,
                    device=tail_scores.device,
                )

            scores_out[
                torch.tensor(tail_positions, dtype=torch.long, device=tail_scores.device)
            ] = tail_scores

        head_positions = [i for i, q in enumerate(queries) if q.mode == "head"]
        if head_positions:
            head_queries = [queries[i] for i in head_positions]
            rt_batch = torch.tensor(
                [[q.rel_id, q.tail_id] for q in head_queries],
                dtype=torch.long,
                device=device,
            )
            head_scores = model.score_h(rt_batch)

            if head_scores.ndim != 2:
                raise RuntimeError(
                    f"Expected 2D head score tensor, got shape={tuple(head_scores.shape)}"
                )

            if scores_out is None:
                scores_out = torch.empty(
                    (len(queries), head_scores.shape[1]),
                    dtype=head_scores.dtype,
                    device=head_scores.device,
                )

            if scores_out.shape[1] != head_scores.shape[1]:
                raise RuntimeError(
                    f"Head/tail score dimension mismatch: "
                    f"{scores_out.shape[1]} vs {head_scores.shape[1]}"
                )

            scores_out[
                torch.tensor(head_positions, dtype=torch.long, device=head_scores.device)
            ] = head_scores

    if scores_out is None:
        raise RuntimeError("No scores were produced for the query batch")

    if scores_out.ndim != 2:
        raise RuntimeError(f"Expected 2D score tensor, got shape={tuple(scores_out.shape)}")

    return scores_out


def get_expected_types_for_query(
    *,
    query: QueryItem,
    ontology: OntologyBundle,
    target_side_policy: str,
) -> set[str]:
    if query.mode == "tail":
        # retrieve tail entities expected by relation range
        if target_side_policy == "strict_relation_side":
            return set(ontology.relation_ranges.get(query.rel_id, set()))
        else:
            return set(ontology.relation_ranges.get(query.rel_id, set()))
    else:
        # retrieve head entities expected by relation domain
        if target_side_policy == "strict_relation_side":
            return set(ontology.relation_domains.get(query.rel_id, set()))
        else:
            return set(ontology.relation_domains.get(query.rel_id, set()))


def evaluate_candidate_type_semantics(
    *,
    candidate_entity_id: int,
    expected_types: set[str],
    ontology: OntologyBundle,
    check_policy: str,
    unknown_penalty: float,
) -> CandidateTypeStatus:
    candidate_types = set(ontology.entity_types.get(candidate_entity_id, set()))

    expected_evaluable = bool(expected_types)
    candidate_evaluable = bool(candidate_types)

    if check_policy == "available_all":
        checkable = expected_evaluable and candidate_evaluable
    elif check_policy == "available_any":
        checkable = expected_evaluable and candidate_evaluable
    else:
        raise ValueError(f"Unsupported check_policy: {check_policy}")

    if checkable:
        admissible = not candidate_types.isdisjoint(expected_types)
        violating = not admissible
        unknown = False
        energy = 0.0 if admissible else 1.0
    else:
        admissible = False
        violating = False
        unknown = True
        energy = float(unknown_penalty)

    return CandidateTypeStatus(
        checkable=checkable,
        admissible=admissible,
        violating=violating,
        unknown=unknown,
        energy=energy,
        expected_type_count=len(expected_types),
        candidate_type_count=len(candidate_types),
    )


# ============================================================
# Control / baselines
# ============================================================

def compute_lambda_star_quota(
    *,
    base_scores: list[float],
    energies: list[float],
    k: int,
    q: int,
) -> tuple[bool, Optional[float], Optional[float]]:
    """
    Minimal finite-window pressure for at least q zero-energy candidates
    in the visible top-k list.

    Up to k-q positive-energy candidates may remain above the q-th
    admissible item. This is therefore minimal for the top-k quota
    objective, not the stricter objective of pushing every positive-energy
    candidate below the q-th admissible score.
    """
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}")
    if q > k:
        raise ValueError(f"q must satisfy q <= k, got q={q}, k={k}")

    zero_idxs = [i for i, e in enumerate(energies) if e == 0.0]
    if len(zero_idxs) < q:
        return False, None, None

    zero_scores = sorted((base_scores[i] for i in zero_idxs), reverse=True)
    tau_q = zero_scores[q - 1]

    crossings: list[float] = []
    for s, e in zip(base_scores, energies):
        if e > 0.0:
            crossing = (s - tau_q) / e
            if crossing > 0.0:
                crossings.append(float(crossing))

    allowed_above = k - q
    crossings.sort(reverse=True)

    if len(crossings) <= allowed_above:
        lam = 0.0
    else:
        lam = crossings[allowed_above]

    return True, float(max(0.0, lam)), float(tau_q)


def stable_rerank_with_lambda(
    *,
    cand_ids: list[int],
    base_scores: list[float],
    energies: list[float],
    lam: float,
) -> list[int]:
    items = []
    for idx, (cid, s, e) in enumerate(zip(cand_ids, base_scores, energies)):
        calibrated = s - lam * e
        items.append((cid, calibrated, e, s, idx))
    items.sort(key=lambda x: (-x[1], x[2], -x[3], x[4]))
    return [x[0] for x in items]


def fill_to_k_from_original_order(
    preferred_ids: list[int],
    original_ids: list[int],
    k: int,
) -> list[int]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    out: list[int] = []
    used: set[int] = set()

    for cid in preferred_ids:
        if cid not in used:
            out.append(cid)
            used.add(cid)
        if len(out) == k:
            return out

    for cid in original_ids:
        if cid not in used:
            out.append(cid)
            used.add(cid)
        if len(out) == k:
            return out

    raise RuntimeError(f"Could not fill result to K={k}; only got {len(out)} items.")


def rerank_admissible_first_quota_fill_to_k(
    *,
    cand_ids: list[int],
    admissible_by_id: dict[int, bool],
    q: int,
    k: int,
) -> list[int]:
    preferred = [cid for cid in cand_ids if admissible_by_id.get(cid, False)]
    preferred = preferred[:q]
    return fill_to_k_from_original_order(preferred, cand_ids, k)


def rerank_hard_validation_fill_to_k(
    *,
    cand_ids: list[int],
    checkable_by_id: dict[int, bool],
    violating_by_id: dict[int, bool],
    k: int,
) -> list[int]:
    kept = [
        cid for cid in cand_ids
        if not (checkable_by_id.get(cid, False) and violating_by_id.get(cid, False))
    ]
    return fill_to_k_from_original_order(kept[:k], cand_ids, k)


# ============================================================
# Metrics
# ============================================================

def compute_pres_at_k(base_topk: list[int], reranked_topk: list[int], k: int) -> float:
    return len(set(base_topk).intersection(set(reranked_topk))) / float(k)


def compute_shift_at_k(base_rank_map: dict[int, int], reranked_topk: list[int], k: int) -> float:
    vals = []
    for out_rank, ent_id in enumerate(reranked_topk, start=1):
        base_rank = base_rank_map.get(ent_id)
        if base_rank is None:
            raise KeyError(f"Entity {ent_id} not found in base_rank_map")
        vals.append(abs(base_rank - out_rank))
    return sum(vals) / float(k)


def compute_topk_set_jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa.union(sb)
    if not union:
        return 1.0
    return len(sa.intersection(sb)) / float(len(union))


def compute_topk_exact_match(a: list[int], b: list[int]) -> float:
    return 1.0 if list(a) == list(b) else 0.0


def compute_topk_set_match(a: list[int], b: list[int]) -> float:
    return 1.0 if set(a) == set(b) else 0.0


def count_admissible_in_topk(
    *,
    returned_topk: list[int],
    admissible_by_id: dict[int, bool],
) -> int:
    return sum(1 for cid in returned_topk if admissible_by_id.get(cid, False))


def evaluate_returned_topk(
    *,
    returned_topk: list[int],
    checkable_by_id: dict[int, bool],
    admissible_by_id: dict[int, bool],
    violating_by_id: dict[int, bool],
    unknown_by_id: dict[int, bool],
    base_topk: list[int],
    base_rank_map: dict[int, int],
    k: int,
) -> dict[str, float]:
    checkable = 0
    admissible = 0
    violating = 0
    unknown = 0

    for cid in returned_topk:
        if checkable_by_id.get(cid, False):
            checkable += 1
            if admissible_by_id.get(cid, False):
                admissible += 1
            elif violating_by_id.get(cid, False):
                violating += 1
        elif unknown_by_id.get(cid, False):
            unknown += 1

    viol_at_k = (violating / checkable) if checkable > 0 else 0.0
    adm_at_k = admissible / float(k)
    cov_at_k = checkable / float(k)
    unknown_at_k = unknown / float(k)
    pres_at_k = compute_pres_at_k(base_topk, returned_topk, k)
    shift_at_k = compute_shift_at_k(base_rank_map, returned_topk, k)

    return {
        "viol_at_k": viol_at_k,
        "adm_at_k": adm_at_k,
        "cov_at_k": cov_at_k,
        "unknown_at_k": unknown_at_k,
        "pres_at_k": pres_at_k,
        "shift_at_k": shift_at_k,
    }


# ============================================================
# CSV / progress
# ============================================================

def write_query_csv_header(path: Path, quotas: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "query_index",
            "query_id",
            "row_index",
            "mode",
            "relation",
            "head_id",
            "rel_id",
            "tail_id",
            "effective_top_m",
            "expected_type_count",
            "target_side",
        ]
        for q in quotas:
            prefix = f"q{q}"
            header.extend([
                f"{prefix}_feasible",
                f"{prefix}_lambda_star",
                f"{prefix}_tau_q",

                f"{prefix}_optq_viol_at_k",
                f"{prefix}_optq_adm_at_k",
                f"{prefix}_optq_cov_at_k",
                f"{prefix}_optq_unknown_at_k",
                f"{prefix}_optq_pres_at_k",
                f"{prefix}_optq_shift_at_k",
                f"{prefix}_optq_adm_count_topk",

                f"{prefix}_afq_viol_at_k",
                f"{prefix}_afq_adm_at_k",
                f"{prefix}_afq_cov_at_k",
                f"{prefix}_afq_unknown_at_k",
                f"{prefix}_afq_pres_at_k",
                f"{prefix}_afq_shift_at_k",
                f"{prefix}_afq_adm_count_topk",

                f"{prefix}_hardval_viol_at_k",
                f"{prefix}_hardval_adm_at_k",
                f"{prefix}_hardval_cov_at_k",
                f"{prefix}_hardval_unknown_at_k",
                f"{prefix}_hardval_pres_at_k",
                f"{prefix}_hardval_shift_at_k",
                f"{prefix}_hardval_adm_count_topk",

                f"{prefix}_optq_vs_afq_exact_match",
                f"{prefix}_optq_vs_afq_set_match",
                f"{prefix}_optq_vs_afq_jaccard",

                f"{prefix}_optq_vs_hardval_exact_match",
                f"{prefix}_optq_vs_hardval_set_match",
                f"{prefix}_optq_vs_hardval_jaccard",
            ])
        writer.writerow(header)


def append_query_rows(path: Path, rows: list[list[Any]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def compute_progress_payload(
    *,
    started_at: float,
    processed_queries: int,
    total_queries: int,
    effective_top_m: int,
    aggs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    elapsed = perf_now() - started_at
    rate = processed_queries / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total_queries - processed_queries)
    eta_sec = (remaining / rate) if rate > 0 else None

    running = {}
    for q, agg in aggs.items():
        n = agg["num_queries"]
        fn = agg["num_feasible"]
        running[f"q{q}_feasible_rate"] = (fn / n) if n > 0 else None
        running[f"q{q}_optq_adm_at_k"] = (agg["optq_adm_sum"] / fn) if fn > 0 else None
        running[f"q{q}_optq_vs_afq_set_match"] = (agg["optq_vs_afq_set_match_sum"] / fn) if fn > 0 else None

    return {
        "status": "running",
        "updated_at_utc": now_iso_utc(),
        "processed_queries": processed_queries,
        "total_queries": total_queries,
        "progress_fraction": processed_queries / total_queries if total_queries else 0.0,
        "elapsed_seconds": round(elapsed, 6),
        "elapsed_human": format_seconds(elapsed),
        "eta_seconds": None if eta_sec is None else round(eta_sec, 6),
        "eta_human": format_seconds(eta_sec),
        "queries_per_second": round(rate, 4),
        "effective_top_m": effective_top_m,
        "running_metrics": running,
    }


# ============================================================
# Main
# ============================================================

def run_exp11(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir / "run.log")

    save_json_atomic(output_dir / "config.json", vars(args))
    set_all_seeds(args.seed)

    started_total = perf_now()

    logger.log("[STEP 1/7] Resolving checkpoint and loading payload")
    checkpoint_path = resolve_checkpoint_path(args.run_dir)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    create_inverse_triples = bool(checkpoint_payload.get("create_inverse_triples", True))
    logger.log(
        f"[STEP 1/7] Done | checkpoint={checkpoint_path} "
        f"create_inverse_triples={create_inverse_triples}"
    )

    logger.log("[STEP 2/7] Loading dataset")
    bundle = load_dataset(args.processed_dir, create_inverse_triples=create_inverse_triples)
    split_tf = {"train": bundle.train_tf, "valid": bundle.valid_tf, "test": bundle.test_tf}[args.split]
    logger.log(
        f"[STEP 2/7] Done | entities={len(bundle.entity_to_id)} "
        f"relations={len(bundle.relation_to_id)} triples={split_tf.num_triples}"
    )

    logger.log("[STEP 3/7] Loading ontology")
    ontology = load_ontology_bundle(args.processed_dir, bundle.entity_to_id, bundle.relation_to_id)
    logger.log(
        f"[STEP 3/7] Done | has_entity_types={ontology.has_entity_types} "
        f"has_relation_constraints={ontology.has_relation_constraints}"
    )

    logger.log("[STEP 4/7] Loading frozen model")
    model = load_model_from_payload(
        payload=checkpoint_payload,
        train_tf=bundle.train_tf,
        device=args.device,
        logger=logger,
    )
    logger.log("[STEP 4/7] Done")

    logger.log("[STEP 5/7] Building query list")
    allowed_query_ids = load_allowed_query_ids(args.query_id_file)
    queries = build_queries(
        mapped_triples=split_tf.mapped_triples,
        id_to_relation=bundle.id_to_relation,
        split_name=args.split,
        mode=args.mode,
        max_queries=args.max_queries,
        allowed_query_ids=allowed_query_ids,
    )
    total_queries = len(queries)
    if total_queries == 0:
        raise RuntimeError("No queries were built. Check split/mode/max_queries/query-id-file.")
    logger.log(f"[STEP 5/7] Done | total_queries={total_queries}")

    quotas = sorted(set(args.quotas))
    if any(q <= 0 for q in quotas):
        raise ValueError(f"All quotas must be positive, got {quotas}")
    if any(q > args.top_k for q in quotas):
        raise ValueError(f"All quotas must be <= top_k={args.top_k}, got {quotas}")

    query_csv_path = output_dir / "query_level.csv"
    summary_csv_path = output_dir / "quota_summary.csv"
    relation_csv_path = output_dir / "relation_level.csv"
    progress_json_path = output_dir / "progress.json"
    summary_json_path = output_dir / "summary.json"

    write_query_csv_header(query_csv_path, quotas)
    env_metadata = get_env_metadata(args.device)

    aggs: dict[int, dict[str, Any]] = {}
    for q in quotas:
        aggs[q] = {
            "num_queries": 0,
            "num_feasible": 0,

            "optq_viol_sum": 0.0,
            "optq_adm_sum": 0.0,
            "optq_cov_sum": 0.0,
            "optq_unknown_sum": 0.0,
            "optq_pres_sum": 0.0,
            "optq_shift_sum": 0.0,
            "optq_adm_count_sum": 0.0,

            "afq_viol_sum": 0.0,
            "afq_adm_sum": 0.0,
            "afq_cov_sum": 0.0,
            "afq_unknown_sum": 0.0,
            "afq_pres_sum": 0.0,
            "afq_shift_sum": 0.0,
            "afq_adm_count_sum": 0.0,

            "hardval_viol_sum": 0.0,
            "hardval_adm_sum": 0.0,
            "hardval_cov_sum": 0.0,
            "hardval_unknown_sum": 0.0,
            "hardval_pres_sum": 0.0,
            "hardval_shift_sum": 0.0,
            "hardval_adm_count_sum": 0.0,

            "optq_vs_afq_exact_match_sum": 0.0,
            "optq_vs_afq_set_match_sum": 0.0,
            "optq_vs_afq_jaccard_sum": 0.0,

            "optq_vs_hardval_exact_match_sum": 0.0,
            "optq_vs_hardval_set_match_sum": 0.0,
            "optq_vs_hardval_jaccard_sum": 0.0,

            "lambda_values": [],
        }

    rel_stats: dict[int, dict[str, Counter]] = {
        q: defaultdict(Counter) for q in quotas
    }

    query_rows_buffer: list[list[Any]] = []
    processed = 0
    step_started = perf_now()
    effective_top_m_global: Optional[int] = None

    logger.log("[STEP 6/7] Running typed entity retrieval generalization benchmark")

    batch_starts = list(range(0, total_queries, args.query_batch_size))
    pbar = tqdm(
        batch_starts,
        desc="EXP-11 batches",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_start in pbar:
        batch_end = min(batch_start + args.query_batch_size, total_queries)
        batch_queries = queries[batch_start:batch_end]

        scores = score_batch(model=model, queries=batch_queries, device=args.device)

        num_candidates = int(scores.shape[1])
        effective_top_m = min(args.top_m, num_candidates)
        if effective_top_m < args.top_k:
            raise RuntimeError(
                f"effective_top_m={effective_top_m} is smaller than top_k={args.top_k}. "
                f"Requested top_m={args.top_m}, num_candidates={num_candidates}."
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

        for row_idx, query in enumerate(batch_queries):
            global_query_index = batch_start + row_idx

            cand_ids = topm_indices[row_idx].detach().cpu().tolist()
            cand_scores = topm_scores[row_idx].detach().cpu().tolist()

            expected_types = get_expected_types_for_query(
                query=query,
                ontology=ontology,
                target_side_policy=args.target_side_policy,
            )

            statuses: list[CandidateTypeStatus] = []
            energies: list[float] = []

            for cand_id in cand_ids:
                st = evaluate_candidate_type_semantics(
                    candidate_entity_id=cand_id,
                    expected_types=expected_types,
                    ontology=ontology,
                    check_policy=args.check_policy,
                    unknown_penalty=args.unknown_penalty,
                )
                statuses.append(st)
                energies.append(st.energy)

            status_by_id = {cid: st for cid, st in zip(cand_ids, statuses)}
            checkable_by_id = {cid: st.checkable for cid, st in status_by_id.items()}
            admissible_by_id = {cid: st.admissible for cid, st in status_by_id.items()}
            violating_by_id = {cid: st.violating for cid, st in status_by_id.items()}
            unknown_by_id = {cid: st.unknown for cid, st in status_by_id.items()}

            base_topk = cand_ids[:args.top_k]
            base_rank_map = {cid: i + 1 for i, cid in enumerate(cand_ids)}

            row = [
                global_query_index,
                query.query_id,
                query.row_index,
                query.mode,
                query.relation_name,
                query.head_id,
                query.rel_id,
                query.tail_id,
                effective_top_m,
                len(expected_types),
                "range" if query.mode == "tail" else "domain",
            ]

            for q in quotas:
                feasible, lambda_star, tau_q = compute_lambda_star_quota(
                    base_scores=cand_scores,
                    energies=energies,
                    k=args.top_k,
                    q=q,
                )

                aggs[q]["num_queries"] += 1
                rel_counter = rel_stats[q][query.relation_name]
                rel_counter["queries"] += 1

                if feasible:
                    optq_ranked = stable_rerank_with_lambda(
                        cand_ids=cand_ids,
                        base_scores=cand_scores,
                        energies=energies,
                        lam=lambda_star,
                    )
                    optq_topk = optq_ranked[:args.top_k]

                    afq_topk = rerank_admissible_first_quota_fill_to_k(
                        cand_ids=cand_ids,
                        admissible_by_id=admissible_by_id,
                        q=q,
                        k=args.top_k,
                    )

                    hardval_topk = rerank_hard_validation_fill_to_k(
                        cand_ids=cand_ids,
                        checkable_by_id=checkable_by_id,
                        violating_by_id=violating_by_id,
                        k=args.top_k,
                    )

                    optq_metrics = evaluate_returned_topk(
                        returned_topk=optq_topk,
                        checkable_by_id=checkable_by_id,
                        admissible_by_id=admissible_by_id,
                        violating_by_id=violating_by_id,
                        unknown_by_id=unknown_by_id,
                        base_topk=base_topk,
                        base_rank_map=base_rank_map,
                        k=args.top_k,
                    )
                    afq_metrics = evaluate_returned_topk(
                        returned_topk=afq_topk,
                        checkable_by_id=checkable_by_id,
                        admissible_by_id=admissible_by_id,
                        violating_by_id=violating_by_id,
                        unknown_by_id=unknown_by_id,
                        base_topk=base_topk,
                        base_rank_map=base_rank_map,
                        k=args.top_k,
                    )
                    hardval_metrics = evaluate_returned_topk(
                        returned_topk=hardval_topk,
                        checkable_by_id=checkable_by_id,
                        admissible_by_id=admissible_by_id,
                        violating_by_id=violating_by_id,
                        unknown_by_id=unknown_by_id,
                        base_topk=base_topk,
                        base_rank_map=base_rank_map,
                        k=args.top_k,
                    )

                    optq_adm_count = count_admissible_in_topk(
                        returned_topk=optq_topk,
                        admissible_by_id=admissible_by_id,
                    )
                    afq_adm_count = count_admissible_in_topk(
                        returned_topk=afq_topk,
                        admissible_by_id=admissible_by_id,
                    )
                    hardval_adm_count = count_admissible_in_topk(
                        returned_topk=hardval_topk,
                        admissible_by_id=admissible_by_id,
                    )

                    optq_vs_afq_exact = compute_topk_exact_match(optq_topk, afq_topk)
                    optq_vs_afq_set = compute_topk_set_match(optq_topk, afq_topk)
                    optq_vs_afq_jaccard = compute_topk_set_jaccard(optq_topk, afq_topk)

                    optq_vs_hv_exact = compute_topk_exact_match(optq_topk, hardval_topk)
                    optq_vs_hv_set = compute_topk_set_match(optq_topk, hardval_topk)
                    optq_vs_hv_jaccard = compute_topk_set_jaccard(optq_topk, hardval_topk)

                    aggs[q]["num_feasible"] += 1

                    aggs[q]["optq_viol_sum"] += optq_metrics["viol_at_k"]
                    aggs[q]["optq_adm_sum"] += optq_metrics["adm_at_k"]
                    aggs[q]["optq_cov_sum"] += optq_metrics["cov_at_k"]
                    aggs[q]["optq_unknown_sum"] += optq_metrics["unknown_at_k"]
                    aggs[q]["optq_pres_sum"] += optq_metrics["pres_at_k"]
                    aggs[q]["optq_shift_sum"] += optq_metrics["shift_at_k"]
                    aggs[q]["optq_adm_count_sum"] += optq_adm_count

                    aggs[q]["afq_viol_sum"] += afq_metrics["viol_at_k"]
                    aggs[q]["afq_adm_sum"] += afq_metrics["adm_at_k"]
                    aggs[q]["afq_cov_sum"] += afq_metrics["cov_at_k"]
                    aggs[q]["afq_unknown_sum"] += afq_metrics["unknown_at_k"]
                    aggs[q]["afq_pres_sum"] += afq_metrics["pres_at_k"]
                    aggs[q]["afq_shift_sum"] += afq_metrics["shift_at_k"]
                    aggs[q]["afq_adm_count_sum"] += afq_adm_count

                    aggs[q]["hardval_viol_sum"] += hardval_metrics["viol_at_k"]
                    aggs[q]["hardval_adm_sum"] += hardval_metrics["adm_at_k"]
                    aggs[q]["hardval_cov_sum"] += hardval_metrics["cov_at_k"]
                    aggs[q]["hardval_unknown_sum"] += hardval_metrics["unknown_at_k"]
                    aggs[q]["hardval_pres_sum"] += hardval_metrics["pres_at_k"]
                    aggs[q]["hardval_shift_sum"] += hardval_metrics["shift_at_k"]
                    aggs[q]["hardval_adm_count_sum"] += hardval_adm_count

                    aggs[q]["optq_vs_afq_exact_match_sum"] += optq_vs_afq_exact
                    aggs[q]["optq_vs_afq_set_match_sum"] += optq_vs_afq_set
                    aggs[q]["optq_vs_afq_jaccard_sum"] += optq_vs_afq_jaccard

                    aggs[q]["optq_vs_hardval_exact_match_sum"] += optq_vs_hv_exact
                    aggs[q]["optq_vs_hardval_set_match_sum"] += optq_vs_hv_set
                    aggs[q]["optq_vs_hardval_jaccard_sum"] += optq_vs_hv_jaccard

                    aggs[q]["lambda_values"].append(lambda_star)

                    rel_counter["num_feasible"] += 1

                    rel_counter["optq_viol_sum"] += optq_metrics["viol_at_k"]
                    rel_counter["optq_adm_sum"] += optq_metrics["adm_at_k"]
                    rel_counter["optq_cov_sum"] += optq_metrics["cov_at_k"]
                    rel_counter["optq_unknown_sum"] += optq_metrics["unknown_at_k"]
                    rel_counter["optq_pres_sum"] += optq_metrics["pres_at_k"]
                    rel_counter["optq_shift_sum"] += optq_metrics["shift_at_k"]
                    rel_counter["optq_adm_count_sum"] += optq_adm_count

                    rel_counter["afq_viol_sum"] += afq_metrics["viol_at_k"]
                    rel_counter["afq_adm_sum"] += afq_metrics["adm_at_k"]
                    rel_counter["afq_cov_sum"] += afq_metrics["cov_at_k"]
                    rel_counter["afq_unknown_sum"] += afq_metrics["unknown_at_k"]
                    rel_counter["afq_pres_sum"] += afq_metrics["pres_at_k"]
                    rel_counter["afq_shift_sum"] += afq_metrics["shift_at_k"]
                    rel_counter["afq_adm_count_sum"] += afq_adm_count

                    rel_counter["hardval_viol_sum"] += hardval_metrics["viol_at_k"]
                    rel_counter["hardval_adm_sum"] += hardval_metrics["adm_at_k"]
                    rel_counter["hardval_cov_sum"] += hardval_metrics["cov_at_k"]
                    rel_counter["hardval_unknown_sum"] += hardval_metrics["unknown_at_k"]
                    rel_counter["hardval_pres_sum"] += hardval_metrics["pres_at_k"]
                    rel_counter["hardval_shift_sum"] += hardval_metrics["shift_at_k"]
                    rel_counter["hardval_adm_count_sum"] += hardval_adm_count

                    rel_counter["optq_vs_afq_exact_match_sum"] += optq_vs_afq_exact
                    rel_counter["optq_vs_afq_set_match_sum"] += optq_vs_afq_set
                    rel_counter["optq_vs_afq_jaccard_sum"] += optq_vs_afq_jaccard

                    rel_counter["optq_vs_hardval_exact_match_sum"] += optq_vs_hv_exact
                    rel_counter["optq_vs_hardval_set_match_sum"] += optq_vs_hv_set
                    rel_counter["optq_vs_hardval_jaccard_sum"] += optq_vs_hv_jaccard

                    rel_counter["lambda_sum"] += lambda_star
                else:
                    lambda_star = None
                    tau_q = None
                    optq_metrics = None
                    afq_metrics = None
                    hardval_metrics = None
                    optq_adm_count = None
                    afq_adm_count = None
                    hardval_adm_count = None
                    optq_vs_afq_exact = None
                    optq_vs_afq_set = None
                    optq_vs_afq_jaccard = None
                    optq_vs_hv_exact = None
                    optq_vs_hv_set = None
                    optq_vs_hv_jaccard = None

                row.extend([
                    int(feasible),
                    lambda_star,
                    tau_q,

                    None if optq_metrics is None else optq_metrics["viol_at_k"],
                    None if optq_metrics is None else optq_metrics["adm_at_k"],
                    None if optq_metrics is None else optq_metrics["cov_at_k"],
                    None if optq_metrics is None else optq_metrics["unknown_at_k"],
                    None if optq_metrics is None else optq_metrics["pres_at_k"],
                    None if optq_metrics is None else optq_metrics["shift_at_k"],
                    optq_adm_count,

                    None if afq_metrics is None else afq_metrics["viol_at_k"],
                    None if afq_metrics is None else afq_metrics["adm_at_k"],
                    None if afq_metrics is None else afq_metrics["cov_at_k"],
                    None if afq_metrics is None else afq_metrics["unknown_at_k"],
                    None if afq_metrics is None else afq_metrics["pres_at_k"],
                    None if afq_metrics is None else afq_metrics["shift_at_k"],
                    afq_adm_count,

                    None if hardval_metrics is None else hardval_metrics["viol_at_k"],
                    None if hardval_metrics is None else hardval_metrics["adm_at_k"],
                    None if hardval_metrics is None else hardval_metrics["cov_at_k"],
                    None if hardval_metrics is None else hardval_metrics["unknown_at_k"],
                    None if hardval_metrics is None else hardval_metrics["pres_at_k"],
                    None if hardval_metrics is None else hardval_metrics["shift_at_k"],
                    hardval_adm_count,

                    optq_vs_afq_exact,
                    optq_vs_afq_set,
                    optq_vs_afq_jaccard,

                    optq_vs_hv_exact,
                    optq_vs_hv_set,
                    optq_vs_hv_jaccard,
                ])

            query_rows_buffer.append(row)
            processed += 1

        append_query_rows(query_csv_path, query_rows_buffer)
        query_rows_buffer.clear()

        progress_payload = compute_progress_payload(
            started_at=step_started,
            processed_queries=processed,
            total_queries=total_queries,
            effective_top_m=effective_top_m,
            aggs=aggs,
        )
        save_json_atomic(progress_json_path, progress_payload)

        postfix = {"q": f"{processed}/{total_queries}"}
        for q in quotas[:3]:
            fr = progress_payload["running_metrics"].get(f"q{q}_feasible_rate")
            if fr is not None:
                postfix[f"q{q}"] = f"{fr:.3f}"
        pbar.set_postfix(postfix)

        batch_number = (batch_start // args.query_batch_size) + 1
        if (batch_number % args.log_every) == 0 or batch_end == total_queries:
            parts = [
                f"[STEP 6/7] processed={processed}/{total_queries}",
                f"elapsed={progress_payload['elapsed_human']}",
                f"eta={progress_payload['eta_human']}",
            ]
            for q in quotas[:4]:
                fr = progress_payload["running_metrics"].get(f"q{q}_feasible_rate")
                if fr is not None:
                    parts.append(f"q{q}_feas={fr:.4f}")
            logger.log(" ".join(parts))

        del scores, topm_scores, topm_indices
        maybe_clear_cuda_cache()

    logger.log("[STEP 6/7] Done")

    logger.log("[STEP 7/7] Building final summaries")
    total_elapsed = perf_now() - started_total
    if effective_top_m_global is None:
        raise RuntimeError("Internal error: effective_top_m_global was never set.")

    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "quota",
            "num_queries",
            "num_feasible",
            "feasible_rate",

            "optq_viol_at_k",
            "optq_adm_at_k",
            "optq_cov_at_k",
            "optq_unknown_at_k",
            "optq_pres_at_k",
            "optq_shift_at_k",
            "optq_adm_count_topk",

            "afq_viol_at_k",
            "afq_adm_at_k",
            "afq_cov_at_k",
            "afq_unknown_at_k",
            "afq_pres_at_k",
            "afq_shift_at_k",
            "afq_adm_count_topk",

            "hardval_viol_at_k",
            "hardval_adm_at_k",
            "hardval_cov_at_k",
            "hardval_unknown_at_k",
            "hardval_pres_at_k",
            "hardval_shift_at_k",
            "hardval_adm_count_topk",

            "optq_vs_afq_exact_match_rate",
            "optq_vs_afq_set_match_rate",
            "optq_vs_afq_jaccard_mean",

            "optq_vs_hardval_exact_match_rate",
            "optq_vs_hardval_set_match_rate",
            "optq_vs_hardval_jaccard_mean",

            "mean_lambda",
            "median_lambda",
            "p95_lambda",
        ])

        for q in quotas:
            agg = aggs[q]
            n = agg["num_queries"]
            fn = agg["num_feasible"]

            writer.writerow([
                q,
                n,
                fn,
                (fn / n) if n > 0 else None,

                (agg["optq_viol_sum"] / fn) if fn > 0 else None,
                (agg["optq_adm_sum"] / fn) if fn > 0 else None,
                (agg["optq_cov_sum"] / fn) if fn > 0 else None,
                (agg["optq_unknown_sum"] / fn) if fn > 0 else None,
                (agg["optq_pres_sum"] / fn) if fn > 0 else None,
                (agg["optq_shift_sum"] / fn) if fn > 0 else None,
                (agg["optq_adm_count_sum"] / fn) if fn > 0 else None,

                (agg["afq_viol_sum"] / fn) if fn > 0 else None,
                (agg["afq_adm_sum"] / fn) if fn > 0 else None,
                (agg["afq_cov_sum"] / fn) if fn > 0 else None,
                (agg["afq_unknown_sum"] / fn) if fn > 0 else None,
                (agg["afq_pres_sum"] / fn) if fn > 0 else None,
                (agg["afq_shift_sum"] / fn) if fn > 0 else None,
                (agg["afq_adm_count_sum"] / fn) if fn > 0 else None,

                (agg["hardval_viol_sum"] / fn) if fn > 0 else None,
                (agg["hardval_adm_sum"] / fn) if fn > 0 else None,
                (agg["hardval_cov_sum"] / fn) if fn > 0 else None,
                (agg["hardval_unknown_sum"] / fn) if fn > 0 else None,
                (agg["hardval_pres_sum"] / fn) if fn > 0 else None,
                (agg["hardval_shift_sum"] / fn) if fn > 0 else None,
                (agg["hardval_adm_count_sum"] / fn) if fn > 0 else None,

                (agg["optq_vs_afq_exact_match_sum"] / fn) if fn > 0 else None,
                (agg["optq_vs_afq_set_match_sum"] / fn) if fn > 0 else None,
                (agg["optq_vs_afq_jaccard_sum"] / fn) if fn > 0 else None,

                (agg["optq_vs_hardval_exact_match_sum"] / fn) if fn > 0 else None,
                (agg["optq_vs_hardval_set_match_sum"] / fn) if fn > 0 else None,
                (agg["optq_vs_hardval_jaccard_sum"] / fn) if fn > 0 else None,

                safe_mean(agg["lambda_values"]),
                statistics.median(agg["lambda_values"]) if agg["lambda_values"] else None,
                percentile(agg["lambda_values"], 95),
            ])

    with relation_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "quota",
            "relation",
            "queries",
            "num_feasible",
            "feasible_rate",

            "optq_adm_at_k",
            "afq_adm_at_k",
            "hardval_adm_at_k",

            "optq_vs_afq_set_match_rate",
            "optq_vs_hardval_set_match_rate",

            "mean_lambda",
        ])

        for q in quotas:
            for rel, c in sorted(rel_stats[q].items()):
                qn = c["queries"]
                fn = c["num_feasible"]
                writer.writerow([
                    q,
                    rel,
                    qn,
                    fn,
                    (fn / qn) if qn > 0 else None,

                    (c["optq_adm_sum"] / fn) if fn > 0 else None,
                    (c["afq_adm_sum"] / fn) if fn > 0 else None,
                    (c["hardval_adm_sum"] / fn) if fn > 0 else None,

                    (c["optq_vs_afq_set_match_sum"] / fn) if fn > 0 else None,
                    (c["optq_vs_hardval_set_match_sum"] / fn) if fn > 0 else None,

                    (c["lambda_sum"] / fn) if fn > 0 else None,
                ])

    summary = {
        "status": "ok",
        "experiment_id": "EXP-11",
        "experiment_name": "typed_entity_retrieval_generalization",
        "saved_at_utc": now_iso_utc(),
        "elapsed_seconds": round(total_elapsed, 6),
        "elapsed_human": format_seconds(total_elapsed),

        "dataset_name": args.dataset_name or args.processed_dir.name,
        "processed_dir": str(args.processed_dir),
        "run_dir": str(args.run_dir),
        "checkpoint_path": str(checkpoint_path),

        "split": args.split,
        "mode": args.mode,
        "top_k": args.top_k,
        "top_m_requested": args.top_m,
        "top_m_effective": effective_top_m_global,
        "query_batch_size": args.query_batch_size,
        "device": args.device,
        "seed": args.seed,
        "max_queries": args.max_queries,

        "retrieval_task": {
            "description": (
                "Typed entity retrieval over frozen rankings. Tail queries retrieve entities "
                "compatible with relation range; head queries retrieve entities compatible with relation domain."
            ),
            "target_side_policy": args.target_side_policy,
        },

        "semantic_policy": {
            "check_policy": args.check_policy,
            "unknown_penalty": args.unknown_penalty,
            "note": (
                "EXP-11 evaluates STC beyond triple-level semantics by using entity-type compatibility "
                "as the external semantic signal over frozen entity rankings."
            ),
        },

        "quota_ladder": quotas,

        "dataset_stats": {
            "num_entities": len(bundle.entity_to_id),
            "num_relations": len(bundle.relation_to_id),
            "train_size": int(bundle.train_tf.num_triples),
            "valid_size": int(bundle.valid_tf.num_triples),
            "test_size": int(bundle.test_tf.num_triples),
        },

        "ontology_sidecars": {
            "has_entity_types": ontology.has_entity_types,
            "has_relation_constraints": ontology.has_relation_constraints,
            "num_entities_with_types": len(ontology.entity_types),
            "num_relations_with_domain_constraints": sum(1 for v in ontology.relation_domains.values() if v),
            "num_relations_with_range_constraints": sum(1 for v in ontology.relation_ranges.values() if v),
        },

        "environment": env_metadata,
        "quotas": {},
        "artifacts": {
            "summary_json": str(summary_json_path),
            "progress_json": str(progress_json_path),
            "query_level_csv": str(query_csv_path),
            "quota_summary_csv": str(summary_csv_path),
            "relation_level_csv": str(relation_csv_path),
            "run_log": str(output_dir / "run.log"),
        },
    }

    for q in quotas:
        agg = aggs[q]
        n = agg["num_queries"]
        fn = agg["num_feasible"]
        summary["quotas"][str(q)] = {
            "num_queries": n,
            "num_feasible": fn,
            "feasible_rate": (fn / n) if n > 0 else None,

            "optq": {
                "viol_at_k": (agg["optq_viol_sum"] / fn) if fn > 0 else None,
                "adm_at_k": (agg["optq_adm_sum"] / fn) if fn > 0 else None,
                "cov_at_k": (agg["optq_cov_sum"] / fn) if fn > 0 else None,
                "unknown_at_k": (agg["optq_unknown_sum"] / fn) if fn > 0 else None,
                "pres_at_k": (agg["optq_pres_sum"] / fn) if fn > 0 else None,
                "shift_at_k": (agg["optq_shift_sum"] / fn) if fn > 0 else None,
                "adm_count_topk": (agg["optq_adm_count_sum"] / fn) if fn > 0 else None,
            },

            "admissible_first_quota": {
                "viol_at_k": (agg["afq_viol_sum"] / fn) if fn > 0 else None,
                "adm_at_k": (agg["afq_adm_sum"] / fn) if fn > 0 else None,
                "cov_at_k": (agg["afq_cov_sum"] / fn) if fn > 0 else None,
                "unknown_at_k": (agg["afq_unknown_sum"] / fn) if fn > 0 else None,
                "pres_at_k": (agg["afq_pres_sum"] / fn) if fn > 0 else None,
                "shift_at_k": (agg["afq_shift_sum"] / fn) if fn > 0 else None,
                "adm_count_topk": (agg["afq_adm_count_sum"] / fn) if fn > 0 else None,
            },

            "hard_validation_fill": {
                "viol_at_k": (agg["hardval_viol_sum"] / fn) if fn > 0 else None,
                "adm_at_k": (agg["hardval_adm_sum"] / fn) if fn > 0 else None,
                "cov_at_k": (agg["hardval_cov_sum"] / fn) if fn > 0 else None,
                "unknown_at_k": (agg["hardval_unknown_sum"] / fn) if fn > 0 else None,
                "pres_at_k": (agg["hardval_pres_sum"] / fn) if fn > 0 else None,
                "shift_at_k": (agg["hardval_shift_sum"] / fn) if fn > 0 else None,
                "adm_count_topk": (agg["hardval_adm_count_sum"] / fn) if fn > 0 else None,
            },

            "optq_vs_admissible_first_quota": {
                "exact_match_rate": (agg["optq_vs_afq_exact_match_sum"] / fn) if fn > 0 else None,
                "set_match_rate": (agg["optq_vs_afq_set_match_sum"] / fn) if fn > 0 else None,
                "mean_jaccard": (agg["optq_vs_afq_jaccard_sum"] / fn) if fn > 0 else None,
            },

            "optq_vs_hard_validation_fill": {
                "exact_match_rate": (agg["optq_vs_hardval_exact_match_sum"] / fn) if fn > 0 else None,
                "set_match_rate": (agg["optq_vs_hardval_set_match_sum"] / fn) if fn > 0 else None,
                "mean_jaccard": (agg["optq_vs_hardval_jaccard_sum"] / fn) if fn > 0 else None,
            },

            "lambda_stats": {
                "mean_lambda": safe_mean(agg["lambda_values"]),
                "median_lambda": statistics.median(agg["lambda_values"]) if agg["lambda_values"] else None,
                "p90_lambda": percentile(agg["lambda_values"], 90),
                "p95_lambda": percentile(agg["lambda_values"], 95),
                "p99_lambda": percentile(agg["lambda_values"], 99),
                "max_lambda": max(agg["lambda_values"]) if agg["lambda_values"] else None,
            },
        }

    save_json_atomic(summary_json_path, summary)
    save_json_atomic(progress_json_path, {**summary, "status": "completed"})

    logger.log("[STEP 7/7] Done")
    logger.log(f"[DONE] quotas={quotas} elapsed={summary['elapsed_human']}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "EXP-11: Generalization of STC to typed entity retrieval over frozen entity rankings."
        )
    )

    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-name", type=str, default=None)

    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--mode", type=str, default="all", choices=["tail", "head", "all"])

    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--top-m", type=int, default=20000)
    p.add_argument("--quotas", type=int, nargs="+", default=[1, 3, 5, 10])

    p.add_argument("--query-batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-queries", type=int, default=None)

    p.add_argument("--check-policy", type=str, default="available_all", choices=["available_any", "available_all"])
    p.add_argument("--unknown-penalty", type=float, default=1.0)
    p.add_argument(
        "--target-side-policy",
        type=str,
        default="strict_relation_side",
        choices=["strict_relation_side"],
        help="How expected types are chosen for retrieval queries.",
    )

    p.add_argument("--log-every", type=int, default=5)

    p.add_argument(
        "--query-id-file",
        type=Path,
        default=None,
        help="Optional file with one allowed query_id per line. If provided, only those queries are evaluated.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    if args.top_m < args.top_k:
        raise SystemExit("--top-m must be >= --top-k")

    if args.query_batch_size <= 0:
        raise SystemExit("--query-batch-size must be positive")

    if args.unknown_penalty < 0:
        raise SystemExit("--unknown-penalty must be >= 0")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir / "run.log")

    try:
        run_exp11(args)
    except Exception as e:
        err_payload = {
            "status": "failed",
            "saved_at_utc": now_iso_utc(),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "config": vars(args),
        }
        logger.log(f"[FAILED] {type(e).__name__}: {e}")
        save_json_atomic(output_dir / "summary.json", err_payload)
        save_json_atomic(output_dir / "progress.json", err_payload)
        raise


if __name__ == "__main__":
    main()