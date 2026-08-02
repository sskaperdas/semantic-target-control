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
    disjoint_pairs: set[tuple[str, str]]
    has_entity_types: bool
    has_relation_constraints: bool
    has_disjoint_pairs: bool


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


def load_disjoint_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()

    raw = load_json(path)
    out: set[tuple[str, str]] = set()

    def add_pair(a: Any, b: Any) -> None:
        aa = _normalize_type_value(a)
        bb = _normalize_type_value(b)
        if aa and bb:
            out.add((aa, bb))
            out.add((bb, aa))

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, list) and len(item) == 2:
                add_pair(item[0], item[1])
            elif isinstance(item, dict):
                a = item.get("left") or item.get("a") or item.get("source") or item.get("type1")
                b = item.get("right") or item.get("b") or item.get("target") or item.get("type2")
                if a is not None and b is not None:
                    add_pair(a, b)

    elif isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, list):
                for x in v:
                    add_pair(k, x)

    return out


def load_ontology_bundle(processed_dir: Path, entity_to_id: dict[str, int], relation_to_id: dict[str, int]) -> OntologyBundle:
    entity_types_path = processed_dir / "entity_types.json"
    relation_constraints_path = processed_dir / "relation_constraints.json"
    disjoint_pairs_path = processed_dir / "disjoint_pairs.json"

    entity_types = load_entity_types(entity_types_path, entity_to_id)
    relation_domains, relation_ranges = load_relation_constraints(relation_constraints_path, relation_to_id)
    disjoint_pairs = load_disjoint_pairs(disjoint_pairs_path)

    return OntologyBundle(
        entity_types=entity_types,
        relation_domains=relation_domains,
        relation_ranges=relation_ranges,
        disjoint_pairs=disjoint_pairs,
        has_entity_types=entity_types_path.exists(),
        has_relation_constraints=relation_constraints_path.exists(),
        has_disjoint_pairs=disjoint_pairs_path.exists(),
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
# Semantics
# ============================================================

@dataclass
class CandidateSemanticStatus:
    checkable: bool
    violated: bool
    admissible: bool
    unknown: bool
    energy: float
    violated_reasons_count: int


def evaluate_candidate_semantics(
    *,
    head_id: int,
    rel_id: int,
    tail_id: int,
    ontology: OntologyBundle,
    check_policy: str,
    use_domain: bool,
    use_range: bool,
    use_disjoint: bool,
    unknown_penalty: float,
) -> CandidateSemanticStatus:
    h_types = ontology.entity_types.get(head_id, set())
    t_types = ontology.entity_types.get(tail_id, set())
    r_dom = ontology.relation_domains.get(rel_id, set())
    r_rng = ontology.relation_ranges.get(rel_id, set())

    violated_reasons = 0
    evaluable_flags: list[bool] = []

    if use_domain:
        dom_evaluable = bool(h_types) and bool(r_dom)
        evaluable_flags.append(dom_evaluable)
        if dom_evaluable and h_types.isdisjoint(r_dom):
            violated_reasons += 1

    if use_range:
        rng_evaluable = bool(t_types) and bool(r_rng)
        evaluable_flags.append(rng_evaluable)
        if rng_evaluable and t_types.isdisjoint(r_rng):
            violated_reasons += 1

    if use_disjoint:
        disj_evaluable = bool(h_types) and bool(t_types) and bool(ontology.disjoint_pairs)
        evaluable_flags.append(disj_evaluable)
        if disj_evaluable:
            found_disjoint = False
            for ht in h_types:
                for tt in t_types:
                    if (ht, tt) in ontology.disjoint_pairs:
                        found_disjoint = True
                        break
                if found_disjoint:
                    break
            if found_disjoint:
                violated_reasons += 1

    if not evaluable_flags:
        checkable = False
    elif check_policy == "available_all":
        checkable = all(evaluable_flags)
    elif check_policy == "available_any":
        checkable = any(evaluable_flags)
    else:
        raise ValueError(f"Unsupported check_policy: {check_policy}")

    violated = checkable and (violated_reasons > 0)
    admissible = checkable and not violated
    unknown = not checkable

    if admissible:
        energy = 0.0
    elif violated:
        energy = float(violated_reasons)
    else:
        energy = float(unknown_penalty)

    return CandidateSemanticStatus(
        checkable=checkable,
        violated=violated,
        admissible=admissible,
        unknown=unknown,
        energy=energy,
        violated_reasons_count=violated_reasons,
    )


# ============================================================
# Query subset helpers
# ============================================================

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


# ============================================================
# Queries
# ============================================================

@dataclass
class QueryItem:
    query_id: str
    mode: str
    head_id: int
    rel_id: int
    tail_id: int
    relation_name: str
    target_entity_id: int
    row_index: int


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
                        mode="tail",
                        head_id=h,
                        rel_id=r,
                        tail_id=t,
                        relation_name=rel_name,
                        target_entity_id=t,
                        row_index=row_index,
                    )
                )

        if mode in ("head", "all"):
            qid = f"{split_name}_head_{row_index:09d}"
            if allowed_query_ids is None or qid in allowed_query_ids:
                queries.append(
                    QueryItem(
                        query_id=qid,
                        mode="head",
                        head_id=h,
                        rel_id=r,
                        tail_id=t,
                        relation_name=rel_name,
                        target_entity_id=h,
                        row_index=row_index,
                    )
                )

        if max_queries is not None and len(queries) >= max_queries:
            queries = queries[:max_queries]
            break

    return queries


# ============================================================
# Ranking helpers
# ============================================================

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

def rank_of_target_from_scores(scores_row: torch.Tensor, target_entity_id: int) -> int:
    target_score = scores_row[target_entity_id].item()
    better = torch.sum(scores_row > target_score).item()
    return int(better) + 1


def compute_pres_at_k(base_topk: list[int], reranked_topk: list[int], k: int) -> float:
    if len(reranked_topk) != k:
        raise ValueError(f"Expected returned top-k length {k}, got {len(reranked_topk)}")
    return len(set(base_topk).intersection(set(reranked_topk))) / float(k)


def compute_shift_at_k(base_rank_map: dict[int, int], reranked_topk: list[int], k: int) -> float:
    if len(reranked_topk) != k:
        raise ValueError(f"Expected returned top-k length {k}, got {len(reranked_topk)}")
    vals = []
    for out_rank, ent_id in enumerate(reranked_topk, start=1):
        base_rank = base_rank_map.get(ent_id)
        if base_rank is None:
            raise KeyError(f"Entity {ent_id} not found in base_rank_map")
        vals.append(abs(base_rank - out_rank))
    return sum(vals) / float(k)


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


def compute_lambda_star_exact(
    *,
    base_scores: list[float],
    energies: list[float],
    k: int,
) -> tuple[bool, Optional[float], Optional[float]]:
    zero_idxs = [i for i, e in enumerate(energies) if e == 0.0]
    if len(zero_idxs) < k:
        return False, None, None

    zero_scores = sorted((base_scores[i] for i in zero_idxs), reverse=True)
    tau_k = zero_scores[k - 1]

    candidates = []
    for s, e in zip(base_scores, energies):
        if e > 0.0:
            candidates.append((s - tau_k) / e)

    lam = max([0.0] + candidates)
    return True, float(lam), float(tau_k)


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

    raise RuntimeError(
        f"Could not fill result to K={k}; only got {len(out)} items."
    )


def rerank_hard_validation_fill_to_k(
    *,
    cand_ids: list[int],
    checkable_by_id: dict[int, bool],
    violated_by_id: dict[int, bool],
    k: int,
) -> list[int]:
    kept = [
        cid
        for cid in cand_ids
        if not (checkable_by_id.get(cid, False) and violated_by_id.get(cid, False))
    ]
    top_pref = kept[:k]
    return fill_to_k_from_original_order(top_pref, cand_ids, k)


def rerank_admissible_first_fill_to_k(
    *,
    cand_ids: list[int],
    admissible_by_id: dict[int, bool],
    k: int,
) -> list[int]:
    preferred = [cid for cid in cand_ids if admissible_by_id.get(cid, False)]
    top_pref = preferred[:k]
    return fill_to_k_from_original_order(top_pref, cand_ids, k)


def evaluate_returned_topk(
    *,
    returned_topk: list[int],
    checkable_by_id: dict[int, bool],
    admissible_by_id: dict[int, bool],
    violated_by_id: dict[int, bool],
    base_topk: list[int],
    base_rank_map: dict[int, int],
    k: int,
) -> dict[str, float]:
    if len(returned_topk) != k:
        raise ValueError(f"Expected returned top-k length {k}, got {len(returned_topk)}")

    checkable = 0
    violating = 0
    admissible = 0

    for cid in returned_topk:
        if checkable_by_id.get(cid, False):
            checkable += 1
            if violated_by_id.get(cid, False):
                violating += 1
            elif admissible_by_id.get(cid, False):
                admissible += 1

    viol_at_k = (violating / checkable) if checkable > 0 else 0.0
    adm_at_k = admissible / float(k)
    cov_at_k = checkable / float(k)
    pres_at_k = compute_pres_at_k(base_topk, returned_topk, k)
    shift_at_k = compute_shift_at_k(base_rank_map, returned_topk, k)

    return {
        "viol_at_k": viol_at_k,
        "adm_at_k": adm_at_k,
        "cov_at_k": cov_at_k,
        "pres_at_k": pres_at_k,
        "shift_at_k": shift_at_k,
    }


# ============================================================
# CSV / progress
# ============================================================

def write_query_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_index",
            "query_id",
            "row_index",
            "mode",
            "relation",
            "head_id",
            "rel_id",
            "tail_id",
            "target_entity_id",
            "base_target_rank",

            "is_blind_broad",
            "is_blind_strict",
            "admissible_in_topm",
            "admissible_in_topk",
            "admissible_in_topm_but_not_topk",

            "exact_feasible",
            "lambda_star",
            "tau_k",

            "base_viol_at_k_on_blind",
            "base_adm_at_k_on_blind",

            "hardval_viol_at_k",
            "hardval_adm_at_k",
            "hardval_cov_at_k",
            "hardval_pres_at_k",
            "hardval_shift_at_k",

            "adfirst_viol_at_k",
            "adfirst_adm_at_k",
            "adfirst_cov_at_k",
            "adfirst_pres_at_k",
            "adfirst_shift_at_k",

            "glincal_viol_at_k",
            "glincal_adm_at_k",
            "glincal_cov_at_k",
            "glincal_pres_at_k",
            "glincal_shift_at_k",

            "optk_viol_at_k",
            "optk_adm_at_k",
            "optk_cov_at_k",
            "optk_pres_at_k",
            "optk_shift_at_k",
        ])


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
    agg: dict[str, Any],
) -> dict[str, Any]:
    elapsed = perf_now() - started_at
    rate = processed_queries / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total_queries - processed_queries)
    eta_sec = (remaining / rate) if rate > 0 else None

    n_blind = agg["num_blind_strict_queries"]
    n_feas = agg["num_feasible"]

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
        "running_metrics": {
            "blind_strict_subset_rate_so_far": (
                agg["num_blind_strict_queries"] / processed_queries if processed_queries else None
            ),
            "blind_broad_subset_rate_so_far": (
                agg["num_blind_broad_queries"] / processed_queries if processed_queries else None
            ),
            "feasible_rate_among_blind_strict": (
                agg["num_feasible"] / n_blind if n_blind > 0 else None
            ),
            "base_viol_at_k_on_blind": (
                agg["base_viol_sum"] / n_blind if n_blind > 0 else None
            ),
            "base_adm_at_k_on_blind": (
                agg["base_adm_sum"] / n_blind if n_blind > 0 else None
            ),
            "hardval_viol_at_k": (
                agg["hardval_viol_sum"] / n_feas if n_feas > 0 else None
            ),
            "hardval_adm_at_k": (
                agg["hardval_adm_sum"] / n_feas if n_feas > 0 else None
            ),
            "adfirst_viol_at_k": (
                agg["adfirst_viol_sum"] / n_feas if n_feas > 0 else None
            ),
            "adfirst_adm_at_k": (
                agg["adfirst_adm_sum"] / n_feas if n_feas > 0 else None
            ),
            "glincal_viol_at_k": (
                agg["glincal_viol_sum"] / n_feas if n_feas > 0 else None
            ),
            "glincal_adm_at_k": (
                agg["glincal_adm_sum"] / n_feas if n_feas > 0 else None
            ),
            "optk_viol_at_k": (
                agg["optk_viol_sum"] / n_feas if n_feas > 0 else None
            ),
            "optk_adm_at_k": (
                agg["optk_adm_sum"] / n_feas if n_feas > 0 else None
            ),
        },
    }


# ============================================================
# Main
# ============================================================

def run_exp02(args: argparse.Namespace) -> None:
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

    if args.use_disjoint and not ontology.has_disjoint_pairs:
        logger.log("[SANITY] Disjoint requested but not materially available; disabling.")
        args.use_disjoint = False

    logger.log(
        f"[STEP 3/7] Done | has_entity_types={ontology.has_entity_types} "
        f"has_relation_constraints={ontology.has_relation_constraints} "
        f"has_disjoint_pairs={ontology.has_disjoint_pairs} "
        f"active_disjoint={args.use_disjoint}"
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
    if allowed_query_ids is None:
        logger.log("[STEP 5/7] No query-id filter provided; using full eligible query set")
    else:
        logger.log(
            f"[STEP 5/7] Loaded query-id filter with {len(allowed_query_ids)} allowed query ids "
            f"from {args.query_id_file}"
        )

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

    if allowed_query_ids is not None:
        matched_query_ids = {q.query_id for q in queries}
        missing_query_count = len(allowed_query_ids - matched_query_ids)
        logger.log(
            f"[STEP 5/7] Query subset match | matched={len(matched_query_ids)} "
            f"missing={missing_query_count}"
        )
        if missing_query_count > 0:
            logger.log(
                f"[STEP 5/7] WARNING | Some query IDs from subset file were not found "
                f"for split={args.split}, mode={args.mode}"
            )

    logger.log(f"[STEP 5/7] Done | total_queries={total_queries}")

    query_csv_path = output_dir / "query_level.csv"
    relation_csv_path = output_dir / "relation_level.csv"
    progress_json_path = output_dir / "progress.json"
    summary_json_path = output_dir / "summary.json"

    write_query_csv_header(query_csv_path)

    env_metadata = get_env_metadata(args.device)

    agg = {
        "num_blind_broad_queries": 0,
        "num_blind_strict_queries": 0,
        "num_feasible": 0,

        "base_viol_sum": 0.0,
        "base_adm_sum": 0.0,

        "hardval_viol_sum": 0.0,
        "hardval_adm_sum": 0.0,
        "hardval_cov_sum": 0.0,
        "hardval_pres_sum": 0.0,
        "hardval_shift_sum": 0.0,

        "adfirst_viol_sum": 0.0,
        "adfirst_adm_sum": 0.0,
        "adfirst_cov_sum": 0.0,
        "adfirst_pres_sum": 0.0,
        "adfirst_shift_sum": 0.0,

        "glincal_viol_sum": 0.0,
        "glincal_adm_sum": 0.0,
        "glincal_cov_sum": 0.0,
        "glincal_pres_sum": 0.0,
        "glincal_shift_sum": 0.0,

        "optk_viol_sum": 0.0,
        "optk_adm_sum": 0.0,
        "optk_cov_sum": 0.0,
        "optk_pres_sum": 0.0,
        "optk_shift_sum": 0.0,

        "lambda_values": [],
        "target_rank_values": [],
    }

    rel_stats: dict[str, Counter] = defaultdict(Counter)
    query_rows_buffer: list[list[Any]] = []

    processed = 0
    step_started = perf_now()
    effective_top_m_global: Optional[int] = None

    global_lambda = float(args.glincal_lambda)
    logger.log(f"[BASELINE] Using GlobalLinearCal lambda = {global_lambda:.6f}")

    logger.log("[STEP 6/7] Scoring and evaluating exact OptK + baselines on strict blind subset")

    batch_starts = list(range(0, total_queries, args.query_batch_size))
    pbar = tqdm(
        batch_starts,
        desc="EXP-02 batches",
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

        for row_idx, q in enumerate(batch_queries):
            global_query_index = batch_start + row_idx

            scores_row = scores[row_idx]
            cand_ids = topm_indices[row_idx].detach().cpu().tolist()
            cand_scores = topm_scores[row_idx].detach().cpu().tolist()

            semantic_statuses: list[CandidateSemanticStatus] = []
            energies: list[float] = []

            base_checkable_topk = 0
            base_violating_topk = 0
            base_admissible_topk = 0
            admissible_in_topm = 0

            for rank_idx, cand_id in enumerate(cand_ids):
                if q.mode == "tail":
                    head_id, rel_id, tail_id = q.head_id, q.rel_id, cand_id
                else:
                    head_id, rel_id, tail_id = cand_id, q.rel_id, q.tail_id

                sem = evaluate_candidate_semantics(
                    head_id=head_id,
                    rel_id=rel_id,
                    tail_id=tail_id,
                    ontology=ontology,
                    check_policy=args.check_policy,
                    use_domain=args.use_domain,
                    use_range=args.use_range,
                    use_disjoint=args.use_disjoint,
                    unknown_penalty=args.unknown_penalty,
                )
                semantic_statuses.append(sem)
                energies.append(sem.energy)

                if sem.admissible:
                    admissible_in_topm += 1

                if rank_idx < args.top_k:
                    if sem.checkable:
                        base_checkable_topk += 1
                        if sem.violated:
                            base_violating_topk += 1
                        elif sem.admissible:
                            base_admissible_topk += 1

            base_viol_at_k = (base_violating_topk / base_checkable_topk) if base_checkable_topk > 0 else 0.0
            base_adm_at_k = base_admissible_topk / float(args.top_k)

            admissible_in_topk = base_admissible_topk
            admissible_in_topm_but_not_topk = admissible_in_topm - admissible_in_topk

            is_blind_broad = (base_violating_topk > 0) and (admissible_in_topm > 0)
            is_blind_strict = (base_violating_topk > 0) and (admissible_in_topm_but_not_topk > 0)

            target_rank = rank_of_target_from_scores(scores_row, q.target_entity_id)

            hardval_metrics = None
            adfirst_metrics = None
            glincal_metrics = None
            optk_metrics = None

            feasible = False
            lambda_star = None
            tau_k = None

            if is_blind_broad:
                agg["num_blind_broad_queries"] += 1

            if is_blind_strict:
                agg["num_blind_strict_queries"] += 1
                agg["base_viol_sum"] += base_viol_at_k
                agg["base_adm_sum"] += base_adm_at_k
                agg["target_rank_values"].append(target_rank)

                rel_counter = rel_stats[q.relation_name]
                rel_counter["blind_broad_queries"] += int(is_blind_broad)
                rel_counter["blind_strict_queries"] += 1
                rel_counter["base_viol_sum"] += base_viol_at_k
                rel_counter["base_adm_sum"] += base_adm_at_k

                feasible, lambda_star, tau_k = compute_lambda_star_exact(
                    base_scores=cand_scores,
                    energies=energies,
                    k=args.top_k,
                )

                if feasible:
                    agg["num_feasible"] += 1
                    agg["lambda_values"].append(lambda_star)

                    rel_counter["num_feasible"] += 1
                    rel_counter["lambda_sum"] += lambda_star

                    base_topk = cand_ids[:args.top_k]
                    base_rank_map = {cid: i + 1 for i, cid in enumerate(cand_ids)}

                    checkable_by_id = {cid: s.checkable for cid, s in zip(cand_ids, semantic_statuses)}
                    admissible_by_id = {cid: s.admissible for cid, s in zip(cand_ids, semantic_statuses)}
                    violated_by_id = {cid: s.violated for cid, s in zip(cand_ids, semantic_statuses)}

                    hardval_topk = rerank_hard_validation_fill_to_k(
                        cand_ids=cand_ids,
                        checkable_by_id=checkable_by_id,
                        violated_by_id=violated_by_id,
                        k=args.top_k,
                    )
                    hardval_metrics = evaluate_returned_topk(
                        returned_topk=hardval_topk,
                        checkable_by_id=checkable_by_id,
                        admissible_by_id=admissible_by_id,
                        violated_by_id=violated_by_id,
                        base_topk=base_topk,
                        base_rank_map=base_rank_map,
                        k=args.top_k,
                    )

                    agg["hardval_viol_sum"] += hardval_metrics["viol_at_k"]
                    agg["hardval_adm_sum"] += hardval_metrics["adm_at_k"]
                    agg["hardval_cov_sum"] += hardval_metrics["cov_at_k"]
                    agg["hardval_pres_sum"] += hardval_metrics["pres_at_k"]
                    agg["hardval_shift_sum"] += hardval_metrics["shift_at_k"]

                    rel_counter["hardval_viol_sum"] += hardval_metrics["viol_at_k"]
                    rel_counter["hardval_adm_sum"] += hardval_metrics["adm_at_k"]
                    rel_counter["hardval_cov_sum"] += hardval_metrics["cov_at_k"]
                    rel_counter["hardval_pres_sum"] += hardval_metrics["pres_at_k"]
                    rel_counter["hardval_shift_sum"] += hardval_metrics["shift_at_k"]

                    adfirst_topk = rerank_admissible_first_fill_to_k(
                        cand_ids=cand_ids,
                        admissible_by_id=admissible_by_id,
                        k=args.top_k,
                    )
                    adfirst_metrics = evaluate_returned_topk(
                        returned_topk=adfirst_topk,
                        checkable_by_id=checkable_by_id,
                        admissible_by_id=admissible_by_id,
                        violated_by_id=violated_by_id,
                        base_topk=base_topk,
                        base_rank_map=base_rank_map,
                        k=args.top_k,
                    )

                    agg["adfirst_viol_sum"] += adfirst_metrics["viol_at_k"]
                    agg["adfirst_adm_sum"] += adfirst_metrics["adm_at_k"]
                    agg["adfirst_cov_sum"] += adfirst_metrics["cov_at_k"]
                    agg["adfirst_pres_sum"] += adfirst_metrics["pres_at_k"]
                    agg["adfirst_shift_sum"] += adfirst_metrics["shift_at_k"]

                    rel_counter["adfirst_viol_sum"] += adfirst_metrics["viol_at_k"]
                    rel_counter["adfirst_adm_sum"] += adfirst_metrics["adm_at_k"]
                    rel_counter["adfirst_cov_sum"] += adfirst_metrics["cov_at_k"]
                    rel_counter["adfirst_pres_sum"] += adfirst_metrics["pres_at_k"]
                    rel_counter["adfirst_shift_sum"] += adfirst_metrics["shift_at_k"]

                    glincal_ranked = stable_rerank_with_lambda(
                        cand_ids=cand_ids,
                        base_scores=cand_scores,
                        energies=energies,
                        lam=global_lambda,
                    )
                    glincal_topk = glincal_ranked[:args.top_k]
                    glincal_metrics = evaluate_returned_topk(
                        returned_topk=glincal_topk,
                        checkable_by_id=checkable_by_id,
                        admissible_by_id=admissible_by_id,
                        violated_by_id=violated_by_id,
                        base_topk=base_topk,
                        base_rank_map=base_rank_map,
                        k=args.top_k,
                    )

                    agg["glincal_viol_sum"] += glincal_metrics["viol_at_k"]
                    agg["glincal_adm_sum"] += glincal_metrics["adm_at_k"]
                    agg["glincal_cov_sum"] += glincal_metrics["cov_at_k"]
                    agg["glincal_pres_sum"] += glincal_metrics["pres_at_k"]
                    agg["glincal_shift_sum"] += glincal_metrics["shift_at_k"]

                    rel_counter["glincal_viol_sum"] += glincal_metrics["viol_at_k"]
                    rel_counter["glincal_adm_sum"] += glincal_metrics["adm_at_k"]
                    rel_counter["glincal_cov_sum"] += glincal_metrics["cov_at_k"]
                    rel_counter["glincal_pres_sum"] += glincal_metrics["pres_at_k"]
                    rel_counter["glincal_shift_sum"] += glincal_metrics["shift_at_k"]

                    optk_ranked = stable_rerank_with_lambda(
                        cand_ids=cand_ids,
                        base_scores=cand_scores,
                        energies=energies,
                        lam=lambda_star,
                    )
                    optk_topk = optk_ranked[:args.top_k]
                    optk_metrics = evaluate_returned_topk(
                        returned_topk=optk_topk,
                        checkable_by_id=checkable_by_id,
                        admissible_by_id=admissible_by_id,
                        violated_by_id=violated_by_id,
                        base_topk=base_topk,
                        base_rank_map=base_rank_map,
                        k=args.top_k,
                    )

                    agg["optk_viol_sum"] += optk_metrics["viol_at_k"]
                    agg["optk_adm_sum"] += optk_metrics["adm_at_k"]
                    agg["optk_cov_sum"] += optk_metrics["cov_at_k"]
                    agg["optk_pres_sum"] += optk_metrics["pres_at_k"]
                    agg["optk_shift_sum"] += optk_metrics["shift_at_k"]

                    rel_counter["optk_viol_sum"] += optk_metrics["viol_at_k"]
                    rel_counter["optk_adm_sum"] += optk_metrics["adm_at_k"]
                    rel_counter["optk_cov_sum"] += optk_metrics["cov_at_k"]
                    rel_counter["optk_pres_sum"] += optk_metrics["pres_at_k"]
                    rel_counter["optk_shift_sum"] += optk_metrics["shift_at_k"]

            query_rows_buffer.append([
                global_query_index,
                q.query_id,
                q.row_index,
                q.mode,
                q.relation_name,
                q.head_id,
                q.rel_id,
                q.tail_id,
                q.target_entity_id,
                target_rank,

                int(is_blind_broad),
                int(is_blind_strict),
                admissible_in_topm,
                admissible_in_topk,
                admissible_in_topm_but_not_topk,

                int(feasible),
                lambda_star,
                tau_k,

                base_viol_at_k if is_blind_strict else None,
                base_adm_at_k if is_blind_strict else None,

                None if hardval_metrics is None else hardval_metrics["viol_at_k"],
                None if hardval_metrics is None else hardval_metrics["adm_at_k"],
                None if hardval_metrics is None else hardval_metrics["cov_at_k"],
                None if hardval_metrics is None else hardval_metrics["pres_at_k"],
                None if hardval_metrics is None else hardval_metrics["shift_at_k"],

                None if adfirst_metrics is None else adfirst_metrics["viol_at_k"],
                None if adfirst_metrics is None else adfirst_metrics["adm_at_k"],
                None if adfirst_metrics is None else adfirst_metrics["cov_at_k"],
                None if adfirst_metrics is None else adfirst_metrics["pres_at_k"],
                None if adfirst_metrics is None else adfirst_metrics["shift_at_k"],

                None if glincal_metrics is None else glincal_metrics["viol_at_k"],
                None if glincal_metrics is None else glincal_metrics["adm_at_k"],
                None if glincal_metrics is None else glincal_metrics["cov_at_k"],
                None if glincal_metrics is None else glincal_metrics["pres_at_k"],
                None if glincal_metrics is None else glincal_metrics["shift_at_k"],

                None if optk_metrics is None else optk_metrics["viol_at_k"],
                None if optk_metrics is None else optk_metrics["adm_at_k"],
                None if optk_metrics is None else optk_metrics["cov_at_k"],
                None if optk_metrics is None else optk_metrics["pres_at_k"],
                None if optk_metrics is None else optk_metrics["shift_at_k"],
            ])

            processed += 1

        append_query_rows(query_csv_path, query_rows_buffer)
        query_rows_buffer.clear()

        progress_payload = compute_progress_payload(
            started_at=step_started,
            processed_queries=processed,
            total_queries=total_queries,
            effective_top_m=effective_top_m,
            agg=agg,
        )
        save_json_atomic(progress_json_path, progress_payload)

        blind_strict_rate = progress_payload["running_metrics"]["blind_strict_subset_rate_so_far"]
        feasible_blind = progress_payload["running_metrics"]["feasible_rate_among_blind_strict"]

        pbar.set_postfix({
            "q": f"{processed}/{total_queries}",
            "blind_strict": f"{blind_strict_rate:.4f}" if blind_strict_rate is not None else "nan",
            "feas_blind": f"{feasible_blind:.4f}" if feasible_blind is not None else "nan",
        })

        batch_number = (batch_start // args.query_batch_size) + 1
        if (batch_number % args.log_every) == 0 or batch_end == total_queries:
            logger.log(
                f"[STEP 6/7] processed={processed}/{total_queries} "
                f"elapsed={progress_payload['elapsed_human']} "
                f"eta={progress_payload['eta_human']} "
                f"blind_strict={blind_strict_rate if blind_strict_rate is not None else 'NA'} "
                f"feasible_blind={feasible_blind if feasible_blind is not None else 'NA'}"
            )

        del scores, topm_scores, topm_indices
        maybe_clear_cuda_cache()

    logger.log("[STEP 6/7] Done")

    logger.log("[STEP 7/7] Building final summaries")
    total_elapsed = perf_now() - started_total
    if effective_top_m_global is None:
        raise RuntimeError("Internal error: effective_top_m_global was never set.")

    with relation_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "relation",
            "blind_broad_queries",
            "blind_strict_queries",
            "feasible_rate_among_blind_strict",

            "base_viol_at_k_on_blind",
            "base_adm_at_k_on_blind",

            "hardval_viol_at_k",
            "hardval_adm_at_k",
            "hardval_cov_at_k",
            "hardval_pres_at_k",
            "hardval_shift_at_k",

            "adfirst_viol_at_k",
            "adfirst_adm_at_k",
            "adfirst_cov_at_k",
            "adfirst_pres_at_k",
            "adfirst_shift_at_k",

            "glincal_viol_at_k",
            "glincal_adm_at_k",
            "glincal_cov_at_k",
            "glincal_pres_at_k",
            "glincal_shift_at_k",

            "optk_viol_at_k",
            "optk_adm_at_k",
            "optk_cov_at_k",
            "optk_pres_at_k",
            "optk_shift_at_k",

            "mean_lambda",
        ])

        for rel, c in sorted(rel_stats.items()):
            blind_n = c["blind_strict_queries"]
            feas_n = c["num_feasible"]

            writer.writerow([
                rel,
                c["blind_broad_queries"],
                blind_n,
                (feas_n / blind_n) if blind_n > 0 else None,

                (c["base_viol_sum"] / blind_n) if blind_n > 0 else None,
                (c["base_adm_sum"] / blind_n) if blind_n > 0 else None,

                (c["hardval_viol_sum"] / feas_n) if feas_n > 0 else None,
                (c["hardval_adm_sum"] / feas_n) if feas_n > 0 else None,
                (c["hardval_cov_sum"] / feas_n) if feas_n > 0 else None,
                (c["hardval_pres_sum"] / feas_n) if feas_n > 0 else None,
                (c["hardval_shift_sum"] / feas_n) if feas_n > 0 else None,

                (c["adfirst_viol_sum"] / feas_n) if feas_n > 0 else None,
                (c["adfirst_adm_sum"] / feas_n) if feas_n > 0 else None,
                (c["adfirst_cov_sum"] / feas_n) if feas_n > 0 else None,
                (c["adfirst_pres_sum"] / feas_n) if feas_n > 0 else None,
                (c["adfirst_shift_sum"] / feas_n) if feas_n > 0 else None,

                (c["glincal_viol_sum"] / feas_n) if feas_n > 0 else None,
                (c["glincal_adm_sum"] / feas_n) if feas_n > 0 else None,
                (c["glincal_cov_sum"] / feas_n) if feas_n > 0 else None,
                (c["glincal_pres_sum"] / feas_n) if feas_n > 0 else None,
                (c["glincal_shift_sum"] / feas_n) if feas_n > 0 else None,

                (c["optk_viol_sum"] / feas_n) if feas_n > 0 else None,
                (c["optk_adm_sum"] / feas_n) if feas_n > 0 else None,
                (c["optk_cov_sum"] / feas_n) if feas_n > 0 else None,
                (c["optk_pres_sum"] / feas_n) if feas_n > 0 else None,
                (c["optk_shift_sum"] / feas_n) if feas_n > 0 else None,

                (c["lambda_sum"] / feas_n) if feas_n > 0 else None,
            ])

    blind_n = agg["num_blind_strict_queries"]
    feas_n = agg["num_feasible"]

    median_lambda = statistics.median(agg["lambda_values"]) if agg["lambda_values"] else None
    median_target_rank = statistics.median(agg["target_rank_values"]) if agg["target_rank_values"] else None

    summary = {
        "status": "ok",
        "experiment_id": "EXP-02",
        "experiment_name": "exact_blind_subset_bench",
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

        "semantic_policy": {
            "check_policy": args.check_policy,
            "use_domain": args.use_domain,
            "use_range": args.use_range,
            "use_disjoint": args.use_disjoint,
            "unknown_penalty": args.unknown_penalty,
            "note": (
                "EXP-02 exact feasibility is defined by zero-energy admissible candidates only; "
                "unknown_penalty affects reranking energies for baselines and exact control but does not count unknowns as feasible admissible items."
            ),
        },

        "query_subset": {
            "query_id_file": str(args.query_id_file) if args.query_id_file is not None else None,
            "subset_filter_active": args.query_id_file is not None,
            "allowed_query_id_count": (len(allowed_query_ids) if allowed_query_ids is not None else None),
            "matched_query_count": total_queries,
        },

        "blindness_definition": {
            "blind_broad": (
                "violation present in returned top-K and at least one admissible candidate exists in top-M"
            ),
            "blind_strict": (
                "violation present in returned top-K and at least one admissible candidate exists outside top-K but inside top-M"
            ),
            "benchmark_subset_used": "blind_strict",
        },

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
            "has_disjoint_pairs": ontology.has_disjoint_pairs,
            "num_entities_with_types": len(ontology.entity_types),
            "num_relations_with_domain_constraints": sum(1 for v in ontology.relation_domains.values() if v),
            "num_relations_with_range_constraints": sum(1 for v in ontology.relation_ranges.values() if v),
            "num_disjoint_pairs": len(ontology.disjoint_pairs),
        },

        "environment": env_metadata,

        "queries": {
            "total_queries": total_queries,
            "blind_broad_queries_count": agg["num_blind_broad_queries"],
            "blind_strict_queries_count": blind_n,
            "blind_broad_subset_rate": (
                agg["num_blind_broad_queries"] / total_queries if total_queries else None
            ),
            "blind_strict_subset_rate": (
                blind_n / total_queries if total_queries else None
            ),
            "num_feasible_among_blind_strict": feas_n,
            "feasible_rate_among_blind_strict": (
                feas_n / blind_n if blind_n > 0 else None
            ),
            "median_target_rank_among_blind_strict": median_target_rank,
        },

        "base_on_blind_strict": {
            "viol_at_k": (agg["base_viol_sum"] / blind_n) if blind_n > 0 else None,
            "adm_at_k": (agg["base_adm_sum"] / blind_n) if blind_n > 0 else None,
        },

        "hardval_on_feasible_blind_strict": {
            "viol_at_k": (agg["hardval_viol_sum"] / feas_n) if feas_n > 0 else None,
            "adm_at_k": (agg["hardval_adm_sum"] / feas_n) if feas_n > 0 else None,
            "cov_at_k": (agg["hardval_cov_sum"] / feas_n) if feas_n > 0 else None,
            "pres_at_k": (agg["hardval_pres_sum"] / feas_n) if feas_n > 0 else None,
            "shift_at_k": (agg["hardval_shift_sum"] / feas_n) if feas_n > 0 else None,
        },

        "admissible_first_on_feasible_blind_strict": {
            "viol_at_k": (agg["adfirst_viol_sum"] / feas_n) if feas_n > 0 else None,
            "adm_at_k": (agg["adfirst_adm_sum"] / feas_n) if feas_n > 0 else None,
            "cov_at_k": (agg["adfirst_cov_sum"] / feas_n) if feas_n > 0 else None,
            "pres_at_k": (agg["adfirst_pres_sum"] / feas_n) if feas_n > 0 else None,
            "shift_at_k": (agg["adfirst_shift_sum"] / feas_n) if feas_n > 0 else None,
        },

        "glincal_on_feasible_blind_strict": {
            "viol_at_k": (agg["glincal_viol_sum"] / feas_n) if feas_n > 0 else None,
            "adm_at_k": (agg["glincal_adm_sum"] / feas_n) if feas_n > 0 else None,
            "cov_at_k": (agg["glincal_cov_sum"] / feas_n) if feas_n > 0 else None,
            "pres_at_k": (agg["glincal_pres_sum"] / feas_n) if feas_n > 0 else None,
            "shift_at_k": (agg["glincal_shift_sum"] / feas_n) if feas_n > 0 else None,
            "global_lambda": args.glincal_lambda,
        },

        "optk_on_feasible_blind_strict": {
            "viol_at_k": (agg["optk_viol_sum"] / feas_n) if feas_n > 0 else None,
            "adm_at_k": (agg["optk_adm_sum"] / feas_n) if feas_n > 0 else None,
            "cov_at_k": (agg["optk_cov_sum"] / feas_n) if feas_n > 0 else None,
            "pres_at_k": (agg["optk_pres_sum"] / feas_n) if feas_n > 0 else None,
            "shift_at_k": (agg["optk_shift_sum"] / feas_n) if feas_n > 0 else None,
            "median_lambda": median_lambda,
        },

        "artifacts": {
            "summary_json": str(summary_json_path),
            "progress_json": str(progress_json_path),
            "query_level_csv": str(query_csv_path),
            "relation_level_csv": str(relation_csv_path),
            "run_log": str(output_dir / "run.log"),
        },
    }

    save_json_atomic(summary_json_path, summary)
    save_json_atomic(progress_json_path, {**summary, "status": "completed"})

    logger.log("[STEP 7/7] Done")
    logger.log(
        f"[DONE] blind_strict_subset_rate={summary['queries']['blind_strict_subset_rate'] if summary['queries']['blind_strict_subset_rate'] is not None else 'NA'} "
        f"feasible_blind_strict={summary['queries']['feasible_rate_among_blind_strict'] if summary['queries']['feasible_rate_among_blind_strict'] is not None else 'NA'} "
        f"base_viol={summary['base_on_blind_strict']['viol_at_k'] if summary['base_on_blind_strict']['viol_at_k'] is not None else 'NA'} "
        f"hardval_viol={summary['hardval_on_feasible_blind_strict']['viol_at_k'] if summary['hardval_on_feasible_blind_strict']['viol_at_k'] is not None else 'NA'} "
        f"adfirst_viol={summary['admissible_first_on_feasible_blind_strict']['viol_at_k'] if summary['admissible_first_on_feasible_blind_strict']['viol_at_k'] is not None else 'NA'} "
        f"glincal_viol={summary['glincal_on_feasible_blind_strict']['viol_at_k'] if summary['glincal_on_feasible_blind_strict']['viol_at_k'] is not None else 'NA'} "
        f"optk_viol={summary['optk_on_feasible_blind_strict']['viol_at_k'] if summary['optk_on_feasible_blind_strict']['viol_at_k'] is not None else 'NA'} "
        f"elapsed={summary['elapsed_human']}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP-02: Exact OptK benchmark on the strict blind subset with baselines."
    )
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-name", type=str, default=None)

    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--mode", type=str, default="all", choices=["tail", "head", "all"])

    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--top-m", type=int, default=20000)

    p.add_argument("--query-batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-queries", type=int, default=None)

    p.add_argument("--check-policy", type=str, default="available_all", choices=["available_any", "available_all"])
    p.add_argument("--use-domain", action="store_true")
    p.add_argument("--use-range", action="store_true")
    p.add_argument("--use-disjoint", action="store_true")
    p.add_argument("--unknown-penalty", type=float, default=1.0)

    p.add_argument(
        "--glincal-lambda",
        type=float,
        default=1.0,
        help="Shared global lambda for the GlobalLinearCal baseline.",
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

    if not (args.use_domain or args.use_range or args.use_disjoint):
        args.use_domain = True
        args.use_range = True

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    if args.top_m < args.top_k:
        raise SystemExit("--top-m must be >= --top-k")

    if args.query_batch_size <= 0:
        raise SystemExit("--query-batch-size must be positive")

    if args.glincal_lambda < 0:
        raise SystemExit("--glincal-lambda must be >= 0")

    if args.unknown_penalty < 0:
        raise SystemExit("--unknown-penalty must be >= 0")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir / "run.log")

    try:
        run_exp02(args)
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