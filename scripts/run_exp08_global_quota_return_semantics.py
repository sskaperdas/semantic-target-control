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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(
        "This script requires matplotlib. Install with: pip install matplotlib"
    ) from e

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


def normalize_quotas(quotas: Optional[list[int]], top_k: int) -> list[int]:
    if quotas is None or len(quotas) == 0:
        quotas = list(range(1, top_k + 1))
    out = sorted(set(int(q) for q in quotas))
    for q in out:
        if q < 1:
            raise ValueError("All quotas must be >= 1")
        if q > top_k:
            raise ValueError("All quotas must satisfy quota <= top-k")
    return out


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
        "matplotlib_version": getattr(matplotlib, "__version__", "unknown"),
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
# Semantics
# ============================================================

@dataclass
class CandidateSemanticStatus:
    checkable: bool
    violated: bool
    admissible: bool
    unknown: bool
    energy: float


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
    elif checkable:
        energy = float(violated_reasons)
    else:
        energy = float(unknown_penalty)

    return CandidateSemanticStatus(
        checkable=checkable,
        violated=violated,
        admissible=admissible,
        unknown=unknown,
        energy=energy,
    )


# ============================================================
# Queries
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
                queries.append(QueryItem(qid, row_index, "tail", h, r, t, rel_name, t))

        if mode in ("head", "all"):
            qid = f"{split_name}_head_{row_index:09d}"
            if allowed_query_ids is None or qid in allowed_query_ids:
                queries.append(QueryItem(qid, row_index, "head", h, r, t, rel_name, h))

        if max_queries is not None and len(queries) >= max_queries:
            queries = queries[:max_queries]
            break

    return queries


# ============================================================
# Scoring / ranking helpers
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


def compute_lambda_star_for_quota(
    *,
    base_scores: list[float],
    energies: list[float],
    k: int,
    quota: int,
) -> tuple[bool, Optional[float], Optional[float]]:
    """
    Minimal finite-window pressure for at least `quota` zero-energy
    candidates in the visible top-k list.

    For a top-k quota, up to k-q positive-energy candidates may remain
    above the q-th admissible item. This makes the controller minimal
    for "at least q admissible candidates in top-k", rather than
    conservatively pushing all positive-energy candidates below the
    q-th admissible score.
    """
    if quota < 1:
        raise ValueError(f"quota must be >= 1, got {quota}")
    if quota > k:
        raise ValueError(f"quota must satisfy quota <= k, got quota={quota}, k={k}")

    zero_idxs = [i for i, e in enumerate(energies) if e == 0.0]
    if len(zero_idxs) < quota:
        return False, None, None

    zero_scores = sorted((base_scores[i] for i in zero_idxs), reverse=True)
    tau_q = zero_scores[quota - 1]

    crossings: list[float] = []
    for s, e in zip(base_scores, energies):
        if e > 0.0:
            crossing = (s - tau_q) / e
            if crossing > 0.0:
                crossings.append(float(crossing))

    allowed_above = k - quota
    crossings.sort(reverse=True)

    if len(crossings) <= allowed_above:
        lam = 0.0
    else:
        lam = crossings[allowed_above]

    return True, float(max(0.0, lam)), float(tau_q)


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


def admissible_first_fallback(cand_ids: list[int], semantic_statuses: list[CandidateSemanticStatus], k: int) -> list[int]:
    admissible = [cid for cid, s in zip(cand_ids, semantic_statuses) if s.admissible]
    if len(admissible) >= k:
        return admissible[:k]
    used = set(admissible)
    fill = []
    for cid in cand_ids:
        if cid not in used:
            fill.append(cid)
        if len(admissible) + len(fill) >= k:
            break
    out = (admissible + fill)[:k]
    if len(out) != k:
        raise RuntimeError(f"Fallback failed to build K={k} outputs; got {len(out)}.")
    return out


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
            "target_entity_id",
            "effective_top_m",
            "base_viol_at_k",
            "base_adm_at_k",
            "base_cov_at_k",
            "base_is_blind",
        ]
        for q in quotas:
            header.extend([
                f"q{q}_feasible",
                f"q{q}_used_optq",
                f"q{q}_lambda_star",
                f"q{q}_tau_q",
                f"q{q}_viol_at_k",
                f"q{q}_adm_at_k",
                f"q{q}_cov_at_k",
                f"q{q}_pres_at_k",
                f"q{q}_shift_at_k",
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
    quota_aggs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    elapsed = perf_now() - started_at
    rate = processed_queries / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total_queries - processed_queries)
    eta_sec = (remaining / rate) if rate > 0 else None

    running = {}
    for q, agg in quota_aggs.items():
        n = agg["num_queries"]
        running[f"q{q}_feasible_rate"] = (agg["num_feasible"] / n) if n > 0 else None
        running[f"q{q}_used_optq_rate"] = (agg["num_used_optq"] / n) if n > 0 else None
        running[f"q{q}_viol_at_k"] = (agg["viol_sum"] / n) if n > 0 else None
        running[f"q{q}_adm_at_k"] = (agg["adm_sum"] / n) if n > 0 else None
        running[f"q{q}_pres_at_k"] = (agg["pres_sum"] / n) if n > 0 else None
        running[f"q{q}_shift_at_k"] = (agg["shift_sum"] / n) if n > 0 else None

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
# Plotting
# ============================================================

def save_frontier_plots(
    *,
    output_dir: Path,
    dataset_name: str,
    quotas: list[int],
    quota_aggs: dict[int, dict[str, Any]],
    fallback: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    xs = quotas
    feas = []
    used = []
    viol = []
    adm = []
    pres = []
    shift = []
    med_lambda = []
    p95_lambda = []

    for q in quotas:
        agg = quota_aggs[q]
        n = agg["num_queries"]

        feas.append((agg["num_feasible"] / n) if n > 0 else np.nan)
        used.append((agg["num_used_optq"] / n) if n > 0 else np.nan)
        viol.append((agg["viol_sum"] / n) if n > 0 else np.nan)
        adm.append((agg["adm_sum"] / n) if n > 0 else np.nan)
        pres.append((agg["pres_sum"] / n) if n > 0 else np.nan)
        shift.append((agg["shift_sum"] / n) if n > 0 else np.nan)

        if agg["lambda_values"]:
            med_lambda.append(statistics.median(agg["lambda_values"]))
            p95_lambda.append(percentile(agg["lambda_values"], 95))
        else:
            med_lambda.append(np.nan)
            p95_lambda.append(np.nan)

    artifacts = {}

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(xs, feas, marker="o", label="Feasible rate")
    ax.plot(xs, used, marker="s", label="Used OptQ rate")
    ax.set_xlabel("Quota q")
    ax.set_ylabel("Rate")
    ax.set_title(f"{dataset_name} | Global quota feasibility ({fallback})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "quota_feasibility.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts["quota_feasibility_plot"] = str(path)

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(xs, adm, marker="o", label="Adm@K")
    ax.plot(xs, viol, marker="s", label="Viol@K")
    ax.set_xlabel("Quota q")
    ax.set_ylabel("Returned-list semantics")
    ax.set_title(f"{dataset_name} | Global quota semantics ({fallback})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "quota_semantics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts["quota_semantics_plot"] = str(path)

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(xs, pres, marker="o", label="Pres@K")
    ax.plot(xs, shift, marker="s", label="Shift@K")
    ax.set_xlabel("Quota q")
    ax.set_ylabel("Perturbation")
    ax.set_title(f"{dataset_name} | Global quota perturbation ({fallback})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "quota_perturbation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts["quota_perturbation_plot"] = str(path)

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(xs, med_lambda, marker="o", label="Median lambda*")
    ax.plot(xs, p95_lambda, marker="s", label="P95 lambda*")
    ax.set_xlabel("Quota q")
    ax.set_ylabel("Intervention strength")
    ax.set_title(f"{dataset_name} | Global quota intervention ({fallback})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "quota_lambda.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts["quota_lambda_plot"] = str(path)

    return artifacts


# ============================================================
# Main
# ============================================================

def run_exp08(args: argparse.Namespace) -> None:
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
    logger.log(f"[STEP 5/7] Done | total_queries={total_queries}")

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

    quotas = normalize_quotas(args.quotas, args.top_k)

    query_csv_path = output_dir / "query_level.csv"
    frontier_csv_path = output_dir / "quota_frontier.csv"
    relation_csv_path = output_dir / "relation_level.csv"
    progress_json_path = output_dir / "progress.json"
    summary_json_path = output_dir / "summary.json"

    write_query_csv_header(query_csv_path, quotas)
    env_metadata = get_env_metadata(args.device)

    base_agg = {
        "num_queries": 0,
        "blind_queries": 0,
        "viol_sum": 0.0,
        "adm_sum": 0.0,
        "cov_sum": 0.0,
    }

    quota_aggs: dict[int, dict[str, Any]] = {
        q: {
            "num_queries": 0,
            "num_feasible": 0,
            "num_used_optq": 0,
            "viol_sum": 0.0,
            "adm_sum": 0.0,
            "cov_sum": 0.0,
            "pres_sum": 0.0,
            "shift_sum": 0.0,
            "lambda_values": [],
        }
        for q in quotas
    }

    rel_stats: dict[str, Counter] = defaultdict(Counter)
    query_rows_buffer: list[list[Any]] = []

    processed = 0
    step_started = perf_now()
    effective_top_m_global: Optional[int] = None

    logger.log("[STEP 6/7] Scoring and evaluating global quota return semantics")

    for batch_start in range(0, total_queries, args.query_batch_size):
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
            base_cov_at_k = base_checkable_topk / float(args.top_k)
            base_is_blind = (base_violating_topk > 0) and (admissible_in_topm > 0)

            base_agg["num_queries"] += 1
            base_agg["blind_queries"] += int(base_is_blind)
            base_agg["viol_sum"] += base_viol_at_k
            base_agg["adm_sum"] += base_adm_at_k
            base_agg["cov_sum"] += base_cov_at_k

            rel_counter = rel_stats[q.relation_name]
            rel_counter["queries"] += 1
            rel_counter["base_viol_sum"] += base_viol_at_k
            rel_counter["base_adm_sum"] += base_adm_at_k
            rel_counter["base_cov_sum"] += base_cov_at_k

            row = [
                global_query_index,
                q.query_id,
                q.row_index,
                q.mode,
                q.relation_name,
                q.head_id,
                q.rel_id,
                q.tail_id,
                q.target_entity_id,
                effective_top_m,
                base_viol_at_k,
                base_adm_at_k,
                base_cov_at_k,
                int(base_is_blind),
            ]

            base_topk = cand_ids[:args.top_k]
            base_rank_map = {cid: i + 1 for i, cid in enumerate(cand_ids)}
            checkable_by_id = {cid: s.checkable for cid, s in zip(cand_ids, semantic_statuses)}
            admissible_by_id = {cid: s.admissible for cid, s in zip(cand_ids, semantic_statuses)}
            violated_by_id = {cid: s.violated for cid, s in zip(cand_ids, semantic_statuses)}

            for quota in quotas:
                quota_aggs[quota]["num_queries"] += 1
                rel_counter[f"q{quota}_queries"] += 1

                feasible, lambda_star, tau_q = compute_lambda_star_for_quota(
                    base_scores=cand_scores,
                    energies=energies,
                    k=args.top_k,
                    quota=quota,
                )

                used_optq = False

                if feasible:
                    reranked = stable_rerank_with_lambda(
                        cand_ids=cand_ids,
                        base_scores=cand_scores,
                        energies=energies,
                        lam=lambda_star,
                    )
                    reranked_topk = reranked[:args.top_k]
                    used_optq = True

                    quota_aggs[quota]["num_feasible"] += 1
                    quota_aggs[quota]["num_used_optq"] += 1
                    quota_aggs[quota]["lambda_values"].append(lambda_star)

                    rel_counter[f"q{quota}_num_feasible"] += 1
                    rel_counter[f"q{quota}_num_used_optq"] += 1
                    rel_counter[f"q{quota}_lambda_sum"] += lambda_star
                else:
                    lambda_star = None
                    tau_q = None
                    if args.fallback == "base":
                        reranked_topk = base_topk
                    elif args.fallback == "admissible_first":
                        reranked_topk = admissible_first_fallback(cand_ids, semantic_statuses, args.top_k)
                    else:
                        raise ValueError(f"Unsupported fallback: {args.fallback}")

                out_checkable = 0
                out_violating = 0
                out_admissible = 0
                for cid in reranked_topk:
                    if checkable_by_id[cid]:
                        out_checkable += 1
                        if violated_by_id[cid]:
                            out_violating += 1
                        elif admissible_by_id[cid]:
                            out_admissible += 1

                viol_at_k = (out_violating / out_checkable) if out_checkable > 0 else 0.0
                adm_at_k = out_admissible / float(args.top_k)
                cov_at_k = out_checkable / float(args.top_k)
                pres_at_k = compute_pres_at_k(base_topk, reranked_topk, args.top_k)
                shift_at_k = compute_shift_at_k(base_rank_map, reranked_topk, args.top_k)

                quota_aggs[quota]["viol_sum"] += viol_at_k
                quota_aggs[quota]["adm_sum"] += adm_at_k
                quota_aggs[quota]["cov_sum"] += cov_at_k
                quota_aggs[quota]["pres_sum"] += pres_at_k
                quota_aggs[quota]["shift_sum"] += shift_at_k

                rel_counter[f"q{quota}_viol_sum"] += viol_at_k
                rel_counter[f"q{quota}_adm_sum"] += adm_at_k
                rel_counter[f"q{quota}_cov_sum"] += cov_at_k
                rel_counter[f"q{quota}_pres_sum"] += pres_at_k
                rel_counter[f"q{quota}_shift_sum"] += shift_at_k

                row.extend([
                    int(feasible),
                    int(used_optq),
                    lambda_star,
                    tau_q,
                    viol_at_k,
                    adm_at_k,
                    cov_at_k,
                    pres_at_k,
                    shift_at_k,
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
            quota_aggs=quota_aggs,
        )
        save_json_atomic(progress_json_path, progress_payload)

        batch_number = (batch_start // args.query_batch_size) + 1
        if (batch_number % args.log_every) == 0 or batch_end == total_queries:
            msg = (
                f"[STEP 6/7] processed={processed}/{total_queries} "
                f"elapsed={progress_payload['elapsed_human']} "
                f"eta={progress_payload['eta_human']}"
            )
            if quotas:
                qmax = quotas[-1]
                qmax_feas = progress_payload["running_metrics"].get(f"q{qmax}_feasible_rate")
                qmax_viol = progress_payload["running_metrics"].get(f"q{qmax}_viol_at_k")
                msg += f" q{qmax}_feas={qmax_feas if qmax_feas is not None else 'NA'}"
                msg += f" q{qmax}_viol={qmax_viol if qmax_viol is not None else 'NA'}"
            logger.log(msg)

        del scores, topm_scores, topm_indices
        maybe_clear_cuda_cache()

    logger.log("[STEP 6/7] Done")

    logger.log("[STEP 7/7] Building frontier tables, plots, and final summaries")
    total_elapsed = perf_now() - started_total
    if effective_top_m_global is None:
        raise RuntimeError("Internal error: effective_top_m_global was never set.")

    with frontier_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "quota",
            "fallback",
            "num_queries",
            "num_feasible",
            "feasible_rate",
            "num_used_optq",
            "used_optq_rate",
            "viol_at_k",
            "adm_at_k",
            "cov_at_k",
            "pres_at_k",
            "shift_at_k",
            "mean_lambda",
            "median_lambda",
            "p95_lambda",
        ])

        for quota in quotas:
            agg = quota_aggs[quota]
            n = agg["num_queries"]

            writer.writerow([
                args.dataset_name or args.processed_dir.name,
                quota,
                args.fallback,
                agg["num_queries"],
                agg["num_feasible"],
                (agg["num_feasible"] / n) if n > 0 else None,
                agg["num_used_optq"],
                (agg["num_used_optq"] / n) if n > 0 else None,
                (agg["viol_sum"] / n) if n > 0 else None,
                (agg["adm_sum"] / n) if n > 0 else None,
                (agg["cov_sum"] / n) if n > 0 else None,
                (agg["pres_sum"] / n) if n > 0 else None,
                (agg["shift_sum"] / n) if n > 0 else None,
                safe_mean(agg["lambda_values"]),
                statistics.median(agg["lambda_values"]) if agg["lambda_values"] else None,
                percentile(agg["lambda_values"], 95),
            ])

    plot_artifacts = save_frontier_plots(
        output_dir=output_dir,
        dataset_name=args.dataset_name or args.processed_dir.name,
        quotas=quotas,
        quota_aggs=quota_aggs,
        fallback=args.fallback,
    )

    with relation_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "relation",
            "queries",
            "base_viol_at_k",
            "base_adm_at_k",
            "base_cov_at_k",
        ]
        for q in quotas:
            header.extend([
                f"q{q}_feasible_rate",
                f"q{q}_used_optq_rate",
                f"q{q}_viol_at_k",
                f"q{q}_adm_at_k",
                f"q{q}_cov_at_k",
                f"q{q}_pres_at_k",
                f"q{q}_shift_at_k",
                f"q{q}_mean_lambda",
            ])
        writer.writerow(header)

        for rel, c in sorted(rel_stats.items()):
            qn = c["queries"]
            row = [
                rel,
                c["queries"],
                (c["base_viol_sum"] / qn) if qn > 0 else None,
                (c["base_adm_sum"] / qn) if qn > 0 else None,
                (c["base_cov_sum"] / qn) if qn > 0 else None,
            ]
            for q in quotas:
                n = c[f"q{q}_queries"]
                used_count = c[f"q{q}_num_used_optq"]
                feas_count = c[f"q{q}_num_feasible"]
                row.extend([
                    (feas_count / n) if n > 0 else None,
                    (used_count / n) if n > 0 else None,
                    (c[f"q{q}_viol_sum"] / n) if n > 0 else None,
                    (c[f"q{q}_adm_sum"] / n) if n > 0 else None,
                    (c[f"q{q}_cov_sum"] / n) if n > 0 else None,
                    (c[f"q{q}_pres_sum"] / n) if n > 0 else None,
                    (c[f"q{q}_shift_sum"] / n) if n > 0 else None,
                    (c[f"q{q}_lambda_sum"] / used_count) if used_count > 0 else None,
                ])
            writer.writerow(row)

    n_base = base_agg["num_queries"]

    summary = {
        "status": "ok",
        "experiment_id": "EXP-08",
        "experiment_name": "global_quota_return_semantics",
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
            "fallback_when_infeasible": args.fallback,
            "note": (
                "EXP-08 evaluates OptQ on the full test set. "
                "If quota q is feasible in the top-M window, OptQ is applied; otherwise the configured fallback is returned."
            ),
        },

        "query_subset": {
            "query_id_file": str(args.query_id_file) if args.query_id_file is not None else None,
            "subset_filter_active": args.query_id_file is not None,
            "allowed_query_id_count": (len(allowed_query_ids) if allowed_query_ids is not None else None),
            "matched_query_count": total_queries,
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
        "quota_ladder": quotas,

        "base": {
            "num_queries": base_agg["num_queries"],
            "blind_subset_rate": (base_agg["blind_queries"] / n_base) if n_base > 0 else None,
            "viol_at_k": (base_agg["viol_sum"] / n_base) if n_base > 0 else None,
            "adm_at_k": (base_agg["adm_sum"] / n_base) if n_base > 0 else None,
            "cov_at_k": (base_agg["cov_sum"] / n_base) if n_base > 0 else None,
        },

        "quotas": {},

        "artifacts": {
            "summary_json": str(summary_json_path),
            "progress_json": str(progress_json_path),
            "query_level_csv": str(query_csv_path),
            "quota_frontier_csv": str(frontier_csv_path),
            "relation_level_csv": str(relation_csv_path),
            "run_log": str(output_dir / "run.log"),
            **plot_artifacts,
        },
    }

    for quota in quotas:
        agg = quota_aggs[quota]
        n = agg["num_queries"]

        summary["quotas"][str(quota)] = {
            "num_queries": agg["num_queries"],
            "num_feasible": agg["num_feasible"],
            "feasible_rate": (agg["num_feasible"] / n) if n > 0 else None,
            "num_used_optq": agg["num_used_optq"],
            "used_optq_rate": (agg["num_used_optq"] / n) if n > 0 else None,
            "viol_at_k": (agg["viol_sum"] / n) if n > 0 else None,
            "adm_at_k": (agg["adm_sum"] / n) if n > 0 else None,
            "cov_at_k": (agg["cov_sum"] / n) if n > 0 else None,
            "pres_at_k": (agg["pres_sum"] / n) if n > 0 else None,
            "shift_at_k": (agg["shift_sum"] / n) if n > 0 else None,
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
    logger.log(
        f"[DONE] base_viol={summary['base']['viol_at_k'] if summary['base']['viol_at_k'] is not None else 'NA'} "
        f"quota_ladder={quotas} "
        f"elapsed={summary['elapsed_human']}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP-08: Global quota returned-list semantics over frozen KGC rankings."
    )
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-name", type=str, default=None)

    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--mode", type=str, default="all", choices=["tail", "head", "all"])

    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--top-m", type=int, default=20000)

    p.add_argument("--quotas", type=int, nargs="*", default=None)

    p.add_argument("--query-batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-queries", type=int, default=None)

    p.add_argument("--check-policy", type=str, default="available_all", choices=["available_any", "available_all"])
    p.add_argument("--use-domain", action="store_true")
    p.add_argument("--use-range", action="store_true")
    p.add_argument("--use-disjoint", action="store_true")
    p.add_argument("--unknown-penalty", type=float, default=1.0)

    p.add_argument("--fallback", type=str, default="base", choices=["base", "admissible_first"])

    p.add_argument(
        "--query-id-file",
        type=Path,
        default=None,
        help="Optional file with one allowed query_id per line. If provided, only those queries are evaluated.",
    )

    p.add_argument("--log-every", type=int, default=5)
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

    if args.unknown_penalty < 0:
        raise SystemExit("--unknown-penalty must be >= 0")

    try:
        args.quotas = normalize_quotas(args.quotas, args.top_k)
    except ValueError as e:
        raise SystemExit(str(e))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir / "run.log")

    try:
        run_exp08(args)
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