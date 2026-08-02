#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
# Time / JSON / logging helpers
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
# Seed / environment
# ============================================================

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
            current_idx = torch.device(device_str).index
            if current_idx is None:
                current_idx = torch.cuda.current_device()
            info["cuda_device_index"] = current_idx
            info["cuda_device_name"] = torch.cuda.get_device_name(current_idx)
            info["cuda_device_capability"] = list(torch.cuda.get_device_capability(current_idx))
        except Exception as e:
            info["cuda_metadata_error"] = f"{type(e).__name__}: {e}"

    return info


# ============================================================
# Dataset loading
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

    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    return DatasetBundle(
        train_tf=train_tf,
        valid_tf=valid_tf,
        test_tf=test_tf,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        id_to_entity=id_to_entity,
        id_to_relation=id_to_relation,
    )


# ============================================================
# Ontology sidecars
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
            if ent_id is None:
                continue
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

            if ent_id is None:
                continue
            out[ent_id] = _extract_type_set(item)

    return out


def load_relation_constraints(
    path: Path,
    relation_to_id: dict[str, int],
) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
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


def load_ontology_bundle(
    processed_dir: Path,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> OntologyBundle:
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
# Checkpoint / model loading
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
# Semantic checking
# ============================================================

@dataclass
class CandidateSemanticStatus:
    checkable: bool
    violated: bool
    admissible: bool
    reasons_checked: list[str]
    reasons_violated: list[str]


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
) -> CandidateSemanticStatus:
    h_types = ontology.entity_types.get(head_id, set())
    t_types = ontology.entity_types.get(tail_id, set())
    r_dom = ontology.relation_domains.get(rel_id, set())
    r_rng = ontology.relation_ranges.get(rel_id, set())

    checked: list[str] = []
    violated: list[str] = []
    evaluable_flags: list[bool] = []

    if use_domain:
        dom_evaluable = bool(h_types) and bool(r_dom)
        evaluable_flags.append(dom_evaluable)
        if dom_evaluable:
            checked.append("domain")
            if h_types.isdisjoint(r_dom):
                violated.append("domain")

    if use_range:
        rng_evaluable = bool(t_types) and bool(r_rng)
        evaluable_flags.append(rng_evaluable)
        if rng_evaluable:
            checked.append("range")
            if t_types.isdisjoint(r_rng):
                violated.append("range")

    if use_disjoint:
        disj_evaluable = bool(h_types) and bool(t_types) and bool(ontology.disjoint_pairs)
        evaluable_flags.append(disj_evaluable)
        if disj_evaluable:
            checked.append("disjoint")
            found_disjoint = False
            for ht in h_types:
                for tt in t_types:
                    if (ht, tt) in ontology.disjoint_pairs:
                        found_disjoint = True
                        break
                if found_disjoint:
                    break
            if found_disjoint:
                violated.append("disjoint")

    if not evaluable_flags:
        checkable = False
    elif check_policy == "available_all":
        checkable = all(evaluable_flags)
    elif check_policy == "available_any":
        checkable = any(evaluable_flags)
    else:
        raise ValueError(f"Unsupported check_policy: {check_policy}")

    is_violated = checkable and bool(violated)
    is_admissible = checkable and not is_violated

    return CandidateSemanticStatus(
        checkable=checkable,
        violated=is_violated,
        admissible=is_admissible,
        reasons_checked=checked,
        reasons_violated=violated,
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
# Query builders
# ============================================================

@dataclass
class QueryItem:
    query_id: str
    mode: str   # "tail" or "head"
    head_id: int
    rel_id: int
    tail_id: int
    relation_name: str
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
                        row_index=row_index,
                    )
                )

        if max_queries is not None and len(queries) >= max_queries:
            queries = queries[:max_queries]
            break

    return queries


# ============================================================
# Running aggregation
# ============================================================

@dataclass
class RunningAgg:
    total_queries: int = 0
    blind_queries: int = 0
    blind_strict_queries: int = 0

    checkable_topk_items: int = 0
    violating_topk_items: int = 0
    admissible_topk_items: int = 0
    unknown_topk_items: int = 0

    feasible_counts: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "blind_queries": self.blind_queries,
            "blind_strict_queries": self.blind_strict_queries,
            "checkable_topk_items": self.checkable_topk_items,
            "violating_topk_items": self.violating_topk_items,
            "admissible_topk_items": self.admissible_topk_items,
            "unknown_topk_items": self.unknown_topk_items,
            "feasible_counts": dict(self.feasible_counts),
        }


# ============================================================
# Scoring
# ============================================================

def score_batch(
    model: Any,
    queries: list[QueryItem],
    device: str,
) -> torch.Tensor:
    if not queries:
        raise ValueError("score_batch() received an empty query batch")

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
                raise RuntimeError(f"Expected 2D tail score tensor, got shape={tuple(tail_scores.shape)}")

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
                raise RuntimeError(f"Expected 2D head score tensor, got shape={tuple(head_scores.shape)}")

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

# ============================================================
# CSV output
# ============================================================

def write_query_csv_header(path: Path, quotas: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        base_cols = [
            "query_index",
            "query_id",
            "row_index",
            "mode",
            "relation",
            "head_id",
            "rel_id",
            "tail_id",
            "blind_at_k",
            "blind_strict_at_k",
            "violating_in_topk",
            "admissible_in_topm",
            "admissible_in_topk",
            "admissible_in_topm_but_not_topk",
            "checkable_in_topm",
            "unknown_in_topm",
            "checkable_in_topk",
            "violating_in_topk_count",
            "admissible_in_topk_count",
            "unknown_in_topk_count",
        ]
        quota_cols = [f"feas_{q}_at_k" for q in quotas]
        writer.writerow(base_cols + quota_cols)


def append_query_rows(path: Path, rows: list[list[Any]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# ============================================================
# Progress payload
# ============================================================

def compute_progress_payload(
    *,
    args: argparse.Namespace,
    started_at: float,
    processed_queries: int,
    total_queries: int,
    agg: RunningAgg,
    quotas: list[int],
    effective_top_m: int,
) -> dict[str, Any]:
    elapsed = perf_now() - started_at
    rate = processed_queries / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total_queries - processed_queries)
    eta_sec = (remaining / rate) if rate > 0 else None

    denom_topk = agg.total_queries * args.top_k if agg.total_queries > 0 else 0

    return {
        "status": "running",
        "updated_at_utc": now_iso_utc(),
        "processed_queries": processed_queries,
        "total_queries": total_queries,
        "progress_fraction": (processed_queries / total_queries) if total_queries > 0 else 0.0,
        "elapsed_seconds": round(elapsed, 6),
        "elapsed_human": format_seconds(elapsed),
        "eta_seconds": None if eta_sec is None else round(eta_sec, 6),
        "eta_human": format_seconds(eta_sec),
        "queries_per_second": round(rate, 4),
        "effective_top_m": effective_top_m,
        "running_metrics": {
            f"Blind@{args.top_k}": (agg.blind_queries / agg.total_queries) if agg.total_queries else None,
            f"BlindStrict@{args.top_k}": (agg.blind_strict_queries / agg.total_queries) if agg.total_queries else None,
            f"Cov@{args.top_k}": (agg.checkable_topk_items / denom_topk) if denom_topk else None,
            f"Viol@{args.top_k}": (
                agg.violating_topk_items / agg.checkable_topk_items
                if agg.checkable_topk_items > 0 else None
            ),
            f"Adm@{args.top_k}": (agg.admissible_topk_items / denom_topk) if denom_topk else None,
            f"Unknown@{args.top_k}": (agg.unknown_topk_items / denom_topk) if denom_topk else None,
            **{
                f"Feas_{q}@{args.top_k}": (
                    agg.feasible_counts.get(q, 0) / agg.total_queries if agg.total_queries else None
                )
                for q in quotas
            },
        },
    }


# ============================================================
# Main EXP-01
# ============================================================

def run_exp01(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir / "run.log")

    save_json_atomic(output_dir / "config.json", vars(args))
    set_all_seeds(args.seed)

    started_total = perf_now()

    logger.log("[STEP 1/7] Resolving and loading checkpoint payload")
    checkpoint_path = resolve_checkpoint_path(args.run_dir)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    create_inverse_triples = bool(checkpoint_payload.get("create_inverse_triples", True))
    logger.log(
        f"[STEP 1/7] Done | checkpoint={checkpoint_path} "
        f"create_inverse_triples={create_inverse_triples}"
    )

    logger.log("[STEP 2/7] Loading dataset")
    bundle = load_dataset(args.processed_dir, create_inverse_triples=create_inverse_triples)
    split_tf = {
        "train": bundle.train_tf,
        "valid": bundle.valid_tf,
        "test": bundle.test_tf,
    }[args.split]

    logger.log(
        f"[STEP 2/7] Done | entities={len(bundle.entity_to_id)} "
        f"relations={len(bundle.relation_to_id)} split={args.split} "
        f"triples={split_tf.num_triples}"
    )

    logger.log("[STEP 3/7] Loading ontology sidecars")
    ontology = load_ontology_bundle(args.processed_dir, bundle.entity_to_id, bundle.relation_to_id)
    logger.log(
        "[STEP 3/7] Done | "
        f"has_entity_types={ontology.has_entity_types} "
        f"has_relation_constraints={ontology.has_relation_constraints} "
        f"has_disjoint_pairs={ontology.has_disjoint_pairs}"
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

    if total_queries == 0:
        raise RuntimeError("No queries were built. Check split/mode/max_queries/query-id-file configuration.")

    quotas = sorted(set(args.quotas))
    query_csv_path = output_dir / "query_level.csv"
    relation_csv_path = output_dir / "relation_level.csv"
    progress_json_path = output_dir / "progress.json"
    summary_json_path = output_dir / "summary.json"

    env_metadata = get_env_metadata(args.device)
    write_query_csv_header(query_csv_path, quotas)

    relation_stats: dict[str, Counter] = defaultdict(Counter)
    agg = RunningAgg(feasible_counts={q: 0 for q in quotas})

    logger.log("[STEP 6/7] Scoring queries and computing blindness diagnostics")
    step_started = perf_now()

    processed_queries = 0
    effective_top_m_global: Optional[int] = None
    batch_rows: list[list[Any]] = []

    batch_starts = list(range(0, total_queries, args.query_batch_size))
    pbar = tqdm(
        batch_starts,
        desc="EXP-01 batches",
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

        _, topm_indices = torch.topk(
            scores,
            k=effective_top_m,
            dim=1,
            largest=True,
            sorted=True,
        )

        del scores
        maybe_clear_cuda_cache()

        for row_idx, q in enumerate(batch_queries):
            global_query_index = batch_start + row_idx
            cand_ids = topm_indices[row_idx].detach().cpu().tolist()

            violating_in_topk = False
            admissible_in_topm = 0
            admissible_in_topk_count = 0
            checkable_in_topm = 0
            unknown_in_topm = 0
            checkable_in_topk = 0
            violating_in_topk_count = 0
            unknown_in_topk_count = 0

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
                )

                if sem.checkable:
                    checkable_in_topm += 1
                    if sem.admissible:
                        admissible_in_topm += 1
                else:
                    unknown_in_topm += 1

                if rank_idx < args.top_k:
                    if sem.checkable:
                        checkable_in_topk += 1
                        if sem.violated:
                            violating_in_topk = True
                            violating_in_topk_count += 1
                        elif sem.admissible:
                            admissible_in_topk_count += 1
                    else:
                        unknown_in_topk_count += 1

            admissible_in_topm_but_not_topk = admissible_in_topm - admissible_in_topk_count

            blind_at_k = violating_in_topk and (admissible_in_topm > 0)
            blind_strict_at_k = violating_in_topk and (admissible_in_topm_but_not_topk > 0)

            agg.total_queries += 1
            agg.blind_queries += int(blind_at_k)
            agg.blind_strict_queries += int(blind_strict_at_k)
            agg.checkable_topk_items += checkable_in_topk
            agg.violating_topk_items += violating_in_topk_count
            agg.admissible_topk_items += admissible_in_topk_count
            agg.unknown_topk_items += unknown_in_topk_count

            for qv in quotas:
                if admissible_in_topm >= qv:
                    agg.feasible_counts[qv] += 1

            rel_counter = relation_stats[q.relation_name]
            rel_counter["queries"] += 1
            rel_counter["blind_queries"] += int(blind_at_k)
            rel_counter["blind_strict_queries"] += int(blind_strict_at_k)
            rel_counter["admissible_topm_total"] += admissible_in_topm
            rel_counter["admissible_topk_total"] += admissible_in_topk_count
            rel_counter["admissible_topm_but_not_topk_total"] += admissible_in_topm_but_not_topk
            rel_counter["checkable_topm_total"] += checkable_in_topm
            rel_counter["unknown_topm_total"] += unknown_in_topm
            rel_counter["checkable_topk_total"] += checkable_in_topk
            rel_counter["violating_topk_total"] += violating_in_topk_count
            rel_counter["unknown_topk_total"] += unknown_in_topk_count
            for qv in quotas:
                rel_counter[f"feas_{qv}"] += int(admissible_in_topm >= qv)

            batch_rows.append([
                global_query_index,
                q.query_id,
                q.row_index,
                q.mode,
                q.relation_name,
                q.head_id,
                q.rel_id,
                q.tail_id,
                int(blind_at_k),
                int(blind_strict_at_k),
                int(violating_in_topk),
                admissible_in_topm,
                admissible_in_topk_count,
                admissible_in_topm_but_not_topk,
                checkable_in_topm,
                unknown_in_topm,
                checkable_in_topk,
                violating_in_topk_count,
                admissible_in_topk_count,
                unknown_in_topk_count,
                *[int(admissible_in_topm >= qv) for qv in quotas],
            ])
            processed_queries += 1

        append_query_rows(query_csv_path, batch_rows)
        batch_rows.clear()

        progress_payload = compute_progress_payload(
            args=args,
            started_at=step_started,
            processed_queries=processed_queries,
            total_queries=total_queries,
            agg=agg,
            quotas=quotas,
            effective_top_m=effective_top_m,
        )
        save_json_atomic(progress_json_path, progress_payload)

        blind_value = progress_payload["running_metrics"][f"Blind@{args.top_k}"]
        strict_blind_value = progress_payload["running_metrics"][f"BlindStrict@{args.top_k}"]
        max_q = quotas[-1]
        feas_value = progress_payload["running_metrics"][f"Feas_{max_q}@{args.top_k}"]

        pbar.set_postfix({
            "q": f"{processed_queries}/{total_queries}",
            "blind": f"{blind_value:.4f}" if blind_value is not None else "nan",
            "strict": f"{strict_blind_value:.4f}" if strict_blind_value is not None else "nan",
            f"feas{max_q}": f"{feas_value:.4f}" if feas_value is not None else "nan",
        })

        batch_number = (batch_start // args.query_batch_size) + 1
        if (batch_number % args.log_every) == 0 or batch_end == total_queries:
            logger.log(
                f"[STEP 6/7] processed={processed_queries}/{total_queries} "
                f"elapsed={progress_payload['elapsed_human']} "
                f"eta={progress_payload['eta_human']} "
                f"Blind@{args.top_k}={blind_value:.6f} "
                f"BlindStrict@{args.top_k}={strict_blind_value:.6f} "
                f"Feas_{max_q}@{args.top_k}={feas_value:.6f}"
            )

        del topm_indices
        maybe_clear_cuda_cache()

    logger.log("[STEP 6/7] Done")

    logger.log("[STEP 7/7] Building final summaries")
    total_elapsed = perf_now() - started_total
    if effective_top_m_global is None:
        raise RuntimeError("Internal error: effective_top_m_global was never set.")

    with relation_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "relation",
            "queries",
            f"Blind@{args.top_k}",
            f"BlindStrict@{args.top_k}",
            "mean_admissible_topm",
            "mean_admissible_topk",
            "mean_admissible_topm_but_not_topk",
            "mean_checkable_topm",
            "mean_unknown_topm",
            "mean_checkable_topk",
            "mean_violating_topk",
            "mean_unknown_topk",
            *[f"Feas_{q}@{args.top_k}" for q in quotas],
            "blind_queries_count",
            "blind_strict_queries_count",
        ]
        writer.writerow(header)

        for rel, c in sorted(relation_stats.items()):
            qn = max(1, c["queries"])
            row = [
                rel,
                c["queries"],
                c["blind_queries"] / qn,
                c["blind_strict_queries"] / qn,
                c["admissible_topm_total"] / qn,
                c["admissible_topk_total"] / qn,
                c["admissible_topm_but_not_topk_total"] / qn,
                c["checkable_topm_total"] / qn,
                c["unknown_topm_total"] / qn,
                c["checkable_topk_total"] / qn,
                c["violating_topk_total"] / qn,
                c["unknown_topk_total"] / qn,
                *[(c[f"feas_{q}"] / qn) for q in quotas],
                c["blind_queries"],
                c["blind_strict_queries"],
            ]
            writer.writerow(row)

    denom_topk = agg.total_queries * args.top_k if agg.total_queries > 0 else 0

    summary = {
        "status": "ok",
        "experiment_id": "EXP-01",
        "experiment_name": "blindness_diagnostics",
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
            "check_policy_description": (
                "candidate is checkable if at least one enabled constraint family is evaluable"
                if args.check_policy == "available_any"
                else "candidate is checkable only if all enabled constraint families are evaluable"
            ),
            "use_domain": args.use_domain,
            "use_range": args.use_range,
            "use_disjoint": args.use_disjoint,
        },
        "query_subset": {
            "query_id_file": str(args.query_id_file) if args.query_id_file is not None else None,
            "subset_filter_active": args.query_id_file is not None,
            "allowed_query_id_count": (len(allowed_query_ids) if allowed_query_ids is not None else None),
            "matched_query_count": total_queries,
        },
        "blindness_definition": {
            f"Blind@{args.top_k}": (
                "violation present in returned top-K and at least one admissible candidate exists in top-M"
            ),
            f"BlindStrict@{args.top_k}": (
                "violation present in returned top-K and at least one admissible candidate exists outside top-K but inside top-M"
            ),
            "feasibility_definition": "Feas_q@K is window-relative and equals fraction of queries with at least q admissible candidates in top-M",
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
            "total": agg.total_queries,
            "blind_queries_count": agg.blind_queries,
            "blind_strict_queries_count": agg.blind_strict_queries,
            f"Blind@{args.top_k}": agg.blind_queries / agg.total_queries if agg.total_queries else None,
            f"BlindStrict@{args.top_k}": agg.blind_strict_queries / agg.total_queries if agg.total_queries else None,
            **{
                f"Feas_{q}@{args.top_k}": (
                    agg.feasible_counts[q] / agg.total_queries if agg.total_queries else None
                )
                for q in quotas
            },
        },
        "topk_semantics": {
            f"Cov@{args.top_k}": (agg.checkable_topk_items / denom_topk) if denom_topk else None,
            f"Unknown@{args.top_k}": (agg.unknown_topk_items / denom_topk) if denom_topk else None,
            f"Viol@{args.top_k}": (
                agg.violating_topk_items / agg.checkable_topk_items
                if agg.checkable_topk_items > 0 else None
            ),
            f"Adm@{args.top_k}": (agg.admissible_topk_items / denom_topk) if denom_topk else None,
            "raw_counts": {
                "checkable_topk_items": agg.checkable_topk_items,
                "violating_topk_items": agg.violating_topk_items,
                "admissible_topk_items": agg.admissible_topk_items,
                "unknown_topk_items": agg.unknown_topk_items,
            },
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
        f"[DONE] "
        f"Blind@{args.top_k}={summary['queries'][f'Blind@{args.top_k}']:.6f} "
        f"BlindStrict@{args.top_k}={summary['queries'][f'BlindStrict@{args.top_k}']:.6f} "
        f"Feas_{quotas[-1]}@{args.top_k}={summary['queries'][f'Feas_{quotas[-1]}@{args.top_k}']:.6f} "
        f"elapsed={summary['elapsed_human']}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP-01: Blindness diagnostics over frozen KGC rankings."
    )
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to best_model dir or directly to base_model_checkpoint.pt",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-name", type=str, default=None)

    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--mode", type=str, default="tail", choices=["tail", "head", "all"])

    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--top-m", type=int, default=10000)
    p.add_argument("--quotas", type=int, nargs="+", default=[1, 3, 5, 10])

    p.add_argument("--query-batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-queries", type=int, default=None)

    p.add_argument(
        "--check-policy",
        type=str,
        default="available_any",
        choices=["available_any", "available_all"],
    )
    p.add_argument("--use-domain", action="store_true")
    p.add_argument("--use-range", action="store_true")
    p.add_argument("--use-disjoint", action="store_true")

    p.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Log every N scoring batches",
    )

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
        args.use_disjoint = True

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    if args.top_m < args.top_k:
        raise SystemExit("--top-m must be >= --top-k")

    if args.query_batch_size <= 0:
        raise SystemExit("--query-batch-size must be positive")

    if not args.quotas:
        raise SystemExit("--quotas must not be empty")

    if min(args.quotas) < 0:
        raise SystemExit("All quotas must be non-negative")

    if max(args.quotas) > args.top_k:
        raise SystemExit("All quotas must satisfy quota <= top-k")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir / "run.log")

    try:
        run_exp01(args)
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