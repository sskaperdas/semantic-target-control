#!/usr/bin/env python3
"""
Preprocess EurostatKG-style RDF/OWL resources into a unified artifact bundle
for OptK / blindness experiments.

Outputs:
  - all_triples.tsv
  - train.tsv / valid.tsv / test.tsv
  - entity2id.json / relation2id.json
  - entity_types.json
  - relation_constraints.json
  - disjoint_pairs.json
  - relation_split_stats.json
  - file_stats.json
  - preflight.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, DefaultDict, Iterator

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

try:
    import orjson  # type: ignore

    JSON_BACKEND = "orjson"
except Exception:
    orjson = None
    JSON_BACKEND = "json"

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


Triple = tuple[str, str, str]

SUPPORTED_SUFFIXES = {
    ".ttl": "turtle",
    ".nt": "nt",
    ".ntriples": "nt",
    ".rdf": "xml",
    ".owl": "xml",
    ".xml": "xml",
    ".n3": "n3",
    ".jsonld": "json-ld",
    ".trig": "trig",
}

SCHEMA_PREDICATES = {
    str(RDF.type),
    str(RDFS.subClassOf),
    str(RDFS.domain),
    str(RDFS.range),
    str(OWL.disjointWith),
}

_NS_PER_SEC = 1_000_000_000.0


# ============================================================
# Timing / formatting helpers
# ============================================================

def _now_ns() -> int:
    return time.perf_counter_ns()


def _ns_to_sec(ns: int) -> float:
    return ns / _NS_PER_SEC


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _append_phase_trace(
    trace: list[dict[str, Any]],
    *,
    phase: str,
    run_start_ns: int,
    phase_start_ns: int,
    phase_end_ns: int,
    extra: dict[str, Any] | None = None,
) -> None:
    row = {
        "phase": phase,
        "start_offset_sec": _ns_to_sec(phase_start_ns - run_start_ns),
        "end_offset_sec": _ns_to_sec(phase_end_ns - run_start_ns),
        "duration_sec": _ns_to_sec(phase_end_ns - phase_start_ns),
    }
    if extra:
        row.update(extra)
    trace.append(row)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess EurostatKG RDF resources")
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--graph-dir", required=True, type=Path, help="Directory with KG graph RDF files")
    p.add_argument("--ontology-dir", type=Path, default=None, help="Directory with ontology/schema RDF files")
    p.add_argument("--output-dir", required=True, type=Path, help="Where processed artifacts will be written")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--min-relation-count-for-split", type=int, default=10)
    p.add_argument("--max-files", type=int, default=None, help="Optional cap for debugging")
    p.add_argument("--include-literals", action="store_true", help="Keep literal objects by stringifying them")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return p.parse_args()


# ============================================================
# IO helpers
# ============================================================

def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_dumps_bytes(data: Any, *, indent: bool) -> bytes:
    if orjson is not None:
        option = 0
        if indent:
            option |= orjson.OPT_INDENT_2
        return orjson.dumps(data, option=option, default=_json_default)

    text = json.dumps(
        data,
        indent=2 if indent else None,
        ensure_ascii=False,
        default=_json_default,
        sort_keys=False,
    )
    return text.encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        f.write(payload)
    tmp_path.replace(path)


def write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, _json_dumps_bytes(obj, indent=True))


def write_tsv(path: Path, triples: list[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")
    tmp_path.replace(path)


# ============================================================
# RDF helpers
# ============================================================

def iter_rdf_files(directory: Path | None) -> Iterator[Path]:
    if directory is None or not directory.exists():
        return iter(())
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return iter(sorted(files))


def guess_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported RDF file suffix: {path}")
    return SUPPORTED_SUFFIXES[suffix]


def node_to_str(node: Any, include_literals: bool) -> str | None:
    if isinstance(node, (URIRef, BNode)):
        return str(node)
    if include_literals and isinstance(node, Literal):
        return f"literal::{str(node)}"
    return None


def load_rdf_graph(path: Path) -> Graph:
    g = Graph()
    fmt = guess_format(path)
    try:
        g.parse(path, format=fmt)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {path} as {fmt}: {exc}") from exc
    return g


# ============================================================
# Dataclasses
# ============================================================

@dataclass(slots=True)
class FileParseStats:
    path: str
    source_kind: str
    rdf_format: str
    parse_sec: float

    triples_seen: int = 0
    kg_triples_added: int = 0
    duplicate_kg_triples: int = 0

    type_triples: int = 0
    subclass_triples: int = 0
    domain_triples: int = 0
    range_triples: int = 0
    disjoint_triples: int = 0

    skipped_nonresource_subject: int = 0
    skipped_nonresource_object: int = 0
    skipped_schema_predicate: int = 0
    skipped_self_disjoint: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Semantics builders
# ============================================================

def build_subclass_closure(subclass_of: dict[str, set[str]]) -> dict[str, set[str]]:
    closure: dict[str, set[str]] = {}

    def dfs(cls_uri: str, seen: set[str]) -> set[str]:
        if cls_uri in closure:
            return closure[cls_uri]

        parents = subclass_of.get(cls_uri, set())
        out = {cls_uri}

        for parent in parents:
            if parent in seen:
                continue
            out.update(dfs(parent, seen | {parent}))

        closure[cls_uri] = out
        return out

    all_classes = set(subclass_of.keys())
    for parents in subclass_of.values():
        all_classes.update(parents)

    for cls_uri in all_classes:
        dfs(cls_uri, {cls_uri})

    return closure


def materialize_entity_types(
    entity_types: dict[str, set[str]],
    subclass_closure: dict[str, set[str]],
) -> dict[str, set[str]]:
    materialized: dict[str, set[str]] = {}
    for entity, types in entity_types.items():
        out: set[str] = set()
        for t in types:
            out.update(subclass_closure.get(t, {t}))
        materialized[entity] = out or set(types)
    return materialized


# ============================================================
# Splitting
# ============================================================

def split_by_relation(
    triples: list[Triple],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    min_relation_count_for_split: int,
) -> tuple[list[Triple], list[Triple], list[Triple], dict[str, dict[str, int]]]:
    ratios_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratios_sum - 1.0) > 1e-8:
        raise ValueError("train/valid/test ratios must sum to 1.0")

    rng = random.Random(seed)
    by_rel: DefaultDict[str, list[Triple]] = defaultdict(list)
    for triple in triples:
        by_rel[triple[1]].append(triple)

    train: list[Triple] = []
    valid: list[Triple] = []
    test: list[Triple] = []
    relation_split_stats: dict[str, dict[str, int]] = {}

    for rel, rel_triples in by_rel.items():
        rel_copy = list(rel_triples)
        rng.shuffle(rel_copy)
        n = len(rel_copy)

        if n < min_relation_count_for_split:
            train.extend(rel_copy)
            relation_split_stats[rel] = {
                "total": n,
                "train": n,
                "valid": 0,
                "test": 0,
                "forced_train_only": 1,
            }
            continue

        # exact, safe, all-triples-preserved partition
        n_valid = int(round(n * valid_ratio))
        n_test = int(round(n * test_ratio))

        n_valid = max(1, min(n_valid, n - 2))
        n_test = max(1, min(n_test, n - n_valid - 1))
        n_train = n - n_valid - n_test

        if n_train < 1:
            n_train = 1
            if n_valid >= n_test and n_valid > 1:
                n_valid -= 1
            else:
                n_test -= 1

        assert n_train >= 1 and n_valid >= 1 and n_test >= 1
        assert n_train + n_valid + n_test == n

        train.extend(rel_copy[:n_train])
        valid.extend(rel_copy[n_train:n_train + n_valid])
        test.extend(rel_copy[n_train + n_valid:])

        relation_split_stats[rel] = {
            "total": n,
            "train": n_train,
            "valid": n_valid,
            "test": n_test,
            "forced_train_only": 0,
        }

    return train, valid, test, relation_split_stats


# ============================================================
# Main processing
# ============================================================

def main() -> int:
    args = parse_args()
    show_progress = not args.no_progress

    if args.train_ratio <= 0 or args.valid_ratio <= 0 or args.test_ratio <= 0:
        print("[ERROR] train/valid/test ratios must all be > 0", file=sys.stderr)
        return 1
    if args.min_relation_count_for_split <= 0:
        print("[ERROR] min_relation_count_for_split must be > 0", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or args.graph_dir.name
    run_start_ns = _now_ns()
    started_at_utc = _utc_now_iso()
    phase_trace: list[dict[str, Any]] = []

    graph_files = list(iter_rdf_files(args.graph_dir))
    ontology_files = list(iter_rdf_files(args.ontology_dir)) if args.ontology_dir else []

    if args.max_files is not None:
        graph_files = graph_files[: args.max_files]
        ontology_files = ontology_files[: args.max_files]

    if not graph_files:
        print("[ERROR] No RDF graph files found.", file=sys.stderr)
        return 1

    entity_types: DefaultDict[str, set[str]] = defaultdict(set)
    subclass_of: DefaultDict[str, set[str]] = defaultdict(set)
    relation_domain: DefaultDict[str, set[str]] = defaultdict(set)
    relation_range: DefaultDict[str, set[str]] = defaultdict(set)
    disjoint_pairs: set[tuple[str, str]] = set()
    kg_triples: set[Triple] = set()

    file_stats: list[dict[str, Any]] = []

    all_files = [(p, "graph") for p in graph_files] + [(p, "ontology") for p in ontology_files]

    RDF_type = RDF.type
    RDFS_subClassOf = RDFS.subClassOf
    RDFS_domain = RDFS.domain
    RDFS_range = RDFS.range
    OWL_disjointWith = OWL.disjointWith

    iterator: Any = all_files
    if show_progress and tqdm is not None:
        iterator = tqdm(
            all_files,
            total=len(all_files),
            desc=f"{dataset_name}-rdf-files",
            unit="file",
            dynamic_ncols=True,
        )

    # --------------------------------------------------------
    # Parse files
    # --------------------------------------------------------
    p0 = _now_ns()
    for idx, (path, source_kind) in enumerate(iterator, start=1):
        file_parse_start_ns = _now_ns()
        fmt = guess_format(path)
        print(f"[{idx}/{len(all_files)}] Parsing {source_kind}: {path}", flush=True)

        g = load_rdf_graph(path)
        file_parse_sec = _ns_to_sec(_now_ns() - file_parse_start_ns)

        stats = FileParseStats(
            path=str(path),
            source_kind=source_kind,
            rdf_format=fmt,
            parse_sec=file_parse_sec,
        )

        et_add = entity_types
        sc_add = subclass_of
        rd_add = relation_domain
        rr_add = relation_range
        dp_add = disjoint_pairs
        kg_add = kg_triples

        for s, p, o in g:
            stats.triples_seen += 1

            s_str = node_to_str(s, include_literals=False)
            if s_str is None:
                stats.skipped_nonresource_subject += 1
                continue

            if p == RDF_type:
                stats.type_triples += 1
                o_str = node_to_str(o, include_literals=False)
                if o_str is not None:
                    et_add[s_str].add(o_str)
                else:
                    stats.skipped_nonresource_object += 1
                continue

            if p == RDFS_subClassOf:
                stats.subclass_triples += 1
                o_str = node_to_str(o, include_literals=False)
                if o_str is not None:
                    sc_add[s_str].add(o_str)
                else:
                    stats.skipped_nonresource_object += 1
                continue

            if p == RDFS_domain:
                stats.domain_triples += 1
                o_str = node_to_str(o, include_literals=False)
                if o_str is not None:
                    rd_add[s_str].add(o_str)
                else:
                    stats.skipped_nonresource_object += 1
                continue

            if p == RDFS_range:
                stats.range_triples += 1
                o_str = node_to_str(o, include_literals=False)
                if o_str is not None:
                    rr_add[s_str].add(o_str)
                else:
                    stats.skipped_nonresource_object += 1
                continue

            if p == OWL_disjointWith:
                stats.disjoint_triples += 1
                o_str = node_to_str(o, include_literals=False)
                if o_str is None:
                    stats.skipped_nonresource_object += 1
                    continue

                if s_str == o_str:
                    stats.skipped_self_disjoint += 1
                    continue

                pair = (s_str, o_str) if s_str <= o_str else (o_str, s_str)
                dp_add.add(pair)
                continue

            o_str = node_to_str(o, include_literals=args.include_literals)
            if o_str is None:
                stats.skipped_nonresource_object += 1
                continue

            p_str = str(p)
            if p_str in SCHEMA_PREDICATES:
                stats.skipped_schema_predicate += 1
                continue

            triple = (s_str, p_str, o_str)
            if triple in kg_add:
                stats.duplicate_kg_triples += 1
            else:
                kg_add.add(triple)
                stats.kg_triples_added += 1

        file_stats.append(stats.to_dict())
        del g

    p1 = _now_ns()
    _append_phase_trace(
        phase_trace,
        phase="parse_rdf_files",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=p1,
        extra={"files_total": len(all_files)},
    )

    if not kg_triples:
        print("[ERROR] No KGC triples extracted after filtering.", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # Build closure / materialize types
    # --------------------------------------------------------
    p0 = _now_ns()
    subclass_closure = build_subclass_closure(subclass_of)
    entity_types_materialized = materialize_entity_types(entity_types, subclass_closure)
    p1 = _now_ns()
    _append_phase_trace(
        phase_trace,
        phase="build_type_closure_and_materialize",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=p1,
        extra={
            "classes_with_explicit_subclass_info": len(subclass_of),
            "closure_class_count": len(subclass_closure),
        },
    )

    # --------------------------------------------------------
    # Sort + split
    # --------------------------------------------------------
    p0 = _now_ns()
    triples_sorted = sorted(kg_triples)
    train, valid, test, relation_split_stats = split_by_relation(
        triples_sorted,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        min_relation_count_for_split=args.min_relation_count_for_split,
    )
    p1 = _now_ns()
    _append_phase_trace(
        phase_trace,
        phase="sort_and_split",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=p1,
        extra={"num_triples": len(triples_sorted)},
    )

    # --------------------------------------------------------
    # Build ids / constraints
    # --------------------------------------------------------
    p0 = _now_ns()
    entities = sorted({s for s, _, _ in triples_sorted} | {o for _, _, o in triples_sorted})
    relations = sorted({p for _, p, _ in triples_sorted})

    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    relation_constraints = {
        rel: {
            "domain": sorted(relation_domain.get(rel, set())),
            "range": sorted(relation_range.get(rel, set())),
        }
        for rel in relations
    }
    p1 = _now_ns()
    _append_phase_trace(
        phase_trace,
        phase="build_ids_and_constraints",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=p1,
        extra={
            "num_entities": len(entities),
            "num_relations": len(relations),
        },
    )

    typed_entities = sum(1 for e in entities if entity_types_materialized.get(e))

    total_schema_type_triples = sum(x["type_triples"] for x in file_stats)
    total_schema_subclass_triples = sum(x["subclass_triples"] for x in file_stats)
    total_schema_domain_triples = sum(x["domain_triples"] for x in file_stats)
    total_schema_range_triples = sum(x["range_triples"] for x in file_stats)
    total_schema_disjoint_triples = sum(x["disjoint_triples"] for x in file_stats)
    total_duplicates = sum(x["duplicate_kg_triples"] for x in file_stats)
    total_skipped_nonresource_subject = sum(x["skipped_nonresource_subject"] for x in file_stats)
    total_skipped_nonresource_object = sum(x["skipped_nonresource_object"] for x in file_stats)

    preflight = {
        "dataset": dataset_name,
        "graph_dir": str(args.graph_dir),
        "ontology_dir": str(args.ontology_dir) if args.ontology_dir is not None else None,
        "output_dir": str(args.output_dir),
        "json_backend": JSON_BACKEND,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now_iso(),
        "include_literals": args.include_literals,
        "seed": args.seed,
        "split_config": {
            "train_ratio": args.train_ratio,
            "valid_ratio": args.valid_ratio,
            "test_ratio": args.test_ratio,
            "min_relation_count_for_split": args.min_relation_count_for_split,
        },
        "files": {
            "num_graph_files": len(graph_files),
            "num_ontology_files": len(ontology_files),
            "num_all_files": len(all_files),
        },
        "counts": {
            "num_triples": len(triples_sorted),
            "num_train": len(train),
            "num_valid": len(valid),
            "num_test": len(test),
            "num_entities": len(entities),
            "num_relations": len(relations),
            "typed_entities": typed_entities,
            "typing_coverage": _safe_div(typed_entities, len(entities)),
            "relations_with_domain": sum(1 for r in relations if relation_domain.get(r)),
            "relations_with_range": sum(1 for r in relations if relation_range.get(r)),
            "num_disjoint_pairs": len(disjoint_pairs),
            "num_classes_with_subclass_edges": len(subclass_of),
            "num_classes_in_subclass_closure": len(subclass_closure),
        },
        "schema_triples": {
            "rdf_type": total_schema_type_triples,
            "rdfs_subClassOf": total_schema_subclass_triples,
            "rdfs_domain": total_schema_domain_triples,
            "rdfs_range": total_schema_range_triples,
            "owl_disjointWith": total_schema_disjoint_triples,
        },
        "skips_and_duplicates": {
            "duplicate_kg_triples": total_duplicates,
            "skipped_nonresource_subject": total_skipped_nonresource_subject,
            "skipped_nonresource_object": total_skipped_nonresource_object,
        },
        "runtime": {
            "total_sec": _ns_to_sec(_now_ns() - run_start_ns),
            "phase_trace": phase_trace,
        },
    }

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------
    p0 = _now_ns()
    write_tsv(args.output_dir / "all_triples.tsv", triples_sorted)
    write_tsv(args.output_dir / "train.tsv", train)
    write_tsv(args.output_dir / "valid.tsv", valid)
    write_tsv(args.output_dir / "test.tsv", test)

    write_json(args.output_dir / "entity2id.json", entity2id)
    write_json(args.output_dir / "relation2id.json", relation2id)
    write_json(
        args.output_dir / "entity_types.json",
        {k: sorted(v) for k, v in entity_types_materialized.items()},
    )
    write_json(args.output_dir / "relation_constraints.json", relation_constraints)
    write_json(
        args.output_dir / "disjoint_pairs.json",
        [list(p) for p in sorted(disjoint_pairs)],
    )
    write_json(args.output_dir / "relation_split_stats.json", relation_split_stats)
    write_json(args.output_dir / "file_stats.json", file_stats)

    preflight["runtime"]["total_sec"] = _ns_to_sec(_now_ns() - run_start_ns)
    write_json(args.output_dir / "preflight.json", preflight)
    p1 = _now_ns()
    _append_phase_trace(
        phase_trace,
        phase="write_outputs",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=p1,
    )

    preflight["runtime"]["total_sec"] = _ns_to_sec(_now_ns() - run_start_ns)
    preflight["runtime"]["outputs_written_sec"] = _ns_to_sec(p1 - p0)
    preflight["runtime"]["phase_trace"] = phase_trace
    write_json(args.output_dir / "preflight.json", preflight)

    print("[OK] EurostatKG preprocessing finished.")
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())