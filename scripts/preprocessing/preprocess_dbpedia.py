#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bz2
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, DefaultDict, Iterable, Iterator

try:
    import orjson  # type: ignore

    JSON_BACKEND = "orjson"
except Exception:
    orjson = None
    JSON_BACKEND = "json"

try:
    from rdflib import Graph, RDF, RDFS, OWL, URIRef
    from rdflib.collection import Collection
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This script requires rdflib. Install it with: pip install rdflib"
    ) from e

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


Triple = tuple[str, str, str]

DEFAULT_SEED = 42
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
    p = argparse.ArgumentParser(
        description="Preprocess DBpedia dataset into a common processed KG format for KGC + constrained decoding."
    )
    p.add_argument("--ontology", type=Path, required=True, help="Path to DBpedia ontology OWL file")
    p.add_argument(
        "--instance-types",
        type=Path,
        required=True,
        help="Path to instance_types_en.ttl.bz2",
    )
    p.add_argument(
        "--mappingbased-objects",
        type=Path,
        required=True,
        help="Path to mappingbased_objects_en.ttl.bz2",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument(
        "--entity-prefix-filter",
        nargs="*",
        default=["http://dbpedia.org/resource/"],
        help="Keep only graph triples whose head/tail start with one of these prefixes",
    )
    p.add_argument(
        "--min-triples-per-relation",
        type=int,
        default=1,
        help="Drop relations with fewer than this many object triples",
    )
    p.add_argument(
        "--deduplicate-triples",
        action="store_true",
        default=False,
        help="Deduplicate mapping-based object triples before writing/splitting",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
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


def write_tsv(path: Path, rows: Iterable[Triple]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for h, r, t in rows:
            f.write(f"{h}\t{r}\t{t}\n")
            count += 1
    tmp_path.replace(path)
    return count


# ============================================================
# RDF loading helpers
# ============================================================

class NamedTextIOProxy:
    """
    Small wrapper so rdflib can access `.name` on bz2-opened text streams.
    """

    def __init__(self, fh, name: str):
        self._fh = fh
        self.name = name

    def read(self, *args, **kwargs):
        return self._fh.read(*args, **kwargs)

    def readline(self, *args, **kwargs):
        return self._fh.readline(*args, **kwargs)

    def readlines(self, *args, **kwargs):
        return self._fh.readlines(*args, **kwargs)

    def __iter__(self):
        return iter(self._fh)

    def seek(self, *args, **kwargs):
        return self._fh.seek(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self._fh.tell(*args, **kwargs)

    def close(self):
        return self._fh.close()

    def __getattr__(self, item):
        return getattr(self._fh, item)


def load_graph_from_any(path: Path, rdf_format: str | None = None) -> Graph:
    g = Graph()

    if path.suffix.lower() == ".bz2":
        inferred_name = str(path.with_suffix(""))
        with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as raw_f:
            wrapped = NamedTextIOProxy(raw_f, inferred_name)
            g.parse(file=wrapped, format=rdf_format or "turtle")
    else:
        if rdf_format is not None:
            g.parse(str(path), format=rdf_format)
        else:
            g.parse(str(path))

    return g


def is_resource_uri(uri: str, allowed_prefixes: tuple[str, ...]) -> bool:
    return any(uri.startswith(pref) for pref in allowed_prefixes)


def relation_local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


# ============================================================
# Dataclasses
# ============================================================

@dataclass(slots=True)
class SourceFileStats:
    path: str
    kind: str
    parse_sec: float
    graph_len: int

    triples_seen: int = 0
    triples_kept: int = 0
    triples_dropped_non_uri: int = 0
    triples_dropped_prefix: int = 0
    duplicate_graph_triples: int = 0

    rdf_type_triples: int = 0
    subclass_triples: int = 0
    domain_triples: int = 0
    range_triples: int = 0
    disjoint_triples: int = 0
    disjoint_union_axioms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Ontology helpers
# ============================================================

def compute_transitive_closure(parent_map: dict[str, set[str]]) -> dict[str, set[str]]:
    closure: dict[str, set[str]] = {}

    def dfs(node: str, stack: set[str] | None = None) -> set[str]:
        if node in closure:
            return closure[node]
        if stack is None:
            stack = set()
        if node in stack:
            return set()

        stack.add(node)
        anc: set[str] = set()
        for p in parent_map.get(node, set()):
            anc.add(p)
            anc |= dfs(p, stack)
        stack.remove(node)

        closure[node] = anc
        return anc

    all_nodes = set(parent_map.keys())
    for parents in parent_map.values():
        all_nodes.update(parents)

    for n in all_nodes:
        dfs(n)

    return closure


def build_splits(
    triples: list[Triple],
    valid_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[Triple], list[Triple], list[Triple], dict[str, dict[str, int]]]:
    by_rel: DefaultDict[str, list[Triple]] = defaultdict(list)
    for h, r, t in triples:
        by_rel[r].append((h, r, t))

    rng = random.Random(seed)
    train: list[Triple] = []
    valid: list[Triple] = []
    test: list[Triple] = []
    relation_split_stats: dict[str, dict[str, int]] = {}

    for r, rel_triples in by_rel.items():
        rel_copy = list(rel_triples)
        rng.shuffle(rel_copy)
        n = len(rel_copy)

        if n < 3:
            n_valid = 0
            n_test = 0
            n_train = n
        else:
            n_valid = int(round(n * valid_frac))
            n_test = int(round(n * test_frac))

            n_valid = max(1, min(n_valid, n - 2))
            n_test = max(1, min(n_test, n - n_valid - 1))
            n_train = n - n_valid - n_test

            if n_train < 1:
                n_train = 1
                if n_valid >= n_test and n_valid > 1:
                    n_valid -= 1
                else:
                    n_test -= 1

        valid.extend(rel_copy[:n_valid])
        test.extend(rel_copy[n_valid:n_valid + n_test])
        train.extend(rel_copy[n_valid + n_test:])

        relation_split_stats[r] = {
            "total": n,
            "train": n_train,
            "valid": n_valid,
            "test": n_test,
        }

    return train, valid, test, relation_split_stats


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

    if not args.ontology.exists():
        raise SystemExit(f"--ontology not found: {args.ontology}")
    if not args.instance_types.exists():
        raise SystemExit(f"--instance-types not found: {args.instance_types}")
    if not args.mappingbased_objects.exists():
        raise SystemExit(f"--mappingbased-objects not found: {args.mappingbased_objects}")
    if args.valid_frac < 0 or args.test_frac < 0:
        raise SystemExit("--valid-frac and --test-frac must be >= 0")
    if args.valid_frac + args.test_frac >= 1.0:
        raise SystemExit("--valid-frac + --test-frac must be < 1.0")
    if args.min_triples_per_relation <= 0:
        raise SystemExit("--min-triples-per-relation must be > 0")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    run_start_ns = _now_ns()
    started_at_utc = _utc_now_iso()
    phase_trace: list[dict[str, Any]] = []
    file_stats: list[dict[str, Any]] = []

    allowed_prefixes = tuple(args.entity_prefix_filter)

    # --------------------------------------------------------
    # Parse ontology
    # --------------------------------------------------------
    print("[1/6] Parsing ontology...", flush=True)
    p0 = _now_ns()
    ont = load_graph_from_any(args.ontology, rdf_format="xml")
    p1 = _now_ns()
    ont_stats = SourceFileStats(
        path=str(args.ontology),
        kind="ontology",
        parse_sec=_ns_to_sec(p1 - p0),
        graph_len=len(ont),
    )

    subclass_of: DefaultDict[str, set[str]] = defaultdict(set)
    relation_domains: DefaultDict[str, set[str]] = defaultdict(set)
    relation_ranges: DefaultDict[str, set[str]] = defaultdict(set)
    disjoint_pairs: set[tuple[str, str]] = set()

    ontology_iter: Any = ont
    if show_progress and tqdm is not None:
        ontology_iter = tqdm(
            ont,
            total=len(ont),
            desc="dbpedia-ontology",
            unit="triple",
            dynamic_ncols=True,
            leave=False,
        )

    for s, p, o in ontology_iter:
        ont_stats.triples_seen += 1

        if p == RDFS.subClassOf:
            ont_stats.subclass_triples += 1
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                subclass_of[str(s)].add(str(o))
            continue

        if p == RDFS.domain:
            ont_stats.domain_triples += 1
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                relation_domains[str(s)].add(str(o))
            continue

        if p == RDFS.range:
            ont_stats.range_triples += 1
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                relation_ranges[str(s)].add(str(o))
            continue

        if p == OWL.disjointWith:
            ont_stats.disjoint_triples += 1
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                a, b = sorted((str(s), str(o)))
                if a != b:
                    disjoint_pairs.add((a, b))
            continue

        if p == OWL.disjointUnionOf:
            ont_stats.disjoint_union_axioms += 1
            try:
                members = [str(x) for x in Collection(ont, o) if isinstance(x, URIRef)]
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = sorted((members[i], members[j]))
                        if a != b:
                            disjoint_pairs.add((a, b))
            except Exception:
                pass

    file_stats.append(ont_stats.to_dict())
    del ont

    _append_phase_trace(
        phase_trace,
        phase="parse_ontology",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=_now_ns(),
    )

    # --------------------------------------------------------
    # Parse instance types
    # --------------------------------------------------------
    print("[2/6] Parsing instance types...", flush=True)
    p0 = _now_ns()
    inst = load_graph_from_any(args.instance_types, rdf_format="turtle")
    p1 = _now_ns()
    inst_stats = SourceFileStats(
        path=str(args.instance_types),
        kind="instance_types",
        parse_sec=_ns_to_sec(p1 - p0),
        graph_len=len(inst),
    )

    entity_types_raw: DefaultDict[str, set[str]] = defaultdict(set)

    inst_iter: Any = inst
    if show_progress and tqdm is not None:
        inst_iter = tqdm(
            inst,
            total=len(inst),
            desc="dbpedia-instance-types",
            unit="triple",
            dynamic_ncols=True,
            leave=False,
        )

    for s, p, o in inst_iter:
        inst_stats.triples_seen += 1

        if p != RDF.type:
            continue

        inst_stats.rdf_type_triples += 1

        if not (isinstance(s, URIRef) and isinstance(o, URIRef)):
            inst_stats.triples_dropped_non_uri += 1
            continue

        s_str = str(s)
        if not is_resource_uri(s_str, allowed_prefixes):
            inst_stats.triples_dropped_prefix += 1
            continue

        entity_types_raw[s_str].add(str(o))
        inst_stats.triples_kept += 1

    file_stats.append(inst_stats.to_dict())
    del inst

    _append_phase_trace(
        phase_trace,
        phase="parse_instance_types",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=_now_ns(),
    )

    # --------------------------------------------------------
    # Parse mapping-based object triples
    # --------------------------------------------------------
    print("[3/6] Parsing mapping-based object triples...", flush=True)
    p0 = _now_ns()
    obj = load_graph_from_any(args.mappingbased_objects, rdf_format="turtle")
    p1 = _now_ns()
    obj_stats = SourceFileStats(
        path=str(args.mappingbased_objects),
        kind="mappingbased_objects",
        parse_sec=_ns_to_sec(p1 - p0),
        graph_len=len(obj),
    )

    if args.deduplicate_triples:
        all_triples_set: set[Triple] = set()
        all_triples_list: list[Triple] = []
    else:
        all_triples_set = set()
        all_triples_list = []

    relation_counter: DefaultDict[str, int] = defaultdict(int)
    graph_entities: set[str] = set()

    obj_iter: Any = obj
    if show_progress and tqdm is not None:
        obj_iter = tqdm(
            obj,
            total=len(obj),
            desc="dbpedia-object-triples",
            unit="triple",
            dynamic_ncols=True,
            leave=False,
        )

    for s, p, o in obj_iter:
        obj_stats.triples_seen += 1

        if not (isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef)):
            obj_stats.triples_dropped_non_uri += 1
            continue

        hs = str(s)
        rs = str(p)
        ts = str(o)

        if not is_resource_uri(hs, allowed_prefixes):
            obj_stats.triples_dropped_prefix += 1
            continue
        if not is_resource_uri(ts, allowed_prefixes):
            obj_stats.triples_dropped_prefix += 1
            continue

        triple = (hs, rs, ts)

        if args.deduplicate_triples:
            if triple in all_triples_set:
                obj_stats.duplicate_graph_triples += 1
                continue
            all_triples_set.add(triple)

        all_triples_list.append(triple)
        relation_counter[rs] += 1
        graph_entities.add(hs)
        graph_entities.add(ts)
        obj_stats.triples_kept += 1

    file_stats.append(obj_stats.to_dict())
    del obj

    _append_phase_trace(
        phase_trace,
        phase="parse_mappingbased_objects",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=_now_ns(),
        extra={"deduplicate_triples": bool(args.deduplicate_triples)},
    )

    # --------------------------------------------------------
    # Build subclass closure + materialize types
    # --------------------------------------------------------
    print("[4/6] Building subclass closure and materialized types...", flush=True)
    p0 = _now_ns()
    subclass_closure = compute_transitive_closure(subclass_of)

    entity_types: dict[str, list[str]] = {}
    for ent, types in entity_types_raw.items():
        materialized: set[str] = set(types)
        for t in types:
            materialized |= subclass_closure.get(t, set())
        entity_types[ent] = sorted(materialized)
    p1 = _now_ns()

    _append_phase_trace(
        phase_trace,
        phase="build_subclass_closure_and_materialize_types",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=p1,
        extra={
            "subclass_nodes": len(subclass_of),
            "closure_nodes": len(subclass_closure),
        },
    )

    # --------------------------------------------------------
    # Filter low-frequency relations
    # --------------------------------------------------------
    if args.min_triples_per_relation > 1:
        p0 = _now_ns()
        keep_relations = {r for r, c in relation_counter.items() if c >= args.min_triples_per_relation}
        all_triples_list = [tr for tr in all_triples_list if tr[1] in keep_relations]
        relation_counter = defaultdict(int)
        graph_entities = set()
        for h, r, t in all_triples_list:
            relation_counter[r] += 1
            graph_entities.add(h)
            graph_entities.add(t)
        p1 = _now_ns()
        _append_phase_trace(
            phase_trace,
            phase="filter_low_frequency_relations",
            run_start_ns=run_start_ns,
            phase_start_ns=p0,
            phase_end_ns=p1,
            extra={"min_triples_per_relation": args.min_triples_per_relation},
        )

    # --------------------------------------------------------
    # IDs / constraints
    # --------------------------------------------------------
    p0 = _now_ns()
    entities = sorted(graph_entities)
    relations = sorted({r for _, r, _ in all_triples_list})
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    entity_types_graph = {e: entity_types.get(e, []) for e in entities if e in entity_types}

    relation_constraints = {
        r: {
            "domain": sorted(relation_domains.get(r, set())),
            "range": sorted(relation_ranges.get(r, set())),
        }
        for r in relations
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

    # --------------------------------------------------------
    # Splits
    # --------------------------------------------------------
    print("[5/6] Building splits...", flush=True)
    p0 = _now_ns()
    train, valid, test, relation_split_stats = build_splits(
        triples=all_triples_list,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    p1 = _now_ns()

    _append_phase_trace(
        phase_trace,
        phase="build_splits",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=p1,
        extra={"num_triples": len(all_triples_list)},
    )

    # --------------------------------------------------------
    # Preflight
    # --------------------------------------------------------
    typed_entities = sum(1 for e in entities if len(entity_types_graph.get(e, [])) > 0)
    typing_coverage = _safe_div(typed_entities, len(entities))
    rel_with_domain = sum(1 for r in relations if relation_constraints.get(r, {}).get("domain"))
    rel_with_range = sum(1 for r in relations if relation_constraints.get(r, {}).get("range"))
    rel_with_both = sum(
        1
        for r in relations
        if relation_constraints.get(r, {}).get("domain") and relation_constraints.get(r, {}).get("range")
    )

    preflight = {
        "dataset": "dbpedia",
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now_iso(),
        "json_backend": JSON_BACKEND,
        "config": {
            "seed": args.seed,
            "valid_frac": args.valid_frac,
            "test_frac": args.test_frac,
            "entity_prefix_filter": list(args.entity_prefix_filter),
            "min_triples_per_relation": args.min_triples_per_relation,
            "deduplicate_triples": bool(args.deduplicate_triples),
        },
        "counts": {
            "num_entities": len(entities),
            "num_relations": len(relations),
            "num_triples": len(all_triples_list),
            "train_size": len(train),
            "valid_size": len(valid),
            "test_size": len(test),
            "typed_entities": typed_entities,
            "typing_coverage": typing_coverage,
            "relations_with_domain": rel_with_domain,
            "relations_with_range": rel_with_range,
            "relations_with_both": rel_with_both,
            "num_disjoint_pairs": len(disjoint_pairs),
            "avg_types_per_typed_entity": (
                _safe_div(sum(len(v) for v in entity_types_graph.values()), typed_entities)
            ),
        },
        "schema": {
            "subclass_edges": sum(len(v) for v in subclass_of.values()),
            "classes_with_subclass_info": len(subclass_of),
            "classes_in_closure": len(subclass_closure),
            "relations_with_domain_assertions": sum(1 for v in relation_domains.values() if v),
            "relations_with_range_assertions": sum(1 for v in relation_ranges.values() if v),
        },
        "top_relations": [
            {"relation": r, "relation_local_name": relation_local_name(r), "count": c}
            for r, c in sorted(relation_counter.items(), key=lambda x: x[1], reverse=True)[:20]
        ],
        "source_files": {
            "ontology": str(args.ontology),
            "instance_types": str(args.instance_types),
            "mappingbased_objects": str(args.mappingbased_objects),
        },
        "runtime": {
            "total_sec": _ns_to_sec(_now_ns() - run_start_ns),
            "phase_trace": phase_trace,
        },
    }

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------
    print("[6/6] Writing processed artifacts...", flush=True)
    p0 = _now_ns()
    write_tsv(out / "all_triples.tsv", all_triples_list)
    write_tsv(out / "train.tsv", train)
    write_tsv(out / "valid.tsv", valid)
    write_tsv(out / "test.tsv", test)

    write_json(out / "entity2id.json", entity2id)
    write_json(out / "relation2id.json", relation2id)
    write_json(out / "entity_types.json", entity_types_graph)
    write_json(out / "relation_constraints.json", relation_constraints)
    write_json(out / "disjoint_pairs.json", [list(x) for x in sorted(disjoint_pairs)])
    write_json(out / "relation_split_stats.json", relation_split_stats)
    write_json(out / "file_stats.json", file_stats)

    preflight["runtime"]["total_sec"] = _ns_to_sec(_now_ns() - run_start_ns)
    write_json(out / "preflight.json", preflight)
    p1 = _now_ns()

    _append_phase_trace(
        phase_trace,
        phase="write_outputs",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=p1,
    )

    preflight["runtime"]["outputs_written_sec"] = _ns_to_sec(p1 - p0)
    preflight["runtime"]["total_sec"] = _ns_to_sec(_now_ns() - run_start_ns)
    preflight["runtime"]["phase_trace"] = phase_trace
    write_json(out / "preflight.json", preflight)

    print("[OK] DBpedia preprocessing finished.", flush=True)
    print(json.dumps(preflight, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()