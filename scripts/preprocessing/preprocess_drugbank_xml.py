#!/usr/bin/env python3
"""
Preprocess DrugBank full_database.xml into a unified artifact bundle
for OptK / blindness experiments.

This script induces a typed KG from the XML structure using entity-valued links:
  - drug_interacts_with: Drug -> Drug
  - drug_has_target: Drug -> Protein
  - drug_has_enzyme: Drug -> Protein
  - drug_has_carrier: Drug -> Protein
  - drug_has_transporter: Drug -> Protein
  - drug_in_pathway: Drug -> Pathway

Outputs:
  - all_triples.tsv
  - train.tsv / valid.tsv / test.tsv
  - entity2id.json / relation2id.json
  - entity_types.json
  - relation_constraints.json
  - disjoint_pairs.json
  - relation_split_stats.json
  - parse_stats.json
  - preflight.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Iterable, Iterator, Sequence

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

RELATION_SIGNATURES = {
    "drug_interacts_with": ("Drug", "Drug"),
    "drug_has_target": ("Drug", "Protein"),
    "drug_has_enzyme": ("Drug", "Protein"),
    "drug_has_carrier": ("Drug", "Protein"),
    "drug_has_transporter": ("Drug", "Protein"),
    "drug_in_pathway": ("Drug", "Pathway"),
}

DISJOINT_PAIRS = [
    ("Drug", "Protein"),
    ("Drug", "Pathway"),
    ("Protein", "Pathway"),
]

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
    p = argparse.ArgumentParser(description="Preprocess DrugBank XML")
    p.add_argument("--input", required=True, type=Path, help="Path to full_database.xml")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--dataset-name", type=str, default="DrugBankKG_from_XML")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--min-relation-count-for-split", type=int, default=10)
    p.add_argument("--max-drugs", type=int, default=None, help="Optional cap for debugging")
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


def write_tsv(path: Path, triples: Sequence[Triple]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")
            count += 1
    tmp_path.replace(path)
    return count


# ============================================================
# XML helpers
# ============================================================

def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def iter_children_by_name(elem: ET.Element, name: str) -> Iterator[ET.Element]:
    for child in elem:
        if strip_ns(child.tag) == name:
            yield child


def find_first_text(elem: ET.Element, path_names: Sequence[str]) -> str | None:
    cur = elem
    for name in path_names:
        nxt = None
        for child in cur:
            if strip_ns(child.tag) == name:
                nxt = child
                break
        if nxt is None:
            return None
        cur = nxt
    text = (cur.text or "").strip()
    return text or None


def get_primary_drugbank_id(drug_elem: ET.Element) -> str | None:
    fallback = None
    for child in drug_elem:
        if strip_ns(child.tag) != "drugbank-id":
            continue
        text = (child.text or "").strip()
        if not text:
            continue
        if child.attrib.get("primary", "").lower() == "true":
            return text
        if fallback is None:
            fallback = text
    return fallback


def extract_partner_ids(container: ET.Element, partner_tag: str) -> Iterator[str]:
    for item in container:
        if strip_ns(item.tag) != partner_tag:
            continue

        partner_ids: list[str] = []
        for sub in item.iter():
            tag = strip_ns(sub.tag)
            if tag == "polypeptide":
                pid = (sub.attrib.get("id") or "").strip()
                if pid:
                    partner_ids.append(pid)
            elif tag == "id":
                txt = (sub.text or "").strip()
                if txt:
                    partner_ids.append(txt)

        seen = set()
        for pid in partner_ids:
            if pid not in seen:
                seen.add(pid)
                yield pid


def extract_drug_interactions(drug_elem: ET.Element) -> Iterator[str]:
    for child in drug_elem:
        if strip_ns(child.tag) != "drug-interactions":
            continue
        for di in child:
            if strip_ns(di.tag) != "drug-interaction":
                continue
            dbid = find_first_text(di, ["drugbank-id"])
            if dbid:
                yield dbid


def extract_pathway_ids(drug_elem: ET.Element) -> Iterator[str]:
    for child in drug_elem:
        if strip_ns(child.tag) != "pathways":
            continue
        for pathway in child:
            if strip_ns(pathway.tag) != "pathway":
                continue
            pid = find_first_text(pathway, ["smpdb-id"])
            if pid:
                yield pid


# ============================================================
# Dataclasses
# ============================================================

@dataclass(slots=True)
class ParseStats:
    drugs_seen: int = 0
    drugs_kept: int = 0
    drugs_skipped_missing_id: int = 0

    triples_total_before_dedup: int = 0
    triples_unique: int = 0

    relation_counts_total: dict[str, int] | None = None
    relation_counts_unique: dict[str, int] | None = None

    unique_drug_entities: int = 0
    unique_protein_entities: int = 0
    unique_pathway_entities: int = 0

    interactions_found: int = 0
    targets_found: int = 0
    enzymes_found: int = 0
    carriers_found: int = 0
    transporters_found: int = 0
    pathways_found: int = 0

    parse_xml_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# KG extraction
# ============================================================

def parse_drugbank_xml(
    xml_path: Path,
    *,
    max_drugs: int | None = None,
    show_progress: bool = False,
) -> tuple[set[Triple], dict[str, set[str]], Counter, ParseStats]:
    triples: set[Triple] = set()
    entity_types: DefaultDict[str, set[str]] = defaultdict(set)
    relation_counts_total: Counter = Counter()

    stats = ParseStats()
    parse_start_ns = _now_ns()

    context = ET.iterparse(xml_path, events=("end",))

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(
            desc="drugbank-drugs",
            unit="drug",
            dynamic_ncols=True,
            leave=False,
        )

    for _, elem in context:
        if strip_ns(elem.tag) != "drug":
            continue

        stats.drugs_seen += 1
        if pbar is not None:
            pbar.update(1)

        if max_drugs is not None and stats.drugs_kept >= max_drugs:
            elem.clear()
            continue

        drug_id = get_primary_drugbank_id(elem)
        if not drug_id:
            stats.drugs_skipped_missing_id += 1
            elem.clear()
            continue

        stats.drugs_kept += 1
        drug_ent = f"drug:{drug_id}"
        entity_types[drug_ent].add("Drug")

        # Drug-drug interactions
        for other_id in extract_drug_interactions(elem):
            other_ent = f"drug:{other_id}"
            entity_types[other_ent].add("Drug")
            triples.add((drug_ent, "drug_interacts_with", other_ent))
            relation_counts_total["drug_interacts_with"] += 1
            stats.interactions_found += 1

        # Protein partners
        for container_name, rel_name, stat_name in [
            ("targets", "drug_has_target", "targets_found"),
            ("enzymes", "drug_has_enzyme", "enzymes_found"),
            ("carriers", "drug_has_carrier", "carriers_found"),
            ("transporters", "drug_has_transporter", "transporters_found"),
        ]:
            for container in iter_children_by_name(elem, container_name):
                for pid in extract_partner_ids(container, container_name[:-1]):
                    protein_ent = f"protein:{pid}"
                    entity_types[protein_ent].add("Protein")
                    triples.add((drug_ent, rel_name, protein_ent))
                    relation_counts_total[rel_name] += 1
                    setattr(stats, stat_name, getattr(stats, stat_name) + 1)

        # Pathways
        for pathway_id in extract_pathway_ids(elem):
            pathway_ent = f"pathway:{pathway_id}"
            entity_types[pathway_ent].add("Pathway")
            triples.add((drug_ent, "drug_in_pathway", pathway_ent))
            relation_counts_total["drug_in_pathway"] += 1
            stats.pathways_found += 1

        elem.clear()

    if pbar is not None:
        pbar.close()

    stats.parse_xml_sec = _ns_to_sec(_now_ns() - parse_start_ns)
    stats.triples_total_before_dedup = int(sum(relation_counts_total.values()))
    stats.triples_unique = len(triples)

    relation_counts_unique: Counter = Counter()
    for _, r, _ in triples:
        relation_counts_unique[r] += 1

    stats.relation_counts_total = dict(relation_counts_total)
    stats.relation_counts_unique = dict(relation_counts_unique)
    stats.unique_drug_entities = sum(1 for e, ts in entity_types.items() if "Drug" in ts)
    stats.unique_protein_entities = sum(1 for e, ts in entity_types.items() if "Protein" in ts)
    stats.unique_pathway_entities = sum(1 for e, ts in entity_types.items() if "Pathway" in ts)

    return triples, entity_types, relation_counts_unique, stats


# ============================================================
# Splitting
# ============================================================

def split_by_relation(
    triples: Sequence[Triple],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    min_relation_count_for_split: int,
) -> tuple[list[Triple], list[Triple], list[Triple], dict[str, dict[str, int]]]:
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-8:
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
# Main
# ============================================================

def main() -> int:
    args = parse_args()
    show_progress = not args.no_progress

    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.train_ratio <= 0 or args.valid_ratio <= 0 or args.test_ratio <= 0:
        print("[ERROR] train/valid/test ratios must all be > 0", file=sys.stderr)
        return 1

    if args.min_relation_count_for_split <= 0:
        print("[ERROR] min_relation_count_for_split must be > 0", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_start_ns = _now_ns()
    started_at_utc = _utc_now_iso()
    phase_trace: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Parse XML
    # --------------------------------------------------------
    print(f"[1/5] Parsing DrugBank XML: {args.input}", flush=True)
    p0 = _now_ns()
    triples, entity_types, relation_counts, parse_stats = parse_drugbank_xml(
        args.input,
        max_drugs=args.max_drugs,
        show_progress=show_progress,
    )
    p1 = _now_ns()

    _append_phase_trace(
        phase_trace,
        phase="parse_drugbank_xml",
        run_start_ns=run_start_ns,
        phase_start_ns=p0,
        phase_end_ns=p1,
        extra={"drugs_kept": parse_stats.drugs_kept},
    )

    if not triples:
        print("[ERROR] No triples extracted from DrugBank XML.", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # Sort + split
    # --------------------------------------------------------
    print("[2/5] Building splits...", flush=True)
    p0 = _now_ns()
    triples_sorted = sorted(triples)
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
    print("[3/5] Building ids and constraints...", flush=True)
    p0 = _now_ns()
    entities = sorted({s for s, _, _ in triples_sorted} | {o for _, _, o in triples_sorted})
    relations = sorted({p for _, p, _ in triples_sorted})
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    relation_constraints = {
        rel: {
            "domain": [RELATION_SIGNATURES[rel][0]],
            "range": [RELATION_SIGNATURES[rel][1]],
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
        extra={"num_entities": len(entities), "num_relations": len(relations)},
    )

    # --------------------------------------------------------
    # Preflight
    # --------------------------------------------------------
    typed_entities = sum(1 for e in entities if entity_types.get(e))
    preflight = {
        "dataset": args.dataset_name,
        "input": str(args.input),
        "json_backend": JSON_BACKEND,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now_iso(),
        "config": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "valid_ratio": args.valid_ratio,
            "test_ratio": args.test_ratio,
            "min_relation_count_for_split": args.min_relation_count_for_split,
            "max_drugs": args.max_drugs,
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
            "num_disjoint_pairs": len(DISJOINT_PAIRS),
        },
        "schema": {
            "schema_source": "induced_from_xml_structure",
            "relation_signatures": RELATION_SIGNATURES,
            "disjoint_pairs": DISJOINT_PAIRS,
        },
        "relation_counts": dict(relation_counts),
        "runtime": {
            "total_sec": _ns_to_sec(_now_ns() - run_start_ns),
            "phase_trace": phase_trace,
        },
    }

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------
    print("[4/5] Writing processed artifacts...", flush=True)
    p0 = _now_ns()
    write_tsv(args.output_dir / "all_triples.tsv", triples_sorted)
    write_tsv(args.output_dir / "train.tsv", train)
    write_tsv(args.output_dir / "valid.tsv", valid)
    write_tsv(args.output_dir / "test.tsv", test)
    write_json(args.output_dir / "entity2id.json", entity2id)
    write_json(args.output_dir / "relation2id.json", relation2id)
    write_json(
        args.output_dir / "entity_types.json",
        {k: sorted(v) for k, v in entity_types.items()},
    )
    write_json(args.output_dir / "relation_constraints.json", relation_constraints)
    write_json(args.output_dir / "disjoint_pairs.json", [list(p) for p in DISJOINT_PAIRS])
    write_json(args.output_dir / "relation_split_stats.json", relation_split_stats)
    write_json(args.output_dir / "parse_stats.json", parse_stats.to_dict())

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

    preflight["runtime"]["outputs_written_sec"] = _ns_to_sec(p1 - p0)
    preflight["runtime"]["total_sec"] = _ns_to_sec(_now_ns() - run_start_ns)
    preflight["runtime"]["phase_trace"] = phase_trace
    write_json(args.output_dir / "preflight.json", preflight)

    print("[5/5] Done.", flush=True)
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())