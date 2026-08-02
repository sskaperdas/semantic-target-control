#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class QuerySlot:
    query_id: str
    split: str
    mode: str          # "head" or "tail"
    row_index: int     # row index inside split file
    head: str
    relation: str
    tail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed stratified evaluation subset of query slots "
            "from a processed KGC split file (typically test.tsv)."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Processed dataset directory, e.g. data/processed/dbpedia",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Which split to sample query slots from (default: test)",
    )
    parser.add_argument(
        "--per-relation-per-mode",
        type=int,
        default=50,
        help=(
            "Maximum number of query slots to sample for each "
            "(relation, mode) stratum"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where subset artifacts will be written",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="dbpedia",
        help="Dataset name for metadata only",
    )
    parser.add_argument(
        "--min-per-stratum-warning",
        type=int,
        default=10,
        help=(
            "Warn if a (relation, mode) stratum has fewer than this many "
            "available examples"
        ),
    )
    return parser.parse_args()


def read_id_map(path: Path) -> Dict[str, str]:
    """
    Reads JSON mapping objects like:
      {"dbo:author": 0, "dbo:birthPlace": 1}
    or
      {"0": "dbo:author", "1": "dbo:birthPlace"}  (less likely)
    Returns a normalized id(str) -> label(str) map when possible.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    id_to_label: Dict[str, str] = {}

    if not isinstance(data, dict):
        return id_to_label

    # Common case: label -> id
    sample_items = list(data.items())[:5]
    if sample_items and isinstance(sample_items[0][0], str) and isinstance(sample_items[0][1], int):
        for label, idx in data.items():
            id_to_label[str(idx)] = str(label)
        return id_to_label

    # Alternate case: id -> label
    for k, v in data.items():
        if isinstance(v, str):
            id_to_label[str(k)] = v

    return id_to_label


def resolve_relation_label(relation_value: str, id_to_relation: Dict[str, str]) -> str:
    """
    If relation_value is an ID stored as text like "12", map it back to label.
    Otherwise keep the raw relation string.
    """
    relation_value = str(relation_value)
    return id_to_relation.get(relation_value, relation_value)


def read_split_rows(split_path: Path) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []

    with split_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(
                    f"Expected 3 tab-separated columns in {split_path}, "
                    f"but got {len(row)} at row {row_idx}: {row!r}"
                )
            h, r, t = row
            rows.append((h, r, t))

    if not rows:
        raise ValueError(f"No rows found in split file: {split_path}")

    return rows


def build_query_slots(
    split_name: str,
    rows: List[Tuple[str, str, str]],
    id_to_relation: Dict[str, str],
) -> List[QuerySlot]:
    """
    For each triple row i = (h, r, t), create:
      - head query: (? , r, t)
      - tail query: (h, r, ?)
    The query_id is deterministic and tied to split + mode + row index.
    """
    slots: List[QuerySlot] = []

    for row_index, (h, r_raw, t) in enumerate(rows):
        r = resolve_relation_label(r_raw, id_to_relation)

        head_qid = f"{split_name}_head_{row_index:09d}"
        tail_qid = f"{split_name}_tail_{row_index:09d}"

        slots.append(
            QuerySlot(
                query_id=head_qid,
                split=split_name,
                mode="head",
                row_index=row_index,
                head=h,
                relation=r,
                tail=t,
            )
        )
        slots.append(
            QuerySlot(
                query_id=tail_qid,
                split=split_name,
                mode="tail",
                row_index=row_index,
                head=h,
                relation=r,
                tail=t,
            )
        )

    return slots


def stratified_sample(
    slots: List[QuerySlot],
    per_relation_per_mode: int,
    seed: int,
) -> Tuple[List[QuerySlot], Dict[Tuple[str, str], int], Dict[Tuple[str, str], int]]:
    """
    Stratify by (relation, mode), sample up to cap per stratum.
    """
    rng = random.Random(seed)

    by_stratum: Dict[Tuple[str, str], List[QuerySlot]] = defaultdict(list)
    for slot in slots:
        key = (slot.relation, slot.mode)
        by_stratum[key].append(slot)

    available_counts = {k: len(v) for k, v in by_stratum.items()}

    selected: List[QuerySlot] = []
    selected_counts: Dict[Tuple[str, str], int] = {}

    for key in sorted(by_stratum.keys()):
        candidates = list(by_stratum[key])
        rng.shuffle(candidates)

        take_n = min(per_relation_per_mode, len(candidates))
        picked = candidates[:take_n]

        selected.extend(picked)
        selected_counts[key] = len(picked)

    # Global deterministic order for stable downstream behavior
    selected.sort(key=lambda s: (s.mode, s.relation, s.row_index, s.query_id))

    return selected, available_counts, selected_counts


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_query_ids(path: Path, slots: List[QuerySlot]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for slot in slots:
            f.write(slot.query_id + "\n")


def write_subset_rows_jsonl(path: Path, slots: List[QuerySlot]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for slot in slots:
            f.write(json.dumps(asdict(slot), ensure_ascii=False) + "\n")


def write_subset_stats_csv(
    path: Path,
    available_counts: Dict[Tuple[str, str], int],
    selected_counts: Dict[Tuple[str, str], int],
) -> None:
    fieldnames = ["relation", "mode", "available", "selected", "sampling_fraction"]

    rows = []
    all_keys = sorted(set(available_counts) | set(selected_counts))
    for relation, mode in all_keys:
        available = available_counts.get((relation, mode), 0)
        selected = selected_counts.get((relation, mode), 0)
        frac = (selected / available) if available > 0 else 0.0
        rows.append(
            {
                "relation": relation,
                "mode": mode,
                "available": available,
                "selected": selected,
                "sampling_fraction": f"{frac:.6f}",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    path: Path,
    *,
    dataset_name: str,
    processed_dir: Path,
    split: str,
    seed: int,
    per_relation_per_mode: int,
    split_rows: int,
    total_query_slots: int,
    selected_query_slots: int,
    selected_head: int,
    selected_tail: int,
    relation_count: int,
    warning_small_strata: List[dict],
) -> None:
    manifest = {
        "dataset_name": dataset_name,
        "processed_dir": str(processed_dir),
        "subset_type": "evaluation_query_subset",
        "subset_scope": "query slots derived from split triples; not a new dataset",
        "sampling_strategy": {
            "name": "stratified_relation_mode_cap",
            "description": (
                "Build head and tail query slots from the chosen split and "
                "sample up to a fixed cap per (relation, mode) stratum."
            ),
            "strata": ["relation", "mode"],
            "per_relation_per_mode": per_relation_per_mode,
            "seed": seed,
        },
        "split": split,
        "split_row_count": split_rows,
        "total_query_slots_before_sampling": total_query_slots,
        "selected_query_slots": selected_query_slots,
        "selected_head_queries": selected_head,
        "selected_tail_queries": selected_tail,
        "relation_count_in_selected_subset": relation_count,
        "warning_small_strata": warning_small_strata,
        "intended_use": (
            "Use the same fixed subset across expensive evaluation experiments "
            "for reproducible and fair comparison."
        ),
    }

    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    processed_dir: Path = args.processed_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    split_path = processed_dir / f"{args.split}.tsv"
    relation2id_path = processed_dir / "relation2id.json"

    if not processed_dir.exists():
        raise SystemExit(f"[ERROR] Processed dir does not exist: {processed_dir}")
    if not split_path.exists():
        raise SystemExit(f"[ERROR] Split file does not exist: {split_path}")

    ensure_dir(output_dir)

    id_to_relation: Dict[str, str] = {}
    if relation2id_path.exists():
        id_to_relation = read_id_map(relation2id_path)

    print("============================================================")
    print("BUILD DBPEDIA EVAL SUBSET")
    print("============================================================")
    print(f"[INFO] Processed dir            : {processed_dir}")
    print(f"[INFO] Split                    : {args.split}")
    print(f"[INFO] Split path               : {split_path}")
    print(f"[INFO] Per relation per mode    : {args.per_relation_per_mode}")
    print(f"[INFO] Seed                     : {args.seed}")
    print(f"[INFO] Output dir               : {output_dir}")
    print("============================================================")

    rows = read_split_rows(split_path)
    slots = build_query_slots(
        split_name=args.split,
        rows=rows,
        id_to_relation=id_to_relation,
    )

    selected, available_counts, selected_counts = stratified_sample(
        slots=slots,
        per_relation_per_mode=args.per_relation_per_mode,
        seed=args.seed,
    )

    query_ids_path = output_dir / "query_ids.txt"
    subset_rows_path = output_dir / "subset_rows.jsonl"
    stats_csv_path = output_dir / "subset_stats.csv"
    manifest_path = output_dir / "subset_manifest.json"

    write_query_ids(query_ids_path, selected)
    write_subset_rows_jsonl(subset_rows_path, selected)
    write_subset_stats_csv(stats_csv_path, available_counts, selected_counts)

    selected_head = sum(1 for s in selected if s.mode == "head")
    selected_tail = sum(1 for s in selected if s.mode == "tail")
    relation_count = len({s.relation for s in selected})

    small_strata = []
    for (relation, mode), available in sorted(available_counts.items()):
        if available < args.min_per_stratum_warning:
            small_strata.append(
                {
                    "relation": relation,
                    "mode": mode,
                    "available": available,
                }
            )

    write_manifest(
        manifest_path,
        dataset_name=args.dataset_name,
        processed_dir=processed_dir,
        split=args.split,
        seed=args.seed,
        per_relation_per_mode=args.per_relation_per_mode,
        split_rows=len(rows),
        total_query_slots=len(slots),
        selected_query_slots=len(selected),
        selected_head=selected_head,
        selected_tail=selected_tail,
        relation_count=relation_count,
        warning_small_strata=small_strata,
    )

    print("")
    print("[DONE] Subset created successfully")
    print(f"[OUT] query_ids.txt        : {query_ids_path}")
    print(f"[OUT] subset_rows.jsonl    : {subset_rows_path}")
    print(f"[OUT] subset_stats.csv     : {stats_csv_path}")
    print(f"[OUT] subset_manifest.json : {manifest_path}")
    print("")
    print("Summary")
    print("------------------------------------------------------------")
    print(f"Split rows                    : {len(rows)}")
    print(f"Total query slots             : {len(slots)}")
    print(f"Selected query slots          : {len(selected)}")
    print(f"Selected head queries         : {selected_head}")
    print(f"Selected tail queries         : {selected_tail}")
    print(f"Relations represented         : {relation_count}")
    print(f"Small strata warnings         : {len(small_strata)}")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()