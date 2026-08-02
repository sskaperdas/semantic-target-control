"""Command-line interface for the lightweight STC reference controller."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .core import Candidate, Status, control_topk


def load_candidates(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"identifier", "score", "status", "energy"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for index, row in enumerate(reader):
            candidates.append(
                Candidate(
                    identifier=row["identifier"],
                    score=float(row["score"]),
                    status=Status(row["status"].strip().lower()),
                    energy=float(row["energy"]),
                    original_index=index,
                )
            )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply OptQ to one finite candidate window."
    )
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quota", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = control_topk(
        load_candidates(args.candidates),
        k=args.top_k,
        q=args.quota,
    )

    payload = {
        "feasible": result.feasible,
        "quota": result.quota,
        "top_k": result.top_k,
        "lambda_star": result.lambda_star,
        "tau_q": result.tau_q,
        "admissible_in_window": result.admissible_in_window,
        "admissible_in_topk": result.admissible_in_topk,
        "reason": result.reason,
        "returned": [
            {
                "rank": item.rank,
                "identifier": item.candidate.identifier,
                "status": item.candidate.normalized_status().value,
                "frozen_score": item.candidate.score,
                "energy": item.candidate.energy,
                "controlled_score": item.controlled_score,
            }
            for item in result.returned
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"feasible: {result.feasible}")
    print(f"lambda*: {result.lambda_star:.6f}")
    print(
        f"admissible in Top-{result.top_k}: "
        f"{result.admissible_in_topk}/{result.top_k}"
    )
    print()
    print("rank  candidate  status       frozen    energy   controlled")
    for row in payload["returned"]:
        print(
            f"{row['rank']:>4}  "
            f"{row['identifier']:<9}  "
            f"{row['status']:<11}  "
            f"{row['frozen_score']:>7.3f}  "
            f"{row['energy']:>6.3f}  "
            f"{row['controlled_score']:>10.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
