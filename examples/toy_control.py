"""Run the lightweight STC controller on the bundled toy window."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stc import Candidate, Status, control_topk  # noqa: E402


def main() -> int:
    path = Path(__file__).with_name("toy_candidates.csv")
    candidates: list[Candidate] = []

    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            candidates.append(
                Candidate(
                    identifier=row["identifier"],
                    score=float(row["score"]),
                    status=Status(row["status"]),
                    energy=float(row["energy"]),
                    original_index=index,
                )
            )

    result = control_topk(candidates, k=5, q=3)

    print("STC toy demonstration")
    print("=====================")
    print(f"feasible: {result.feasible}")
    print(f"lambda*: {result.lambda_star:.6f}")
    print(
        f"quota certificate: {result.admissible_in_topk} admissible "
        f"candidates in Top-{result.top_k} (requested q={result.quota})"
    )
    print()

    for item in result.returned:
        candidate = item.candidate
        print(
            f"{item.rank:>2}. {candidate.identifier:<9} "
            f"{candidate.normalized_status().value:<11} "
            f"frozen={candidate.score:.3f} "
            f"energy={candidate.energy:.2f} "
            f"controlled={item.controlled_score:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
