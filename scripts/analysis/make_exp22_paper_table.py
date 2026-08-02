"""Select the publication-level learned-reranker rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SELECTORS = {
    "Base",
    "OptQ",
    "Learned nearest OptQ-Pres",
    "Learned quota-safe",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    subset = frame[
        (frame["scope"] == "blind_strict")
        & (frame["selector"].isin(SELECTORS))
    ].copy()
    columns = [
        "dataset",
        "selector",
        "alpha",
        "quota_success",
        "viol_at_k",
        "adm_at_k",
        "pres_at_k",
        "shift_at_k",
        "hit10",
        "mrr10",
        "num_queries",
    ]
    result = subset[columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
