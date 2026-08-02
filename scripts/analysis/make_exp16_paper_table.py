"""Select the full-scope EXP-16 paper columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    subset = frame[frame["scope"] == "full"].copy()
    columns = [
        "dataset",
        "scorer",
        "variant",
        "feasible_rate",
        "delta_hit10",
        "delta_mrr10",
        "delta_viol",
        "delta_adm",
        "delta_unknown",
        "optq_pres",
        "optq_shift",
    ]
    result = subset[columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
