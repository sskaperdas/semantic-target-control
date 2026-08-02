"""Select one informative relation-support threshold per dataset."""

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
    keep = (
        ((frame["dataset"] == "EurostatKG") & (frame["min_support"] == 500))
        | ((frame["dataset"] == "DBpedia") & (frame["min_support"] == 30))
        | ((frame["dataset"] == "DrugBank") & (frame["min_support"] == 1))
    )
    result = frame[keep].copy()
    columns = [
        "dataset",
        "scorer",
        "min_support",
        "num_relations_all_feasible",
        "num_relations_kept",
        "kept_query_share",
        "max_relation_support_share",
        "top3_relation_support_share",
        "macro_optq_adm",
        "weighted_optq_adm",
        "macro_optq_pres",
        "weighted_optq_pres",
        "share_relations_adm_ge_0p5",
        "share_relations_viol_le_0p5",
    ]
    result = result[columns]
    for column in result.columns:
        if column not in {
            "dataset",
            "scorer",
            "min_support",
            "num_relations_all_feasible",
            "num_relations_kept",
        }:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            ).round(3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
