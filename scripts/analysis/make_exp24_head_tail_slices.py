"""Create q=5 head/tail query slices from EXP-12 query-level outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


USE_COLUMNS = [
    "mode",
    "is_blind",
    "blind_strict",
    "q5_feasible",
    "q5_optq_viol_at_k",
    "q5_optq_adm_at_k",
    "q5_optq_unknown_at_k",
    "q5_optq_pres_at_k",
    "q5_optq_shift_at_k",
    "q5_optq_adm_count_topk",
    "q5_afq_viol_at_k",
    "q5_afq_adm_at_k",
    "q5_afq_pres_at_k",
    "q5_afq_shift_at_k",
    "q5_hardval_viol_at_k",
    "q5_hardval_adm_at_k",
    "q5_hardval_pres_at_k",
    "q5_hardval_shift_at_k",
    "q5_lambda_star",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eurostat", type=Path, required=True)
    parser.add_argument("--dbpedia", type=Path, required=True)
    parser.add_argument("--drugbank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = [
        ("EurostatKG", "PairRE-main", args.eurostat),
        ("DBpedia", "ComplEx-main", args.dbpedia),
        ("DrugBank", "RotatE-main", args.drugbank),
    ]
    rows = []

    for dataset, scorer, path in inputs:
        frame = pd.read_csv(path, usecols=USE_COLUMNS)
        scopes = {
            "full_feasible": frame[frame["q5_feasible"] == 1],
            "blind_strict_feasible": frame[
                (frame["q5_feasible"] == 1)
                & (frame["blind_strict"] == 1)
            ],
        }

        for scope, scoped in scopes.items():
            for mode in ("head", "tail"):
                subset = scoped[scoped["mode"] == mode]
                if subset.empty:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "scorer": scorer,
                        "scope": scope,
                        "mode": mode,
                        "num_queries": int(len(subset)),
                        "optq_viol": subset[
                            "q5_optq_viol_at_k"
                        ].mean(),
                        "optq_adm": subset[
                            "q5_optq_adm_at_k"
                        ].mean(),
                        "optq_unknown": subset[
                            "q5_optq_unknown_at_k"
                        ].mean(),
                        "optq_pres": subset[
                            "q5_optq_pres_at_k"
                        ].mean(),
                        "optq_shift": subset[
                            "q5_optq_shift_at_k"
                        ].mean(),
                        "optq_adm_count": subset[
                            "q5_optq_adm_count_topk"
                        ].mean(),
                        "afq_adm": subset[
                            "q5_afq_adm_at_k"
                        ].mean(),
                        "afq_pres": subset[
                            "q5_afq_pres_at_k"
                        ].mean(),
                        "hardval_adm": subset[
                            "q5_hardval_adm_at_k"
                        ].mean(),
                        "hardval_pres": subset[
                            "q5_hardval_pres_at_k"
                        ].mean(),
                        "mean_lambda": subset[
                            "q5_lambda_star"
                        ].mean(),
                        "median_lambda": subset[
                            "q5_lambda_star"
                        ].median(),
                    }
                )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
