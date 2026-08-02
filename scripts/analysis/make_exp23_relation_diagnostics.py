"""Create relation-level q=5 diagnostics for the three datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def short_relation(value: object) -> str:
    text = str(value)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rstrip("/").rsplit("/", 1)[-1]
    return text


def weighted_mean(
    frame: pd.DataFrame,
    column: str,
    weight_column: str = "num_feasible",
) -> float:
    data = frame[[column, weight_column]].dropna()
    if data.empty:
        return float("nan")
    weights = data[weight_column].astype(float)
    if float(weights.sum()) == 0:
        return float("nan")
    return float((data[column].astype(float) * weights).sum() / weights.sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eurostat", type=Path, required=True)
    parser.add_argument("--dbpedia", type=Path, required=True)
    parser.add_argument("--drugbank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = [
        ("EurostatKG", "PairRE-main", args.eurostat),
        ("DBpedia", "ComplEx-main", args.dbpedia),
        ("DrugBank", "RotatE-main", args.drugbank),
    ]

    all_rows = []
    summaries = []
    top_support_rows = []
    top_bottom_rows = []

    for dataset, scorer, path in inputs:
        frame = pd.read_csv(path)
        frame = frame[
            (frame["quota"] == 5)
            & (frame["num_feasible"].fillna(0) > 0)
        ].copy()
        if frame.empty:
            raise SystemExit(f"No feasible q=5 rows in {path}")

        frame["dataset"] = dataset
        frame["scorer"] = scorer
        frame["relation_short"] = frame["relation"].map(short_relation)
        frame["support_share"] = (
            frame["num_feasible"] / frame["num_feasible"].sum()
        )
        all_rows.append(frame)

        for minimum_support in (1, 10, 30, 100, 500):
            subset = frame[frame["num_feasible"] >= minimum_support]
            if subset.empty:
                continue
            summaries.append(
                {
                    "dataset": dataset,
                    "scorer": scorer,
                    "quota": 5,
                    "min_support": minimum_support,
                    "num_relations_all_feasible": int(len(frame)),
                    "num_relations_kept": int(len(subset)),
                    "total_feasible_queries": int(
                        frame["num_feasible"].sum()
                    ),
                    "kept_feasible_queries": int(
                        subset["num_feasible"].sum()
                    ),
                    "kept_query_share": float(
                        subset["num_feasible"].sum()
                        / frame["num_feasible"].sum()
                    ),
                    "max_relation_support_share": float(
                        frame["support_share"].max()
                    ),
                    "top3_relation_support_share": float(
                        frame.sort_values(
                            "support_share", ascending=False
                        )["support_share"].head(3).sum()
                    ),
                    "macro_optq_viol": float(
                        subset["optq_viol_at_k"].mean()
                    ),
                    "macro_optq_adm": float(
                        subset["optq_adm_at_k"].mean()
                    ),
                    "macro_optq_pres": float(
                        subset["optq_pres_at_k"].mean()
                    ),
                    "macro_optq_shift": float(
                        subset["optq_shift_at_k"].mean()
                    ),
                    "weighted_optq_viol": weighted_mean(
                        subset, "optq_viol_at_k"
                    ),
                    "weighted_optq_adm": weighted_mean(
                        subset, "optq_adm_at_k"
                    ),
                    "weighted_optq_pres": weighted_mean(
                        subset, "optq_pres_at_k"
                    ),
                    "weighted_optq_shift": weighted_mean(
                        subset, "optq_shift_at_k"
                    ),
                    "share_relations_adm_ge_0p5": float(
                        (subset["optq_adm_at_k"] >= 0.5).mean()
                    ),
                    "share_relations_viol_le_0p5": float(
                        (subset["optq_viol_at_k"] <= 0.5).mean()
                    ),
                }
            )

        columns = [
            "dataset",
            "scorer",
            "relation_short",
            "relation",
            "queries",
            "num_feasible",
            "support_share",
            "feasible_rate",
            "optq_viol_at_k",
            "optq_adm_at_k",
            "optq_pres_at_k",
            "optq_shift_at_k",
            "mean_lambda",
        ]
        top_support_rows.append(
            frame.sort_values("num_feasible", ascending=False)
            .head(10)[columns]
        )

        supported = frame[frame["num_feasible"] >= 30]
        if supported.empty:
            supported = frame
        top_bottom_rows.extend(
            [
                supported.sort_values(
                    "optq_adm_at_k", ascending=False
                ).head(10)[columns].assign(rank_type="highest_optq_adm"),
                supported.sort_values(
                    "optq_pres_at_k", ascending=True
                ).head(10)[columns].assign(rank_type="lowest_optq_pres"),
            ]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(
        args.output_dir / "exp23_relation_level_q5_all_relations.csv",
        index=False,
    )
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "exp23_relation_level_q5_summary.csv",
        index=False,
    )
    pd.concat(top_support_rows, ignore_index=True).to_csv(
        args.output_dir / "exp23_relation_level_q5_top_support.csv",
        index=False,
    )
    pd.concat(top_bottom_rows, ignore_index=True).to_csv(
        args.output_dir / "exp23_relation_level_q5_top_bottom.csv",
        index=False,
    )
    print(f"Wrote relation diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
