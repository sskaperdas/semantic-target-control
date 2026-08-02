"""Create the compact EXP-16 operational-energy ablation table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/stc_upgrade_corrected/exp16_energy_ablation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/generated/exp16_corrected_energy_ablation_compact.csv"
        ),
    )
    args = parser.parse_args()

    dataset_map = {
        "eurostat_pairre": ("EurostatKG", "PairRE-main"),
        "dbpedia_complex": ("DBpedia", "ComplEx-main"),
    }
    variant_order = {
        "domain_range_u1": 0,
        "domain_only_u1": 1,
        "range_only_u1": 2,
        "domain_range_u0p5": 3,
        "domain_range_u2": 4,
    }

    rows = []
    for csv_path in sorted(
        args.input_root.glob("*/*/factual_utility_summary.csv")
    ):
        job_key = csv_path.parent.parent.name
        variant = csv_path.parent.name
        dataset, scorer = dataset_map.get(job_key, (job_key, ""))
        frame = pd.read_csv(csv_path)

        for scope in ("full", "blind_strict"):
            subset = frame[frame["scope"] == scope]
            base_rows = subset[subset["method"] == "base"]
            optq_rows = subset[subset["method"] == "optq"]
            if base_rows.empty or optq_rows.empty:
                continue

            base = base_rows.iloc[0]
            optq = optq_rows.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "scorer": scorer,
                    "variant": variant,
                    "scope": scope,
                    "num_queries": int(optq["num_queries"]),
                    "num_feasible": int(optq["num_feasible"]),
                    "feasible_rate": optq["feasible_rate"],
                    "target_in_topm_rate": optq["target_in_topm_rate"],
                    "delta_hit10": optq["hit10"] - base["hit10"],
                    "delta_mrr10": optq["mrr10"] - base["mrr10"],
                    "base_viol": base["viol_at_k"],
                    "optq_viol": optq["viol_at_k"],
                    "delta_viol": optq["viol_at_k"] - base["viol_at_k"],
                    "base_adm": base["adm_at_k"],
                    "optq_adm": optq["adm_at_k"],
                    "delta_adm": optq["adm_at_k"] - base["adm_at_k"],
                    "base_unknown": base["unknown_at_k"],
                    "optq_unknown": optq["unknown_at_k"],
                    "delta_unknown": (
                        optq["unknown_at_k"] - base["unknown_at_k"]
                    ),
                    "optq_pres": optq["pres_at_k"],
                    "optq_shift": optq["shift_at_k"],
                }
            )

    if not rows:
        raise SystemExit(
            f"No factual_utility_summary.csv files found under "
            f"{args.input_root}"
        )

    output = pd.DataFrame(rows)
    output["_variant_order"] = (
        output["variant"].map(variant_order).fillna(999)
    )
    output = output.sort_values(
        ["dataset", "scorer", "scope", "_variant_order"]
    ).drop(columns=["_variant_order"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
