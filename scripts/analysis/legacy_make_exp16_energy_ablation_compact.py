from pathlib import Path
import pandas as pd

root = Path("outputs/stc_upgrade_corrected/exp16_energy_ablation")

dataset_map = {
    "eurostat_pairre": ("EurostatKG", "PairRE-main"),
    "dbpedia_complex": ("DBpedia", "ComplEx-main"),
}

rows = []

for csv_path in sorted(root.glob("*/*/factual_utility_summary.csv")):
    job_key = csv_path.parent.parent.name
    variant = csv_path.parent.name

    dataset, scorer = dataset_map.get(job_key, (job_key, ""))

    df = pd.read_csv(csv_path)

    for scope in ["full", "blind_strict"]:
        sub = df[df["scope"] == scope].copy()
        if sub.empty:
            continue

        base_rows = sub[sub["method"] == "base"]
        optq_rows = sub[sub["method"] == "optq"]

        if base_rows.empty or optq_rows.empty:
            continue

        base = base_rows.iloc[0]
        optq = optq_rows.iloc[0]

        rows.append({
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
            "delta_unknown": optq["unknown_at_k"] - base["unknown_at_k"],

            "optq_pres": optq["pres_at_k"],
            "optq_shift": optq["shift_at_k"],
        })

out = pd.DataFrame(rows)

# Useful ordering
variant_order = {
    "domain_range_u1": 0,
    "domain_only_u1": 1,
    "range_only_u1": 2,
    "domain_range_u0p5": 3,
    "domain_range_u2": 4,
}
out["variant_order"] = out["variant"].map(variant_order).fillna(999)
out = out.sort_values(["dataset", "scorer", "scope", "variant_order"]).drop(columns=["variant_order"])

out_dir = Path("artifacts/stc_final_tables_corrected")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "exp16_corrected_energy_ablation_compact.csv"
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print("\nWrote", out_path)
