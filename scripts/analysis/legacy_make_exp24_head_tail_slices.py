from pathlib import Path
import pandas as pd
import numpy as np

inputs = [
    ("EurostatKG", "PairRE-main", Path("outputs/stc_upgrade_corrected/exp12_quota_baselines/eurostat_m20000_full_q13510/query_level.csv")),
    ("DBpedia", "ComplEx-main", Path("outputs/stc_upgrade_corrected/exp12_quota_baselines/dbpedia_m5000_subset_q13510/query_level.csv")),
    ("DrugBank", "RotatE-main", Path("outputs/stc_upgrade_corrected/exp12_quota_baselines/drugbank_m5000_full_q13510/query_level.csv")),
]

rows = []

for dataset, scorer, path in inputs:
    if not path.exists():
        print("MISSING:", path)
        continue

    usecols = [
        "mode", "is_blind", "blind_strict",
        "q5_feasible",
        "q5_optq_viol_at_k", "q5_optq_adm_at_k", "q5_optq_unknown_at_k",
        "q5_optq_pres_at_k", "q5_optq_shift_at_k", "q5_optq_adm_count_topk",
        "q5_afq_viol_at_k", "q5_afq_adm_at_k", "q5_afq_pres_at_k", "q5_afq_shift_at_k",
        "q5_hardval_viol_at_k", "q5_hardval_adm_at_k", "q5_hardval_pres_at_k", "q5_hardval_shift_at_k",
        "q5_lambda_star",
    ]

    df = pd.read_csv(path, usecols=usecols)

    scopes = {
        "full_feasible": df[df["q5_feasible"] == 1].copy(),
        "blind_strict_feasible": df[(df["q5_feasible"] == 1) & (df["blind_strict"] == 1)].copy(),
    }

    for scope, sub0 in scopes.items():
        for mode in ["head", "tail"]:
            sub = sub0[sub0["mode"] == mode].copy()
            if sub.empty:
                continue

            rows.append({
                "dataset": dataset,
                "scorer": scorer,
                "scope": scope,
                "mode": mode,
                "num_queries": int(len(sub)),

                "optq_viol": sub["q5_optq_viol_at_k"].mean(),
                "optq_adm": sub["q5_optq_adm_at_k"].mean(),
                "optq_unknown": sub["q5_optq_unknown_at_k"].mean(),
                "optq_pres": sub["q5_optq_pres_at_k"].mean(),
                "optq_shift": sub["q5_optq_shift_at_k"].mean(),
                "optq_adm_count": sub["q5_optq_adm_count_topk"].mean(),

                "afq_adm": sub["q5_afq_adm_at_k"].mean(),
                "afq_pres": sub["q5_afq_pres_at_k"].mean(),

                "hardval_adm": sub["q5_hardval_adm_at_k"].mean(),
                "hardval_pres": sub["q5_hardval_pres_at_k"].mean(),

                "mean_lambda": sub["q5_lambda_star"].mean(),
                "median_lambda": sub["q5_lambda_star"].median(),
            })

out = pd.DataFrame(rows)

for c in out.columns:
    if c not in ["dataset", "scorer", "scope", "mode"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

out_dir = Path("artifacts/stc_final_tables_corrected")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "exp24_head_tail_slice_q5_summary.csv"
out.to_csv(out_path, index=False)

paper = out.copy()
for c in paper.columns:
    if c not in ["dataset", "scorer", "scope", "mode", "num_queries"]:
        paper[c] = pd.to_numeric(paper[c], errors="coerce").round(3)

paper_path = out_dir / "exp24_paper_head_tail_slice_q5_summary.csv"
paper.to_csv(paper_path, index=False)

print(paper.to_string(index=False))
print("\nWrote")
print(out_path)
print(paper_path)
