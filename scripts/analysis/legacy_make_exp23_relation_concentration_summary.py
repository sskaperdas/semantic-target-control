from pathlib import Path
import pandas as pd

p = Path("artifacts/stc_final_tables_corrected/exp23_relation_level_q5_summary.csv")
df = pd.read_csv(p)

# Use one informative threshold per dataset.
keep = (
    ((df["dataset"] == "EurostatKG") & (df["min_support"] == 500)) |
    ((df["dataset"] == "DBpedia") & (df["min_support"] == 30)) |
    ((df["dataset"] == "DrugBank") & (df["min_support"] == 1))
)

out = df[keep].copy()

cols = [
    "dataset", "scorer", "min_support",
    "num_relations_all_feasible", "num_relations_kept",
    "kept_query_share", "max_relation_support_share", "top3_relation_support_share",
    "macro_optq_adm", "weighted_optq_adm",
    "macro_optq_pres", "weighted_optq_pres",
    "share_relations_adm_ge_0p5", "share_relations_viol_le_0p5"
]

out = out[cols]

for c in out.columns:
    if c not in ["dataset", "scorer", "min_support", "num_relations_all_feasible", "num_relations_kept"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(3)

out_path = Path("artifacts/stc_final_tables_corrected/exp23_paper_relation_concentration_summary.csv")
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print("\nWrote", out_path)
