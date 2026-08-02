from pathlib import Path
import pandas as pd

p = Path("artifacts/stc_final_tables_corrected/exp16_corrected_energy_ablation_compact.csv")
df = pd.read_csv(p)

sub = df[df["scope"] == "full"].copy()

cols = [
    "dataset", "scorer", "variant",
    "feasible_rate", "delta_hit10", "delta_mrr10",
    "delta_viol", "delta_adm", "delta_unknown",
    "optq_pres", "optq_shift"
]

out = sub[cols]

out_path = Path("artifacts/stc_final_tables_corrected/exp16_paper_fullscope_energy_ablation.csv")
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print("\nWrote", out_path)
