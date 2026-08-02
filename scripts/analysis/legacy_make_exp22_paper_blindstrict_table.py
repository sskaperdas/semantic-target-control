from pathlib import Path
import pandas as pd

p = Path("artifacts/stc_final_tables_corrected/exp22_all_datasets_learned_reranker_selected.csv")
df = pd.read_csv(p)

sub = df[
    (df["scope"] == "blind_strict") &
    (df["selector"].isin([
        "Base",
        "OptQ",
        "Learned nearest OptQ-Pres",
        "Learned quota-safe",
    ]))
].copy()

cols = [
    "dataset", "selector", "alpha", "quota_success",
    "viol_at_k", "adm_at_k", "pres_at_k", "shift_at_k",
    "hit10", "mrr10", "num_queries"
]

sub = sub[cols]

out = Path("artifacts/stc_final_tables_corrected/exp22_paper_blindstrict_learned_baseline.csv")
sub.to_csv(out, index=False)

print(sub.to_string(index=False))
print("\nWrote", out)
