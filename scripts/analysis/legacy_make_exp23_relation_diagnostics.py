from pathlib import Path
import pandas as pd
import numpy as np

inputs = [
    (
        "EurostatKG",
        "PairRE-main",
        Path("outputs/stc_upgrade_corrected/exp12_quota_baselines/eurostat_m20000_full_q13510/relation_level.csv"),
    ),
    (
        "DBpedia",
        "ComplEx-main",
        Path("outputs/stc_upgrade_corrected/exp12_quota_baselines/dbpedia_m5000_subset_q13510/relation_level.csv"),
    ),
    (
        "DrugBank",
        "RotatE-main",
        Path("outputs/stc_upgrade_corrected/exp12_quota_baselines/drugbank_m5000_full_q13510/relation_level.csv"),
    ),
]

out_dir = Path("artifacts/stc_final_tables_corrected")
out_dir.mkdir(parents=True, exist_ok=True)

def short_rel(x):
    x = str(x)
    if "#" in x:
        return x.rsplit("#", 1)[-1]
    if "/" in x:
        return x.rstrip("/").rsplit("/", 1)[-1]
    return x

def wmean(df, col, weight_col="num_feasible"):
    d = df[[col, weight_col]].dropna()
    if d.empty:
        return np.nan
    w = d[weight_col].astype(float)
    if float(w.sum()) == 0:
        return np.nan
    return float((d[col].astype(float) * w).sum() / w.sum())

all_rel_rows = []
summary_rows = []
top_support_rows = []
top_improve_rows = []
bottom_pres_rows = []

for dataset, scorer, path in inputs:
    if not path.exists():
        print("MISSING:", path)
        continue

    df = pd.read_csv(path)

    # Corrected q=5 only.
    df = df[df["quota"] == 5].copy()

    # Keep relations with feasible q=5 metrics.
    df = df[df["num_feasible"].fillna(0) > 0].copy()

    if df.empty:
        print("EMPTY after filter:", dataset, path)
        continue

    df["dataset"] = dataset
    df["scorer"] = scorer
    df["relation_short"] = df["relation"].map(short_rel)
    df["support_share"] = df["num_feasible"] / df["num_feasible"].sum()

    # Differences vs hard validation endpoint and AFQ-like quota baseline.
    # Note: AFQ can collapse to OptQ as a set in binary-energy cases; use carefully.
    df["optq_minus_hardval_pres"] = df["optq_pres_at_k"] - df["hardval_pres_at_k"]
    df["optq_minus_hardval_shift"] = df["optq_shift_at_k"] - df["hardval_shift_at_k"]
    df["optq_minus_afq_pres"] = df["optq_pres_at_k"] - df["afq_pres_at_k"]
    df["optq_minus_afq_shift"] = df["optq_shift_at_k"] - df["afq_shift_at_k"]

    all_rel_rows.append(df)

    for min_support in [1, 10, 30, 100, 500]:
        sub = df[df["num_feasible"] >= min_support].copy()
        if sub.empty:
            continue

        total_feasible = int(df["num_feasible"].sum())

        summary_rows.append({
            "dataset": dataset,
            "scorer": scorer,
            "quota": 5,
            "min_support": min_support,
            "num_relations_all_feasible": int(len(df)),
            "num_relations_kept": int(len(sub)),
            "total_feasible_queries": total_feasible,
            "kept_feasible_queries": int(sub["num_feasible"].sum()),
            "kept_query_share": float(sub["num_feasible"].sum() / df["num_feasible"].sum()),
            "max_relation_support_share": float(df["support_share"].max()),
            "top3_relation_support_share": float(df.sort_values("support_share", ascending=False)["support_share"].head(3).sum()),

            "macro_optq_viol": float(sub["optq_viol_at_k"].mean()),
            "macro_optq_adm": float(sub["optq_adm_at_k"].mean()),
            "macro_optq_pres": float(sub["optq_pres_at_k"].mean()),
            "macro_optq_shift": float(sub["optq_shift_at_k"].mean()),

            "weighted_optq_viol": wmean(sub, "optq_viol_at_k"),
            "weighted_optq_adm": wmean(sub, "optq_adm_at_k"),
            "weighted_optq_pres": wmean(sub, "optq_pres_at_k"),
            "weighted_optq_shift": wmean(sub, "optq_shift_at_k"),

            "relations_with_adm_ge_0p5": int((sub["optq_adm_at_k"] >= 0.5).sum()),
            "relations_with_viol_le_0p5": int((sub["optq_viol_at_k"] <= 0.5).sum()),
            "share_relations_adm_ge_0p5": float((sub["optq_adm_at_k"] >= 0.5).mean()),
            "share_relations_viol_le_0p5": float((sub["optq_viol_at_k"] <= 0.5).mean()),
        })

    top_support_rows.append(
        df.sort_values("num_feasible", ascending=False)
          .head(10)
          [["dataset", "scorer", "relation_short", "relation", "queries", "num_feasible", "support_share",
            "feasible_rate", "optq_viol_at_k", "optq_adm_at_k", "optq_pres_at_k", "optq_shift_at_k",
            "mean_lambda"]]
    )

    # Only reasonably supported relations for best/worst diagnostics.
    supported = df[df["num_feasible"] >= 30].copy()
    if supported.empty:
        supported = df.copy()

    top_improve_rows.append(
        supported.sort_values("optq_adm_at_k", ascending=False)
          .head(10)
          [["dataset", "scorer", "relation_short", "relation", "queries", "num_feasible", "support_share",
            "optq_viol_at_k", "optq_adm_at_k", "optq_pres_at_k", "optq_shift_at_k", "mean_lambda"]]
          .assign(rank_type="highest_optq_adm")
    )

    bottom_pres_rows.append(
        supported.sort_values("optq_pres_at_k", ascending=True)
          .head(10)
          [["dataset", "scorer", "relation_short", "relation", "queries", "num_feasible", "support_share",
            "optq_viol_at_k", "optq_adm_at_k", "optq_pres_at_k", "optq_shift_at_k", "mean_lambda"]]
          .assign(rank_type="lowest_optq_pres")
    )

all_rel = pd.concat(all_rel_rows, ignore_index=True) if all_rel_rows else pd.DataFrame()
summary = pd.DataFrame(summary_rows)
top_support = pd.concat(top_support_rows, ignore_index=True) if top_support_rows else pd.DataFrame()
top_bottom = pd.concat(top_improve_rows + bottom_pres_rows, ignore_index=True) if top_improve_rows else pd.DataFrame()

all_rel_path = out_dir / "exp23_relation_level_q5_all_relations.csv"
summary_path = out_dir / "exp23_relation_level_q5_summary.csv"
top_support_path = out_dir / "exp23_relation_level_q5_top_support.csv"
top_bottom_path = out_dir / "exp23_relation_level_q5_top_bottom.csv"

all_rel.to_csv(all_rel_path, index=False)
summary.to_csv(summary_path, index=False)
top_support.to_csv(top_support_path, index=False)
top_bottom.to_csv(top_bottom_path, index=False)

print("\n=== EXP23 relation-level q=5 summary ===")
print(summary.to_string(index=False))

print("\n=== Top support relations ===")
print(top_support.to_string(index=False))

print("\n=== Top/bottom diagnostics ===")
print(top_bottom.to_string(index=False))

print("\nWrote")
print(all_rel_path)
print(summary_path)
print(top_support_path)
print(top_bottom_path)
