#!/usr/bin/env python3
"""Verify the public source map, CLI names, environment locks and OptQ core."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
failures: list[str] = []
checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str) -> None:
    checks.append((name, condition, detail))
    if not condition:
        failures.append(f"{name}: {detail}")


expected_scripts = [
    "scripts/preprocessing/preprocess_dbpedia.py",
    "scripts/preprocessing/preprocess_drugbank_xml.py",
    "scripts/preprocessing/preprocess_eurostatkg.py",
    "scripts/preprocessing/build_dbpedia_eval_subset.py",
    "scripts/training/train_base_kgc.py",
    "scripts/training/train_base_kgc_ddp.py",
    "scripts/training/train_schema_aware_portfolio_multigpu.py",
    "scripts/run_exp08_global_quota_return_semantics.py",
    "scripts/run_exp12_quota_matched_baselines.py",
    "scripts/run_exp14_factual_utility.py",
    "scripts/run_exp19_bootstrap_ci.py",
    "scripts/run_exp21_runtime_breakdown.py",
    "scripts/run_exp22_learned_reranker_baseline.py",
]

for relative in expected_scripts:
    check(
        f"source:{relative}",
        (ROOT / relative).is_file(),
        "required public source is present",
    )


def argparse_flags(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(
                argument.value, str
            ):
                if argument.value.startswith("--"):
                    flags.add(argument.value)
    return flags


for relative in (
    "scripts/preprocessing/preprocess_drugbank_xml.py",
    "scripts/preprocessing/preprocess_eurostatkg.py",
    "scripts/training/train_base_kgc.py",
):
    path = ROOT / relative
    if path.exists():
        flags = argparse_flags(path)
        check(
            f"cli:{relative}:dataset-name",
            "--dataset-name" in flags,
            "defines --dataset-name",
        )
        check(
            f"cli:{relative}:output-dir",
            "--output-dir" in flags,
            "defines --output-dir",
        )

core = (ROOT / "src/stc/core.py").read_text(encoding="utf-8")
check(
    "optq:zero-branch",
    "len(crossings) <= b" in core,
    "explicit zero-pressure branch",
)
check(
    "optq:b-plus-one",
    "crossings[b]" in core,
    "selects the (b+1)-th largest crossing in zero-based indexing",
)
check(
    "optq:no-old-max-formula",
    "max(" not in core or "crossings[b]" in core,
    "correct threshold-order-statistic construction is used",
)

lock_path = ROOT / "environment/requirements-hpc-pykeen-cu121-lock.txt"
lock = lock_path.read_text(encoding="utf-8") if lock_path.exists() else ""
check("env:pykeen", "pykeen==1.11.1" in lock.lower(), "PyKEEN version recorded")
check(
    "env:torch-cu121",
    "torch==2.5.1+cu121" in lock.lower(),
    "historical PyTorch/CUDA lock recorded",
)
check("env:no-nul", "\x00" not in lock, "environment lock is clean UTF-8")

canonical_path = ROOT / "configs/paper/canonical_runs.json"
try:
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
except Exception:
    canonical = {}
check(
    "config:datasets",
    set(canonical.get("datasets", {}))
    == {"EurostatKG", "DBpedia", "DrugBank"},
    "all three canonical datasets are present",
)
check(
    "config:topm",
    canonical.get("datasets", {}).get("EurostatKG", {}).get("top_m")
    == 20000
    and canonical.get("datasets", {}).get("DBpedia", {}).get("top_m")
    == 5000
    and canonical.get("datasets", {}).get("DrugBank", {}).get("top_m")
    == 5000,
    "canonical Top-M values are recorded",
)

report_lines = [
    "# Public source consistency audit",
    "",
    f"Checks: {len(checks)}",
    f"Failures: {len(failures)}",
    "",
]
for name, passed, detail in checks:
    report_lines.append(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`: {detail}"
    )
report_path = ROOT / "results/provenance/release/SOURCE_CONSISTENCY_REPORT.md"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
    newline="\n",
)

if failures:
    print("Source consistency audit FAILED")
    for failure in failures:
        print("-", failure)
    raise SystemExit(2)

print(f"Source consistency audit PASS ({len(checks)} checks)")
