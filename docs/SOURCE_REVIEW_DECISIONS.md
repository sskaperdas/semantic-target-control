# Source-review integration decisions

The uploaded review bundle contained 27/27 selected files, with no exclusions or missing files.

| Historical file | Decision | Public treatment |
|---|---|---|
| `HPC Scripts/preprocess_dbpedia.py` | **include** | scripts/preprocessing/preprocess_dbpedia.py |
| `HPC Scripts/preprocess_drugbank_xml.py` | **include** | scripts/preprocessing/preprocess_drugbank_xml.py |
| `HPC Scripts/preprocess_eurostatkg.py` | **include** | scripts/preprocessing/preprocess_eurostatkg.py |
| `HPC Scripts/build_dbpedia_eval_subset.py` | **include** | scripts/preprocessing/build_dbpedia_eval_subset.py |
| `HPC Scripts/train_base_kgc.py` | **include** | scripts/training/train_base_kgc.py |
| `HPC Scripts/train_base_kgc_ddp.py` | **include** | scripts/training/train_base_kgc_ddp.py |
| `HPC Scripts/train_schema_aware_portfolio_multigpu.py` | **include** | scripts/training/train_schema_aware_portfolio_multigpu.py |
| `HPC Scripts/pyproject.toml` | **replace** | root pyproject.toml with correct STC metadata |
| `HPC Scripts/requirements.txt` | **normalize** | environment/requirements-hpc-pykeen-cu121-lock.txt |
| `HPC Scripts/requirements-dglke.txt` | **include** | environment/requirements-dglke-cu121.txt |
| `HPC Scripts/run_all_drugbank.ps1` | **replace** | commands/run_stc_evaluation.ps1 |
| `HPC Scripts/run_all_eurostat.ps1` | **replace** | commands/run_stc_evaluation.ps1 |
| `scripts/run_all_dbpedia.ps1` | **replace** | commands/run_stc_evaluation.ps1 |
| `scripts/run_all_drugbank.ps1` | **replace** | commands/run_stc_evaluation.ps1 |
| `scripts/run_all_eurostat.ps1` | **replace** | commands/run_stc_evaluation.ps1 |
| `scripts/run_dbpedia_subset_suite.ps1` | **replace** | commands/run_stc_evaluation.ps1 |
| `artifacts/stc_audit_clean/corrected_rerun_config_inventory.txt` | **omit** | internal path/timestamp index; no executable configuration |
| `artifacts/stc_audit_clean/key_stc_configs_dump.txt` | **normalize** | configs/provenance/key_stc_configs_dump.txt |
| `artifacts/stc_audit_clean/make_exp16_energy_ablation_compact.py` | **include+replace** | legacy exact source plus portable scripts/analysis version |
| `artifacts/stc_audit_clean/make_exp16_paper_fullscope.py` | **include+replace** | legacy exact source plus portable scripts/analysis version |
| `artifacts/stc_audit_clean/make_exp22_paper_blindstrict_table.py` | **include+replace** | legacy exact source plus portable scripts/analysis version |
| `artifacts/stc_audit_clean/make_exp23_relation_diagnostics.py` | **include+replace** | legacy exact source plus portable scripts/analysis version |
| `artifacts/stc_audit_clean/make_exp23_paper_relation_concentration_summary.py` | **include+replace** | legacy exact source plus portable scripts/analysis version |
| `artifacts/stc_audit_clean/make_exp24_head_tail_slices.py` | **include+replace** | legacy exact source plus portable scripts/analysis version |
| `artifacts/stc_audit_clean/run_exp16_energy_ablation_corrected.ps1` | **include** | commands/legacy/run_exp16_energy_ablation_corrected.ps1 |
| `artifacts/stc_final_evidence_v1/window_sensitivity_canonical.csv` | **include** | results/frozen/window_sensitivity_canonical.csv |
| `dbpedia_ddp/.../config.json` | **rewrite** | configs/training/dbpedia_transe_ddp_reference.json |

Hard-coded orchestration files were not copied verbatim. Their experimental
entry points remain public, while portable wrappers accept repository and
dataset paths as parameters.
