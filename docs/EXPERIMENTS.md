# Experiment map

## Dataset preparation and frozen scoring

| Path | Purpose |
|---|---|
| `scripts/preprocessing/preprocess_eurostatkg.py` | Build EurostatKG triples and semantic sidecars |
| `scripts/preprocessing/preprocess_dbpedia.py` | Build DBpedia triples and semantic sidecars |
| `scripts/preprocessing/preprocess_drugbank_xml.py` | Extract the authorized DrugBank XML |
| `scripts/preprocessing/build_dbpedia_eval_subset.py` | Construct the fixed DBpedia evaluation subset |
| `scripts/training/train_base_kgc.py` | Single-device PyKEEN portfolio |
| `scripts/training/train_schema_aware_portfolio_multigpu.py` | Multi-GPU portfolio scheduling |
| `scripts/training/train_base_kgc_ddp.py` | Explicit torch-distributed scorer training |

## STC evaluation scripts

| Family | Public entry point |
|---|---|
| Semantic blindness | `run_exp01_blindness_diagnostics.py` |
| Exact blind-subset control | `run_exp02_exact_blind_subset_bench.py` |
| Global exact semantics | `run_exp03_global_return_semantics_exact.py` |
| Runtime practicality | `run_exp04_practicality_exact.py` |
| Unknown-aware control | `run_exp05_unknown_aware_global.py` |
| Constraint-family ablations | `run_exp06_ablations_exact.py` |
| Quota control | `run_exp07_quota_blind_subset_bench.py`, `run_exp08_global_quota_return_semantics.py` |
| Filtering comparison | `run_exp09_quota_vs_filtering_partial.py` |
| Candidate-window sensitivity | `run_exp10_collapse_sensitivity_analysis.py` |
| Typed retrieval | `run_exp11_typed_entity_retrieval_generalization.py` |
| Quota-matched baselines | `run_exp12_quota_matched_baselines.py` |
| Fixed-pressure grid | `run_exp13_fixed_lambda_grid.py` |
| Factual utility / scorers | `run_exp14_factual_utility.py` |
| Bootstrap intervals | `run_exp19_bootstrap_ci.py` |
| Runtime decomposition | `run_exp21_runtime_breakdown.py` |
| Learned LightGBM policies | `run_exp22_learned_reranker_baseline.py` |

## Publication post-processing

Portable scripts under `scripts/analysis/` regenerate EXP-16, EXP-22, EXP-23
and EXP-24 publication tables from their query/relation-level CSV outputs.
Files prefixed with `legacy_` preserve the exact historical scripts.

## Portable wrappers

- `commands/preprocess_dataset.ps1`
- `commands/train_frozen_scorer.ps1`
- `commands/run_stc_evaluation.ps1`

The wrappers accept paths as parameters and contain no user-specific absolute
directory.
