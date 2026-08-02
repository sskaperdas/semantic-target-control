# Full reproduction workflow

End-to-end reproduction is intentionally separated into bounded stages.

## 1. Obtain source datasets

Obtain EurostatKG and DBpedia source resources under their original terms.
Obtain DrugBank XML through an authorized DrugBank account.

## 2. Preprocess

Use `commands/preprocess_dataset.ps1` or call the Python entry points directly.
Each preprocessing script writes split files, entity/relation maps, semantic
sidecars and a preflight summary.

## 3. Train or restore a frozen scorer

```powershell
.\commands\train_frozen_scorer.ps1 `
  -Trainer pykeen `
  -ProcessedDir data\processed\eurostatkg `
  -OutputDir runs\eurostat_schema_portfolio_v1 `
  -DatasetName EurostatKG `
  -Portfolio medium `
  -Devices cuda
```

For DDP, launch `scripts/training/train_base_kgc_ddp.py` through `torchrun`.

## 4. Materialize and evaluate one fixed Top-M window

Example EXP-12 run:

```powershell
.\commands\run_stc_evaluation.ps1 `
  -Experiment 12 `
  -ProcessedDir data\processed\eurostatkg `
  -RunDir runs\eurostat_schema_portfolio_v1\best_model `
  -OutputDir outputs\exp12_eurostat `
  -DatasetName EurostatKG `
  -TopM 20000 `
  -QueryBatchSize 32
```

The controller does not retrieve additional candidates during a run.

## 5. Rebuild publication summaries

Use the portable `scripts/analysis/` commands. The immutable publication-level
summaries are retained in `results/frozen/` for direct comparison.

## 6. Verify provenance

```bash
python tools/source_consistency_audit.py
python tools/verify_manifest.py
```

The final camera-ready migration passed 54/54 content and provenance checks.
