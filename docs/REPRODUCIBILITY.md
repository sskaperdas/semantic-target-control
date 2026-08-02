# Reproducibility levels

## Level 0 — Conceptual reference implementation

`src/stc/core.py`, the toy CSV and unit tests reproduce the finite-window
controller without a KG model or GPU.

## Level 1 — Repository integrity

```bash
python tools/repository_audit.py
python tools/source_consistency_audit.py
python tools/verify_manifest.py
```

## Level 2 — Frozen publication evidence

The CSV files under `results/frozen/` expose the publication-level aggregate
evidence used by the paper: quota frontiers, fixed pressure, factual utility,
energy ablation, bootstrap intervals, learned policies, relation diagnostics,
head/tail slices, runtime and window sensitivity.

## Level 3 — Selected experiment reproduction

Use an existing processed dataset and frozen checkpoint with the matching
command wrapper. Compare generated summaries with `results/frozen/`.

## Level 4 — End-to-end reconstruction

Rebuild a dataset, train a frozen scorer, materialize the same candidate-window
protocol and rerun STC. This level requires source datasets, substantial
compute and—where applicable—licensed data.

## Scope of the guarantee

The OptQ guarantee is conditional on the finite window and supplied operational
semantic energy. It does not claim factual truth, complete OWL reasoning or
global retrieval completeness.
