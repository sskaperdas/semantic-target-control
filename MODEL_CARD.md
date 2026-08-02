# Model and controller card

## Intended use

STC is intended for research and deployment analysis of bounded Top-k outputs from frozen KGC scorers.

## Inputs

- frozen candidate scores;
- a materialized Top-M window;
- candidate-level operational semantic statuses/energies;
- visible size `k` and requested quota `q`.

## Outputs

- feasibility status;
- query-specific pressure;
- controlled Top-k list;
- deterministic control certificate.

## Limitations

The guarantee is relative to the supplied finite window and operational semantic evidence. It does not establish factual truth, global ontology consistency, or completeness of the semantic metadata.
