# Hardware guidance

## Lightweight demo and tests

- CPU only
- under one minute
- negligible memory

## Preprocessing

Requirements depend on source-dataset size and parsing format. DBpedia
decompression and RDF parsing are primarily CPU/RAM/I/O workloads.

## PyKEEN portfolios

The single-device and multi-GPU portfolio scripts can be expensive. Start with
`--portfolio small`, validate preprocessing metadata and monitor VRAM before
running the complete portfolio.

## DDP training

`train_base_kgc_ddp.py` is intended for `torchrun` and records world size,
batching, Top-M dump settings and CUDA allocator configuration.

## Reproducing paper-scale windows

EurostatKG uses a larger canonical Top-M than DBpedia and DrugBank. Candidate
materialization may dominate memory and runtime even when the controller itself
is lightweight.
