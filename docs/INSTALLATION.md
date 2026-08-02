# Installation

## Lightweight package

The reference OptQ implementation has no runtime dependency:

```bash
python -m pip install -e .
```

## Preprocessing and result analysis

```bash
python -m pip install -r environment/requirements-core.txt
```

## Frozen-scorer training

The paper used GPU-specific PyTorch/PyKEEN environments. See
[`environment/README.md`](../environment/README.md) before installing a lock.

```bash
python -m pip install   -r environment/requirements-hpc-pykeen-cu121-lock.txt
```

The DGL-KE lock is a separate environment and must not be mixed with the
PyKEEN lock.

## Supported Python

The lightweight package targets Python 3.10 or later. Historical training
environments are documented as provenance snapshots and may require their
original Python/CUDA combination.
