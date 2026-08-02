# Environment profiles

The repository deliberately separates lightweight control, preprocessing,
PyKEEN training, and DGL-KE environments.

## Lightweight reference controller

```bash
python -m pip install -e .
python examples/toy_control.py
```

No third-party runtime dependency is required by `src/stc`.

## Preprocessing and publication analysis

```bash
python -m pip install -r environment/requirements-core.txt
```

## Historical PyKEEN/CUDA 12.1 environment

`requirements-hpc-pykeen-cu121-lock.txt` is a normalized copy of the actual
historical environment snapshot used by the HPC training workflow. It pins
PyTorch 2.5.1+cu121 and PyKEEN 1.11.1.

Install it only in a fresh CUDA-12.1-compatible environment.

## DGL-KE environment

`requirements-dglke-cu121.txt` uses a different PyTorch/DGL stack. Do not
install it into the PyKEEN environment. Create a separate virtual environment.

## Portability note

CUDA wheels, GPU drivers and cluster launchers are platform-specific. The
repository records the exact historical stacks but does not claim that one lock
file is appropriate for every current accelerator.
