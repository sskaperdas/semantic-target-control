# Quick start

## Thirty-second CPU demonstration

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -e .
python examples/toy_control.py
```

Expected certificate:

```text
feasible: True
lambda*: 0.050000
quota certificate: 3 admissible candidates in Top-5 (requested q=3)
```

The demo uses only a fixed CSV candidate window. It does not download a
dataset, train a scorer, invoke a reasoner, or contact an external service.

## Command-line interface

```bash
stc-control examples/toy_candidates.csv --top-k 5 --quota 3
stc-control examples/toy_candidates.csv --top-k 5 --quota 3 --json
```

## Repository checks

```bash
python -m unittest discover -s tests -v
python tools/check_local_imports.py
python tools/source_consistency_audit.py
python tools/release_readiness.py
python tools/repository_audit.py
python tools/verify_manifest.py
```
