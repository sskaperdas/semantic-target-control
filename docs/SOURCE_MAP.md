# Public source map

The final public structure replaces the historical `HPC Scripts/` directory
with function-oriented locations.

| Historical source | Public destination |
|---|---|
| preprocessing scripts | `scripts/preprocessing/` |
| frozen-scorer trainers | `scripts/training/` |
| final table builders | `scripts/analysis/` |
| hard-coded PowerShell suites | replaced by `commands/` parameterized wrappers |
| exact environment freeze | `environment/requirements-hpc-pykeen-cu121-lock.txt` |
| DGL-KE environment | `environment/requirements-dglke-cu121.txt` |
| STC configuration dump | `configs/provenance/key_stc_configs_dump.txt` |
| DDP example config | `configs/training/dbpedia_transe_ddp_reference.json` |

The historical internal `pyproject.toml` was not copied because it identified
the package as proprietary and anonymous. The root public `pyproject.toml`
contains the correct project and author metadata.
