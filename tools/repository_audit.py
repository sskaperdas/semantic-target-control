#!/usr/bin/env python3
"""Final public-repository privacy, structure, text-integrity, and release audit."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPOSITORY = "https://github.com/sskaperdas/semantic-target-control"
OBSOLETE_REPOSITORY_OWNER = "estratios" + "skap"

REQUIRED = [
    "README.md", "LICENSE", "CITATION.cff", "pyproject.toml",
    "artifact_manifest.json", "src/stc/core.py", "src/stc/cli.py",
    "examples/toy_control.py", "examples/toy_candidates.csv",
    "tests/test_core.py",
    "scripts/preprocessing/preprocess_eurostatkg.py",
    "scripts/preprocessing/preprocess_dbpedia.py",
    "scripts/preprocessing/preprocess_drugbank_xml.py",
    "scripts/training/train_base_kgc.py",
    "scripts/training/train_base_kgc_ddp.py",
    "scripts/training/train_schema_aware_portfolio_multigpu.py",
    "scripts/analysis/make_exp16_energy_ablation.py",
    "scripts/analysis/make_exp23_relation_diagnostics.py",
    "scripts/analysis/make_exp24_head_tail_slices.py",
    "commands/preprocess_dataset.ps1", "commands/train_frozen_scorer.ps1",
    "commands/run_stc_evaluation.ps1", "configs/paper/canonical_runs.json",
    "environment/README.md",
    "environment/requirements-hpc-pykeen-cu121-lock.txt",
    "results/frozen/window_sensitivity_canonical.csv",
    "results/provenance/release/SOURCE_CONSISTENCY_REPORT.md",
    "paper/STC_ISWC_2026.pdf",
    "paper/STC_ISWC_2026_supplementary.pdf",
    "assets/stc-social-preview.png", "assets/stc-pipeline.png",
    "assets/stc-optq-intuition.png", "assets/stc-feasibility-certificate.png",
    "assets/stc-semantic-status.png", "assets/stc-results-overview.png",
    "assets/stc-artifact-overview.png", ".github/workflows/ci.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
]

ROOT_CLUTTER = [
    "COPY_REPORT.csv", "EXCLUSION_REPORT.csv", "SECURITY_SCAN_REPORT.md",
    "PUBLIC_SOURCE_INTEGRATION_REPORT.md", "SOURCE_CONSISTENCY_REPORT.md",
    "RELEASE_READINESS_REPORT.md",
]
TEXT_EXTENSIONS = {
    ".py", ".ps1", ".md", ".txt", ".json", ".jsonl", ".yml", ".yaml",
    ".csv", ".tsv", ".toml", ".cff", ".ini", ".cfg", ".sh", ".bat",
}
TEXT_NAMES = {"LICENSE", ".gitignore", ".gitattributes"}
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "build", "dist",
    "htmlcov",
}
PRIVATE_PATH_PATTERNS = [
    re.compile(r"(?i)[A-Z]:\\Users\\[^\\\s\"';,]+"),
    re.compile(r"(?i)/scratch/[a-z]/[a-z0-9_-]+/"),
]
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|secret|password)"
    r"\s*[:=]\s*[\"'][^\"']{8,}[\"']"
)
MOJIBAKE_TOKENS = tuple(
    chr(codepoint)
    for codepoint in (0x00EF, 0x00E2, 0x00C3, 0xFFFD)
)
PLACEHOLDER = "YOUR_" + "GITHUB_USERNAME"


def excluded(relative: Path) -> bool:
    return any(
        part in EXCLUDED_PARTS or part.endswith(".egg-info")
        for part in relative.parts
    )


errors: list[str] = []

for relative in REQUIRED:
    if not (ROOT / relative).exists():
        errors.append(f"missing required file: {relative}")

for relative in ROOT_CLUTTER:
    if (ROOT / relative).exists():
        errors.append(f"internal build report remains in repository root: {relative}")

if (ROOT / ".release").exists():
    errors.append("internal .release template directory remains after finalization")

for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(ROOT)
    if excluded(relative):
        continue
    if path.stat().st_size > 50 * 1024 * 1024:
        errors.append(f"file exceeds 50 MiB policy: {relative}")
    if path.suffix.lower() not in TEXT_EXTENSIONS and path.name not in TEXT_NAMES:
        continue

    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append(f"UTF-8 BOM found in public text artifact: {relative}")
    if b"\x00" in raw:
        errors.append(f"NUL bytes in text artifact: {relative}")
        continue
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"invalid UTF-8 in {relative}: {exc}")
        continue

    controls = sorted(
        {ord(char) for char in text if ord(char) < 32 and char not in "\n\r\t"}
    )
    if controls:
        codes = ", ".join(f"0x{code:02x}" for code in controls)
        errors.append(f"disallowed control character(s) {codes} in: {relative}")
    if any(token in text for token in MOJIBAKE_TOKENS):
        errors.append(f"possible mojibake in: {relative}")
    if PLACEHOLDER in text:
        errors.append(f"GitHub username placeholder remains in: {relative}")
    if OBSOLETE_REPOSITORY_OWNER in text:
        errors.append(f"obsolete GitHub owner remains in: {relative}")

    for pattern in PRIVATE_PATH_PATTERNS:
        if pattern.search(text):
            errors.append(f"private/cluster path found: {relative}")
            break
    if SECRET_PATTERN.search(text):
        errors.append(f"possible secret found: {relative}")

for pattern in (
    "**/*drugbank*.xml", "**/*.ckpt", "**/*.safetensors", "**/*.pth",
    "**/*topm*.jsonl",
):
    for path in ROOT.glob(pattern):
        if path.is_file():
            errors.append(
                f"restricted or large artifact present: {path.relative_to(ROOT)}"
            )

readme_path = ROOT / "README.md"
if readme_path.exists():
    readme = readme_path.read_text(encoding="utf-8")
    for heading in (
        "## Overview", "## Run STC in thirty seconds", "## Canonical results",
        "## Reproduction paths", "## Citation", "## License",
    ):
        if heading not in readme:
            errors.append(f"README missing section: {heading}")
    if EXPECTED_REPOSITORY not in readme:
        errors.append("README does not contain the canonical GitHub repository URL")

for relative in ("CITATION.cff", "pyproject.toml"):
    path = ROOT / relative
    if path.exists() and EXPECTED_REPOSITORY not in path.read_text(encoding="utf-8"):
        errors.append(f"canonical GitHub repository URL missing from: {relative}")

license_path = ROOT / "LICENSE"
if license_path.exists() and license_path.stat().st_size < 900:
    errors.append("root LICENSE appears incomplete")

if errors:
    print("Repository audit FAILED")
    for error in errors:
        print("-", error)
    raise SystemExit(2)

print("Repository audit PASS")
