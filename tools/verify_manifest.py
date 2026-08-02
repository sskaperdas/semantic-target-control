#!/usr/bin/env python3
"""Strictly verify hashes, sizes, coverage, and uniqueness of the manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifact_manifest.json"
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".idea", ".vscode", "build", "dist", "htmlcov",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_NAMES = {".coverage"}


def excluded(relative: Path) -> bool:
    return (
        any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in relative.parts
        )
        or relative.name in EXCLUDED_NAMES
        or relative.suffix.lower() in EXCLUDED_SUFFIXES
    )


if not MANIFEST.exists():
    raise SystemExit("Missing artifact_manifest.json")
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
entries = manifest.get("files", [])
failures: list[str] = []
if not isinstance(entries, list):
    raise SystemExit("Manifest field 'files' must be a list")

paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
for path, count in Counter(paths).items():
    if path is None:
        failures.append("manifest entry without a path")
    elif count > 1:
        failures.append(f"duplicate manifest entry: {path}")

expected_paths = {path for path in paths if isinstance(path, str)}
actual_paths = {
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_file()
    and path != MANIFEST
    and not excluded(path.relative_to(ROOT))
}
for path in sorted(expected_paths - actual_paths):
    failures.append(f"manifest references missing/excluded file: {path}")
for path in sorted(actual_paths - expected_paths):
    failures.append(f"unlisted repository file: {path}")

for entry in entries:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        continue
    path = ROOT / entry["path"]
    if not path.is_file():
        continue
    if path.stat().st_size != entry.get("bytes"):
        failures.append(f"size mismatch: {entry['path']}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.get("sha256"):
        failures.append(f"hash mismatch: {entry['path']}")

if failures:
    print("Manifest verification FAILED")
    for failure in failures:
        print("-", failure)
    raise SystemExit(2)
print(f"Manifest verification PASS ({len(entries)} files; strict coverage)")
