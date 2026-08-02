#!/usr/bin/env python3
"""Regenerate the public artifact manifest without transient build files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
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


files = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == MANIFEST:
        continue
    relative = path.relative_to(ROOT)
    if excluded(relative):
        continue
    files.append(
        {
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )

payload = {
    "schema_version": "1.0",
    "artifact": "STC ISWC 2026 public research artifact",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "source_project": "local research workspace (path intentionally omitted)",
    "files": files,
}
MANIFEST.write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
)
print(f"Updated artifact_manifest.json ({len(files)} files)")
