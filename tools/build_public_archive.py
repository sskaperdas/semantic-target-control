#!/usr/bin/env python3
"""Build a deterministic public ZIP including hidden GitHub metadata."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".idea", ".vscode", "build", "dist", "htmlcov",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_NAMES = {".coverage"}
ARCHIVE_TIMESTAMP = (2026, 8, 2, 0, 0, 0)


def excluded(relative: Path) -> bool:
    return (
        any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in relative.parts
        )
        or relative.name in EXCLUDED_NAMES
        or relative.suffix.lower() in EXCLUDED_SUFFIXES
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    output = args.output.resolve()
    checksum = output.with_suffix(output.suffix + ".sha256")
    output.parent.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if excluded(relative) or path.resolve() in {output, checksum}:
            continue
        files.append(path)

    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative)
            info.date_time = ARCHIVE_TIMESTAMP
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum.write_text(
        f"{digest}  {output.name}\n", encoding="utf-8", newline="\n"
    )
    print(f"Created {output} ({len(files)} files)")
    print(f"SHA256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
