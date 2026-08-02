#!/usr/bin/env python3
"""Compile every public Python source and detect encoding damage."""

from __future__ import annotations

import ast
import sys
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__"}
errors: list[str] = []
checked = 0

for path in sorted(ROOT.rglob("*.py")):
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        continue

    raw = path.read_bytes()
    if b"\x00" in raw:
        errors.append(f"NUL bytes in Python source: {relative}")
        continue
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append(f"UTF-8 BOM remains in Python source: {relative}")
        continue

    try:
        with tokenize.open(path) as handle:
            source = handle.read()
        ast.parse(source, filename=str(relative))
        compile(source, str(relative), "exec")
    except Exception as exc:
        errors.append(f"{relative}: {exc}")
        continue
    checked += 1

if errors:
    print("Python source audit FAILED")
    for error in errors:
        print("-", error)
    raise SystemExit(2)

print(f"Python source audit PASS ({checked} files)")
