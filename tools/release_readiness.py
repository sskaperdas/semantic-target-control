#!/usr/bin/env python3
"""Report administrative release blockers without modifying source content."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TITLE = (
    "STC: Inference-Time Semantic Target Control for Knowledge Graph Completion Rankings"
)
EXPECTED_REPOSITORY = "https://github.com/sskaperdas/semantic-target-control"
EXPECTED_TAG = "v1.0.0-iswc2026"
blockers: list[str] = []
warnings: list[str] = []

license_path = ROOT / "LICENSE"
if not license_path.exists():
    blockers.append("Select and add a root software LICENSE.")
elif license_path.stat().st_size < 900:
    warnings.append("The root LICENSE is unusually short; confirm that it is complete.")

texts: dict[str, str] = {}
for relative in ("README.md", "CITATION.cff", "pyproject.toml", "CHANGELOG.md"):
    path = ROOT / relative
    if not path.exists():
        blockers.append(f"Missing release metadata file: {relative}")
        continue
    texts[relative] = path.read_text(encoding="utf-8")

for relative in ("README.md", "CITATION.cff"):
    text = texts.get(relative, "")
    if "YOUR_" + "GITHUB_USERNAME" in text:
        blockers.append(f"Replace the GitHub username placeholder in {relative}.")
    if EXPECTED_TITLE not in text:
        blockers.append(f"Final paper title is missing from {relative}.")
    if EXPECTED_REPOSITORY not in text:
        blockers.append(f"Canonical repository URL is missing from {relative}.")

if EXPECTED_REPOSITORY not in texts.get("pyproject.toml", ""):
    blockers.append("Canonical repository URL is missing from pyproject.toml.")
if EXPECTED_TAG not in texts.get("README.md", ""):
    blockers.append("Release tag is missing from README.md.")
if EXPECTED_TAG not in texts.get("CHANGELOG.md", ""):
    blockers.append("Release tag is missing from CHANGELOG.md.")
obsolete_owner = "estratios" + "skap"
if any(obsolete_owner in text for text in texts.values()):
    blockers.append("Obsolete GitHub owner remains in release metadata.")

for relative in (
    "paper/STC_ISWC_2026.pdf",
    "paper/STC_ISWC_2026_supplementary.pdf",
):
    path = ROOT / relative
    if not path.exists():
        blockers.append(f"Missing final PDF: {relative}")
    elif path.stat().st_size < 10_000:
        blockers.append(f"PDF appears unexpectedly small: {relative}")

lines = [
    "# Release readiness", "", f"Administrative blockers: {len(blockers)}",
    f"Warnings: {len(warnings)}", "",
]
if blockers:
    lines.append("## Blockers")
    lines.extend(f"- {item}" for item in blockers)
    lines.append("")
if warnings:
    lines.append("## Warnings")
    lines.extend(f"- {item}" for item in warnings)
    lines.append("")
if not blockers:
    lines.append("The repository has no detected administrative release blocker.")
    lines.append("")

report_path = ROOT / "results/provenance/release/RELEASE_READINESS_REPORT.md"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
)
print(f"Release blockers: {len(blockers)}")
for blocker in blockers:
    print("-", blocker)
for warning in warnings:
    print("WARNING:", warning)
raise SystemExit(2 if blockers else 0)
