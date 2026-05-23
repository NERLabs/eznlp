#!/usr/bin/env python3
"""Validate figure and table numbering in the red-jujube NER manuscript."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EXPECTED_FIGURES = [1, 2, 3, 4, 5]
EXPECTED_TABLES = [1, 2, 3, 4, 5, 6, 7, 8]


def _next_non_empty(lines: list[str], start: int) -> str:
    for line in lines[start + 1 :]:
        if line.strip():
            return line.strip()
    return ""


def _caption_numbers(lines: list[str], cn_prefix: str, en_prefix: str) -> list[int]:
    numbers: list[int] = []
    pattern = re.compile(rf"^{cn_prefix} ([0-9]+)\s+")
    for idx, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match and _next_non_empty(lines, idx).startswith(f"{en_prefix} {match.group(1)}"):
            numbers.append(int(match.group(1)))
    return numbers


def _has_prior_mention(lines: list[str], caption_index: int, prefix: str, number: int) -> bool:
    needle = f"{prefix} {number}"
    return any(needle in line for line in lines[:caption_index])


def _caption_indices(lines: list[str], cn_prefix: str, en_prefix: str) -> dict[int, int]:
    indices: dict[int, int] = {}
    pattern = re.compile(rf"^{cn_prefix} ([0-9]+)\s+")
    for idx, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match and _next_non_empty(lines, idx).startswith(f"{en_prefix} {match.group(1)}"):
            indices[int(match.group(1))] = idx
    return indices


def _image_refs(text: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)


def _resolve_image(md_path: Path, ref: str) -> tuple[bool, str]:
    local = md_path.parent / ref
    if local.exists() and local.stat().st_size > 0:
        return True, str(local)
    path = Path(ref)
    if path.suffix.lower() == ".svg":
        png_name = path.with_suffix(".png").name
        fallback = md_path.parent / "figures_png" / png_name
        if fallback.exists() and fallback.stat().st_size > 0:
            return True, str(fallback)
    return False, str(local)


def _table_has_markdown_body(lines: list[str], caption_index: int) -> bool:
    seen_en_caption = False
    for line in lines[caption_index + 1 : caption_index + 8]:
        stripped = line.strip()
        if stripped.startswith("Table "):
            seen_en_caption = True
            continue
        if seen_en_caption and stripped.startswith("|"):
            return True
        if seen_en_caption and stripped and not stripped.startswith("|"):
            return False
    return False


def _ok(results: list[str], message: str) -> None:
    results.append(f"- PASS: {message}")


def _fail(results: list[str], message: str) -> None:
    results.append(f"- FAIL: {message}")


def validate(md_path: Path) -> tuple[bool, str]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    results: list[str] = []
    success = True

    fig_numbers = _caption_numbers(lines, "图", "Fig.")
    table_numbers = _caption_numbers(lines, "表", "Table")
    fig_indices = _caption_indices(lines, "图", "Fig.")
    table_indices = _caption_indices(lines, "表", "Table")

    if fig_numbers == EXPECTED_FIGURES:
        _ok(results, "figure captions are continuous: 图 1-5")
    else:
        _fail(results, f"figure captions are {fig_numbers}, expected {EXPECTED_FIGURES}")
        success = False

    if table_numbers == EXPECTED_TABLES:
        _ok(results, "table captions are continuous: 表 1-8")
    else:
        _fail(results, f"table captions are {table_numbers}, expected {EXPECTED_TABLES}")
        success = False

    for number in EXPECTED_FIGURES:
        idx = fig_indices.get(number)
        if idx is not None and _has_prior_mention(lines, idx, "图", number):
            _ok(results, f"图 {number} is mentioned before its caption")
        else:
            _fail(results, f"图 {number} lacks a prior in-text mention")
            success = False

    for number in EXPECTED_TABLES:
        idx = table_indices.get(number)
        if idx is not None and _has_prior_mention(lines, idx, "表", number):
            _ok(results, f"表 {number} is mentioned before its caption")
        else:
            _fail(results, f"表 {number} lacks a prior in-text mention")
            success = False
        if idx is not None and _table_has_markdown_body(lines, idx):
            _ok(results, f"表 {number} has a Markdown table body")
        else:
            _fail(results, f"表 {number} lacks a Markdown table body after the bilingual caption")
            success = False

    refs = _image_refs(text)
    if len(refs) == len(EXPECTED_FIGURES):
        _ok(results, "manuscript contains 5 figure image references")
    else:
        _fail(results, f"manuscript contains {len(refs)} image references, expected 5")
        success = False

    for expected_number, ref in zip(EXPECTED_FIGURES, refs):
        if f"fig{expected_number}_" in ref:
            _ok(results, f"figure reference {expected_number} filename matches figure number")
        else:
            _fail(results, f"figure reference {expected_number} filename mismatch: {ref}")
            success = False
        exists, resolved = _resolve_image(md_path, ref)
        if exists:
            _ok(results, f"figure file available for 图 {expected_number}: {resolved}")
        else:
            _fail(results, f"missing figure file for 图 {expected_number}: {resolved}")
            success = False

    report = ["# Figure and Table Validation", ""]
    report.extend(results)
    return success, "\n".join(report) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    success, report = validate(args.manuscript)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
