#!/usr/bin/env python3
"""Validate basic GB/T 7714-style reference formatting gates."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REF_LINE_RE = re.compile(r"^\[([0-9]+)\]\s+(.+)$", re.MULTILINE)
TYPE_RE = re.compile(r"\[(J|C|M|EB/OL|D|R)\]")
YEAR_RE = re.compile(r"(19|20)[0-9]{2}")
DOI_RE = re.compile(r"DOI:\s*\S+", re.IGNORECASE)
URL_RE = re.compile(r"URL:\s*https?://\S+", re.IGNORECASE)


def _ok(results: list[str], message: str) -> None:
    results.append(f"- PASS: {message}")


def _fail(results: list[str], message: str) -> None:
    results.append(f"- FAIL: {message}")


def _is_english_reference(entry: str) -> bool:
    head = entry.split(".", 1)[0]
    return bool(re.fullmatch(r"[A-Z ,\-]+(?:et al)?", head.strip()))


def _publication_year(entry: str) -> int | None:
    years = [int(match.group(0)) for match in YEAR_RE.finditer(entry)]
    plausible = [year for year in years if 1990 <= year <= 2026]
    return plausible[0] if plausible else (years[0] if years else None)


def validate(md_path: Path) -> tuple[bool, str]:
    text = md_path.read_text(encoding="utf-8")
    if "\n## 参考文献" not in text:
        return False, "# Reference Format Validation\n\n- FAIL: missing reference section\n"
    refs_text = text.split("\n## 参考文献", 1)[1]
    entries = [(int(num), body.strip()) for num, body in REF_LINE_RE.findall(refs_text)]
    results: list[str] = []
    success = True

    if len(entries) >= 25:
        _ok(results, f"reference count is {len(entries)}, at least 25")
    else:
        _fail(results, f"reference count is {len(entries)}, expected at least 25")
        success = False

    expected = list(range(1, len(entries) + 1))
    actual = [num for num, _ in entries]
    if actual == expected:
        _ok(results, "reference numbering is continuous")
    else:
        _fail(results, f"reference numbering is not continuous: {actual}")
        success = False

    type_counts: dict[str, int] = {}
    recent_count = 0
    english_count = 0
    for num, body in entries:
        type_match = TYPE_RE.search(body)
        if type_match:
            ref_type = type_match.group(1)
            type_counts[ref_type] = type_counts.get(ref_type, 0) + 1
            _ok(results, f"[{num}] has literature type marker [{ref_type}]")
        else:
            _fail(results, f"[{num}] missing literature type marker")
            success = False

        year = _publication_year(body)
        if year is not None:
            if year >= 2020:
                recent_count += 1
            _ok(results, f"[{num}] has publication year {year}")
        else:
            _fail(results, f"[{num}] missing publication year")
            success = False

        if DOI_RE.search(body) or URL_RE.search(body):
            _ok(results, f"[{num}] has DOI or URL")
        else:
            _fail(results, f"[{num}] missing DOI or URL")
            success = False

        if _is_english_reference(body):
            english_count += 1
            author_head = body.split(".", 1)[0]
            normalized_author_head = author_head.replace("et al", "ET AL")
            if normalized_author_head == normalized_author_head.upper():
                _ok(results, f"[{num}] English author names are uppercase")
            else:
                _fail(results, f"[{num}] English author names are not uppercase")
                success = False

        if "张华洋" in body:
            _fail(results, f"[{num}] contains stale author name 张华洋")
            success = False

    if type_counts.get("J", 0) >= 1 and type_counts.get("C", 0) >= 1:
        _ok(results, "reference list includes both journal and conference literature")
    else:
        _fail(results, "reference list lacks journal or conference literature")
        success = False

    if recent_count >= 10:
        _ok(results, f"recent references since 2020: {recent_count}")
    else:
        _fail(results, f"recent references since 2020: {recent_count}, expected at least 10")
        success = False

    if english_count >= 10:
        _ok(results, f"foreign-language references: {english_count}")
    else:
        _fail(results, f"foreign-language references: {english_count}, expected at least 10")
        success = False

    report = ["# Reference Format Validation", ""]
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
