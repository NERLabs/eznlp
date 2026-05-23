#!/usr/bin/env python3
"""Validate in-text citation order and reference-list continuity."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CITE_RE = re.compile(r"\[([0-9]+(?:\s*[-,，]\s*[0-9]+)*)\]")
REF_RE = re.compile(r"^\[([0-9]+)\]", re.MULTILINE)


def expand_citation(raw: str) -> list[int]:
    numbers: list[int] = []
    for part in re.split(r"[,，]", raw):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = [int(x.strip()) for x in part.split("-", 1)]
            if left <= right:
                numbers.extend(range(left, right + 1))
            else:
                numbers.extend(range(left, right - 1, -1))
        else:
            numbers.append(int(part))
    return numbers


def validate(md_path: Path) -> tuple[bool, str]:
    text = md_path.read_text(encoding="utf-8")
    if "\n## 参考文献" not in text:
        return False, "# Reference Validation\n\n- FAIL: missing reference section\n"
    body, refs_text = text.split("\n## 参考文献", 1)
    citation_groups = [expand_citation(m.group(1)) for m in CITE_RE.finditer(body)]
    citations = [num for group in citation_groups for num in group]
    refs = [int(num) for num in REF_RE.findall(refs_text)]

    lines: list[str] = ["# Reference Validation", ""]
    success = True

    expected_refs = list(range(1, max(refs) + 1)) if refs else []
    if refs == expected_refs:
        lines.append(f"- PASS: reference list is continuous [1]-[{len(refs)}]")
    else:
        lines.append(f"- FAIL: reference list is not continuous: {refs}")
        success = False

    if citations:
        lines.append(f"- PASS: found {len(citations)} expanded in-text citations")
    else:
        lines.append("- FAIL: no in-text citations found before reference section")
        success = False

    cited_unique = sorted(set(citations))
    if cited_unique == refs:
        lines.append("- PASS: every reference is cited in the body")
    else:
        missing = sorted(set(refs) - set(citations))
        extra = sorted(set(citations) - set(refs))
        if missing:
            lines.append(f"- FAIL: uncited references: {missing}")
            success = False
        if extra:
            lines.append(f"- FAIL: citations without reference entries: {extra}")
            success = False

    first_positions: dict[int, int] = {}
    for idx, num in enumerate(citations):
        first_positions.setdefault(num, idx)
    first_order = [num for num, _ in sorted(first_positions.items(), key=lambda item: item[1])]
    if first_order == sorted(first_order):
        lines.append("- PASS: first citation order is ascending")
    else:
        lines.append(f"- FAIL: first citation order is not ascending: {first_order}")
        success = False

    if citations and min(citations) >= 1 and max(citations) <= (max(refs) if refs else 0):
        lines.append(f"- PASS: citation range is [{min(citations)}]-[{max(citations)}]")

    lines.append("")
    return success, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    success, report = validate(args.md_path)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
