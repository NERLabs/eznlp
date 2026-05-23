#!/usr/bin/env python3
"""Validate author/front-matter JSON before final DOCX generation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_KEYS = [
    "authors_cn",
    "authors_en",
    "affiliations_cn",
    "affiliations_en",
    "clc",
    "document_code",
    "article_id",
    "funding",
    "author_bio",
    "corresponding_author",
]

PLACEHOLDER_PATTERNS = [
    "待作者补充",
    "待作者确认",
    "To be completed",
    "（待作者确认）",
]


def _ok(results: list[str], message: str) -> None:
    results.append(f"- PASS: {message}")


def _fail(results: list[str], message: str) -> None:
    results.append(f"- FAIL: {message}")


def load_json(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("submission info must be a JSON object")
    return {str(key): str(value).strip() for key, value in data.items()}


def _author_count(authors_cn: str) -> int:
    normalized = re.sub(r"[，、;；]+", ",", authors_cn)
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    return len(parts) if parts else 0


def _clc_count(clc: str) -> int:
    normalized = re.sub(r"[，、;；]+", ",", clc)
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    return len(parts) if parts else 0


def validate_info(path: Path, allow_placeholders: bool = False) -> tuple[bool, str]:
    data = load_json(path)
    results: list[str] = []
    success = True

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if not missing:
        _ok(results, "all required keys are present")
    else:
        _fail(results, "missing required keys: " + ", ".join(missing))
        success = False

    for key in REQUIRED_KEYS:
        value = data.get(key, "")
        if value:
            _ok(results, f"{key} is non-empty")
        else:
            _fail(results, f"{key} is empty")
            success = False
        if not allow_placeholders and any(pattern in value for pattern in PLACEHOLDER_PATTERNS):
            _fail(results, f"{key} still contains placeholder text")
            success = False

    authors = _author_count(data.get("authors_cn", ""))
    if 1 <= authors <= 6:
        _ok(results, f"Chinese author count is {authors}, within 6")
    elif allow_placeholders:
        _ok(results, "Chinese author count check deferred for template JSON")
    else:
        _fail(results, f"Chinese author count is {authors}, expected 1-6")
        success = False

    clc_num = _clc_count(data.get("clc", ""))
    if 1 <= clc_num <= 2:
        _ok(results, f"CLC count is {clc_num}, within 2")
    elif allow_placeholders:
        _ok(results, "CLC count check deferred for template JSON")
    else:
        _fail(results, f"CLC count is {clc_num}, expected 1-2")
        success = False

    if data.get("document_code") == "A":
        _ok(results, "document_code is A")
    else:
        _fail(results, "document_code should be A")
        success = False

    if "编辑部填写" in data.get("article_id", ""):
        _ok(results, "article_id is marked for editorial office")
    elif data.get("article_id"):
        _ok(results, "article_id is provided")

    if not allow_placeholders:
        for key in ["author_bio", "corresponding_author"]:
            value = data.get(key, "")
            if "E-mail" in value or "邮箱" in value or "@" in value:
                _ok(results, f"{key} includes email information")
            else:
                _fail(results, f"{key} should include email information")
                success = False

    report = ["# Submission Info Validation", ""]
    report.extend(results)
    return success, "\n".join(report) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("info_json", type=Path)
    parser.add_argument("--draft", action="store_true", help="Allow placeholder template values.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    success, report = validate_info(args.info_json, allow_placeholders=args.draft)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
