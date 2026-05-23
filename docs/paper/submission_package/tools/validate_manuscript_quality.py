#!/usr/bin/env python3
"""Validate journal-facing manuscript quality gates."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REQUIRED_HEADINGS = [
    "# 基于专家词典与边界预测的红枣栽培命名实体识别方法",
    "## 摘要",
    "## Abstract",
    "## 0 引言",
    "## 1 材料与方法",
    "## 2 结果与分析",
    "## 3 讨论",
    "## 4 结论",
    "## 参考文献",
]


def _ok(results: list[str], message: str) -> None:
    results.append(f"- PASS: {message}")


def _fail(results: list[str], message: str) -> None:
    results.append(f"- FAIL: {message}")


def _section(text: str, start: str, end: str | None = None) -> str:
    begin = text.index(start)
    if end is None:
        return text[begin:]
    finish = text.index(end, begin)
    return text[begin:finish]


def _cjk_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _keyword_count(line: str, sep: str) -> int:
    return len([part.strip() for part in line.split("：", 1)[-1].split(sep) if part.strip()])


def validate(md_path: Path) -> tuple[bool, str]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    results: list[str] = []
    success = True

    title = next((line[2:].strip() for line in lines if line.startswith("# ")), "")
    title_len = _cjk_count(title)
    if 0 < title_len <= 24:
        _ok(results, f"title CJK length is {title_len}, within 24")
    else:
        _fail(results, f"title CJK length is {title_len}, expected 1-24")
        success = False

    last_pos = -1
    for heading in REQUIRED_HEADINGS:
        pos = text.find(heading)
        if pos > last_pos:
            _ok(results, f"required heading appears in order: {heading}")
            last_pos = pos
        else:
            _fail(results, f"required heading missing or out of order: {heading}")
            success = False

    keyword_line = next((line for line in lines if line.startswith("关键词：")), "")
    keyword_num = _keyword_count(keyword_line, "；") if keyword_line else 0
    if 1 <= keyword_num <= 6:
        _ok(results, f"Chinese keyword count is {keyword_num}, within 6")
    else:
        _fail(results, f"Chinese keyword count is {keyword_num}, expected 1-6")
        success = False

    english_keyword_line = next((line for line in lines if line.startswith("Keywords:")), "")
    english_keyword_num = _keyword_count(english_keyword_line.replace(":", "：", 1), ";") if english_keyword_line else 0
    if 1 <= english_keyword_num <= 6:
        _ok(results, f"English keyword count is {english_keyword_num}, within 6")
    else:
        _fail(results, f"English keyword count is {english_keyword_num}, expected 1-6")
        success = False

    abstract = _section(text, "## 摘要", "## Abstract")
    english_abstract = _section(text, "## Abstract", "## 0 引言")
    if "[" not in abstract and "$$" not in abstract:
        _ok(results, "Chinese abstract contains no citations or display formulas")
    else:
        _fail(results, "Chinese abstract contains citation markers or display formulas")
        success = False
    if all(needle in english_abstract for needle in ["First", "Second", "Third", "results"]):
        _ok(results, "English abstract covers method sequence and results")
    else:
        _fail(results, "English abstract lacks method sequence or result wording")
        success = False

    refs = re.findall(r"^\[[0-9]+\]", text, flags=re.MULTILINE)
    if len(refs) >= 25:
        _ok(results, f"reference count is {len(refs)}, at least 25")
    else:
        _fail(results, f"reference count is {len(refs)}, expected at least 25")
        success = False

    forbidden = {
        "Mermaid code fence": "```mermaid",
        "generic code fence": "```",
        "old model abbreviation EDBS": "EDBS",
        "project-management TODO marker": "TODO",
        "draft placeholder": "待补",
    }
    for label, needle in forbidden.items():
        if needle not in text:
            _ok(results, f"no {label}")
        else:
            _fail(results, f"found {label}: {needle}")
            success = False

    if "训练集自动抽取" in text and "人工专家知识库" not in text:
        _ok(results, "expert dictionary wording avoids claiming a manual expert knowledge base")
    else:
        _fail(results, "expert dictionary wording may overstate knowledge-base construction")
        success = False

    if all(needle in text for needle in ["红枣栽培知识图谱", "农业技术智能检索", "问答系统"]):
        _ok(results, "agricultural information application context is explicit")
    else:
        _fail(results, "agricultural information application context is incomplete")
        success = False

    report = ["# Manuscript Quality Validation", ""]
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
