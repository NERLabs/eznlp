#!/usr/bin/env python3
"""Validate key numeric claims in the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED = {
    "main_f1": "88.28%±0.22%",
    "main_f1_table": "88.28±0.22",
    "bilstm_gain": "9.59",
    "bert_wwm_gain": "2.80",
    "macbert_gain": "2.71",
    "representative_prf": ["89.51", "87.58", "88.54"],
    "public": {
        "MSRA": "95.19±0.22",
        "WeiboNER": "72.27±1.03",
        "ResumeNER": "96.13±0.29",
        "Boson": "85.60±0.12",
        "CLUENER": "80.06±0.38",
    },
}


def section(text: str, start: str, end: str | None = None) -> str:
    i = text.index(start)
    if end is None:
        return text[i:]
    j = text.index(end, i)
    return text[i:j]


def contains_numeric(haystack: str, value: str) -> bool:
    return value in haystack or value in haystack.replace("%", "")


def validate(md_path: Path) -> tuple[bool, str]:
    text = md_path.read_text(encoding="utf-8")
    report: list[str] = ["# Numeric Consistency Validation", ""]
    success = True

    abstract = section(text, "## 摘要", "## Abstract")
    english_abstract = section(text, "## Abstract", "## 0 引言")
    results = section(text, "## 2 结果与分析", "## 3 讨论")
    conclusion = section(text, "## 4 结论", "## 参考文献")

    checks = [
        ("Chinese abstract main F1", EXPECTED["main_f1"], abstract),
        ("English abstract main F1", EXPECTED["main_f1"], english_abstract),
        ("Table/result main F1", EXPECTED["main_f1_table"], results),
        ("Conclusion main F1", EXPECTED["main_f1"], conclusion),
        ("BiLSTM-CRF gain", EXPECTED["bilstm_gain"], abstract + english_abstract + results),
        ("BERT-wwm-ext gain", EXPECTED["bert_wwm_gain"], abstract + english_abstract + results),
        ("MacBERT gain", EXPECTED["macbert_gain"], abstract + english_abstract + results + conclusion),
    ]
    for label, needle, haystack in checks:
        if needle in haystack:
            report.append(f"- PASS: {label} contains `{needle}`")
        else:
            report.append(f"- FAIL: {label} missing `{needle}`")
            success = False

    for value in EXPECTED["representative_prf"]:
        if value in results:
            report.append(f"- PASS: representative analysis contains `{value}`")
        else:
            report.append(f"- FAIL: representative analysis missing `{value}`")
            success = False

    for dataset, value in EXPECTED["public"].items():
        if dataset in abstract and contains_numeric(abstract, value) and dataset in results and contains_numeric(results, value):
            report.append(f"- PASS: public dataset {dataset} value `{value}` appears in abstract and Table 8")
        else:
            report.append(f"- FAIL: public dataset {dataset} value `{value}` missing from abstract or Table 8")
            success = False

    lexicon_values = ["5 317", "77.75", "31.04", "16.94", "62.76"]
    for value in lexicon_values:
        if value in results:
            report.append(f"- PASS: lexicon strategy section contains `{value}`")
        else:
            report.append(f"- FAIL: lexicon strategy section missing `{value}`")
            success = False

    if "统计口径与表 3 的三种子均值不同" in results:
        report.append("- PASS: representative-run table states its different statistical scope")
    else:
        report.append("- FAIL: representative-run table does not state different statistical scope")
        success = False

    report.append("")
    return success, "\n".join(report)


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
