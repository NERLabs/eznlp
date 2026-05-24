#!/usr/bin/env python3
"""Validate the LibreOffice-rendered PDF preview."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


REQUIRED_TEXT = [
    "基于专家词典与边界预测的红枣栽培命名实体识别方法",
    "Named Entity Recognition Method for Red Jujube Cultivation",
    "88.16%",
    "表 5 词典构建策略对比",
    "表 8 公开数据集泛化实验结果",
    "参考文献",
    "10.6041/j.issn.1000-1298.2025.11.050",
]


def _ok(results: list[str], message: str) -> None:
    results.append(f"- PASS: {message}")


def _fail(results: list[str], message: str) -> None:
    results.append(f"- FAIL: {message}")


def _run(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def _pdf_pages(info: str) -> int | None:
    match = re.search(r"^Pages:\s+([0-9]+)", info, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def _image_rows(images_output: str) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for line in images_output.splitlines():
        parts = line.split()
        if len(parts) < 14 or parts[2] != "image":
            continue
        try:
            rows.append((int(parts[12]), int(parts[13])))
        except ValueError:
            continue
    return rows


def validate(pdf: Path) -> tuple[bool, str]:
    results: list[str] = []
    success = True

    if pdf.exists() and pdf.stat().st_size > 0:
        _ok(results, f"PDF exists: {pdf}")
    else:
        _fail(results, f"missing PDF: {pdf}")
        return False, "# Rendered PDF Validation\n\n" + "\n".join(results) + "\n"

    try:
        info = _run(["pdfinfo", str(pdf)])
    except Exception as exc:  # noqa: BLE001
        return False, f"# Rendered PDF Validation\n\n- FAIL: pdfinfo failed: {exc}\n"

    pages = _pdf_pages(info)
    if pages is not None and pages >= 5:
        _ok(results, f"PDF page count is {pages}, at least 5")
    else:
        _fail(results, f"PDF page count is {pages}, expected at least 5")
        success = False

    if "Page size:" in info and "595." in info and "841." in info:
        _ok(results, "PDF page size is A4-like")
    else:
        _fail(results, "PDF page size is not A4-like")
        success = False

    try:
        text = _run(["pdftotext", str(pdf), "-"])
    except Exception as exc:  # noqa: BLE001
        text = ""
        _fail(results, f"pdftotext failed: {exc}")
        success = False

    for needle in REQUIRED_TEXT:
        if needle in text:
            _ok(results, f"PDF text contains: {needle}")
        else:
            _fail(results, f"PDF text missing: {needle}")
            success = False

    try:
        images_output = _run(["pdfimages", "-list", str(pdf)])
        rows = _image_rows(images_output)
        if len(rows) == 5:
            _ok(results, "PDF contains 5 raster figure images")
        else:
            _fail(results, f"PDF contains {len(rows)} raster figure images, expected 5")
            success = False
        low_res = [(x_ppi, y_ppi) for x_ppi, y_ppi in rows if x_ppi < 250 or y_ppi < 250]
        if not low_res:
            _ok(results, "all raster figure images are at least 250 ppi")
        else:
            _fail(results, f"low-resolution figure images found: {low_res}")
            success = False
    except Exception as exc:  # noqa: BLE001
        _fail(results, f"pdfimages failed: {exc}")
        success = False

    report = ["# Rendered PDF Validation", ""]
    report.extend(results)
    return success, "\n".join(report) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    success, report = validate(args.pdf)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
