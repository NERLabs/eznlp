#!/usr/bin/env python3
"""Validate display equations and the official symbol appendix."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


REQUIRED_SYMBOLS = [
    "$X$",
    "$x_i$",
    "$n$",
    "$d_h$",
    "$C$",
    "$D$",
    "$D_c$",
    "$H^{bert}$",
    "$H^{dict}$",
    "$H^{fused}$",
    "$f_{c,p}(x_i)$",
    "$v_i^{dict}$",
    "$e_i^{dict}$",
    "$H^{start}$",
    "$H^{end}$",
    "$h_i^{start}$",
    "$h_j^{end}$",
    "$s_{i,j,c}$",
    "$W_s$",
    "$W_e$",
    "$U_c$",
    "$W_c$",
    "$b_s$",
    "$b_e$",
    "$b_c$",
    "$p_t$",
    "$\\alpha_t$",
    "$\\gamma$",
]


def _ok(results: list[str], message: str) -> None:
    results.append(f"- PASS: {message}")


def _fail(results: list[str], message: str) -> None:
    results.append(f"- FAIL: {message}")


def _display_equations(text: str) -> list[str]:
    return re.findall(r"^\$\$\n(.*?)\n\$\$", text, flags=re.MULTILINE | re.DOTALL)


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        return zf.read("word/document.xml").decode("utf-8")


def validate(manuscript: Path, appendix: Path, docx: Path | None = None) -> tuple[bool, str]:
    text = manuscript.read_text(encoding="utf-8")
    appendix_text = appendix.read_text(encoding="utf-8")
    equations = _display_equations(text)
    results: list[str] = []
    success = True

    if len(equations) == 10:
        _ok(results, "manuscript contains 10 display equations")
    else:
        _fail(results, f"manuscript contains {len(equations)} display equations, expected 10")
        success = False

    equation_checks = {
        "input sequence": "X=\\{x_1,x_2,\\ldots,x_n\\}",
        "MacBERT representation": "H^{bert}=\\operatorname{MacBERT}(X)",
        "expert dictionary": "D=\\{D_1,D_2,\\ldots,D_C\\}",
        "BMES indicator": "f_{c,p}(x_i)=",
        "dictionary vector": "v_i^{dict}=",
        "dictionary embedding": "e_i^{dict}=",
        "start projection": "H^{start}=W_sH^{fused}+b_s",
        "end projection": "H^{end}=W_eH^{fused}+b_e",
        "biaffine score": "s_{i,j,c}=h_i^{start}U_c(h_j^{end})^T",
        "Focal Loss": "L=-\\alpha_t(1-p_t)^\\gamma \\log(p_t)",
    }
    equation_blob = "\n".join(equations)
    for label, needle in equation_checks.items():
        if needle in equation_blob:
            _ok(results, f"equation present: {label}")
        else:
            _fail(results, f"equation missing: {label}")
            success = False

    missing_symbols = [symbol for symbol in REQUIRED_SYMBOLS if symbol not in appendix_text]
    if not missing_symbols:
        _ok(results, f"symbol appendix covers {len(REQUIRED_SYMBOLS)} required symbols")
    else:
        _fail(results, "symbol appendix missing: " + ", ".join(missing_symbols))
        success = False

    if "无量纲" in appendix_text:
        _ok(results, "symbol appendix includes unit/dimension column values")
    else:
        _fail(results, "symbol appendix lacks unit/dimension information")
        success = False

    if docx is not None and docx.exists():
        xml = _docx_text(docx)
        missing_numbers = [f"({idx})" for idx in range(1, len(equations) + 1) if f"({idx})" not in xml]
        if not missing_numbers:
            _ok(results, "DOCX contains continuous equation numbers")
        else:
            _fail(results, "DOCX missing equation numbers: " + ", ".join(missing_numbers))
            success = False

    report = ["# Equation and Symbol Validation", ""]
    report.extend(results)
    return success, "\n".join(report) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("appendix", type=Path)
    parser.add_argument("--docx", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    success, report = validate(args.manuscript, args.appendix, args.docx)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
