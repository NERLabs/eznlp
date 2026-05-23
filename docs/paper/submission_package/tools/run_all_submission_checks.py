#!/usr/bin/env python3
"""Run all submission checks for the red-jujube NER paper."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "农业机械学报_红枣NER_投稿稿.md"
APPENDIX = ROOT / "农业机械学报_红枣NER_官方核对附录.md"
DOCX = ROOT / "农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx"
PACKAGE_DIR = ROOT / "submission_package"
ZIP_PATH = ROOT / "农业机械学报_红枣NER_投稿交接包_2026-05-23.zip"
EVIDENCE_DIR = ROOT / "submission_package" / "evidence"


CHECKS = [
    (
        "reference citation order",
        ["tools/validate_references.py", str(MANUSCRIPT)],
        ROOT / "reference_citation_validation_2026-05-23.md",
    ),
    (
        "reference format",
        ["tools/validate_reference_format.py", str(MANUSCRIPT)],
        ROOT / "reference_format_validation_2026-05-23.md",
    ),
    (
        "numeric consistency",
        ["tools/validate_numeric_consistency.py", str(MANUSCRIPT)],
        ROOT / "numeric_consistency_validation_2026-05-23.md",
    ),
    (
        "figure/table consistency",
        ["tools/validate_figures_tables.py", str(MANUSCRIPT)],
        ROOT / "figure_table_validation_2026-05-23.md",
    ),
    (
        "equation/symbol consistency",
        ["tools/validate_equations_symbols.py", str(MANUSCRIPT), str(APPENDIX), "--docx", str(DOCX)],
        ROOT / "equation_symbol_validation_2026-05-23.md",
    ),
    (
        "DOCX layout",
        ["tools/validate_docx_layout.py", str(DOCX)],
        ROOT / "docx_layout_validation_2026-05-23.md",
    ),
    (
        "manuscript quality",
        ["tools/validate_manuscript_quality.py", str(MANUSCRIPT)],
        ROOT / "manuscript_quality_validation_2026-05-23.md",
    ),
    (
        "submission info template",
        ["tools/validate_submission_info.py", str(ROOT / "submission_info.example.json"), "--draft"],
        ROOT / "submission_info_template_validation_2026-05-23.md",
    ),
    (
        "reproducibility paths",
        [
            "tools/validate_reproducibility_paths.py",
            "--root",
            str(ROOT.parent.parent),
            "--statement",
            str(ROOT / "reproducibility_statement.md"),
        ],
        ROOT / "reproducibility_validation_2026-05-23.md",
    ),
    (
        "rendered PDF quality",
        [
            "tools/validate_rendered_pdf.py",
            str(ROOT / "rendered" / "农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf"),
        ],
        ROOT / "rendered_pdf_validation_2026-05-23.md",
    ),
    (
        "submission package",
        ["tools/validate_submission_package.py", str(PACKAGE_DIR)],
        EVIDENCE_DIR / "package_validation_draft_2026-05-23.md",
    ),
]


def run_check(args: list[str], report: Path) -> tuple[bool, str]:
    cmd = [sys.executable, *args, "--report", str(report)]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    output = proc.stdout
    if proc.stderr:
        output += "\nSTDERR:\n" + proc.stderr
    return proc.returncode == 0, output.strip()


def zip_integrity(zip_path: Path) -> tuple[bool, str]:
    if not zip_path.exists():
        return False, f"- FAIL: zip not found: {zip_path}"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
        if bad is None:
            return True, "- PASS: handoff zip integrity test passed"
        return False, f"- FAIL: handoff zip corrupt member: {bad}"
    except Exception as exc:  # noqa: BLE001
        return False, f"- FAIL: handoff zip validation failed: {exc}"


def build_report(final: bool = False) -> tuple[bool, str]:
    lines = [f"# Full Submission Check Report ({date.today().isoformat()})", ""]
    success = True
    for label, args, report in CHECKS:
        ok, output = run_check(args, report)
        if ok:
            lines.append(f"## PASS: {label}")
        else:
            lines.append(f"## FAIL: {label}")
            success = False
        lines.append("")
        lines.append(output)
        lines.append("")

    zip_ok, zip_report = zip_integrity(ZIP_PATH)
    lines.append("## " + ("PASS" if zip_ok else "FAIL") + ": zip integrity")
    lines.append("")
    lines.append(zip_report)
    lines.append("")
    success = success and zip_ok

    if final:
        lines.append("## Final Mode")
        lines.append("")
        lines.append("- INFO: final mode requires a validated real submission_info.json and a no-placeholder DOCX.")
        lines.append("")
    return success, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "full_submission_check_report_2026-05-23.md",
    )
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    success, report = build_report(final=args.final)
    args.report.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
