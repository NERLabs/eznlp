#!/usr/bin/env python3
"""Validate the red-jujube NER submission package.

The default mode validates the current handoff package, where author/front
matter placeholders are allowed. Use ``--final`` after filling author metadata
to require that placeholders have been removed from the DOCX.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path


PLACEHOLDERS = [
    "待作者补充",
    "待作者确认",
    "To be completed",
]

REQUIRED_FILES = [
    "README.md",
    "农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx",
    "农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf",
    "农业机械学报_红枣NER_投稿稿.md",
    "农业机械学报_红枣NER_投稿信息补全表.md",
    "submission_info.example.json",
    "tools/fill_docx_front_matter.py",
    "tools/validate_submission_info.py",
    "tools/validate_references.py",
    "tools/validate_reference_format.py",
    "tools/validate_numeric_consistency.py",
    "tools/validate_figures_tables.py",
    "tools/validate_equations_symbols.py",
    "tools/validate_docx_layout.py",
    "tools/validate_manuscript_quality.py",
    "tools/md_to_html.py",
    "tools/run_all_submission_checks.py",
    "tools/validate_reproducibility_paths.py",
    "tools/validate_rendered_pdf.py",
    "official_docs/论文编改专项核对表_2018-8-20.pdf",
    "official_docs/论文写作模板_2019.pdf",
    "evidence/reference_audit_round1.md",
    "evidence/render_audit_2026-05-23.md",
    "evidence/paper_result_registry.md",
    "evidence/农业机械学报_红枣NER_官方核对附录.md",
    "evidence/full_submission_check_report_2026-05-23.md",
    "evidence/reproducibility_statement.md",
    "evidence/reproducibility_validation_2026-05-23.md",
    "evidence/rendered_pdf_validation_2026-05-23.md",
    "evidence/docx_layout_validation_2026-05-23.md",
]

FIGURES = [
    "figures_png/fig1_edbp_architecture.png",
    "figures_png/fig2_bmes_dictionary_encoding.png",
    "figures_png/fig3_boundary_prediction_decoder.png",
    "figures_png/fig4_entity_f1_by_category.png",
    "figures_png/fig5_boundary_error_cases.png",
]


def ok(results: list[str], message: str) -> None:
    results.append(f"- PASS: {message}")


def fail(results: list[str], message: str) -> None:
    results.append(f"- FAIL: {message}")


def warn(results: list[str], message: str) -> None:
    results.append(f"- WARN: {message}")


def read_docx_xml(path: Path) -> tuple[str, list[str]]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        xml = zf.read("word/document.xml").decode("utf-8")
    return xml, names


def run_cmd(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def validate(package_dir: Path, final: bool) -> tuple[bool, str]:
    results: list[str] = []
    success = True

    for rel in REQUIRED_FILES + FIGURES:
        path = package_dir / rel
        if path.exists() and path.stat().st_size > 0:
            ok(results, f"required file exists: {rel}")
        else:
            fail(results, f"missing or empty required file: {rel}")
            success = False

    docx = package_dir / "农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx"
    if docx.exists():
        try:
            xml, names = read_docx_xml(docx)
            media_count = sum(name.startswith("word/media/") for name in names)
            table_count = xml.count("<w:tbl>")
            if media_count == 5:
                ok(results, "DOCX contains 5 embedded media files")
            else:
                fail(results, f"DOCX embedded media count is {media_count}, expected 5")
                success = False
            if table_count == 8:
                ok(results, "DOCX contains 8 tables")
            else:
                fail(results, f"DOCX table count is {table_count}, expected 8")
                success = False
            for needle in [
                "基于专家词典与边界预测的红枣栽培命名实体识别方法",
                "Named Entity Recognition Method for Red Jujube Cultivation",
                "88.16%",
                "10.6041/j.issn.1000-1298.2025.11.050",
            ]:
                if needle in xml:
                    ok(results, f"DOCX contains: {needle}")
                else:
                    fail(results, f"DOCX missing: {needle}")
                    success = False
            found_placeholders = [p for p in PLACEHOLDERS if p in xml]
            if found_placeholders and final:
                fail(results, "DOCX still contains placeholders: " + ", ".join(found_placeholders))
                success = False
            elif found_placeholders:
                warn(results, "DOCX draft placeholders remain: " + ", ".join(found_placeholders))
            else:
                ok(results, "DOCX contains no front-matter placeholders")
        except Exception as exc:  # noqa: BLE001
            fail(results, f"DOCX validation failed: {exc}")
            success = False

    md = package_dir / "农业机械学报_红枣NER_投稿稿.md"
    if md.exists():
        text = md.read_text(encoding="utf-8")
        refs = re.findall(r"^\[[0-9]+\]", text, flags=re.MULTILINE)
        if len(refs) == 28:
            ok(results, "Markdown contains 28 references")
        else:
            fail(results, f"Markdown reference count is {len(refs)}, expected 28")
            success = False
        if "张华洋" in text:
            fail(results, "Markdown still contains stale author name 张华洋")
            success = False
        else:
            ok(results, "Markdown does not contain stale author name 张华洋")
        try:
            import importlib.util

            validator_path = package_dir / "tools" / "validate_references.py"
            if validator_path.exists():
                spec = importlib.util.spec_from_file_location("validate_references", validator_path)
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                ref_success, _ = module.validate(md)
                if ref_success:
                    ok(results, "Markdown in-text citations cover references in ascending order")
                else:
                    fail(results, "Markdown citation/reference validation failed")
                    success = False
            else:
                warn(results, "reference validator script not found in package")
        except Exception as exc:  # noqa: BLE001
            warn(results, f"reference validation skipped or failed: {exc}")
        try:
            import importlib.util

            numeric_validator_path = package_dir / "tools" / "validate_numeric_consistency.py"
            if numeric_validator_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "validate_numeric_consistency", numeric_validator_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                numeric_success, _ = module.validate(md)
                if numeric_success:
                    ok(results, "Markdown key numeric claims are internally consistent")
                else:
                    fail(results, "Markdown numeric consistency validation failed")
                    success = False
            else:
                warn(results, "numeric consistency validator script not found in package")
        except Exception as exc:  # noqa: BLE001
            warn(results, f"numeric consistency validation skipped or failed: {exc}")
        try:
            import importlib.util

            ref_format_validator_path = package_dir / "tools" / "validate_reference_format.py"
            if ref_format_validator_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "validate_reference_format", ref_format_validator_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                ref_format_success, _ = module.validate(md)
                if ref_format_success:
                    ok(results, "Markdown references pass basic format gates")
                else:
                    fail(results, "Markdown reference format validation failed")
                    success = False
            else:
                warn(results, "reference format validator script not found in package")
        except Exception as exc:  # noqa: BLE001
            warn(results, f"reference format validation skipped or failed: {exc}")
        try:
            import importlib.util

            figure_table_validator_path = package_dir / "tools" / "validate_figures_tables.py"
            if figure_table_validator_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "validate_figures_tables", figure_table_validator_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                figure_table_success, _ = module.validate(md)
                if figure_table_success:
                    ok(results, "Markdown figure/table numbering, mentions, and files are consistent")
                else:
                    fail(results, "Markdown figure/table validation failed")
                    success = False
            else:
                warn(results, "figure/table validator script not found in package")
        except Exception as exc:  # noqa: BLE001
            warn(results, f"figure/table validation skipped or failed: {exc}")
        try:
            import importlib.util

            equation_validator_path = package_dir / "tools" / "validate_equations_symbols.py"
            appendix_path = package_dir / "evidence" / "农业机械学报_红枣NER_官方核对附录.md"
            if equation_validator_path.exists() and appendix_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "validate_equations_symbols", equation_validator_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                equation_success, _ = module.validate(md, appendix_path, docx)
                if equation_success:
                    ok(results, "Markdown equations, symbol appendix, and DOCX equation numbers are consistent")
                else:
                    fail(results, "equation/symbol validation failed")
                    success = False
            else:
                warn(results, "equation/symbol validator or appendix not found in package")
        except Exception as exc:  # noqa: BLE001
            warn(results, f"equation/symbol validation skipped or failed: {exc}")
        try:
            import importlib.util

            docx_layout_validator_path = package_dir / "tools" / "validate_docx_layout.py"
            if docx_layout_validator_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "validate_docx_layout", docx_layout_validator_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                docx_layout_success, _ = module.validate(docx)
                if docx_layout_success:
                    ok(results, "DOCX layout gates pass")
                else:
                    fail(results, "DOCX layout validation failed")
                    success = False
            else:
                warn(results, "DOCX layout validator script not found in package")
        except Exception as exc:  # noqa: BLE001
            warn(results, f"DOCX layout validation skipped or failed: {exc}")
        try:
            import importlib.util

            manuscript_validator_path = package_dir / "tools" / "validate_manuscript_quality.py"
            if manuscript_validator_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "validate_manuscript_quality", manuscript_validator_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                manuscript_success, _ = module.validate(md)
                if manuscript_success:
                    ok(results, "Markdown passes manuscript quality gates")
                else:
                    fail(results, "Markdown manuscript quality validation failed")
                    success = False
            else:
                warn(results, "manuscript quality validator script not found in package")
        except Exception as exc:  # noqa: BLE001
            warn(results, f"manuscript quality validation skipped or failed: {exc}")

    info_example = package_dir / "submission_info.example.json"
    if info_example.exists():
        try:
            import importlib.util

            info_validator_path = package_dir / "tools" / "validate_submission_info.py"
            if info_validator_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "validate_submission_info", info_validator_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                info_success, _ = module.validate_info(info_example, allow_placeholders=True)
                if info_success:
                    ok(results, "submission_info.example.json has the required schema")
                else:
                    fail(results, "submission_info.example.json schema validation failed")
                    success = False
            else:
                warn(results, "submission info validator script not found in package")
        except Exception as exc:  # noqa: BLE001
            warn(results, f"submission info validation skipped or failed: {exc}")

    pdf = package_dir / "农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf"
    if pdf.exists():
        try:
            info = run_cmd(["pdfinfo", str(pdf)])
            if "Pages:" in info and "Page size:" in info:
                ok(results, "PDF metadata readable")
            else:
                fail(results, "PDF metadata missing page information")
                success = False
            text = run_cmd(["pdftotext", str(pdf), "-"])
            for needle in ["88.16%", "参考文献", "10.6041/j.issn.1000-1298.2025.11.050"]:
                if needle in text:
                    ok(results, f"PDF text contains: {needle}")
                else:
                    fail(results, f"PDF text missing: {needle}")
                    success = False
        except Exception as exc:  # noqa: BLE001
            warn(results, f"PDF validation skipped or failed: {exc}")
        try:
            import importlib.util

            rendered_pdf_validator_path = package_dir / "tools" / "validate_rendered_pdf.py"
            if rendered_pdf_validator_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "validate_rendered_pdf", rendered_pdf_validator_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                pdf_success, _ = module.validate(pdf)
                if pdf_success:
                    ok(results, "PDF render quality gates pass")
                else:
                    fail(results, "PDF render quality validation failed")
                    success = False
            else:
                warn(results, "rendered PDF validator script not found in package")
        except Exception as exc:  # noqa: BLE001
            warn(results, f"rendered PDF validation skipped or failed: {exc}")

    zip_candidates = [
        package_dir.parent / "农业机械学报_红枣NER_投稿交接包_2026-05-23.zip",
        package_dir.with_suffix(".zip"),
    ]
    zip_path = next((path for path in zip_candidates if path.exists()), None)
    if zip_path is not None:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                bad = zf.testzip()
                if bad is None:
                    ok(results, "handoff zip integrity test passed")
                else:
                    fail(results, f"handoff zip corrupt member: {bad}")
                    success = False
        except Exception as exc:  # noqa: BLE001
            fail(results, f"handoff zip validation failed: {exc}")
            success = False
    else:
        warn(results, "handoff zip not found next to package directory")

    mode = "final" if final else "draft"
    report = [f"# Submission Package Validation ({mode})", ""]
    report.extend(results)
    return success, "\n".join(report) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", nargs="?", default="docs/paper/submission_package")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    success, report = validate(Path(args.package_dir), final=args.final)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
