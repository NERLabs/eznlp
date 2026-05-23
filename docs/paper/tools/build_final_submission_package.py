#!/usr/bin/env python3
"""Build a final submission package after author metadata is provided.

Input:
    docs/paper/submission_info.json

Output:
    docs/paper/final_submission_package/
    docs/paper/农业机械学报_红枣NER_终稿交接包_YYYY-MM-DD.zip

The script fills front-matter placeholders, renders a PDF with LibreOffice,
copies evidence/figures, and runs final validation.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

from fill_docx_front_matter import fill_docx
from validate_submission_info import validate_info
from validate_submission_package import validate


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "final_submission_package"
DOCX_NAME = "农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx"
PDF_NAME = "农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf"


def run(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def make_zip(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            zf.write(path, path.relative_to(src_dir.parent))


def build(info_json: Path, output_dir: Path, zip_path: Path) -> None:
    info_success, info_report = validate_info(info_json, allow_placeholders=False)
    if not info_success:
        print(info_report)
        raise SystemExit("Submission info validation failed")

    source_package = ROOT / "submission_package"
    if not source_package.exists():
        raise SystemExit(f"Missing source package: {source_package}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    copy_tree_contents(source_package, output_dir)

    template_docx = source_package / DOCX_NAME
    final_docx = output_dir / DOCX_NAME
    fill_docx(template_docx, info_json, final_docx)

    pdf_path = output_dir / PDF_NAME
    if pdf_path.exists():
        pdf_path.unlink()
    run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(final_docx)])

    make_zip(output_dir, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise SystemExit(f"Final zip corrupt member: {bad}")

    report_path = output_dir / "evidence" / f"package_validation_final_{date.today().isoformat()}.md"
    success, report = validate(output_dir, final=True)
    report_path.write_text(report, encoding="utf-8")
    if not success:
        print(report)
        raise SystemExit("Final validation failed")

    make_zip(output_dir, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise SystemExit(f"Final zip corrupt member after adding report: {bad}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--info",
        type=Path,
        default=ROOT / "submission_info.json",
        help="Filled author/front-matter JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / PACKAGE_NAME,
        help="Final package directory.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=ROOT / f"农业机械学报_红枣NER_终稿交接包_{date.today().isoformat()}.zip",
        help="Final zip path.",
    )
    args = parser.parse_args()
    build(args.info, args.output_dir, args.zip)
    print(f"Final package: {args.output_dir}")
    print(f"Final zip: {args.zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
