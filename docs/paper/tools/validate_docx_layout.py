#!/usr/bin/env python3
"""Validate Word layout details that are easy to regress in generated DOCX."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
CAPTION_RE = re.compile(
    r"^(图|表)\s*\d+\s+(?!(为|给出|显示|进一步|所示|见)\b)"
    r"|^(Fig\.|Table)\s*\d+\b"
)
BODY_GUIDE_RE = re.compile(r"^(图|表)\s*\d+\s*(为|给出|显示|进一步|所示|见)\b")
ORDERED_RE = re.compile(r"^\d+\.\s+")


def paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(".//w:t", NS)).strip()


def paragraph_alignment(p: ET.Element) -> str | None:
    jc = p.find("./w:pPr/w:jc", NS)
    if jc is None:
        return None
    return jc.attrib.get(f"{{{NS['w']}}}val")


def read_paragraphs(docx: Path) -> list[ET.Element]:
    with zipfile.ZipFile(docx) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    return root.findall(".//w:p", NS)


def ok(lines: list[str], message: str) -> None:
    lines.append(f"- PASS: {message}")


def fail(lines: list[str], message: str) -> None:
    lines.append(f"- FAIL: {message}")


def validate(docx: Path) -> tuple[bool, str]:
    results = ["# DOCX Layout Validation", ""]
    success = True
    paragraphs = read_paragraphs(docx)
    texts = [paragraph_text(p) for p in paragraphs]

    dollar_paragraphs = [t for t in texts if "$" in t]
    if dollar_paragraphs:
        fail(results, f"DOCX still contains Markdown math markers in {len(dollar_paragraphs)} paragraphs")
        success = False
    else:
        ok(results, "DOCX contains no Markdown math markers")

    captions = [(paragraph_text(p), paragraph_alignment(p)) for p in paragraphs if CAPTION_RE.match(paragraph_text(p))]
    if captions and all(align == "center" for _, align in captions):
        ok(results, f"figure/table captions are centered: {len(captions)} paragraphs")
    else:
        bad = [text for text, align in captions if align != "center"]
        fail(results, "uncentered captions: " + "; ".join(bad[:5]))
        success = False

    centered_guides = [
        paragraph_text(p)
        for p in paragraphs
        if BODY_GUIDE_RE.match(paragraph_text(p)) and paragraph_alignment(p) == "center"
    ]
    if centered_guides:
        fail(results, "body guide sentences wrongly centered: " + "; ".join(centered_guides[:5]))
        success = False
    else:
        ok(results, "body guide sentences are not treated as captions")

    ordered = [t for t in texts if ORDERED_RE.match(t)]
    merged_ordered = [t for t in ordered if re.search(r"\s2\.\s+", t)]
    if len(ordered) >= 8 and not merged_ordered:
        ok(results, f"ordered list items are separate paragraphs: {len(ordered)} items")
    else:
        fail(results, "ordered list items may be merged into paragraphs")
        success = False

    return success, "\n".join(results) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    success, report = validate(args.docx)
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
