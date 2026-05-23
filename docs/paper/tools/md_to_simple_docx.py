#!/usr/bin/env python3
"""Create a basic docx from the paper Markdown using only stdlib.

This is intentionally small and conservative. It preserves headings, paragraphs,
Markdown tables, and image references as textual placeholders. The generated
file is a Word-transfer draft, not the final journal template.
"""

from __future__ import annotations

import html
import re
import sys
import zipfile
from pathlib import Path


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def clean_inline(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[图：\1；文件：\2]", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = text.replace("**", "")
    text = text.replace("`", "")
    return text


def paragraph(text: str, style: str | None = None) -> str:
    text = clean_inline(text.strip())
    if not text:
        return ""
    style_xml = ""
    if style:
        style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
    return (
        "<w:p>"
        f"{style_xml}"
        "<w:r><w:t xml:space=\"preserve\">"
        f"{esc(text)}"
        "</w:t></w:r></w:p>"
    )


def table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    grid = "".join("<w:gridCol w:w=\"2400\"/>" for _ in rows[0])
    trs = []
    for row in rows:
        cells = []
        for cell in row:
            cells.append(
                "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
                f"{paragraph(cell) or '<w:p/>'}</w:tc>"
            )
        trs.append("<w:tr>" + "".join(cells) + "</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/></w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        + "".join(trs)
        + "</w:tbl>"
    )


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip()
        parts = [p.strip() for p in raw.strip("|").split("|")]
        is_sep = all(re.fullmatch(r":?-{2,}:?", p or "") for p in parts)
        if not is_sep:
            rows.append(parts)
        i += 1
    return table(rows), i


def md_to_body(md: str) -> str:
    lines = md.splitlines()
    blocks: list[str] = []
    buf: list[str] = []
    i = 0

    def flush() -> None:
        nonlocal buf
        if buf:
            blocks.append(paragraph(" ".join(x.strip() for x in buf)))
            buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        if stripped.startswith("|"):
            flush()
            tbl, i = parse_table(lines, i)
            blocks.append(tbl)
            continue
        if stripped.startswith("# "):
            flush()
            blocks.append(paragraph(stripped[2:], "Title"))
            i += 1
            continue
        if stripped.startswith("## "):
            flush()
            blocks.append(paragraph(stripped[3:], "Heading1"))
            i += 1
            continue
        if stripped.startswith("### "):
            flush()
            blocks.append(paragraph(stripped[4:], "Heading2"))
            i += 1
            continue
        if stripped.startswith("#### "):
            flush()
            blocks.append(paragraph(stripped[5:], "Heading3"))
            i += 1
            continue
        if stripped.startswith("$$"):
            flush()
            eq_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                eq_lines.append(lines[i])
                i += 1
            blocks.append(paragraph("[公式] " + " ".join(eq_lines)))
            if i < len(lines):
                i += 1
            continue
        if stripped.startswith("!["):
            flush()
            blocks.append(paragraph(stripped))
            i += 1
            continue
        buf.append(line)
        i += 1
    flush()
    return "".join(blocks)


def content_types() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def document_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def styles() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:eastAsia="宋体" w:ascii="Times New Roman"/><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:b/><w:rFonts w:eastAsia="黑体" w:ascii="Times New Roman"/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:rFonts w:eastAsia="黑体" w:ascii="Times New Roman"/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:rFonts w:eastAsia="黑体" w:ascii="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:rFonts w:eastAsia="黑体" w:ascii="Times New Roman"/><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>"""


def document(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>"""


def core_props() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>基于专家词典与边界预测的红枣栽培命名实体识别方法</dc:title>
  <dc:creator>Codex draft generator</dc:creator>
</cp:coreProperties>"""


def app_props() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex</Application></Properties>"""


def build_docx(md_path: Path, out_path: Path) -> None:
    body = md_to_body(md_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types())
        zf.writestr("_rels/.rels", root_rels())
        zf.writestr("word/_rels/document.xml.rels", document_rels())
        zf.writestr("word/document.xml", document(body))
        zf.writestr("word/styles.xml", styles())
        zf.writestr("docProps/core.xml", core_props())
        zf.writestr("docProps/app.xml", app_props())


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: md_to_simple_docx.py INPUT.md OUTPUT.docx", file=sys.stderr)
        return 2
    build_docx(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
