#!/usr/bin/env python3
"""Render the manuscript Markdown to a simple standalone HTML preview."""

from __future__ import annotations

import argparse
from pathlib import Path

import markdown


CSS = """
body {
  font-family: "Times New Roman", "SimSun", serif;
  line-height: 1.65;
  margin: 40px auto;
  max-width: 980px;
  color: #111;
}
h1 { text-align: center; font-size: 24px; }
h2 { font-size: 20px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
h3 { font-size: 17px; }
table { border-collapse: collapse; margin: 14px auto; width: 100%; }
th, td { border: 1px solid #999; padding: 5px 8px; }
th { background: #f5f5f5; }
img { display: block; max-width: 92%; margin: 12px auto; }
code { font-family: Consolas, monospace; }
"""


def build_html(md_path: Path, out_path: Path) -> None:
    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    html = (
        "<!doctype html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>基于专家词典与边界预测的红枣栽培命名实体识别方法</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_html(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
