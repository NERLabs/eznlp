#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看预测结果文件工具（支持把实体标注回原始文本）
"""

import argparse  # 确保在文件最顶部全局导入 argparse
import os
import sys
from pathlib import Path

import torch
import html

# 添加项目根目录，便于导入 research.training.*
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 替换掉原来的 `from data.redjujube import RedJujubeDataLoader`
from research.training.redjujube_data_loader import RedJujubeDataLoader  # noqa: E402


def format_entities_in_text(tokens, spans):
    """
    在原文中用 [TYPE]...[/TYPE] 标注实体，便于直观对比。

    tokens:
        - 可以是 TokenSequence（有 raw_text 属性）
        - 也可以是字符/词列表
    spans:
        - 元素形如 (type, start, end) 或 (text, type, start, end)
    """
    # 取出字符序列
    if hasattr(tokens, "raw_text"):
        chars = tokens.raw_text
    else:
        chars = list(tokens)

    n = len(chars)
    norm_spans = []
    for span in spans:
        if len(span) == 3:
            typ, s, e = span
        elif len(span) == 4:
            _, typ, s, e = span
        else:
            continue
        if 0 <= s < e <= n:
            norm_spans.append((typ, s, e))

    # 按起始位置排序，方便插标记
    norm_spans.sort(key=lambda x: (x[1], x[2]))

    start_marks = {i: [] for i in range(n)}
    end_marks = {i: [] for i in range(n)}
    for typ, s, e in norm_spans:
        start_marks[s].append(f"[{typ}]")
        end_marks[e - 1].append(f"[/{typ}]")

    pieces = []
    for i, ch in enumerate(chars):
        for m in start_marks.get(i, []):
            pieces.append(m)
        pieces.append(ch)
        for m in end_marks.get(i, []):
            pieces.append(m)
    return "".join(pieces)


def compare_predictions(
    pred_file1,
    pred_file2,
    num_samples=10,
    show_errors_only=False,
    test_data=None,
    html_out=None,  # 新增：输出 HTML 文件路径
):
    """
    对比两个预测结果文件（例如 baseline vs expert_dict）：

    - 对齐同一条样本（按索引）
    - 显示 Gold / 模型1 / 模型2
    - 标出：
        * baseline 漏检 / 误检
        * 词典模型 漏检 / 误检
        * baseline 正确且词典错误的实体
        * 词典正确且 baseline 错误的实体
    """
    print("\n" + "=" * 70)
    print(f"预测结果文件1: {pred_file1}")
    print(f"预测结果文件2: {pred_file2}")
    print("=" * 70 + "\n")

    data1 = torch.load(pred_file1, map_location="cpu", weights_only=False)
    data2 = torch.load(pred_file2, map_location="cpu", weights_only=False)

    n1, n2 = len(data1), len(data2)
    n = min(n1, n2)
    print(f"文件1样本数: {n1}")
    print(f"文件2样本数: {n2}")
    print(f"按前 {n} 条样本对齐对比\n")

    # 简单统计：每个模型完全正确的样本数
    def count_full_correct(data):
        cnt = 0
        for item in data:
            if isinstance(item, dict) and "chunks" in item and "chunks_pred" in item:
                if set(item["chunks"]) == set(item["chunks_pred"]):
                    cnt += 1
        return cnt

    c1 = count_full_correct(data1)
    c2 = count_full_correct(data2)
    print(f"模型1 完全正确样本: {c1}/{n1} ({c1 / n1 * 100:.2f}%)")
    print(f"模型2 完全正确样本: {c2}/{n2} ({c2 / n2 * 100:.2f}%)\n")

    print("=" * 70)
    print(f"样本详情对比 (最多显示前 {num_samples} 个)")
    print("=" * 70 + "\n")

    html_blocks = [] if html_out is not None else None  # 用于收集 HTML 片段

    shown = 0
    for i in range(n):
        item1 = data1[i]
        item2 = data2[i]

        if not (isinstance(item1, dict) and isinstance(item2, dict)):
            continue
        if "chunks" not in item1 or "chunks_pred" not in item1 or "chunks_pred" not in item2:
            continue

        gold = set(item1["chunks"])
        pred1 = set(item1["chunks_pred"])
        pred2 = set(item2["chunks_pred"])

        model1_correct = (pred1 == gold)
        model2_correct = (pred2 == gold)

        # 只看错误样本：如果两个模型都完全正确，就跳过
        if show_errors_only and model1_correct and model2_correct:
            continue

        if shown >= num_samples:
            break

        # 尝试用样本索引 i + test_data 还原原文
        tokens = None
        doc_idx = item1.get("doc_idx", None)  # 仅用于打印参考
        if test_data is not None and 0 <= i < len(test_data) and "tokens" in test_data[i]:
            tokens = test_data[i]["tokens"]

        print(f"样本 {i+1} (索引 {i}, doc_idx={doc_idx}):")
        print(f"  模型1是否完全正确: {'是' if model1_correct else '否'}")
        print(f"  模型2是否完全正确: {'是' if model2_correct else '否'}")

        raw_text = ""
        if tokens is not None:
            if hasattr(tokens, "raw_text"):
                chars = tokens.raw_text
            else:
                chars = list(tokens)
            raw_text = "".join(chars)

            print(f"  原文: {raw_text}")
            print(f"  Gold: {format_entities_in_text(chars, item1['chunks'])}")
            print(f"  模型1: {format_entities_in_text(chars, item1['chunks_pred'])}")
            print(f"  模型2: {format_entities_in_text(chars, item2['chunks_pred'])}")
        else:
            # 回退只打印 span
            print(f"  Gold: {item1['chunks']}")
            print(f"  模型1预测: {item1['chunks_pred']}")
            print(f"  模型2预测: {item2['chunks_pred']}")

        # 各自的漏检 / 误检
        miss1 = gold - pred1
        wrong1 = pred1 - gold
        miss2 = gold - pred2
        wrong2 = pred2 - gold

        # 小工具：把 span 集合格式化为 “[TYPE] 文本 @ (type,start,end)” 形式
        def _fmt_span_set(span_set, chars_for_text):
            lines = []
            for span in span_set:
                if len(span) == 3:
                    typ, s, e = span
                elif len(span) == 4:
                    _, typ, s, e = span
                else:
                    continue
                if chars_for_text is not None and 0 <= s < e <= len(chars_for_text):
                    text = "".join(chars_for_text[s:e])
                else:
                    text = "<?>"  # 越界或无文本时的兜底
                lines.append(f"[{typ}] {text} @ {span}")
            return lines

        # 如果有原文字符，就打印具体文字；否则退回原来的集合打印
        if tokens is not None:
            if miss1:
                print("    模型1 漏检:")
                for line in _fmt_span_set(miss1, chars):
                    print("      " + line)
            if wrong1:
                print("    模型1 误检:")
                for line in _fmt_span_set(wrong1, chars):
                    print("      " + line)
            if miss2:
                print("    模型2 漏检:")
                for line in _fmt_span_set(miss2, chars):
                    print("      " + line)
            if wrong2:
                print("    模型2 误检:")
                for line in _fmt_span_set(wrong2, chars):
                    print("      " + line)
        else:
            if miss1:
                print(f"    模型1 漏检: {miss1}")
            if wrong1:
                print(f"    模型1 误检: {wrong1}")
            if miss2:
                print(f"    模型2 漏检: {miss2}")
            if wrong2:
                print(f"    模型2 误检: {wrong2}")

        # 重点：基线成功 / 词典失败 & 反之
        # 这里默认 pred_file1 是 baseline，pred_file2 是 词典模型
        only1_correct = (pred1 & gold) - (pred2 & gold)
        only2_correct = (pred2 & gold) - (pred1 & gold)

        if tokens is not None:
            if only1_correct:
                print("    基线正确但词典错误的实体:")
                for line in _fmt_span_set(only1_correct, chars):
                    print("      " + line)
            if only2_correct:
                print("    词典正确但基线错误的实体:")
                for line in _fmt_span_set(only2_correct, chars):
                    print("      " + line)
        else:
            if only1_correct:
                print(f"    基线正确但词典错误的实体: {only1_correct}")
            if only2_correct:
                print(f"    词典正确但基线错误的实体: {only2_correct}")

        # 同时收集 HTML 片段
        if html_blocks is not None:
            block = []
            block.append("<div class='sample'>")
            block.append(
                f"<h3>样本 {i+1} (索引 {i}, doc_idx={html.escape(str(doc_idx))})</h3>"
            )
            if raw_text:
                block.append(
                    "<pre>"
                    + html.escape("原文: " + raw_text)
                    + "</pre>"
                )
                block.append(
                    "<pre>"
                    + html.escape(
                        "Gold: " + format_entities_in_text(chars, item1["chunks"])
                    )
                    + "</pre>"
                )
                block.append(
                    "<pre>"
                    + html.escape(
                        "模型1: " + format_entities_in_text(chars, item1["chunks_pred"])
                    )
                    + "</pre>"
                )
                block.append(
                    "<pre>"
                    + html.escape(
                        "模型2: " + format_entities_in_text(chars, item2["chunks_pred"])
                    )
                    + "</pre>"
                )
            else:
                block.append(
                    "<pre>" + html.escape(f"Gold: {item1['chunks']}") + "</pre>"
                )
                block.append(
                    "<pre>"
                    + html.escape(f"模型1预测: {item1['chunks_pred']}")
                    + "</pre>"
                )
                block.append(
                    "<pre>"
                    + html.escape(f"模型2预测: {item2['chunks_pred']}")
                    + "</pre>"
                )

            def _add_span_section(title, span_set):
                if not span_set:
                    return
                block.append(f"<p>{html.escape(title)}</p>")
                block.append("<ul>")
                for line in _fmt_span_set(span_set, chars if tokens is not None else None):
                    block.append("<li>" + html.escape(line) + "</li>")
                block.append("</ul>")

            if tokens is not None:
                _add_span_section("模型1 漏检", miss1)
                _add_span_section("模型1 误检", wrong1)
                _add_span_section("模型2 漏检", miss2)
                _add_span_section("模型2 误检", wrong2)
                _add_span_section("基线正确但词典错误的实体", only1_correct)
                _add_span_section("词典正确但基线错误的实体", only2_correct)

            block.append("</div>")
            html_blocks.append("\n".join(block))

        print()
        shown += 1

    if show_errors_only:
        print("(只显示存在错误的样本)")
    print("=" * 70 + "\n")

    # 写出 HTML 文件
    if html_blocks:
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'/>",
            "<title>Baseline vs 词典模型 对比</title>",
            "<style>",
            "body { font-family: Menlo, monospace; }",
            ".sample { border: 1px solid #ccc; padding: 8px; margin: 8px 0; }",
            "pre { background: #f7f7f7; padding: 4px; overflow-x: auto; }",
            "</style>",
            "</head>",
            "<body>",
        ]
        html_content.extend(html_blocks)
        html_content.append("</body></html>")

        with open(html_out, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))
        print(f"已导出对比结果到 HTML 文件: {html_out}\n")


def main():
    parser = argparse.ArgumentParser(description="查看预测结果文件")
    parser.add_argument("pred_file", type=str, help="预测结果文件路径（例如 baseline）")
    parser.add_argument(
        "-n", "--num_samples", type=int, default=10, help="显示样本数量"
    )
    parser.add_argument(
        "--errors_only", action="store_true", help="只显示预测错误的样本"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="（可选）原始数据目录，用于还原文本（RedJujube: datasets/raw/RedJujube）",
    )
    parser.add_argument(
        "--compare_with",
        type=str,
        default=None,
        help="（可选）第二个预测结果文件路径，用于与 pred_file 做对比（如词典模型）",
    )
    parser.add_argument(
        "--html_out",
        type=str,
        default=None,
        help="（可选）将对比结果导出为 HTML 文件路径",
    )

    args = parser.parse_args()

    if not Path(args.pred_file).exists():
        print(f"❌ 文件不存在: {args.pred_file}")
        return

    # 如果提供了 data_dir，则按 RedJujube 的方式加载测试集
    test_data = None
    if args.data_dir is not None:
        loader = RedJujubeDataLoader(args.data_dir)
        _, _, test_data = loader.load_data()
        print(f"已从 {args.data_dir} 加载测试集，共 {len(test_data)} 条样本\n")

    # 如果提供了第二个预测文件，走“对比模式”
    if args.compare_with is not None:
        if not Path(args.compare_with).exists():
            print(f"❌ 文件不存在: {args.compare_with}")
            return
        compare_predictions(
            args.pred_file,
            args.compare_with,
            num_samples=args.num_samples,
            show_errors_only=args.errors_only,
            test_data=test_data,
            html_out=args.html_out,
        )
    else:
        # 默认单文件查看模式（目前不导出 HTML）
        view_predictions(
            args.pred_file,
            num_samples=args.num_samples,
            show_errors_only=args.errors_only,
            test_data=test_data,
        )


if __name__ == "__main__":
    main()
