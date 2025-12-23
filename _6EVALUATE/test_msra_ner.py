#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSRA NER 专用详细测试与错误分析脚本

功能：
- 加载训练好的 MSRA 模型（config + model）
- 按训练时一致的数据预处理流程构建 MSRA 测试集
- 计算宏/微平均 P/R/F1 以及各实体类型的详细指标
- 统计 Top 漏检/误检实体，并输出若干典型错误样本
- 可选：导出预测结果文件，配合 _6EVALUATE/view_predictions.py 做人工检查

适用场景：
- 使用 scripts/entity_recognition.py 在 MSRA-ER 上训练的模型
  （典型目录结构：cache/MSRA-ER/xxxxxx-xxxxxx/xxx-config.pth + xxx.pth + results.json）
"""

import argparse
import json
import os
import sys
from collections import Counter

import torch

# 添加项目根目录到 sys.path，才能导入 eznlp.*
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from eznlp.dataset import Dataset
from eznlp.metrics import precision_recall_f1_report
from eznlp.training import Trainer
from eznlp.io import ConllIO


def span_text(entry, span):
    """
    span 可能有两种格式：
      (type, start, end) 或 (text, type, start, end)
    返回: (type, text)
    """
    if len(span) == 3:
        typ, s, e = span
    elif len(span) == 4:
        # 假设格式为 (text, type, start, end)
        _, typ, s, e = span
    else:
        return "UNK", "<?>"

    tokens = entry["tokens"]
    text = "".join(tokens.raw_text[s:e])
    return typ, text


def format_entities_in_text(entry, spans):
    """
    在原文上标注实体，格式类似：
    原文：中国石化集团公司主营汽油、煤油、柴油等石油产品的生产和销售。
    Gold：在[ORG]中国石化集团公司[/ORG]主营汽油、煤油、柴油等石油产品的生产和销售。

    spans 中元素可以是 (type, start, end) 或 (text, type, start, end)。
    """
    tokens = entry["tokens"].raw_text
    n = len(tokens)

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

    norm_spans.sort(key=lambda x: (x[1], x[2]))

    start_marks = {i: [] for i in range(n)}
    end_marks = {i: [] for i in range(n)}
    for typ, s, e in norm_spans:
        start_marks[s].append(f"[{typ}]")
        end_marks[e - 1].append(f"[/{typ}]")  # e 是右开区间

    pieces = []
    for i, ch in enumerate(tokens):
        for m in start_marks.get(i, []):
            pieces.append(m)
        pieces.append(ch)
        for m in end_marks.get(i, []):
            pieces.append(m)

    return "".join(pieces)


def load_config_and_model(save_dir, device):
    """
    从 save_dir 中自动寻找 *-config.pth 和对应的 *.pth 模型并加载。
    兼容 scripts/entity_recognition.py 在 MSRA-ER 上的默认命名方式。
    """
    files = os.listdir(save_dir)

    config_files = [f for f in files if f.endswith("-config.pth")]
    if not config_files:
        raise FileNotFoundError(f"未在目录中找到 *-config.pth 文件: {save_dir}")
    if len(config_files) > 1:
        raise RuntimeError(
            f"在目录 {save_dir} 中找到多个 *-config.pth，"
            f"请确保一次只评估一个模型，或手动指定目录。"
        )

    config_file = config_files[0]
    model_file = config_file.replace("-config.pth", ".pth")
    if model_file not in files:
        raise FileNotFoundError(
            f"未找到模型文件: {model_file} (目录: {save_dir})，"
            f"请确认训练脚本是否保存了对应的 .pth 模型。"
        )

    config_path = os.path.join(save_dir, config_file)
    model_path = os.path.join(save_dir, model_file)

    print(f"加载配置: {config_path}")
    print(f"加载模型: {model_path}")

    # 显式 weights_only=False，兼容新版 torch 的安全默认值
    config = torch.load(config_path, map_location=device, weights_only=False)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)

    return config, model, config_path, model_path


def load_msra_data(data_dir="data/MSRA"):
    """
    加载 MSRA-ER 数据集（BMES 字符级标注）。

    data_dir 下期望存在：
      - hz_train.bmes
      - hz_dev.bmes
      - hz_test.bmes

    若你的数据路径不同，可通过命令行参数 --data_dir 覆盖。
    """
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="",
    )

    train_path = os.path.join(data_dir, "hz_train.bmes")
    dev_path = os.path.join(data_dir, "hz_dev.bmes")
    test_path = os.path.join(data_dir, "hz_test.bmes")

    train_data = io.read(train_path)
    dev_data = io.read(dev_path)
    test_data = io.read(test_path)

    return train_data, dev_data, test_data


def infer_args_from_results(save_dir):
    """
    从 results.json 恢复 data_dir / batch_size 等关键信息，并返回完整 args 字典。
    若不存在 results.json，则使用默认值。
    """
    results_path = os.path.join(save_dir, "results.json")
    data_dir = "data/MSRA"
    batch_size = 32
    args = {}

    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        args = results.get("args", {}) or {}
        data_dir = args.get("data_dir", data_dir)
        batch_size = args.get("batch_size", batch_size)

        # 某些脚本只存 dataset，不存 data_dir，这里兜底一下
        dataset = args.get("dataset")
        if dataset and dataset.lower() == "msra" and "data_dir" not in args:
            data_dir = "data/MSRA"

    return data_dir, batch_size, args


def evaluate_msra_and_analyze(save_dir, data_dir=None, export_predictions=True):
    """
    MSRA 模型详细测试与错误分析：
    - 加载 MSRA-ER 训练出的模型（config + model）
    - 按 MSRA 训练脚本一致的数据预处理构建测试集
    - 计算宏/微平均 P/R/F1 及各类型指标
    - 统计 Top 漏检/误检实体 + 打印典型错误样本
    - 可选：导出 predictions_test.pt，配合 view_predictions.py 查看
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, model, _, _ = load_config_and_model(save_dir, device)
    inferred_data_dir, batch_size, args = infer_args_from_results(save_dir)

    if data_dir is None:
        data_dir = inferred_data_dir

    print(f"使用 data_dir = {data_dir}")
    print(f"使用 batch_size = {batch_size}")
    print("数据集: MSRA-ER")

    # 1. 构建 MSRA 测试集
    _, _, test_data = load_msra_data(data_dir)
    test_set = Dataset(test_data, config, training=False)

    # 2. 预测
    trainer = Trainer(model, device=device)
    print("\n===== MSRA 测试集预测 =====")
    pred_chunks = trainer.predict(test_set, batch_size=batch_size)
    gold_chunks = [ex["chunks"] for ex in test_set.data]

    # 3. 计算指标
    scores, ave_scores = precision_recall_f1_report(
        gold_chunks, pred_chunks, macro_over="types"
    )

    print("\n===== 宏 / 微 平均指标 (MSRA) =====")
    macro = ave_scores["macro"]
    micro = ave_scores["micro"]
    print(f"Macro: P={macro['precision']:.4f} R={macro['recall']:.4f} F1={macro['f1']:.4f}")
    print(f"Micro: P={micro['precision']:.4f} R={micro['recall']:.4f} F1={micro['f1']:.4f}")

    print("\n===== 各实体类型指标 (MSRA) =====")
    print(f"{'Type':<15} {'P':>8} {'R':>8} {'F1':>8} {'Gold':>8} {'Pred':>8} {'TP':>8}")
    for t, s in sorted(scores.items(), key=lambda kv: kv[0]):
        print(
            f"{t:<15} {s['precision']:.4f} {s['recall']:.4f} {s['f1']:.4f} "
            f"{s['n_gold']:>8} {s['n_pred']:>8} {s['n_true_positive']:>8}"
        )

    # 4. 错误分析
    miss_counter = Counter()   # (type, text) -> count
    wrong_counter = Counter()  # (type, text) -> count
    error_cases = []           # 存储有错误的样本

    for idx, (entry, g_list, p_list) in enumerate(
        zip(test_set.data, gold_chunks, pred_chunks)
    ):
        g_set, p_set = set(g_list), set(p_list)
        if g_set == p_set:
            continue

        missed = g_set - p_set
        wrong = p_set - g_set

        case_info = {
            "index": idx,
            "text": "".join(entry["tokens"].raw_text),
            "tokens": entry["tokens"],
            "gold": g_list,
            "pred": p_list,
            "missed": [],
            "wrong": [],
        }

        for span in missed:
            typ, txt = span_text(entry, span)
            miss_counter[(typ, txt)] += 1
            case_info["missed"].append((typ, txt, span))

        for span in wrong:
            typ, txt = span_text(entry, span)
            wrong_counter[(typ, txt)] += 1
            case_info["wrong"].append((typ, txt, span))

        error_cases.append(case_info)

    print(f"\n样本级有错误的数量: {len(error_cases)} / {len(test_set.data)}")

    print("\n===== Top 20 漏检实体 (type, text, count) =====")
    for (typ, txt), c in miss_counter.most_common(20):
        print(f"[{typ}] {txt} : {c}")

    print("\n===== Top 20 误检实体 (type, text, count) =====")
    for (typ, txt), c in wrong_counter.most_common(20):
        print(f"[{typ}] {txt} : {c}")

    print("\n===== 典型错误案例（前 5 条，MSRA） =====")
    for case in error_cases[:5]:
        print("-" * 60)
        print(f"样本索引: {case['index']}")
        print(f"原文: {case['text']}")
        print("Gold 标注:", format_entities_in_text(case, case["gold"]))
        print("Pred 预测:", format_entities_in_text(case, case["pred"]))
        if case["missed"]:
            print("漏检实体列表:")
            for typ, txt, span in case["missed"]:
                print(f"  [{typ}] {txt} @ {span}")
        if case["wrong"]:
            print("误检实体列表:")
            for typ, txt, span in case["wrong"]:
                print(f"  [{typ}] {txt} @ {span}")
        print()

    # 5. 导出预测结果，方便用 view_predictions.py 进一步查看
    if export_predictions:
        pred_dump_path = os.path.join(save_dir, "predictions_test_msra.pt")
        pred_dump = []
        for entry, p in zip(test_set.data, pred_chunks):
            d = {k: v for k, v in entry.items() if k != "tokens"}
            d["chunks_pred"] = p
            pred_dump.append(d)

        torch.save(pred_dump, pred_dump_path)
        print(f"\n已保存 MSRA 预测详情到: {pred_dump_path}")
        print("可以用以下命令查看错误样本：")
        print(
            f"  python _6EVALUATE/view_predictions.py "
            f"{pred_dump_path} --errors_only -n 20"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="MSRA NER 模型详细测试与错误分析"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="cache/MSRA-ER/20250822-195241-646229",
        help="训练输出目录（包含 *-config.pth / *.pth / results.json）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/MSRA",
        help="MSRA 数据目录（包含 hz_train.bmes / hz_dev.bmes / hz_test.bmes）",
    )
    parser.add_argument(
        "--no_export_predictions",
        action="store_true",
        help="不导出 predictions_test_msra.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_msra_and_analyze(
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        export_predictions=not args.no_export_predictions,
    )