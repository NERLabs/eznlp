#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube Baseline 模型详细测试与错误分析脚本

功能：
- 加载训练好的 Baseline 模型（config + model）
- 按训练时一致的数据预处理流程构建 RedJujube 测试集
- 计算宏/微平均 P/R/F1 以及各实体类型的详细指标
- 统计 Top 漏检/误检实体，并输出若干典型错误样本
- 可选：导出预测结果文件，配合 _6EVALUATE/view_predictions.py 做人工检查
"""

import argparse
import json
import os
from collections import Counter

import torch
import sys


# 添加项目根目录到 sys.path，才能导入 _5TRAIN.*
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from eznlp.dataset import Dataset
from eznlp.metrics import precision_recall_f1_report
from eznlp.training import Trainer
from _5TRAIN.redjujube_data_loader import RedJujubeDataLoader, DataPreparationPipeline


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
    # RedJujube 是字级别标注，直接拼接字符
    text = "".join(tokens.raw_text[s:e])
    return typ, text

def format_entities_in_text(entry, spans):
    """
    在原文上标注实体，格式类似：
    原文：在灵宝产区,4月中旬萌芽...
    Gold：在[GEO]灵宝[/GEO]产区,4月中旬萌芽...

    spans 中元素可以是 (type, start, end) 或 (text, type, start, end)。
    """
    tokens = entry["tokens"].raw_text
    n = len(tokens)

    # 统一格式，并按起止位置排序
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

    # 在 token 级别打开始/结束标记
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
    files = os.listdir(save_dir)

    config_files = [f for f in files if f.endswith("-config.pth")]
    if not config_files:
        raise FileNotFoundError(f"未在目录中找到 *-config.pth 文件: {save_dir}")
    config_file = config_files[0]

    model_file = config_file.replace("-config.pth", ".pth")
    if model_file not in files:
        raise FileNotFoundError(f"未找到模型文件: {model_file} (目录: {save_dir})")

    config_path = os.path.join(save_dir, config_file)
    model_path = os.path.join(save_dir, model_file)

    print(f"加载配置: {config_path}")
    print(f"加载模型: {model_path}")

    # 显式 weights_only=False，兼容新版 torch 的安全默认值
    config = torch.load(config_path, map_location=device, weights_only=False)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)

    return config, model, config_path, model_path


def infer_args_from_results(save_dir):
    """
    从 results.json 恢复 data_dir / batch_size 等关键信息，并返回完整 args 字典
    """
    results_path = os.path.join(save_dir, "results.json")
    data_dir = "_2DATA/RedJujube"
    batch_size = 16
    args = {}

    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        args = results.get("args", {}) or {}
        data_dir = args.get("data_dir", data_dir)
        batch_size = args.get("batch_size", batch_size)

    return data_dir, batch_size, args


def evaluate_and_analyze(save_dir, model_type="baseline", export_predictions=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, model, _, _ = load_config_and_model(save_dir, device)
    data_dir, batch_size, args = infer_args_from_results(save_dir)

    print(f"使用 data_dir = {data_dir}")
    print(f"使用 batch_size = {batch_size}")
    print(f"使用 model_type = {model_type}")

    # 1. 构建 RedJujube 测试集（按指定模型类型流程）
    loader = RedJujubeDataLoader(data_dir)
    pipeline = DataPreparationPipeline(loader)

    if model_type == "baseline":
        train_data, dev_data, test_data = pipeline.prepare("baseline")

    elif model_type in ["expert_dict", "expert_dict_auto", "expert_dict_manual"]:
        # 与训练脚本保持一致：auto 用 expert_dict_auto_path，其它用 expert_dict_path
        if "auto" in model_type:
            expert_dict_path = args.get(
                "expert_dict_auto_path",
                "_2DATA/RedJujube/expert_lexicon_auto.txt",
            )
        else:
            expert_dict_path = args.get(
                "expert_dict_path",
                "_2DATA/RedJujube/expert_lexicon.txt",
            )
        print(f"使用 expert_dict_path = {expert_dict_path}")
        train_data, dev_data, test_data = pipeline.prepare(
            model_type, expert_dict_path=expert_dict_path
        )

    else:
        raise NotImplementedError(
            f"当前测试脚本暂未支持的模型类型: {model_type}，"
            f"如需支持请在 evaluate_and_analyze 中补充相应分支。"
        )

    test_set = Dataset(test_data, config, training=False)

    # 2. 预测
    trainer = Trainer(model, device=device)
    print("\n===== 预测测试集 =====")
    pred_chunks = trainer.predict(test_set, batch_size=batch_size)
    gold_chunks = [ex["chunks"] for ex in test_set.data]

    # 3. 计算指标
    scores, ave_scores = precision_recall_f1_report(
        gold_chunks, pred_chunks, macro_over="types"
    )

    print("\n===== 宏 / 微 平均指标 =====")
    macro = ave_scores["macro"]
    micro = ave_scores["micro"]
    print(f"Macro: P={macro['precision']:.4f} R={macro['recall']:.4f} F1={macro['f1']:.4f}")
    print(f"Micro: P={micro['precision']:.4f} R={micro['recall']:.4f} F1={micro['f1']:.4f}")

    print("\n===== 各实体类型指标 =====")
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

    print("\n===== 典型错误案例（前 5 条） =====")
    for case in error_cases[:100]:
        print("-" * 60)
        print(f"样本索引: {case['index']}")
        print(f"原文: {case['text']}")
        # 在文本中高亮 Gold / Pred 实体，便于对比
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
        pred_dump_path = os.path.join(save_dir, "predictions_test.pt")
        pred_dump = []
        for entry, p in zip(test_set.data, pred_chunks):
            d = {k: v for k, v in entry.items() if k != "tokens"}
            d["chunks_pred"] = p
            pred_dump.append(d)

        torch.save(pred_dump, pred_dump_path)
        print(f"\n已保存预测详情到: {pred_dump_path}")
        print("可以用以下命令查看错误样本：")
        print(
            f"  python _6EVALUATE/view_predictions.py "
            f"{pred_dump_path} --errors_only -n 20"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="RedJujube 模型详细测试与错误分析"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="cache/redjujube_expert_dict_auto_new/expert_dict_auto_20251215-204832",
        help="训练输出目录（包含 *-config.pth / *.pth / results.json）",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="expert_dict_auto",
        help="模型类型，例如: baseline, expert_dict_auto, softlexicon 等",
    )
    parser.add_argument(
        "--no_export_predictions",
        action="store_true",
        help="不导出 predictions_test.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_and_analyze(
        save_dir=args.save_dir,
        model_type=args.model_type,
        export_predictions=not args.no_export_predictions,
    )