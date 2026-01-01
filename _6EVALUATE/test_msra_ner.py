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
from eznlp.token import LexiconTokenizer
from eznlp.model import BertLikePreProcessor  # 新增：用于 BERT 预处理（句子切分）


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


def load_msra_data(data_dir="data/MSRA", prefix=""):
    """
    加载 BMES 字符级标注数据集。

    data_dir 下期望存在：
      - {prefix}train.bmes
      - {prefix}dev.bmes
      - {prefix}test.bmes

    MSRA 默认 prefix="" 且文件名为 train.char.bmes / dev.char.bmes / test.char.bmes，
    RedJujube 可使用 prefix="redjujube_".
    """
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="",
    )

    if prefix:
        train_name = f"{prefix}train.bmes"
        dev_name   = f"{prefix}dev.bmes"
        test_name  = f"{prefix}test.bmes"
    else:
        train_name = "train.char.bmes"
        dev_name   = "dev.char.bmes"
        test_name  = "test.char.bmes"

    train_path = os.path.join(data_dir, train_name)
    dev_path   = os.path.join(data_dir, dev_name)
    test_path  = os.path.join(data_dir, test_name)

    train_data = io.read(train_path)
    dev_data   = io.read(dev_path)
    test_data  = io.read(test_path)
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

    # 打印模型结构
    print("\n===== 模型结构 (MSRA-ER) =====")
    print(model)

    inferred_data_dir, batch_size, args = infer_args_from_results(save_dir)

    if data_dir is None:
        data_dir = inferred_data_dir

    print(f"使用 data_dir = {data_dir}")
    print(f"使用 batch_size = {batch_size}")

    # 简单根据路径名判断：RedJujube 或 MSRA
    if "RedJujube" in data_dir:
        print("数据集: RedJujube-ER（BMES）")
        train_data, dev_data, test_data = load_msra_data(data_dir, prefix="redjujube_")
    else:
        print("数据集: MSRA-ER")
        train_data, dev_data, test_data = load_msra_data(data_dir)

    # 1.5 先做与训练一致的 BERT 句子切分（避免超长）
    if getattr(config, "bert_like", None) is not None:
        # 优先从训练 args 中恢复 bert_max_length，其次从 config 里取，最后默认 512
        bert_max_len = args.get("bert_max_length", getattr(config, "bert_max_length", 512))

        preprocessor = BertLikePreProcessor(
            config.bert_like.tokenizer,
            model_max_length=bert_max_len,
            verbose=True,
        )
        test_data = preprocessor.segment_sentences_for_data(
            test_data, update_raw_idx=True
        )

    # 2. 专家词典特征：本脚本改为“只测试 ExpertDict 架构”
    nested_ohots = getattr(config, "nested_ohots", None)

    # 强制要求存在 expert_dict 通道，否则报错
    if not (nested_ohots is not None and hasattr(nested_ohots, "keys") and "expert_dict" in nested_ohots.keys()):
        raise ValueError(
            "当前模型配置中未找到 nested_ohots['expert_dict']，"
            "本测试脚本仅适用于带 ExpertDict 特征的 MSRA-ER 模型。"
        )

    print("\n[ExpertDict] 检测到 ExpertDict 特征通道，开始加载专家词典...")
    # 优先使用命令行参数指定的路径
    cli_args = parse_args()
    if hasattr(cli_args, 'expert_dict_path') and cli_args.expert_dict_path:
        expert_dict_path = cli_args.expert_dict_path
    else:
        # 其次从训练 args 中恢复
        expert_dict_path = args.get("expert_dict_path") or args.get("expert_dict_auto_path")
    
    if not expert_dict_path or not os.path.exists(expert_dict_path):
        # 检测是否是带类型的模型（从路径判断）
        if "typed" in save_dir:
            expert_dict_path = os.path.join(data_dir, "expert_lexicon_typed.txt")
        else:
            expert_dict_path = os.path.join(data_dir, "expert_lexicon_auto.txt")

    print(f"[ExpertDict] 加载专家词典: {expert_dict_path}")
    lexicon = []
    with open(expert_dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            word = parts[0]
            if word:
                # 如果有类型信息，拼接成 word_TYPE
                if len(parts) >= 2 and parts[1]:
                    word = f"{word}_{parts[1]}"
                lexicon.append(word)
    print(f"[ExpertDict] 词典大小: {len(lexicon)}")

    tokenizer = LexiconTokenizer(lexicon, max_len=10)
    print("[ExpertDict] 为 train/dev/test 添加 expert_dict 特征...")
    for data in (train_data, dev_data, test_data):
        for entry in data:
            entry["tokens"].build_expert_dict_tags(tokenizer.tokenize)

    # 如需完全对齐训练时的频率统计，可在此补充，但默认不再重复构建
    # ed_cfg = nested_ohots["expert_dict"]
    # if hasattr(ed_cfg, "build_freqs"):
    #     print("[ExpertDict] 构建 expert_dict 词频统计...")
    #     ed_cfg.build_freqs(train_data, dev_data)

    # 3. 构建测试集 Dataset
    test_set = Dataset(test_data, config, training=False)

    # 4. 预测
    trainer = Trainer(model, device=device)
    print("\n===== MSRA 测试集预测 =====")
    pred_chunks = trainer.predict(test_set, batch_size=batch_size)
    gold_chunks = [ex["chunks"] for ex in test_set.data]

    # 5. 计算指标
    scores, ave_scores = precision_recall_f1_report(
        gold_chunks, pred_chunks, macro_over="types"
    )

    print("\n===== 宏 / 微 平均指标 (MSRA) =====")
    macro = ave_scores["macro"]
    micro = ave_scores["micro"]
    print(f"Macro: P={macro['precision']:.4f} R={macro['recall']:.4f} F1={macro['f1']:.4f}")
    print(f"Micro: P={micro['precision']:.4f} R={micro['recall']:.4f} F1={micro['f1']:.4f}")

    # 各实体类型指标：NR/NS/NT 等，列宽统一对齐
    print("\n===== 各实体类型指标 (MSRA) =====")
    print(f"{'Type':<6} {'P':>9} {'R':>9} {'F1':>9} {'Gold':>8} {'Pred':>8} {'TP':>8}")

    # 若想优先按 NR/NS/NT 顺序展示
    order = ["NR", "NS", "NT"]
    types_in_scores = list(scores.keys())
    ordered_types = [t for t in order if t in types_in_scores] + [
        t for t in sorted(types_in_scores) if t not in order
    ]

    for t in ordered_types:
        s = scores[t]
        print(
            f"{t:<6} "
            f"{s['precision']:>9.4f} {s['recall']:>9.4f} {s['f1']:>9.4f} "
            f"{s['n_gold']:>8d} {s['n_pred']:>8d} {s['n_true_positive']:>8d}"
        )

    # 6. 错误分析
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
    parser = argparse.ArgumentParser(description="MSRA/RedJujube NER 模型测试")
    parser.add_argument("--save_dir", type=str, required=True, help="模型保存目录")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录")
    parser.add_argument("--expert_dict_path", type=str, default=None, help="专家词典路径（可选，覆盖默认）")
    parser.add_argument("--no_export_predictions", action="store_true", help="不导出预测结果")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_msra_and_analyze(
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        export_predictions=not args.no_export_predictions,
    )