#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用模型评估脚本

支持两种模型类型：
1. CRF 模型：文件名包含 BERT-LSTM-BMES-CRF
2. BS 模型（边界选择）：文件名包含 BERT-FFN-SB

使用方法：
    python evaluate_all_models.py --save_dir <模型目录> [--model_type auto|crf|bs]
"""

import argparse
import json
import os
import sys

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from eznlp.io import ConllIO
from eznlp.dataset import Dataset
from eznlp.metrics import precision_recall_f1_report
from eznlp.training import Trainer
from eznlp.token import LexiconTokenizer


def detect_model_type(save_dir):
    """
    自动检测模型类型（通过查看 pth 文件名）
    
    Returns:
        str: "crf" 或 "bs"
    """
    files = os.listdir(save_dir)
    pth_files = [f for f in files if f.endswith(".pth") and not f.endswith("-config.pth")]
    
    if not pth_files:
        raise FileNotFoundError(f"未找到模型文件 (*.pth): {save_dir}")
    
    model_file = pth_files[0]
    
    if "CRF" in model_file:
        return "crf"
    elif "SB" in model_file:
        return "bs"
    else:
        raise ValueError(f"无法从文件名判断模型类型: {model_file}")


def detect_expert_dict_usage(save_dir):
    """
    检测模型是否使用了专家词典
    
    Returns:
        bool: 是否使用专家词典
    """
    files = os.listdir(save_dir)
    pth_files = [f for f in files if f.endswith(".pth") and not f.endswith("-config.pth")]
    
    if not pth_files:
        return False
    
    model_file = pth_files[0]
    return "expert_dict" in model_file


def load_config_and_model(save_dir, device):
    """
    加载模型配置和权重
    """
    files = os.listdir(save_dir)
    
    config_files = [f for f in files if f.endswith("-config.pth")]
    if not config_files:
        raise FileNotFoundError(f"未找到配置文件 (*-config.pth): {save_dir}")
    config_file = config_files[0]
    
    model_file = config_file.replace("-config.pth", ".pth")
    if model_file not in files:
        raise FileNotFoundError(f"未找到模型文件: {model_file}")
    
    config_path = os.path.join(save_dir, config_file)
    model_path = os.path.join(save_dir, model_file)
    
    print(f"加载配置: {config_path}")
    print(f"加载模型: {model_path}")
    
    config = torch.load(config_path, map_location=device, weights_only=False)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    return config, model


def infer_args_from_results(save_dir):
    """
    从 results.json 恢复参数
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


def load_data_conllio(data_dir):
    """
    使用 ConllIO 加载 RedJujube 数据集
    """
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="<pad>",
    )
    train_data = io.read(os.path.join(data_dir, "redjujube_train.bmes"))
    dev_data = io.read(os.path.join(data_dir, "redjujube_dev.bmes"))
    test_data = io.read(os.path.join(data_dir, "redjujube_test.bmes"))
    return train_data, dev_data, test_data


def load_expert_lexicon(dict_path):
    """
    加载专家词典
    """
    lexicon = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            word = parts[0]
            if word:
                lexicon.append(word)
    return lexicon


def add_expert_dict_features(data_partitions, lexicon, max_len=10):
    """
    为数据添加专家词典特征
    """
    tokenizer = LexiconTokenizer(lexicon, max_len=max_len)
    for partition in data_partitions:
        for entry in partition:
            entry["tokens"].build_expert_dict_tags(tokenizer.tokenize)


def truncate_long_sequences(datasets, max_char_len=510):
    """
    截断过长序列
    """
    num_truncated = 0
    for data in datasets:
        for entry in data:
            tokens = entry["tokens"]
            if len(tokens) > max_char_len:
                num_truncated += 1
                entry["tokens"] = tokens[:max_char_len]
                chunks = entry.get("chunks", [])
                new_chunks = []
                for label, start, end in chunks:
                    if end <= max_char_len:
                        new_chunks.append((label, start, end))
                entry["chunks"] = new_chunks
    return num_truncated


def prepare_test_data(data_dir, use_expert_dict, expert_dict_path=None):
    """
    准备测试数据
    
    Args:
        data_dir: 数据目录
        use_expert_dict: 是否使用专家词典
        expert_dict_path: 专家词典路径
    
    Returns:
        test_data: 测试数据列表
    """
    train_data, dev_data, test_data = load_data_conllio(data_dir)
    
    # 截断过长序列
    truncate_long_sequences([train_data, dev_data, test_data], max_char_len=510)
    
    if use_expert_dict and expert_dict_path:
        lexicon = load_expert_lexicon(expert_dict_path)
        print(f"加载专家词典: {len(lexicon)} 个词")
        add_expert_dict_features([train_data, dev_data, test_data], lexicon)
        print("专家词典特征添加完成")
    
    return test_data


def evaluate_model(save_dir, model_type="auto", expert_dict_path=None):
    """
    评估模型
    
    Args:
        save_dir: 模型保存目录
        model_type: 模型类型 (auto/crf/bs)
        expert_dict_path: 专家词典路径（覆盖默认路径）
    
    Returns:
        dict: 评估结果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 自动检测模型类型
    if model_type == "auto":
        model_type = detect_model_type(save_dir)
    
    # 检测是否使用专家词典
    use_expert_dict = detect_expert_dict_usage(save_dir)
    
    print(f"\n{'=' * 70}")
    print(f"模型目录: {save_dir}")
    print(f"模型类型: {model_type.upper()}")
    print(f"使用专家词典: {use_expert_dict}")
    print(f"{'=' * 70}")
    
    # 加载配置和模型
    config, model = load_config_and_model(save_dir, device)
    data_dir, batch_size, args = infer_args_from_results(save_dir)
    
    print(f"数据目录: {data_dir}")
    print(f"批次大小: {batch_size}")
    
    # 确定专家词典路径
    if use_expert_dict:
        if expert_dict_path is None:
            # 从 args 中获取或使用默认路径
            expert_dict_path = args.get("expert_dict_path")
            if expert_dict_path is None:
                # 检查是否有 auto_lexicon.txt
                auto_lexicon_path = os.path.join(save_dir, "auto_lexicon.txt")
                if os.path.exists(auto_lexicon_path):
                    expert_dict_path = auto_lexicon_path
                else:
                    expert_dict_path = "_2DATA/RedJujube/expert_lexicon_auto_min1.txt"
        print(f"专家词典路径: {expert_dict_path}")
    
    # 准备测试数据
    test_data = prepare_test_data(
        data_dir,
        use_expert_dict=use_expert_dict,
        expert_dict_path=expert_dict_path if use_expert_dict else None
    )
    
    # 构建测试集
    test_set = Dataset(test_data, config, training=False)
    print(f"测试集样本数: {len(test_set.data)}")
    
    # 预测
    trainer = Trainer(model, device=device)
    print("\n正在预测测试集...")
    pred_chunks = trainer.predict(test_set, batch_size=batch_size)
    gold_chunks = [ex["chunks"] for ex in test_set.data]
    
    # 计算指标
    scores, ave_scores = precision_recall_f1_report(
        gold_chunks, pred_chunks, macro_over="types"
    )
    
    # 输出结果
    print("\n" + "=" * 70)
    print("整体评估指标")
    print("=" * 70)
    
    macro = ave_scores["macro"]
    micro = ave_scores["micro"]
    
    print(f"\nMicro Average: P={micro['precision']:.4f}  R={micro['recall']:.4f}  F1={micro['f1']:.4f}")
    print(f"Macro Average: P={macro['precision']:.4f}  R={macro['recall']:.4f}  F1={macro['f1']:.4f}")
    
    print("\n" + "=" * 70)
    print("各实体类别评估指标")
    print("=" * 70)
    print(f"\n{'Type':<15} {'P':>8} {'R':>8} {'F1':>8} {'Gold':>8} {'Pred':>8} {'TP':>8}")
    print("-" * 70)
    
    for entity_type, s in sorted(scores.items(), key=lambda kv: kv[0]):
        print(
            f"{entity_type:<15} {s['precision']:>8.4f} {s['recall']:>8.4f} {s['f1']:>8.4f} "
            f"{s['n_gold']:>8} {s['n_pred']:>8} {s['n_true_positive']:>8}"
        )
    
    print("-" * 70)
    print(f"{'Total':<15} {micro['precision']:>8.4f} {micro['recall']:>8.4f} {micro['f1']:>8.4f}")
    
    # 返回结果
    return {
        "save_dir": save_dir,
        "model_type": model_type,
        "use_expert_dict": use_expert_dict,
        "micro": micro,
        "macro": macro,
        "scores_by_type": scores,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="通用模型评估脚本 - 支持 CRF 和 BS 模型"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="模型保存目录",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "crf", "bs"],
        help="模型类型 (auto=自动检测)",
    )
    parser.add_argument(
        "--expert_dict_path",
        type=str,
        default=None,
        help="专家词典路径（覆盖默认路径）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_model(
        save_dir=args.save_dir,
        model_type=args.model_type,
        expert_dict_path=args.expert_dict_path,
    )
    
    print("\n" + "=" * 70)
    print("评估完成!")
    print(f"Micro F1: {results['micro']['f1']:.4f}")
    print("=" * 70)
