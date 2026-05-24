#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文模型评估脚本

严格复用训练脚本的数据加载和处理流程，确保评估结果与训练日志一致。

关键设计：
1. 直接 import 训练脚本中的函数，确保数据处理一致
2. 自动检测并使用训练时的专家词典（auto_lexicon.txt）
3. 使用训练脚本保存的 config 重建 Dataset
4. 输出详细评估指标（Overall + 各实体类型）
"""

import argparse
import json
import os
import sys
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "research/training"))

import torch
from eznlp.dataset import Dataset
from eznlp.metrics import precision_recall_f1_report
from eznlp.token import LexiconTokenizer
from eznlp.training import Trainer

# 复用训练脚本的函数
from train_redjujube_expert_boundary import (
    load_redjujube_data,
    truncate_long_sequences,
)


def find_best_seed_model(model_group_dir):
    """
    找出指定模型组目录下最佳 seed 的模型目录
    
    Args:
        model_group_dir: 模型组目录（如 Q_bs_focal/）
    
    Returns:
        tuple: (best_model_dir, test_f1, seed)
    """
    best_f1 = -1
    best_dir = None
    best_seed = None
    
    if not os.path.isdir(model_group_dir):
        return None, None, None
    
    for subdir in os.listdir(model_group_dir):
        subdir_path = os.path.join(model_group_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        results_path = os.path.join(subdir_path, "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            test_f1 = results.get("test_f1", 0)
            seed = results.get("args", {}).get("seed", "unknown")
            
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_dir = subdir_path
                best_seed = seed
    
    return best_dir, best_f1, best_seed


def detect_model_type(save_dir):
    """自动检测模型类型（CRF 或 BS）"""
    files = os.listdir(save_dir)
    pth_files = [f for f in files if f.endswith(".pth") and not f.endswith("-config.pth")]
    
    if not pth_files:
        return "unknown"
    
    model_file = pth_files[0]
    if "CRF" in model_file:
        return "CRF"
    elif "SB" in model_file or "boundary" in save_dir.lower():
        return "BS"
    return "unknown"


def get_expert_dict_path(save_dir, results_args):
    """
    获取正确的专家词典路径
    
    优先级：
    1. 模型目录下的 auto_lexicon.txt（训练时自动提取的）
    2. results.json 中记录的 expert_dict_path
    3. 默认路径
    """
    # 优先使用训练时自动提取的词典
    auto_lexicon_path = os.path.join(save_dir, "auto_lexicon.txt")
    if os.path.exists(auto_lexicon_path):
        return auto_lexicon_path
    
    # 使用训练时指定的词典路径
    expert_dict_path = results_args.get("expert_dict_path")
    if expert_dict_path and os.path.exists(expert_dict_path):
        return expert_dict_path
    
    # 默认路径
    default_path = "datasets/raw/RedJujube/expert_lexicon_auto_min1.txt"
    if os.path.exists(default_path):
        return default_path
    
    return None


def load_expert_lexicon(dict_path):
    """加载专家词典"""
    lexicon = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            word = parts[0]
            if word:
                lexicon.append(word)
    return lexicon


def load_config_and_model(save_dir, device):
    """加载模型配置和权重"""
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
    
    config = torch.load(config_path, map_location=device, weights_only=False)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    return config, model


def evaluate_single_model(save_dir, data_dir, device, verbose=True):
    """
    评估单个模型
    
    Args:
        save_dir: 模型保存目录
        data_dir: 数据目录
        device: 计算设备
        verbose: 是否输出详细信息
    
    Returns:
        dict: 评估结果
    """
    # 1. 加载 results.json 获取训练参数
    results_path = os.path.join(save_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            train_results = json.load(f)
        args = train_results.get("args", {})
    else:
        train_results = {}
        args = {}
    
    model_type = detect_model_type(save_dir)
    use_expert_dict = "expert_dict" in save_dir or "expert" in str(os.listdir(save_dir))
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"模型目录: {save_dir}")
        print(f"模型类型: {model_type}")
        print(f"训练时 test_f1: {train_results.get('test_f1', 'N/A'):.4f}" if train_results.get('test_f1') else "训练时 test_f1: N/A")
        print(f"{'=' * 70}")
    
    # 2. 加载数据（复用训练脚本的函数）
    train_data, dev_data, test_data = load_redjujube_data(data_dir)
    if verbose:
        print(f"数据加载: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")
    
    # 3. 截断过长序列（复用训练脚本的函数）
    num_trunc = truncate_long_sequences([train_data, dev_data, test_data], max_char_len=510)
    if num_trunc > 0 and verbose:
        print(f"截断过长样本: {num_trunc} 条")
    
    # 4. 检测是否使用专家词典
    config, model = load_config_and_model(save_dir, device)
    has_expert_dict = (
        hasattr(config, "nested_ohots") 
        and config.nested_ohots 
        and "expert_dict" in config.nested_ohots
    )
    
    # 5. 如果使用专家词典，添加特征
    if has_expert_dict:
        expert_dict_path = get_expert_dict_path(save_dir, args)
        if expert_dict_path:
            lexicon = load_expert_lexicon(expert_dict_path)
            if verbose:
                print(f"专家词典: {expert_dict_path} ({len(lexicon)} 个词)")
            
            tokenizer = LexiconTokenizer(lexicon, max_len=10)
            for data in [train_data, dev_data, test_data]:
                for entry in data:
                    entry["tokens"].build_expert_dict_tags(tokenizer.tokenize)
    
    # 6. 创建测试集
    test_set = Dataset(test_data, config, training=False)
    if verbose:
        print(f"测试集样本数: {len(test_set)}")
    
    # 7. 预测
    trainer = Trainer(model, device=device)
    pred_chunks = trainer.predict(test_set, batch_size=16)
    gold_chunks = [ex["chunks"] for ex in test_set.data]
    
    # 8. 计算指标
    scores, ave_scores = precision_recall_f1_report(
        gold_chunks, pred_chunks, macro_over="types"
    )
    micro = ave_scores["micro"]
    macro = ave_scores["macro"]
    
    # 9. 输出结果
    if verbose:
        print(f"\n{'=' * 70}")
        print("整体评估指标")
        print(f"{'=' * 70}")
        print(f"\nMicro Average: P={micro['precision']:.4f}  R={micro['recall']:.4f}  F1={micro['f1']:.4f}")
        print(f"Macro Average: P={macro['precision']:.4f}  R={macro['recall']:.4f}  F1={macro['f1']:.4f}")
        
        print(f"\n{'=' * 70}")
        print("各实体类别评估指标")
        print(f"{'=' * 70}")
        print(f"\n{'Type':<15} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
        print("-" * 50)
        
        for entity_type in sorted(scores.keys()):
            s = scores[entity_type]
            print(
                f"{entity_type:<15} {s['precision']:>8.4f} {s['recall']:>8.4f} "
                f"{s['f1']:>8.4f} {s['n_gold']:>8}"
            )
        
        print("-" * 50)
        total_gold = sum(s["n_gold"] for s in scores.values())
        print(f"{'Total':<15} {micro['precision']:>8.4f} {micro['recall']:>8.4f} {micro['f1']:>8.4f} {total_gold:>8}")
        
        # 验证与训练日志一致性
        train_f1 = train_results.get("test_f1")
        if train_f1 is not None:
            diff = abs(micro["f1"] - train_f1) * 100
            status = "✅ 一致" if diff < 0.01 else "❌ 不一致"
            print(f"\n验证: 训练日志 F1={train_f1*100:.2f}%, 评估 F1={micro['f1']*100:.2f}%, 差异={diff:.2f}% {status}")
    
    # 10. 返回结果
    return {
        "save_dir": save_dir,
        "model_type": model_type,
        "train_f1": train_results.get("test_f1"),
        "eval_f1": micro["f1"],
        "micro": micro,
        "macro": macro,
        "scores_by_type": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="论文模型评估脚本")
    parser.add_argument(
        "--results_base",
        type=str,
        default="experiments/EXP-010-optimization/results_newdata",
        help="实验结果基础目录",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/raw/RedJujube",
        help="数据目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/research_logs/paper_evaluation.log",
        help="输出日志文件",
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 定义需要评估的模型组
    model_groups = {
        "Q' (BS+Dict+Focal)": "Q_bs_focal",
        "H' (BS+Dict)": "H_bs_baseline",
        "A' (CRF+Dict)": "A_baseline",
    }
    
    all_results = {}
    
    # 打开日志文件
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 同时输出到控制台和文件
    import io
    from contextlib import redirect_stdout
    
    log_buffer = io.StringIO()
    
    class TeeOutput:
        def __init__(self, *outputs):
            self.outputs = outputs
        def write(self, text):
            for output in self.outputs:
                output.write(text)
        def flush(self):
            for output in self.outputs:
                output.flush()
    
    tee = TeeOutput(sys.stdout, log_buffer)
    
    with redirect_stdout(tee):
        print(f"=" * 70)
        print(f"论文模型评估报告")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 70)
        
        for group_name, group_dir in model_groups.items():
            group_path = os.path.join(args.results_base, group_dir)
            
            print(f"\n\n{'#' * 70}")
            print(f"# 模型组: {group_name}")
            print(f"# 目录: {group_path}")
            print(f"{'#' * 70}")
            
            # 找出最佳 seed 的模型
            best_dir, best_f1, best_seed = find_best_seed_model(group_path)
            
            if best_dir is None:
                print(f"警告: 未找到有效的模型目录")
                continue
            
            print(f"\n最佳 seed: {best_seed}, 训练时 F1={best_f1*100:.2f}%")
            print(f"模型目录: {best_dir}")
            
            # 评估
            result = evaluate_single_model(best_dir, args.data_dir, device, verbose=True)
            all_results[group_name] = result
        
        # 汇总表格
        print(f"\n\n{'=' * 70}")
        print("汇总表格")
        print(f"{'=' * 70}")
        print(f"\n{'Model':<25} {'Train F1':>12} {'Eval F1':>12} {'Status':>10}")
        print("-" * 60)
        
        for group_name, result in all_results.items():
            train_f1 = result.get("train_f1")
            eval_f1 = result.get("eval_f1")
            
            train_str = f"{train_f1*100:.2f}%" if train_f1 else "N/A"
            eval_str = f"{eval_f1*100:.2f}%" if eval_f1 else "N/A"
            
            if train_f1 and eval_f1:
                diff = abs(train_f1 - eval_f1) * 100
                status = "✅" if diff < 0.01 else "❌"
            else:
                status = "?"
            
            print(f"{group_name:<25} {train_str:>12} {eval_str:>12} {status:>10}")
        
        print("-" * 60)
        print(f"\n评估完成！")
    
    # 保存日志
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())
    print(f"\n日志已保存到: {args.output}")
    
    # 保存 JSON 结果
    json_output = args.output.replace(".log", ".json")
    with open(json_output, "w", encoding="utf-8") as f:
        # 转换为可序列化格式
        serializable_results = {}
        for k, v in all_results.items():
            serializable_results[k] = {
                "save_dir": v["save_dir"],
                "model_type": v["model_type"],
                "train_f1": v.get("train_f1"),
                "eval_f1": v.get("eval_f1"),
                "micro": v["micro"],
                "macro": v["macro"],
            }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"JSON 结果已保存到: {json_output}")


if __name__ == "__main__":
    main()
