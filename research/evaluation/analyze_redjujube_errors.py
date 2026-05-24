#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube NER 错误分析脚本

功能：
- 加载训练好的模型，在测试集上运行预测
- 按实体类型统计 P/R/F1（每类单独统计）
- 分析错误类型：漏检(FN)、错检(FP)
- 区分边界错误、类型错误、完全漏检/错检
- 输出典型错误样本
- 支持保存分析结果到文件

用法：
    python research/evaluation/analyze_redjujube_errors.py \
        --model_path experiments/EXP-001-expert-dict/results/xxx/Extractor.pth \
        --config_path experiments/EXP-001-expert-dict/results/xxx/Extractor-config.pth \
        --data_dir datasets/raw/RedJujube \
        --output_path results/error_analysis.txt
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import torch

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from eznlp.dataset import Dataset
from eznlp.io import ConllIO
from eznlp.metrics import precision_recall_f1_report
from eznlp.training import Trainer
from eznlp.token import LexiconTokenizer


# ====================== 数据加载 ======================

def load_redjujube_data(data_dir: str):
    """
    加载 RedJujube 数据集（BMES格式）
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        tuple: (train_data, dev_data, test_data)
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


def load_expert_lexicon(dict_path: str, with_type: bool = False):
    """
    加载专家词典
    
    Args:
        dict_path: 词典路径
        with_type: 是否带类型（word\ttype 格式）
        
    Returns:
        list: 词典列表
    """
    lexicon = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            word = parts[0]
            if word:
                if with_type and len(parts) >= 2 and parts[1]:
                    word = f"{word}_{parts[1]}"
                lexicon.append(word)
    return lexicon


# ====================== 模型加载与预测 ======================

def load_model_and_config(model_path: str, config_path: str, device: torch.device):
    """
    加载模型和配置
    
    Args:
        model_path: 模型 .pth 文件路径
        config_path: 配置 .pth 文件路径
        device: 设备
        
    Returns:
        tuple: (model, config)
    """
    print(f"加载配置: {config_path}")
    config = torch.load(config_path, map_location=device, weights_only=False)
    
    print(f"加载模型: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    
    return model, config


def run_prediction(model, config, test_data, device: torch.device, batch_size: int = 32):
    """
    运行预测
    
    Args:
        model: 模型
        config: 配置
        test_data: 测试数据
        device: 设备
        batch_size: 批次大小
        
    Returns:
        tuple: (pred_chunks, gold_chunks) - 预测结果和金标列表
    """
    # 构建测试数据集
    test_set = Dataset(test_data, config, training=False)
    
    # 创建训练器并预测
    trainer = Trainer(model, device=device)
    pred_chunks = trainer.predict(test_set, batch_size=batch_size)
    
    # 获取金标
    gold_chunks = [ex["chunks"] for ex in test_set.data]
    
    return pred_chunks, gold_chunks


# ====================== 指标计算 ======================

def compute_per_type_metrics(gold_chunks_list: List, pred_chunks_list: List) -> Dict:
    """
    计算每个实体类型的 P/R/F1
    
    Args:
        gold_chunks_list: 金标 chunks 列表
        pred_chunks_list: 预测 chunks 列表
        
    Returns:
        dict: 每个类型的指标，以及宏/微平均
    """
    scores, ave_scores = precision_recall_f1_report(
        gold_chunks_list, pred_chunks_list, macro_over="types"
    )
    return scores, ave_scores


# ====================== 错误分类分析 ======================

def classify_error(gold_span: Tuple, pred_span: Tuple) -> str:
    """
    判断两个 span 之间的错误类型
    
    Args:
        gold_span: 金标 (type, start, end)
        pred_span: 预测 (type, start, end)
        
    Returns:
        str: 错误类型 ('boundary_error', 'type_error', 'exact_match', 'no_overlap')
    """
    g_type, g_start, g_end = gold_span
    p_type, p_start, p_end = pred_span
    
    # 计算是否有重叠
    overlap_start = max(g_start, p_start)
    overlap_end = min(g_end, p_end)
    has_overlap = overlap_start < overlap_end
    
    if not has_overlap:
        return "no_overlap"
    
    # 位置完全相同
    same_position = (g_start == p_start) and (g_end == p_end)
    # 类型相同
    same_type = (g_type == p_type)
    
    if same_position and same_type:
        return "exact_match"
    elif same_position and not same_type:
        return "type_error"  # 位置正确但类型错误
    elif has_overlap and same_type:
        return "boundary_error"  # 类型正确但边界偏移
    elif has_overlap and not same_type:
        return "both_error"  # 边界和类型都错
    
    return "no_overlap"


def analyze_errors(
    test_data: List,
    gold_chunks_list: List,
    pred_chunks_list: List
) -> Dict:
    """
    分析错误模式
    
    Args:
        test_data: 测试数据
        gold_chunks_list: 金标 chunks 列表
        pred_chunks_list: 预测 chunks 列表
        
    Returns:
        dict: 错误分析结果
    """
    # 错误统计
    error_stats = {
        "total_fn": 0,          # 总漏检数
        "total_fp": 0,          # 总错检数
        "boundary_error": 0,     # 边界错误（位置偏移但类型正确）
        "type_error": 0,         # 类型错误（位置正确但类型错误）
        "both_error": 0,         # 边界和类型都错
        "complete_fn": 0,        # 完全漏检
        "complete_fp": 0,        # 完全错检
    }
    
    # 按类型统计错误
    fn_by_type = Counter()  # 每种类型的漏检数
    fp_by_type = Counter()  # 每种类型的错检数
    
    # 收集错误样本
    error_samples = {
        "boundary_error": [],
        "type_error": [],
        "complete_fn": [],
        "complete_fp": [],
    }
    
    for idx, (entry, golds, preds) in enumerate(zip(test_data, gold_chunks_list, pred_chunks_list)):
        gold_set = set(golds)
        pred_set = set(preds)
        
        # 完全正确的样本跳过
        if gold_set == pred_set:
            continue
        
        # 漏检：gold 中有但 pred 中没有
        fn_spans = gold_set - pred_set
        # 错检：pred 中有但 gold 中没有
        fp_spans = pred_set - gold_set
        
        error_stats["total_fn"] += len(fn_spans)
        error_stats["total_fp"] += len(fp_spans)
        
        # 获取原文
        tokens = entry["tokens"]
        raw_text = "".join(tokens.raw_text) if hasattr(tokens, "raw_text") else str(tokens)
        
        # 分析每个漏检实体
        for fn_span in fn_spans:
            fn_type = fn_span[0]
            fn_by_type[fn_type] += 1
            
            # 检查是否有对应的边界错误或类型错误
            matched_error_type = "complete_fn"
            for fp_span in fp_spans:
                err_type = classify_error(fn_span, fp_span)
                if err_type == "boundary_error":
                    matched_error_type = "boundary_error"
                    break
                elif err_type == "type_error":
                    matched_error_type = "type_error"
                    break
                elif err_type == "both_error":
                    matched_error_type = "both_error"
                    break
            
            error_stats[matched_error_type] += 1
            
            # 收集样本
            if matched_error_type != "complete_fn" or len(error_samples["complete_fn"]) < 5:
                if matched_error_type in error_samples and len(error_samples[matched_error_type]) < 5:
                    entity_text = "".join(tokens.raw_text[fn_span[1]:fn_span[2]]) if hasattr(tokens, "raw_text") else "<?>"
                    error_samples[matched_error_type].append({
                        "index": idx,
                        "raw_text": raw_text,
                        "gold": list(golds),
                        "pred": list(preds),
                        "error_span": fn_span,
                        "entity_text": entity_text,
                        "error_type": matched_error_type,
                    })
        
        # 分析每个错检实体
        for fp_span in fp_spans:
            fp_type = fp_span[0]
            fp_by_type[fp_type] += 1
            
            # 检查是否已被边界/类型错误计数
            is_counted = False
            for fn_span in fn_spans:
                err_type = classify_error(fn_span, fp_span)
                if err_type in ["boundary_error", "type_error", "both_error"]:
                    is_counted = True
                    break
            
            if not is_counted:
                error_stats["complete_fp"] += 1
                
                # 收集样本
                if len(error_samples["complete_fp"]) < 5:
                    entity_text = "".join(tokens.raw_text[fp_span[1]:fp_span[2]]) if hasattr(tokens, "raw_text") else "<?>"
                    error_samples["complete_fp"].append({
                        "index": idx,
                        "raw_text": raw_text,
                        "gold": list(golds),
                        "pred": list(preds),
                        "error_span": fp_span,
                        "entity_text": entity_text,
                        "error_type": "complete_fp",
                    })
    
    return {
        "stats": error_stats,
        "fn_by_type": fn_by_type,
        "fp_by_type": fp_by_type,
        "samples": error_samples,
    }


# ====================== 格式化输出 ======================

def format_entities_in_text(tokens, spans):
    """
    在原文中用 [TYPE]...[/TYPE] 标注实体
    
    Args:
        tokens: TokenSequence 或字符列表
        spans: 实体 span 列表，元素为 (type, start, end)
        
    Returns:
        str: 标注后的文本
    """
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


def print_metrics_table(scores: Dict, ave_scores: Dict, output_file=None):
    """
    打印指标表格
    
    Args:
        scores: 每个类型的指标
        ave_scores: 平均指标
        output_file: 输出文件对象（可选）
    """
    def _print(text):
        print(text)
        if output_file:
            output_file.write(text + "\n")
    
    _print("\n" + "=" * 70)
    _print("各实体类型 P/R/F1 指标")
    _print("=" * 70)
    _print(f"{'Type':<8} {'P':>9} {'R':>9} {'F1':>9} {'Gold':>8} {'Pred':>8} {'TP':>8}")
    _print("-" * 70)
    
    # 按 Gold 数量排序（从多到少）
    sorted_types = sorted(scores.keys(), key=lambda t: -scores[t].get('n_gold', 0))
    
    for t in sorted_types:
        s = scores[t]
        _print(
            f"{t:<8} "
            f"{s['precision']:>9.4f} {s['recall']:>9.4f} {s['f1']:>9.4f} "
            f"{s['n_gold']:>8d} {s['n_pred']:>8d} {s['n_true_positive']:>8d}"
        )
    
    _print("-" * 70)
    
    # 宏/微平均
    macro = ave_scores["macro"]
    micro = ave_scores["micro"]
    _print(f"{'Macro':<8} {macro['precision']:>9.4f} {macro['recall']:>9.4f} {macro['f1']:>9.4f}")
    _print(f"{'Micro':<8} {micro['precision']:>9.4f} {micro['recall']:>9.4f} {micro['f1']:>9.4f}")
    _print("=" * 70 + "\n")


def print_error_analysis(error_result: Dict, output_file=None):
    """
    打印错误分析结果
    
    Args:
        error_result: 错误分析结果
        output_file: 输出文件对象（可选）
    """
    def _print(text):
        print(text)
        if output_file:
            output_file.write(text + "\n")
    
    stats = error_result["stats"]
    fn_by_type = error_result["fn_by_type"]
    fp_by_type = error_result["fp_by_type"]
    samples = error_result["samples"]
    
    # 错误统计表
    _print("=" * 70)
    _print("错误分类统计")
    _print("=" * 70)
    _print(f"  总漏检 (FN): {stats['total_fn']}")
    _print(f"  总错检 (FP): {stats['total_fp']}")
    _print("-" * 40)
    _print(f"  边界错误（位置偏移，类型正确）: {stats['boundary_error']}")
    _print(f"  类型错误（位置正确，类型错误）: {stats['type_error']}")
    _print(f"  边界+类型错误: {stats['both_error']}")
    _print(f"  完全漏检（无重叠）: {stats['complete_fn']}")
    _print(f"  完全错检（无重叠）: {stats['complete_fp']}")
    _print("=" * 70 + "\n")
    
    # 按类型统计漏检
    _print("=" * 70)
    _print("漏检(FN)按实体类型分布（Top 15）")
    _print("=" * 70)
    for entity_type, count in fn_by_type.most_common(15):
        _print(f"  [{entity_type}]: {count}")
    _print("")
    
    # 按类型统计错检
    _print("=" * 70)
    _print("错检(FP)按实体类型分布（Top 15）")
    _print("=" * 70)
    for entity_type, count in fp_by_type.most_common(15):
        _print(f"  [{entity_type}]: {count}")
    _print("")


def print_error_samples(error_result: Dict, test_data: List, output_file=None):
    """
    打印典型错误样本
    
    Args:
        error_result: 错误分析结果
        test_data: 测试数据
        output_file: 输出文件对象（可选）
    """
    def _print(text):
        print(text)
        if output_file:
            output_file.write(text + "\n")
    
    samples = error_result["samples"]
    
    error_type_names = {
        "boundary_error": "边界错误（位置偏移，类型正确）",
        "type_error": "类型错误（位置正确，类型错误）",
        "complete_fn": "完全漏检",
        "complete_fp": "完全错检",
    }
    
    for error_type, sample_list in samples.items():
        if not sample_list:
            continue
        
        _print("=" * 70)
        _print(f"典型错误样本 - {error_type_names.get(error_type, error_type)}（最多5个）")
        _print("=" * 70)
        
        for i, sample in enumerate(sample_list[:5]):
            _print(f"\n样本 {i+1} (索引: {sample['index']})")
            _print(f"  原文: {sample['raw_text']}")
            
            entry = test_data[sample["index"]]
            tokens = entry["tokens"]
            
            _print(f"  Gold: {format_entities_in_text(tokens, sample['gold'])}")
            _print(f"  Pred: {format_entities_in_text(tokens, sample['pred'])}")
            
            err_span = sample["error_span"]
            _print(f"  错误实体: [{err_span[0]}] \"{sample['entity_text']}\" @ ({err_span[1]}, {err_span[2]})")
            _print("-" * 40)
        
        _print("")


# ====================== 主程序 ======================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RedJujube NER 错误分析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python research/evaluation/analyze_redjujube_errors.py \\
      --model_path experiments/EXP-001-expert-dict/results/xxx/Extractor.pth \\
      --config_path experiments/EXP-001-expert-dict/results/xxx/Extractor-config.pth \\
      --data_dir datasets/raw/RedJujube \\
      --output_path results/error_analysis.txt
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型 .pth 文件路径"
    )
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="配置 .pth 文件路径"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="RedJujube 数据目录路径"
    )
    
    # 可选参数
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="分析结果保存路径（可选）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="预测批次大小（默认: 32）"
    )
    parser.add_argument(
        "--expert_dict_path", type=str, default=None,
        help="专家词典路径（可选，用于带词典特征的模型）"
    )
    parser.add_argument(
        "--with_type", action="store_true", default=False,
        help="词典是否带类型信息"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载模型和配置
    print("\n" + "=" * 70)
    print("加载模型和配置")
    print("=" * 70)
    model, config = load_model_and_config(args.model_path, args.config_path, device)
    
    # 2. 加载数据
    print("\n" + "=" * 70)
    print("加载 RedJujube 数据集")
    print("=" * 70)
    train_data, dev_data, test_data = load_redjujube_data(args.data_dir)
    print(f"数据集大小: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")
    
    # 3. 如果模型使用了专家词典特征，需要添加相应特征
    nested_ohots = getattr(config, "nested_ohots", None)
    if nested_ohots is not None and "expert_dict" in nested_ohots:
        print("\n检测到 ExpertDict 特征，加载专家词典...")
        
        # 确定词典路径
        if args.expert_dict_path:
            expert_dict_path = args.expert_dict_path
        else:
            # 默认词典路径
            if args.with_type:
                expert_dict_path = os.path.join(args.data_dir, "expert_lexicon_typed.txt")
            else:
                expert_dict_path = os.path.join(args.data_dir, "expert_lexicon_auto.txt")
        
        print(f"加载专家词典: {expert_dict_path}")
        lexicon = load_expert_lexicon(expert_dict_path, with_type=args.with_type)
        print(f"词典大小: {len(lexicon)}")
        
        # 为数据添加专家词典特征
        tokenizer = LexiconTokenizer(lexicon, max_len=10)
        for data in (train_data, dev_data, test_data):
            for entry in data:
                entry["tokens"].build_expert_dict_tags(tokenizer.tokenize)
        print("已添加专家词典特征")
    
    # 4. 运行预测
    print("\n" + "=" * 70)
    print("在测试集上运行预测")
    print("=" * 70)
    pred_chunks, gold_chunks = run_prediction(
        model, config, test_data, device, batch_size=args.batch_size
    )
    print(f"预测完成，共 {len(pred_chunks)} 个样本")
    
    # 5. 准备输出文件
    output_file = None
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        output_file = open(args.output_path, "w", encoding="utf-8")
        output_file.write(f"RedJujube NER 错误分析报告\n")
        output_file.write(f"模型: {args.model_path}\n")
        output_file.write(f"配置: {args.config_path}\n")
        output_file.write(f"数据: {args.data_dir}\n\n")
    
    # 6. 计算并打印指标
    print("\n" + "=" * 70)
    print("计算评估指标")
    print("=" * 70)
    scores, ave_scores = compute_per_type_metrics(gold_chunks, pred_chunks)
    print_metrics_table(scores, ave_scores, output_file)
    
    # 7. 错误分析
    print("\n" + "=" * 70)
    print("错误分析")
    print("=" * 70)
    error_result = analyze_errors(test_data, gold_chunks, pred_chunks)
    print_error_analysis(error_result, output_file)
    
    # 8. 打印典型错误样本
    print_error_samples(error_result, test_data, output_file)
    
    # 关闭输出文件
    if output_file:
        output_file.close()
        print(f"\n分析结果已保存到: {args.output_path}")
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
