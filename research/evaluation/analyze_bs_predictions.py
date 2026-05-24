#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H 组 BS 模型（Boundary Selection）预测错误分析脚本

分析内容：
1. 总体指标：Test P/R/F1
2. 按实体类型分解：每个实体类型的 P/R/F1、Support 数量（按 F1 从低到高排序）
3. 漏检分析（False Negatives）：按类型统计 + 典型案例
4. 错检分析（False Positives）：按类型统计 + 典型案例
5. 边界错误：类型正确但边界不完全匹配的情况
6. 按实体长度的错误分布：1字、2字、3字、4-5字、6字以上
7. 类型混淆矩阵

使用：
    cd /home/shiwenlong/NERlabs/eznlp
    PYTHONPATH=/home/shiwenlong/NERlabs/eznlp python research/evaluation/analyze_bs_predictions.py
"""

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


# ====================== 配置 ======================

# 模型文件路径
CONFIG_PATH = "experiments/EXP-010-optimization/results_newdata/R_bs_typeaware_dict/expert_boundary_20260331-141939/text-expert_dict-BERT-FFN-SB(0.10, 2)-config.pth"
MODEL_PATH = "experiments/EXP-010-optimization/results_newdata/R_bs_typeaware_dict/expert_boundary_20260331-141939/text-expert_dict-BERT-FFN-SB(0.10, 2).pth"

# 数据目录
DATA_DIR = "datasets/raw/RedJujube"

# 自动词典路径（需要与训练时一致）
AUTO_LEXICON_PATH = "experiments/EXP-010-optimization/results_newdata/R_bs_typeaware_dict/expert_boundary_20260331-141939/auto_lexicon.txt"

# 批次大小
BATCH_SIZE = 16


# ====================== 数据加载 ======================

def load_redjujube_data(data_dir: str):
    """加载 RedJujube 数据集（BMES格式）"""
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


def extract_auto_lexicon(train_data, min_freq=2):
    """从训练数据自动提取词典（与训练时一致）"""
    entity_counter = Counter()
    for entry in train_data:
        chunks = entry.get("chunks", [])
        tokens = entry["tokens"]
        for label, start, end in chunks:
            entity_text = "".join(str(tokens[i]) for i in range(start, end))
            if entity_text:
                entity_counter[entity_text] += 1
    lexicon = [word for word, count in entity_counter.items() if count >= min_freq]
    return lexicon


def load_expert_lexicon(dict_path: str):
    """加载专家词典文件"""
    lexicon = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                lexicon.append(word)
    return lexicon


# ====================== 模型加载与预测 ======================

def load_model_and_config(model_path: str, config_path: str, device: torch.device):
    """加载模型和配置"""
    print(f"加载配置: {config_path}")
    config = torch.load(config_path, map_location=device, weights_only=False)
    
    print(f"加载模型: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    return model, config


def run_prediction(model, config, test_data, device: torch.device, batch_size: int = 16):
    """运行预测"""
    test_set = Dataset(test_data, config, training=False)
    trainer = Trainer(model, device=device)
    pred_chunks = trainer.predict(test_set, batch_size=batch_size)
    gold_chunks = [ex["chunks"] for ex in test_set.data]
    return pred_chunks, gold_chunks, test_set


# ====================== 工具函数 ======================

def get_entity_text(entry, span):
    """获取实体文本"""
    if len(span) == 3:
        typ, start, end = span
    elif len(span) == 4:
        _, typ, start, end = span
    else:
        return "<?>"
    
    tokens = entry["tokens"]
    if hasattr(tokens, "raw_text"):
        return "".join(tokens.raw_text[start:end])
    else:
        return "".join(str(tokens[i]) for i in range(start, end))


def get_context(entry, start, end, context_chars=5):
    """获取实体的上下文"""
    tokens = entry["tokens"]
    if hasattr(tokens, "raw_text"):
        chars = tokens.raw_text
    else:
        chars = [str(tokens[i]) for i in range(len(tokens))]
    
    left_start = max(0, start - context_chars)
    right_end = min(len(chars), end + context_chars)
    
    left_ctx = "".join(chars[left_start:start])
    entity = "".join(chars[start:end])
    right_ctx = "".join(chars[end:right_end])
    
    return f"...{left_ctx}【{entity}】{right_ctx}..."


def get_entity_length(span):
    """获取实体长度"""
    if len(span) == 3:
        _, start, end = span
    elif len(span) == 4:
        _, _, start, end = span
    else:
        return 0
    return end - start


def get_length_bucket(length):
    """将长度映射到分桶"""
    if length == 1:
        return "1字"
    elif length == 2:
        return "2字"
    elif length == 3:
        return "3字"
    elif 4 <= length <= 5:
        return "4-5字"
    else:
        return "6字+"


def format_entities_in_text(tokens, spans):
    """在原文中用 [TYPE]...[/TYPE] 标注实体"""
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


# ====================== 分析函数 ======================

def analyze_boundary_errors(test_data, gold_chunks_list, pred_chunks_list):
    """
    分析边界错误：类型正确但边界不完全匹配的情况
    
    Returns:
        dict: {
            'total': 边界错误总数,
            'left_error': 左边界错误数,
            'right_error': 右边界错误数,
            'both_error': 两边都错,
            'samples': 典型样本列表
        }
    """
    boundary_errors = {
        'total': 0,
        'left_error': 0,  # 左边界错误（右边界正确）
        'right_error': 0,  # 右边界错误（左边界正确）
        'both_error': 0,  # 两边都错
        'expand': 0,  # 预测范围扩大
        'shrink': 0,  # 预测范围缩小
        'samples': []
    }
    
    for idx, (entry, golds, preds) in enumerate(zip(test_data, gold_chunks_list, pred_chunks_list)):
        gold_set = set(golds)
        pred_set = set(preds)
        
        # 对于每个金标实体，检查是否有类型匹配但边界不同的预测
        for g_span in golds:
            g_type, g_start, g_end = g_span[:3] if len(g_span) == 3 else g_span[1:4]
            
            # 如果完全匹配，跳过
            if g_span in pred_set:
                continue
            
            # 查找同类型且有重叠的预测
            for p_span in preds:
                p_type, p_start, p_end = p_span[:3] if len(p_span) == 3 else p_span[1:4]
                
                if p_type != g_type:
                    continue
                
                # 检查是否有重叠
                overlap_start = max(g_start, p_start)
                overlap_end = min(g_end, p_end)
                if overlap_start >= overlap_end:
                    continue
                
                # 有重叠且类型相同 -> 边界错误
                boundary_errors['total'] += 1
                
                left_diff = p_start - g_start  # 正数表示预测左边界右移（缩小）
                right_diff = p_end - g_end  # 正数表示预测右边界右移（扩大）
                
                if left_diff != 0 and right_diff == 0:
                    boundary_errors['left_error'] += 1
                elif left_diff == 0 and right_diff != 0:
                    boundary_errors['right_error'] += 1
                else:
                    boundary_errors['both_error'] += 1
                
                # 判断是扩大还是缩小
                pred_len = p_end - p_start
                gold_len = g_end - g_start
                if pred_len > gold_len:
                    boundary_errors['expand'] += 1
                elif pred_len < gold_len:
                    boundary_errors['shrink'] += 1
                
                # 收集样本
                if len(boundary_errors['samples']) < 20:
                    gold_text = get_entity_text(entry, g_span)
                    pred_text = get_entity_text(entry, p_span)
                    boundary_errors['samples'].append({
                        'index': idx,
                        'gold_span': g_span,
                        'pred_span': p_span,
                        'gold_text': gold_text,
                        'pred_text': pred_text,
                        'context': get_context(entry, min(g_start, p_start), max(g_end, p_end)),
                        'left_diff': left_diff,
                        'right_diff': right_diff,
                    })
                
                break  # 每个金标只匹配一个预测
    
    return boundary_errors


def analyze_errors_by_length(test_data, gold_chunks_list, pred_chunks_list):
    """
    按实体长度分析错误分布
    
    Returns:
        dict: {
            bucket: {
                'gold': gold 数量,
                'pred': pred 数量,
                'tp': 正确预测数量,
                'precision': P,
                'recall': R,
                'f1': F1
            }
        }
    """
    length_stats = defaultdict(lambda: {'gold': 0, 'pred': 0, 'tp': 0})
    
    for golds, preds in zip(gold_chunks_list, pred_chunks_list):
        gold_set = set(golds)
        pred_set = set(preds)
        tp_set = gold_set & pred_set
        
        for span in golds:
            length = get_entity_length(span)
            bucket = get_length_bucket(length)
            length_stats[bucket]['gold'] += 1
        
        for span in preds:
            length = get_entity_length(span)
            bucket = get_length_bucket(length)
            length_stats[bucket]['pred'] += 1
        
        for span in tp_set:
            length = get_entity_length(span)
            bucket = get_length_bucket(length)
            length_stats[bucket]['tp'] += 1
    
    # 计算 P/R/F1
    for bucket in length_stats:
        stats = length_stats[bucket]
        p = stats['tp'] / stats['pred'] if stats['pred'] > 0 else 0
        r = stats['tp'] / stats['gold'] if stats['gold'] > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        stats['precision'] = p
        stats['recall'] = r
        stats['f1'] = f1
    
    return dict(length_stats)


def analyze_type_confusion(test_data, gold_chunks_list, pred_chunks_list):
    """
    分析类型混淆矩阵
    
    Returns:
        dict: {
            (gold_type, pred_type): count
        }
    """
    confusion = Counter()
    
    for idx, (entry, golds, preds) in enumerate(zip(test_data, gold_chunks_list, pred_chunks_list)):
        # 对于每个金标实体，找是否有位置完全匹配但类型不同的预测
        for g_span in golds:
            g_type, g_start, g_end = g_span[:3] if len(g_span) == 3 else g_span[1:4]
            
            for p_span in preds:
                p_type, p_start, p_end = p_span[:3] if len(p_span) == 3 else p_span[1:4]
                
                # 位置完全匹配但类型不同
                if g_start == p_start and g_end == p_end and g_type != p_type:
                    confusion[(g_type, p_type)] += 1
    
    return confusion


def collect_fn_fp_samples(test_data, gold_chunks_list, pred_chunks_list, max_samples_per_type=3):
    """
    收集漏检（FN）和错检（FP）样本
    
    Returns:
        tuple: (fn_by_type, fp_by_type, fn_samples, fp_samples)
    """
    fn_by_type = Counter()
    fp_by_type = Counter()
    fn_samples = defaultdict(list)  # type -> list of samples
    fp_samples = defaultdict(list)  # type -> list of samples
    
    for idx, (entry, golds, preds) in enumerate(zip(test_data, gold_chunks_list, pred_chunks_list)):
        gold_set = set(golds)
        pred_set = set(preds)
        
        # 漏检：gold 中有但 pred 中没有
        fn_spans = gold_set - pred_set
        # 错检：pred 中有但 gold 中没有
        fp_spans = pred_set - gold_set
        
        for span in fn_spans:
            entity_type = span[0] if len(span) == 3 else span[1]
            fn_by_type[entity_type] += 1
            
            if len(fn_samples[entity_type]) < max_samples_per_type:
                entity_text = get_entity_text(entry, span)
                start = span[1] if len(span) == 3 else span[2]
                end = span[2] if len(span) == 3 else span[3]
                fn_samples[entity_type].append({
                    'index': idx,
                    'text': entity_text,
                    'context': get_context(entry, start, end),
                    'span': span
                })
        
        for span in fp_spans:
            entity_type = span[0] if len(span) == 3 else span[1]
            fp_by_type[entity_type] += 1
            
            if len(fp_samples[entity_type]) < max_samples_per_type:
                entity_text = get_entity_text(entry, span)
                start = span[1] if len(span) == 3 else span[2]
                end = span[2] if len(span) == 3 else span[3]
                fp_samples[entity_type].append({
                    'index': idx,
                    'text': entity_text,
                    'context': get_context(entry, start, end),
                    'span': span
                })
    
    return fn_by_type, fp_by_type, fn_samples, fp_samples


# ====================== 输出函数 ======================

def print_section_header(title):
    """打印分隔线和标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_overall_metrics(scores, ave_scores):
    """打印总体指标"""
    print_section_header("1. 总体指标 (Test P/R/F1)")
    
    macro = ave_scores["macro"]
    micro = ave_scores["micro"]
    
    print(f"\n  {'Metric':<10} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("  " + "-" * 50)
    print(f"  {'Micro':<10} {micro['precision']:>12.4f} {micro['recall']:>12.4f} {micro['f1']:>12.4f}")
    print(f"  {'Macro':<10} {macro['precision']:>12.4f} {macro['recall']:>12.4f} {macro['f1']:>12.4f}")


def print_per_type_metrics(scores):
    """打印按实体类型分解的指标（按 F1 从低到高排序）"""
    print_section_header("2. 按实体类型分解 (按 F1 从低到高排序)")
    
    # 按 F1 从低到高排序
    sorted_types = sorted(scores.keys(), key=lambda t: scores[t]['f1'])
    
    print(f"\n  {'Type':<10} {'P':>9} {'R':>9} {'F1':>9} {'Gold':>8} {'Pred':>8} {'TP':>8}")
    print("  " + "-" * 68)
    
    for t in sorted_types:
        s = scores[t]
        print(f"  {t:<10} {s['precision']:>9.4f} {s['recall']:>9.4f} {s['f1']:>9.4f} "
              f"{s['n_gold']:>8d} {s['n_pred']:>8d} {s['n_true_positive']:>8d}")


def print_fn_analysis(fn_by_type, fn_samples):
    """打印漏检分析"""
    print_section_header("3. 漏检分析 (False Negatives)")
    
    print("\n  按实体类型统计漏检数量:")
    print(f"  {'Type':<10} {'FN Count':>10}")
    print("  " + "-" * 25)
    for entity_type, count in fn_by_type.most_common():
        print(f"  {entity_type:<10} {count:>10}")
    
    print(f"\n  总漏检数: {sum(fn_by_type.values())}")
    
    print("\n  典型漏检案例 (每类最多3个):")
    for entity_type in fn_by_type.keys():
        if fn_samples[entity_type]:
            print(f"\n  [{entity_type}] 类型漏检案例:")
            for i, sample in enumerate(fn_samples[entity_type][:3], 1):
                print(f"    {i}. \"{sample['text']}\" {sample['context']}")


def print_fp_analysis(fp_by_type, fp_samples):
    """打印错检分析"""
    print_section_header("4. 错检分析 (False Positives)")
    
    print("\n  按实体类型统计错检数量:")
    print(f"  {'Type':<10} {'FP Count':>10}")
    print("  " + "-" * 25)
    for entity_type, count in fp_by_type.most_common():
        print(f"  {entity_type:<10} {count:>10}")
    
    print(f"\n  总错检数: {sum(fp_by_type.values())}")
    
    print("\n  典型错检案例 (每类最多3个):")
    for entity_type in fp_by_type.keys():
        if fp_samples[entity_type]:
            print(f"\n  [{entity_type}] 类型错检案例:")
            for i, sample in enumerate(fp_samples[entity_type][:3], 1):
                print(f"    {i}. \"{sample['text']}\" {sample['context']}")


def print_boundary_errors(boundary_errors):
    """打印边界错误分析"""
    print_section_header("5. 边界错误分析 (类型正确但边界不完全匹配)")
    
    print(f"\n  边界错误总数: {boundary_errors['total']}")
    print(f"    - 仅左边界错误: {boundary_errors['left_error']}")
    print(f"    - 仅右边界错误: {boundary_errors['right_error']}")
    print(f"    - 两边都错: {boundary_errors['both_error']}")
    print(f"    - 预测范围扩大: {boundary_errors['expand']}")
    print(f"    - 预测范围缩小: {boundary_errors['shrink']}")
    
    print("\n  典型边界错误案例 (最多10个):")
    for i, sample in enumerate(boundary_errors['samples'][:10], 1):
        gold_text = sample['gold_text']
        pred_text = sample['pred_text']
        direction = ""
        if sample['left_diff'] < 0:
            direction += "左扩"
        elif sample['left_diff'] > 0:
            direction += "左缩"
        if sample['right_diff'] > 0:
            direction += "右扩"
        elif sample['right_diff'] < 0:
            direction += "右缩"
        
        print(f"    {i}. Gold=[{gold_text}] → Pred=[{pred_text}] ({direction})")
        print(f"       {sample['context']}")


def print_length_analysis(length_stats):
    """打印按实体长度的错误分布"""
    print_section_header("6. 按实体长度的错误分布")
    
    # 按固定顺序显示
    bucket_order = ["1字", "2字", "3字", "4-5字", "6字+"]
    
    print(f"\n  {'长度':<8} {'Gold':>8} {'Pred':>8} {'TP':>8} {'P':>10} {'R':>10} {'F1':>10}")
    print("  " + "-" * 70)
    
    for bucket in bucket_order:
        if bucket in length_stats:
            stats = length_stats[bucket]
            print(f"  {bucket:<8} {stats['gold']:>8d} {stats['pred']:>8d} {stats['tp']:>8d} "
                  f"{stats['precision']:>10.4f} {stats['recall']:>10.4f} {stats['f1']:>10.4f}")


def print_confusion_matrix(confusion):
    """打印类型混淆矩阵"""
    print_section_header("7. 类型混淆矩阵 (位置匹配但类型错误)")
    
    if not confusion:
        print("\n  没有发现类型混淆错误（所有位置匹配的实体类型都正确）")
        return
    
    print(f"\n  {'Gold Type':<10} {'Pred Type':<10} {'Count':>8}")
    print("  " + "-" * 35)
    for (gold_type, pred_type), count in confusion.most_common():
        print(f"  {gold_type:<10} {pred_type:<10} {count:>8d}")
    
    print(f"\n  总计: {sum(confusion.values())} 个类型混淆错误")


# ====================== 主程序 ======================

def main():
    print("=" * 80)
    print("  H 组 BS 模型 (Boundary Selection) 预测错误分析")
    print("  模型: seed=42, Test F1=86.80%")
    print("=" * 80)
    
    # 设备配置 - 强制 CPU
    device = torch.device("cpu")
    print(f"\n运行设备: {device}")
    
    # 1. 加载模型和配置
    print("\n" + "-" * 40)
    print("加载模型和配置...")
    print("-" * 40)
    
    model, config = load_model_and_config(MODEL_PATH, CONFIG_PATH, device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 2. 加载数据
    print("\n" + "-" * 40)
    print("加载 RedJujube 数据集...")
    print("-" * 40)
    
    train_data, dev_data, test_data = load_redjujube_data(DATA_DIR)
    print(f"数据集大小: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")
    
    # 3. 添加专家词典特征（与训练时一致）
    print("\n" + "-" * 40)
    print("添加专家词典特征...")
    print("-" * 40)
    
    # 优先使用保存的自动词典，如果不存在则重新提取
    if os.path.exists(AUTO_LEXICON_PATH):
        lexicon = load_expert_lexicon(AUTO_LEXICON_PATH)
        print(f"从文件加载自动词典: {AUTO_LEXICON_PATH}")
    else:
        lexicon = extract_auto_lexicon(train_data, min_freq=2)
        print(f"从训练集提取自动词典 (min_freq=2)")
    print(f"词典大小: {len(lexicon)} 个词")
    
    tokenizer = LexiconTokenizer(lexicon, max_len=10)
    for data in (train_data, dev_data, test_data):
        for entry in data:
            entry["tokens"].build_expert_dict_tags(tokenizer.tokenize)
    print("专家词典特征添加完成")
    
    # 4. 运行预测
    print("\n" + "-" * 40)
    print("在测试集上运行预测...")
    print("-" * 40)
    
    pred_chunks, gold_chunks, test_set = run_prediction(
        model, config, test_data, device, batch_size=BATCH_SIZE
    )
    print(f"预测完成，共 {len(pred_chunks)} 个样本")
    
    # 5. 计算指标
    print("\n" + "-" * 40)
    print("计算评估指标...")
    print("-" * 40)
    
    scores, ave_scores = precision_recall_f1_report(
        gold_chunks, pred_chunks, macro_over="types"
    )
    
    # 6. 错误分析
    print("\n" + "-" * 40)
    print("进行错误分析...")
    print("-" * 40)
    
    # 收集 FN/FP 样本
    fn_by_type, fp_by_type, fn_samples, fp_samples = collect_fn_fp_samples(
        test_data, gold_chunks, pred_chunks, max_samples_per_type=3
    )
    
    # 边界错误分析
    boundary_errors = analyze_boundary_errors(test_data, gold_chunks, pred_chunks)
    
    # 按长度分析
    length_stats = analyze_errors_by_length(test_data, gold_chunks, pred_chunks)
    
    # 类型混淆矩阵
    confusion = analyze_type_confusion(test_data, gold_chunks, pred_chunks)
    
    # 7. 输出结果
    print("\n\n")
    print("#" * 80)
    print("#" + " " * 30 + "分析结果报告" + " " * 30 + "#")
    print("#" * 80)
    
    print_overall_metrics(scores, ave_scores)
    print_per_type_metrics(scores)
    print_fn_analysis(fn_by_type, fn_samples)
    print_fp_analysis(fp_by_type, fp_samples)
    print_boundary_errors(boundary_errors)
    print_length_analysis(length_stats)
    print_confusion_matrix(confusion)
    
    print("\n" + "=" * 80)
    print("  分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
