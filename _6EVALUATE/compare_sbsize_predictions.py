#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H 组（sb_size=2）vs M 组（sb_size=3）模型对比分析脚本

分析内容：
1. 按实体长度的 F1 对比（1字、2字、3字、4-5字、6+字）
2. 按实体类型的 F1 对比（按 H-M 的 F1 差值排序）
3. 长实体（6+字）的详细漏检案例对比

使用：
    cd /home/shiwenlong/NERlabs/eznlp
    export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
    PYTHONPATH=/home/shiwenlong/NERlabs/eznlp python _6EVALUATE/compare_sbsize_predictions.py
"""

import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

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

# H 组（sb_size=2）
H_CONFIG_PATH = "experiments/EXP-010-optimization/results/H_bs_baseline/expert_boundary_20260318-163821/text-expert_dict-BERT-FFN-SB(0.10, 2)-config.pth"
H_MODEL_PATH = "experiments/EXP-010-optimization/results/H_bs_baseline/expert_boundary_20260318-163821/text-expert_dict-BERT-FFN-SB(0.10, 2).pth"
H_LEXICON_PATH = "experiments/EXP-010-optimization/results/H_bs_baseline/expert_boundary_20260318-163821/auto_lexicon.txt"

# M 组（sb_size=3）
M_CONFIG_PATH = "experiments/EXP-010-optimization/results/M_bs_sbsize3/expert_boundary_20260318-212213/text-expert_dict-BERT-FFN-SB(0.10, 3)-config.pth"
M_MODEL_PATH = "experiments/EXP-010-optimization/results/M_bs_sbsize3/expert_boundary_20260318-212213/text-expert_dict-BERT-FFN-SB(0.10, 3).pth"
M_LEXICON_PATH = "experiments/EXP-010-optimization/results/M_bs_sbsize3/expert_boundary_20260318-212213/auto_lexicon.txt"

# 数据目录
DATA_DIR = "_2DATA/RedJujube"
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
    config = torch.load(config_path, map_location=device, weights_only=False)
    
    # 兼容旧模型：添加缺失的属性
    if hasattr(config, 'decoder') and not hasattr(config.decoder, 'sb_size_map'):
        config.decoder.sb_size_map = None
    
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


def get_context(entry, start, end, context_chars=8):
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


# ====================== 分析函数 ======================

def analyze_by_length(gold_chunks_list, pred_chunks_list):
    """按实体长度分析"""
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


def collect_long_entity_fn(test_data, gold_chunks_list, pred_chunks_list, min_length=6):
    """收集长实体（>=min_length字）的漏检案例"""
    fn_list = []
    
    for idx, (entry, golds, preds) in enumerate(zip(test_data, gold_chunks_list, pred_chunks_list)):
        gold_set = set(golds)
        pred_set = set(preds)
        fn_spans = gold_set - pred_set
        
        for span in fn_spans:
            length = get_entity_length(span)
            if length >= min_length:
                entity_type = span[0] if len(span) == 3 else span[1]
                start = span[1] if len(span) == 3 else span[2]
                end = span[2] if len(span) == 3 else span[3]
                entity_text = get_entity_text(entry, span)
                context = get_context(entry, start, end)
                
                fn_list.append({
                    'index': idx,
                    'type': entity_type,
                    'text': entity_text,
                    'length': length,
                    'context': context,
                    'span': span
                })
    
    return fn_list


# ====================== 主程序 ======================

def main():
    print("=" * 80)
    print("  H 组 (sb_size=2) vs M 组 (sb_size=3) 模型对比分析")
    print("=" * 80)
    
    # 设备配置 - 强制 CPU
    device = torch.device("cpu")
    print(f"\n运行设备: {device}")
    
    # 1. 加载数据
    print("\n" + "-" * 60)
    print("加载 RedJujube 数据集...")
    print("-" * 60)
    
    train_data, dev_data, test_data = load_redjujube_data(DATA_DIR)
    print(f"数据集大小: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")
    
    # 2. 加载 H 组模型并预测
    print("\n" + "-" * 60)
    print("【H 组 (sb_size=2)】加载模型...")
    print("-" * 60)
    
    # 加载 H 组词典
    h_lexicon = load_expert_lexicon(H_LEXICON_PATH)
    print(f"H 组词典大小: {len(h_lexicon)} 个词")
    
    # 重新加载数据，添加 H 组词典特征
    train_data_h, dev_data_h, test_data_h = load_redjujube_data(DATA_DIR)
    h_tokenizer = LexiconTokenizer(h_lexicon, max_len=10)
    for data in (train_data_h, dev_data_h, test_data_h):
        for entry in data:
            entry["tokens"].build_expert_dict_tags(h_tokenizer.tokenize)
    
    # 加载 H 组模型
    print(f"加载 H 组模型: {H_MODEL_PATH}")
    h_model, h_config = load_model_and_config(H_MODEL_PATH, H_CONFIG_PATH, device)
    
    # H 组预测
    print("H 组预测中...")
    h_pred_chunks, h_gold_chunks, h_test_set = run_prediction(
        h_model, h_config, test_data_h, device, batch_size=BATCH_SIZE
    )
    print(f"H 组预测完成，共 {len(h_pred_chunks)} 个样本")
    
    # 3. 加载 M 组模型并预测
    print("\n" + "-" * 60)
    print("【M 组 (sb_size=3)】加载模型...")
    print("-" * 60)
    
    # 加载 M 组词典
    m_lexicon = load_expert_lexicon(M_LEXICON_PATH)
    print(f"M 组词典大小: {len(m_lexicon)} 个词")
    
    # 重新加载数据，添加 M 组词典特征
    train_data_m, dev_data_m, test_data_m = load_redjujube_data(DATA_DIR)
    m_tokenizer = LexiconTokenizer(m_lexicon, max_len=10)
    for data in (train_data_m, dev_data_m, test_data_m):
        for entry in data:
            entry["tokens"].build_expert_dict_tags(m_tokenizer.tokenize)
    
    # 加载 M 组模型
    print(f"加载 M 组模型: {M_MODEL_PATH}")
    m_model, m_config = load_model_and_config(M_MODEL_PATH, M_CONFIG_PATH, device)
    
    # M 组预测
    print("M 组预测中...")
    m_pred_chunks, m_gold_chunks, m_test_set = run_prediction(
        m_model, m_config, test_data_m, device, batch_size=BATCH_SIZE
    )
    print(f"M 组预测完成，共 {len(m_pred_chunks)} 个样本")
    
    # 4. 计算指标
    print("\n" + "-" * 60)
    print("计算评估指标...")
    print("-" * 60)
    
    h_scores, h_ave_scores = precision_recall_f1_report(h_gold_chunks, h_pred_chunks, macro_over="types")
    m_scores, m_ave_scores = precision_recall_f1_report(m_gold_chunks, m_pred_chunks, macro_over="types")
    
    # ==================== 输出分析结果 ====================
    
    print("\n")
    print("#" * 80)
    print("#" + " " * 25 + "对比分析结果报告" + " " * 25 + "#")
    print("#" * 80)
    
    # --------------- 总体指标 ---------------
    print("\n" + "=" * 80)
    print("  0. 总体指标对比")
    print("=" * 80)
    
    h_micro = h_ave_scores["micro"]
    m_micro = m_ave_scores["micro"]
    
    print(f"\n  {'Model':<12} {'sb_size':<10} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("  " + "-" * 60)
    print(f"  {'H 组':<12} {'2':<10} {h_micro['precision']:>12.4f} {h_micro['recall']:>12.4f} {h_micro['f1']:>12.4f}")
    print(f"  {'M 组':<12} {'3':<10} {m_micro['precision']:>12.4f} {m_micro['recall']:>12.4f} {m_micro['f1']:>12.4f}")
    print(f"  {'差值(H-M)':<12} {'':<10} {h_micro['precision']-m_micro['precision']:>+12.4f} {h_micro['recall']-m_micro['recall']:>+12.4f} {h_micro['f1']-m_micro['f1']:>+12.4f}")
    
    # --------------- 1. 按实体长度的 F1 对比 ---------------
    print("\n" + "=" * 80)
    print("  1. 按实体长度的 F1 对比")
    print("=" * 80)
    
    h_length_stats = analyze_by_length(h_gold_chunks, h_pred_chunks)
    m_length_stats = analyze_by_length(m_gold_chunks, m_pred_chunks)
    
    bucket_order = ["1字", "2字", "3字", "4-5字", "6字+"]
    
    print(f"\n  {'长度':<8} {'Gold':>8} │ {'H组 F1':>10} {'H组 P':>10} {'H组 R':>10} │ {'M组 F1':>10} {'M组 P':>10} {'M组 R':>10} │ {'H-M F1':>10}")
    print("  " + "-" * 105)
    
    for bucket in bucket_order:
        h_stats = h_length_stats.get(bucket, {'gold': 0, 'f1': 0, 'precision': 0, 'recall': 0})
        m_stats = m_length_stats.get(bucket, {'gold': 0, 'f1': 0, 'precision': 0, 'recall': 0})
        
        gold_count = h_stats.get('gold', 0)
        h_f1 = h_stats.get('f1', 0)
        h_p = h_stats.get('precision', 0)
        h_r = h_stats.get('recall', 0)
        m_f1 = m_stats.get('f1', 0)
        m_p = m_stats.get('precision', 0)
        m_r = m_stats.get('recall', 0)
        diff = h_f1 - m_f1
        
        diff_str = f"{diff:>+10.4f}"
        print(f"  {bucket:<8} {gold_count:>8} │ {h_f1:>10.4f} {h_p:>10.4f} {h_r:>10.4f} │ {m_f1:>10.4f} {m_p:>10.4f} {m_r:>10.4f} │ {diff_str}")
    
    # --------------- 2. 按实体类型的 F1 对比 ---------------
    print("\n" + "=" * 80)
    print("  2. 按实体类型的 F1 对比 (按 H-M 差值排序)")
    print("=" * 80)
    
    # 合并所有类型
    all_types = set(h_scores.keys()) | set(m_scores.keys())
    
    # 计算差值并排序
    type_diffs = []
    for t in all_types:
        h_f1 = h_scores.get(t, {}).get('f1', 0)
        m_f1 = m_scores.get(t, {}).get('f1', 0)
        diff = h_f1 - m_f1
        type_diffs.append((t, h_f1, m_f1, diff))
    
    type_diffs.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\n  {'类型':<15} {'Gold':>8} │ {'H组 F1':>10} {'H组 P':>10} {'H组 R':>10} │ {'M组 F1':>10} {'M组 P':>10} {'M组 R':>10} │ {'H-M F1':>10}")
    print("  " + "-" * 115)
    
    for t, h_f1, m_f1, diff in type_diffs:
        h_s = h_scores.get(t, {'precision': 0, 'recall': 0, 'f1': 0, 'n_gold': 0})
        m_s = m_scores.get(t, {'precision': 0, 'recall': 0, 'f1': 0, 'n_gold': 0})
        
        gold_count = h_s.get('n_gold', 0)
        h_p = h_s.get('precision', 0)
        h_r = h_s.get('recall', 0)
        m_p = m_s.get('precision', 0)
        m_r = m_s.get('recall', 0)
        
        diff_str = f"{diff:>+10.4f}"
        print(f"  {t:<15} {gold_count:>8} │ {h_f1:>10.4f} {h_p:>10.4f} {h_r:>10.4f} │ {m_f1:>10.4f} {m_p:>10.4f} {m_r:>10.4f} │ {diff_str}")
    
    # --------------- 3. 长实体（6+字）的详细漏检案例 ---------------
    print("\n" + "=" * 80)
    print("  3. 长实体（6+字）漏检案例对比")
    print("=" * 80)
    
    h_long_fn = collect_long_entity_fn(test_data_h, h_gold_chunks, h_pred_chunks, min_length=6)
    m_long_fn = collect_long_entity_fn(test_data_m, m_gold_chunks, m_pred_chunks, min_length=6)
    
    print(f"\n  H 组（sb_size=2）漏检长实体: {len(h_long_fn)} 个")
    print(f"  M 组（sb_size=3）漏检长实体: {len(m_long_fn)} 个")
    
    # 创建漏检实体文本集合，用于对比
    h_fn_texts = {(fn['text'], fn['type']) for fn in h_long_fn}
    m_fn_texts = {(fn['text'], fn['type']) for fn in m_long_fn}
    
    # H 漏检但 M 正确（M 比 H 好的地方）
    h_only_fn = h_fn_texts - m_fn_texts
    # M 漏检但 H 正确（H 比 M 好的地方）
    m_only_fn = m_fn_texts - h_fn_texts
    # 两者都漏检
    both_fn = h_fn_texts & m_fn_texts
    
    print(f"\n  仅 H 组漏检（M 组正确识别）: {len(h_only_fn)} 个")
    print(f"  仅 M 组漏检（H 组正确识别）: {len(m_only_fn)} 个")
    print(f"  两组都漏检: {len(both_fn)} 个")
    
    # 详细显示 H 独有漏检（说明 sb_size=3 更好）
    print("\n  " + "-" * 70)
    print("  【H 组独有漏检】(sb_size=3 比 sb_size=2 更好的地方)")
    print("  " + "-" * 70)
    
    h_only_fn_details = [fn for fn in h_long_fn if (fn['text'], fn['type']) in h_only_fn]
    h_only_fn_details.sort(key=lambda x: x['length'], reverse=True)
    
    for i, fn in enumerate(h_only_fn_details[:15], 1):
        print(f"  {i:2d}. [{fn['type']}] \"{fn['text']}\" (长度={fn['length']}字)")
        print(f"      {fn['context']}")
    
    if len(h_only_fn_details) > 15:
        print(f"  ... 还有 {len(h_only_fn_details) - 15} 个")
    
    # 详细显示 M 独有漏检（说明 sb_size=2 更好）
    print("\n  " + "-" * 70)
    print("  【M 组独有漏检】(sb_size=2 比 sb_size=3 更好的地方)")
    print("  " + "-" * 70)
    
    m_only_fn_details = [fn for fn in m_long_fn if (fn['text'], fn['type']) in m_only_fn]
    m_only_fn_details.sort(key=lambda x: x['length'], reverse=True)
    
    for i, fn in enumerate(m_only_fn_details[:15], 1):
        print(f"  {i:2d}. [{fn['type']}] \"{fn['text']}\" (长度={fn['length']}字)")
        print(f"      {fn['context']}")
    
    if len(m_only_fn_details) > 15:
        print(f"  ... 还有 {len(m_only_fn_details) - 15} 个")
    
    # 两组都漏检的案例
    print("\n  " + "-" * 70)
    print("  【两组都漏检】(两个 sb_size 都无法识别)")
    print("  " + "-" * 70)
    
    both_fn_details = [fn for fn in h_long_fn if (fn['text'], fn['type']) in both_fn]
    both_fn_details.sort(key=lambda x: x['length'], reverse=True)
    
    for i, fn in enumerate(both_fn_details[:15], 1):
        print(f"  {i:2d}. [{fn['type']}] \"{fn['text']}\" (长度={fn['length']}字)")
        print(f"      {fn['context']}")
    
    if len(both_fn_details) > 15:
        print(f"  ... 还有 {len(both_fn_details) - 15} 个")
    
    # --------------- 总结 ---------------
    print("\n" + "=" * 80)
    print("  分析总结")
    print("=" * 80)
    
    print(f"""
  【sb_size 对长实体识别的影响分析】
  
  1. 总体表现:
     - H 组 (sb_size=2): Micro F1 = {h_micro['f1']:.4f}
     - M 组 (sb_size=3): Micro F1 = {m_micro['f1']:.4f}
     - 差异: {h_micro['f1'] - m_micro['f1']:+.4f}
  
  2. 长实体 (6+字) 表现:
     - H 组 F1: {h_length_stats.get('6字+', {}).get('f1', 0):.4f} (Recall: {h_length_stats.get('6字+', {}).get('recall', 0):.4f})
     - M 组 F1: {m_length_stats.get('6字+', {}).get('f1', 0):.4f} (Recall: {m_length_stats.get('6字+', {}).get('recall', 0):.4f})
     - 差异: {h_length_stats.get('6字+', {}).get('f1', 0) - m_length_stats.get('6字+', {}).get('f1', 0):+.4f}
  
  3. 漏检对比:
     - H 独有漏检 (M 正确): {len(h_only_fn)} 个长实体
     - M 独有漏检 (H 正确): {len(m_only_fn)} 个长实体
     - 共同漏检: {len(both_fn)} 个长实体
    """)
    
    print("=" * 80)
    print("  分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
