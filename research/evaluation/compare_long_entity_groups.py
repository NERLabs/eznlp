#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W-Y 组（长实体优化技术）vs H 组（基线）和 Q 组（Focal Loss）长实体对比分析
"""

import os
import sys
import json
from collections import defaultdict

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from eznlp.dataset import Dataset
from eznlp.io import ConllIO
from eznlp.metrics import precision_recall_f1_report
from eznlp.training import Trainer
from eznlp.token import LexiconTokenizer

# ====================== 配置 ======================

MODEL_CONFIGS = {
    'H': {
        'name': 'H 基线 (sb_size=2)',
        'config': 'experiments/EXP-010-optimization/results/H_bs_baseline/expert_boundary_20260318-163821/text-expert_dict-BERT-FFN-SB(0.10, 2)-config.pth',
        'model': 'experiments/EXP-010-optimization/results/H_bs_baseline/expert_boundary_20260318-163821/text-expert_dict-BERT-FFN-SB(0.10, 2).pth',
        'lexicon': 'experiments/EXP-010-optimization/results/H_bs_baseline/expert_boundary_20260318-163821/auto_lexicon.txt',
    },
    'Q': {
        'name': 'Q Focal Loss (fl_gamma=2.0)',
        'config': 'experiments/EXP-010-optimization/results/Q_bs_focal/expert_boundary_20260319-103810/text-expert_dict-BERT-FFN-SB(0.10, 2)-config.pth',
        'model': 'experiments/EXP-010-optimization/results/Q_bs_focal/expert_boundary_20260319-103810/text-expert_dict-BERT-FFN-SB(0.10, 2).pth',
        'lexicon': 'experiments/EXP-010-optimization/results/Q_bs_focal/expert_boundary_20260319-103810/auto_lexicon.txt',
    },
    'W': {
        'name': 'W Enhanced Size Emb (size_emb_dim=50)',
        'config': 'experiments/EXP-010-optimization/results/W_enhanced_size_emb/expert_boundary_20260319-143927/text-expert_dict-BERT-FFN-SB(0.10, 2)-config.pth',
        'model': 'experiments/EXP-010-optimization/results/W_enhanced_size_emb/expert_boundary_20260319-143927/text-expert_dict-BERT-FFN-SB(0.10, 2).pth',
        'lexicon': 'experiments/EXP-010-optimization/results/W_enhanced_size_emb/expert_boundary_20260319-143927/auto_lexicon.txt',
    },
    'Y': {
        'name': 'Y Size+Focal+SpanWidth (max_span=50)',
        'config': 'experiments/EXP-010-optimization/results/Y_size_focal_spanwidth/expert_boundary_20260319-154017/text-expert_dict-BERT-FFN-SB(0.10, 2)-config.pth',
        'model': 'experiments/EXP-010-optimization/results/Y_size_focal_spanwidth/expert_boundary_20260319-154017/text-expert_dict-BERT-FFN-SB(0.10, 2).pth',
        'lexicon': 'experiments/EXP-010-optimization/results/Y_size_focal_spanwidth/expert_boundary_20260319-154017/auto_lexicon.txt',
    },
}

TEST_DATA_PATH = "datasets/raw/RedJujube/redjujube_test.bmes.orig"
BATCH_SIZE = 16

# ====================== 数据加载 ======================

def load_test_data(test_path: str):
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="<pad>",
    )
    test_data = io.read(test_path)
    return test_data

def load_expert_lexicon(dict_path: str):
    lexicon = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                lexicon.append(word)
    return lexicon

# ====================== 模型加载与预测 ======================

def load_model_and_config(model_path: str, config_path: str, device: torch.device):
    config = torch.load(config_path, map_location=device, weights_only=False)
    
    # 为旧config添加缺失的属性（新版本中添加的）
    if hasattr(config, 'decoder'):
        decoder_config = config.decoder
        # 添加长实体优化属性
        if not hasattr(decoder_config, 'sb_size_map'):
            decoder_config.sb_size_map = None
        if not hasattr(decoder_config, 'enhanced_size_emb'):
            decoder_config.enhanced_size_emb = False
        if not hasattr(decoder_config, 'use_lognscaling'):
            decoder_config.use_lognscaling = False
        if not hasattr(decoder_config, 'lognscaling_base'):
            decoder_config.lognscaling_base = 512
        if not hasattr(decoder_config, 'max_span_width'):
            decoder_config.max_span_width = None
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    # 为 decoder 设置 config 对象
    if hasattr(model, 'decoder') and hasattr(config, 'decoder'):
        model.decoder.config = config.decoder
    
    # 关键：为旧模型的 decoder 添加缺失的属性
    if hasattr(model, 'decoder'):
        decoder = model.decoder
        # size_mlp 属性：新版本的增强大小嵌入功能
        if not hasattr(decoder, 'size_mlp'):
            decoder.size_mlp = None
        # max_span_width 属性
        if not hasattr(decoder, 'max_span_width'):
            decoder.max_span_width = None
        # lognscaling 相关属性
        if not hasattr(decoder, 'use_lognscaling'):
            decoder.use_lognscaling = False
        if not hasattr(decoder, 'lognscaling_base'):
            decoder.lognscaling_base = 512
    
    return model, config

def run_prediction(model, config, test_data, device: torch.device, batch_size: int = 16):
    test_set = Dataset(test_data, config, training=False)
    trainer = Trainer(model, device=device)
    pred_chunks = trainer.predict(test_set, batch_size=batch_size)
    gold_chunks = [ex["chunks"] for ex in test_set.data]
    return pred_chunks, gold_chunks

# ====================== 工具函数 ======================

def get_entity_length(span):
    if len(span) == 3:
        _, start, end = span
    elif len(span) == 4:
        _, _, start, end = span
    else:
        return 0
    return end - start

def get_length_bucket(length):
    """将实体长度映射到分桶（用户指定: 1-2字, 3-5字, 6-8字, 9+字）"""
    if 1 <= length <= 2:
        return "1-2字"
    elif 3 <= length <= 5:
        return "3-5字"
    elif 6 <= length <= 8:
        return "6-8字"
    else:
        return "9+字"

# ====================== 分析函数 ======================

def analyze_by_length(gold_chunks_list, pred_chunks_list):
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
    
    for bucket in length_stats:
        stats = length_stats[bucket]
        p = stats['tp'] / stats['pred'] if stats['pred'] > 0 else 0
        r = stats['tp'] / stats['gold'] if stats['gold'] > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        stats['precision'] = p
        stats['recall'] = r
        stats['f1'] = f1
    
    return dict(length_stats)

# ====================== 主程序 ======================

def main():
    print("=" * 90)
    print("  W-Y 组（长实体优化）vs H 组（基线）和 Q 组（Focal Loss）长实体对比分析")
    print("=" * 90)
    
    device = torch.device("cpu")
    print(f"\n运行设备: {device}")
    
    print("\n" + "-" * 70)
    print("加载旧版测试集...")
    print("-" * 70)
    
    test_data = load_test_data(TEST_DATA_PATH)
    print(f"旧版测试集大小: {len(test_data)}")
    
    predictions = {}
    
    for group_id in ['H', 'Q', 'W', 'Y']:
        print(f"\n" + "-" * 70)
        print(f"【{group_id} 组】{MODEL_CONFIGS[group_id]['name']}")
        print("-" * 70)
        
        config_path = MODEL_CONFIGS[group_id]['config']
        model_path = MODEL_CONFIGS[group_id]['model']
        lexicon_path = MODEL_CONFIGS[group_id]['lexicon']
        
        if not os.path.exists(model_path) or not os.path.exists(config_path) or not os.path.exists(lexicon_path):
            print(f"  [ERROR] 文件不完整")
            continue
        
        lexicon = load_expert_lexicon(lexicon_path)
        print(f"  加载词典: {len(lexicon)} 个词")
        
        test_data_group = load_test_data(TEST_DATA_PATH)
        tokenizer = LexiconTokenizer(lexicon, max_len=10)
        for entry in test_data_group:
            entry["tokens"].build_expert_dict_tags(tokenizer.tokenize)
        
        print(f"  加载模型...")
        model, config = load_model_and_config(model_path, config_path, device)
        
        print(f"  预测中...")
        pred_chunks, gold_chunks = run_prediction(
            model, config, test_data_group, device, batch_size=BATCH_SIZE
        )
        print(f"  预测完成: {len(pred_chunks)} 个样本")
        
        predictions[group_id] = {
            'pred_chunks': pred_chunks,
            'gold_chunks': gold_chunks,
        }
        
        del model, config, test_data_group
        torch.cuda.empty_cache()
    
    print("\n" + "-" * 70)
    print("计算评估指标...")
    print("-" * 70)
    
    all_scores = {}
    for group_id in ['H', 'Q', 'W', 'Y']:
        if group_id not in predictions:
            continue
        gold = predictions[group_id]['gold_chunks']
        pred = predictions[group_id]['pred_chunks']
        scores, ave_scores = precision_recall_f1_report(gold, pred, macro_over="types")
        all_scores[group_id] = {
            'detailed': scores,
            'average': ave_scores,
        }
    
    # ==================== 输出分析结果 ====================
    
    print("\n")
    print("#" * 90)
    print("#" + " " * 28 + "对比分析结果报告" + " " * 28 + "#")
    print("#" * 90)
    
    print("\n" + "=" * 90)
    print("  0. 总体指标对比")
    print("=" * 90)
    
    print(f"\n  {'组别':<12} {'说明':<30} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("  " + "-" * 85)
    
    for group_id in ['H', 'Q', 'W', 'Y']:
        if group_id not in all_scores:
            continue
        micro = all_scores[group_id]['average']['micro']
        name = MODEL_CONFIGS[group_id]['name']
        print(f"  {group_id:<12} {name:<30} {micro['precision']:>12.4f} {micro['recall']:>12.4f} {micro['f1']:>12.4f}")
    
    print("\n" + "=" * 90)
    print("  1. 按实体长度的 F1 对比")
    print("=" * 90)
    
    length_analysis = {}
    for group_id in ['H', 'Q', 'W', 'Y']:
        if group_id not in predictions:
            continue
        gold = predictions[group_id]['gold_chunks']
        pred = predictions[group_id]['pred_chunks']
        length_stats = analyze_by_length(gold, pred)
        length_analysis[group_id] = length_stats
    
    bucket_order = ["1-2字", "3-5字", "6-8字", "9+字"]
    
    print(f"\n  {'长度':<8} {'Gold':>8} │ {'H F1':>8} {'Q F1':>8} {'W F1':>8} {'Y F1':>8} │ {'W-H':>8} {'Y-H':>8} {'W-Q':>8} {'Y-Q':>8}")
    print("  " + "-" * 105)
    
    for bucket in bucket_order:
        h_stats = length_analysis.get('H', {}).get(bucket, {'gold': 0, 'f1': 0})
        q_stats = length_analysis.get('Q', {}).get(bucket, {'gold': 0, 'f1': 0})
        w_stats = length_analysis.get('W', {}).get(bucket, {'gold': 0, 'f1': 0})
        y_stats = length_analysis.get('Y', {}).get(bucket, {'gold': 0, 'f1': 0})
        
        gold_count = h_stats.get('gold', 0)
        h_f1 = h_stats.get('f1', 0)
        q_f1 = q_stats.get('f1', 0)
        w_f1 = w_stats.get('f1', 0)
        y_f1 = y_stats.get('f1', 0)
        
        w_h_diff = w_f1 - h_f1
        y_h_diff = y_f1 - h_f1
        w_q_diff = w_f1 - q_f1
        y_q_diff = y_f1 - q_f1
        
        print(f"  {bucket:<8} {gold_count:>8} │ {h_f1:>8.4f} {q_f1:>8.4f} {w_f1:>8.4f} {y_f1:>8.4f} │ {w_h_diff:>+8.4f} {y_h_diff:>+8.4f} {w_q_diff:>+8.4f} {y_q_diff:>+8.4f}")
    
    print(f"\n【详细数据 - Precision/Recall】:")
    for bucket in bucket_order:
        print(f"\n  {bucket}:")
        for group_id in ['H', 'Q', 'W', 'Y']:
            if group_id not in length_analysis:
                continue
            stats = length_analysis[group_id].get(bucket, {'f1': 0, 'precision': 0, 'recall': 0})
            print(f"    {group_id}: P={stats.get('precision', 0):.4f}, R={stats.get('recall', 0):.4f}, F1={stats.get('f1', 0):.4f}")
    
    print("\n" + "=" * 90)
    print("  分析总结")
    print("=" * 90)
    
    h_f1 = all_scores.get('H', {}).get('average', {}).get('micro', {}).get('f1', 0)
    q_f1 = all_scores.get('Q', {}).get('average', {}).get('micro', {}).get('f1', 0)
    w_f1 = all_scores.get('W', {}).get('average', {}).get('micro', {}).get('f1', 0)
    y_f1 = all_scores.get('Y', {}).get('average', {}).get('micro', {}).get('f1', 0)
    
    # 长实体统计：6-8字 + 9+字
    h_f1_68 = length_analysis.get('H', {}).get('6-8字', {}).get('f1', 0)
    q_f1_68 = length_analysis.get('Q', {}).get('6-8字', {}).get('f1', 0)
    w_f1_68 = length_analysis.get('W', {}).get('6-8字', {}).get('f1', 0)
    y_f1_68 = length_analysis.get('Y', {}).get('6-8字', {}).get('f1', 0)
    
    h_f1_9p = length_analysis.get('H', {}).get('9+字', {}).get('f1', 0)
    q_f1_9p = length_analysis.get('Q', {}).get('9+字', {}).get('f1', 0)
    w_f1_9p = length_analysis.get('W', {}).get('9+字', {}).get('f1', 0)
    y_f1_9p = length_analysis.get('Y', {}).get('9+字', {}).get('f1', 0)
    
    print(f"""
  【长实体优化技术的效果评估】
  
  1. 总体表现:
     - H 基线: F1 = {h_f1:.4f}
     - Q Focal: F1 = {q_f1:.4f} (vs H: {q_f1 - h_f1:+.4f})
     - W 增强大小嵌入: F1 = {w_f1:.4f} (vs H: {w_f1 - h_f1:+.4f})
     - Y Size+Focal+SpanWidth: F1 = {y_f1:.4f} (vs H: {y_f1 - h_f1:+.4f})
  
  2. 长实体（6-8字）表现:
     - H 基线: F1 = {h_f1_68:.4f}
     - Q Focal: F1 = {q_f1_68:.4f} (vs H: {q_f1_68 - h_f1_68:+.4f})
     - W 增强大小嵌入: F1 = {w_f1_68:.4f} (vs H: {w_f1_68 - h_f1_68:+.4f})
     - Y Size+Focal+SpanWidth: F1 = {y_f1_68:.4f} (vs H: {y_f1_68 - h_f1_68:+.4f})
  
  3. 超长实体（9+字）表现:
     - H 基线: F1 = {h_f1_9p:.4f}
     - Q Focal: F1 = {q_f1_9p:.4f} (vs H: {q_f1_9p - h_f1_9p:+.4f})
     - W 增强大小嵌入: F1 = {w_f1_9p:.4f} (vs H: {w_f1_9p - h_f1_9p:+.4f})
     - Y Size+Focal+SpanWidth: F1 = {y_f1_9p:.4f} (vs H: {y_f1_9p - h_f1_9p:+.4f})
  
  4. 结论:
     6-8字实体: W {'有改善' if w_f1_68 > h_f1_68 else '无改善'}, Y {'有改善' if y_f1_68 > h_f1_68 else '无改善'}, Q {'有改善' if q_f1_68 > h_f1_68 else '无改善'}
     9+字实体: W {'有改善' if w_f1_9p > h_f1_9p else '无改善'}, Y {'有改善' if y_f1_9p > h_f1_9p else '无改善'}, Q {'有改善' if q_f1_9p > h_f1_9p else '无改善'}
    """)
    
    print("=" * 90)
    print("  分析完成!")
    print("=" * 90)


if __name__ == "__main__":
    main()
