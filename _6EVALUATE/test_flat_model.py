#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【实验】FLAT 模型测试评估脚本

使用与基线模型相同的测试函数进行评估，确保指标计算一致性
"""

import argparse
import os
import sys
import json
import torch
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from _4MODELS.models.flat_data_processor import FLATDataProcessor, load_word_list
from _4MODELS.models.flat_extractor import FLATModel
from _5TRAIN.redjujube_trainer import RedJujubeNERTrainer
from eznlp.dataset import Dataset
from eznlp.token import Token


def load_bmes_data(file_path):
    """加载 BMES 格式数据（每行一个字符）"""
    data = []
    current_chars = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_chars:
                    data.append({'chars': current_chars, 'labels': current_labels})
                    current_chars = []
                    current_labels = []
                continue
            
            parts = line.split()
            if len(parts) == 2:
                char, label = parts
                current_chars.append(char)
                current_labels.append(label)
        
        # 添加最后一句
        if current_chars:
            data.append({'chars': current_chars, 'labels': current_labels})
    
    return data


def load_flat_model(model_path, model_config, processor, device):
    """加载训练好的 FLAT 模型"""
    model = FLATModel(
        vocab_size=len(processor.char_vocab),
        label_size=len(processor.label_vocab),
        hidden_size=model_config['hidden_size'],
        embed_size=50,
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        max_seq_len=256,
        dropout=model_config['dropout'],
        use_bigram=False,
        use_bert=False,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def extract_entities_from_bmes(labels):
    """从 BMES 标签序列中提取实体块
    
    Args:
        labels: List[str] BMES 格式标签列表
        
    Returns:
        List[Tuple[str, int, int]]: [(entity_type, start, end), ...]
    """
    entities = []
    current_entity = None
    current_start = None
    
    for i, label in enumerate(labels):
        if label == 'O':
            if current_entity:
                entities.append((current_entity, current_start, i))
                current_entity = None
        elif label.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_start, i))
            current_entity = label[2:]
            current_start = i
        elif label.startswith('M-'):
            if current_entity != label[2:]:
                if current_entity:
                    entities.append((current_entity, current_start, i))
                current_entity = None
        elif label.startswith('E-'):
            if current_entity == label[2:]:
                entities.append((current_entity, current_start, i + 1))
            current_entity = None
        elif label.startswith('S-'):
            if current_entity:
                entities.append((current_entity, current_start, i))
            entities.append((label[2:], i, i + 1))
            current_entity = None
    
    if current_entity:
        entities.append((current_entity, current_start, len(labels)))
    
    return entities


def convert_predictions_to_entity_tuples(predictions, char_list, processor):
    """将 FLAT 预测转换为实体元组格式
    
    Args:
        predictions: List[List[int]] FLAT 模型预测的标签 ID
        char_list: List[List[str]] 字符列表
        processor: FLATDataProcessor 实例
        
    Returns:
        List[List[Tuple]]: 每个句子的实体元组列表 [(type, start, end), ...]
    """
    all_entities = []
    
    for chars, pred_ids in zip(char_list, predictions):
        # 转换标签 ID 为标签字符串
        pred_labels = [processor.idx2label[i] for i in pred_ids[:len(chars)]]
        
        # 提取实体元组
        entities = extract_entities_from_bmes(pred_labels)
        all_entities.append(entities)
    
    return all_entities


def convert_gold_to_entity_tuples(test_data):
    """将金标数据转换为实体元组格式
    
    Returns:
        List[List[Tuple]]: 每个句子的实体元组列表
    """
    all_entities = []
    for data in test_data:
        entities = extract_entities_from_bmes(data['labels'])
        all_entities.append(entities)
    return all_entities


def evaluate_with_baseline_function(model, processor, test_data, device, bert_model=None, bert_tokenizer=None):
    """使用基线模型相同的评估函数"""
    from tqdm import tqdm
    
    # 1. 获取 FLAT 模型预测
    all_predictions = []
    all_chars = []
    
    model.eval()
    print(f"\n正在预测 {len(test_data)} 条测试数据...")
    with torch.no_grad():
        for data in tqdm(test_data, desc="Predicting"):
            chars = data['chars']
            processed = processor.process_sentence(chars, data['labels'])
            
            # 转换为 tensor
            lattice = torch.tensor([processed['lattice_ids']], dtype=torch.long).to(device)
            pos_s = torch.tensor([processed['pos_s']], dtype=torch.long).to(device)
            pos_e = torch.tensor([processed['pos_e']], dtype=torch.long).to(device)
            seq_len = torch.tensor([processed['seq_len']], dtype=torch.long).to(device)
            lex_num = torch.tensor([processed['lex_num']], dtype=torch.long).to(device)
            
            # 预测
            output = model(lattice, seq_len, lex_num, pos_s, pos_e)
            preds = output['pred'][0][:processed['seq_len']]
            
            if isinstance(preds, torch.Tensor):
                all_predictions.append(preds.cpu().tolist())
            else:
                all_predictions.append(preds)
            all_chars.append(chars)
    
    # 2. 转换为实体元组格式
    pred_entity_tuples = convert_predictions_to_entity_tuples(all_predictions, all_chars, processor)
    gold_entity_tuples = convert_gold_to_entity_tuples(test_data)
    
    # 3. 使用 eznlp 评估函数
    from eznlp.metrics import precision_recall_f1_report
    
    # 计算实体级别 F1
    scores, ave_scores = precision_recall_f1_report(gold_entity_tuples, pred_entity_tuples)
    
    # 返回 micro 平均指标（与基线一致）
    micro = ave_scores['micro']
    return {
        'precision': micro['precision'],
        'recall': micro['recall'],
        'f1': micro['f1'],
        'n_gold': micro['n_gold'],
        'n_pred': micro['n_pred'],
        'n_true_positive': micro['n_true_positive'],
        'per_type_scores': scores
    }


def main():
    parser = argparse.ArgumentParser(description='FLAT 模型测试评估')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='模型保存目录')
    parser.add_argument('--data_dir', type=str, default='_2DATA/RedJujube',
                        help='数据目录')
    parser.add_argument('--word_file', type=str, default='assets/vectors/ctb.50d.vec',
                        help='词表文件')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    args = parser.parse_args()
    
    print("="*70)
    print("【实验】FLAT 模型测试评估（使用基线相同的评估函数）")
    print("="*70)
    
    # 加载结果配置
    results_path = os.path.join(args.model_dir, 'results.json')
    if not os.path.exists(results_path):
        print(f"❌ 未找到结果文件: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    model_config = results['model_config']
    print(f"\n模型配置:")
    print(f"  Hidden Size: {model_config['hidden_size']}")
    print(f"  Num Layers: {model_config['num_layers']}")
    print(f"  Num Heads: {model_config['num_heads']}")
    print(f"  Dropout: {model_config['dropout']}")
    
    # 加载词表
    print("\n[1/4] 加载词表...")
    softlex_file = os.path.join(args.data_dir, 'softlexicon_train.txt')
    if os.path.exists(softlex_file):
        word_list = load_word_list(softlex_file)
    else:
        print(f"⚠️ 未找到词表文件，使用空词表")
        word_list = []
    
    # 创建数据处理器
    processor = FLATDataProcessor(word_list, max_seq_len=256)
    
    # 加载数据（需要全部数据来构建与训练时一致的词汇表）
    print("\n[2/4] 加载数据...")
    train_file = os.path.join(args.data_dir, 'redjujube_train.bmes')
    dev_file = os.path.join(args.data_dir, 'redjujube_dev.bmes')
    test_file = os.path.join(args.data_dir, 'redjujube_test.bmes')
    
    train_data = load_bmes_data(train_file)
    dev_data = load_bmes_data(dev_file)
    test_data = load_bmes_data(test_file)
    
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(dev_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    # 构建词汇表（使用全部数据）
    print("\n[3/4] 构建词汇表...")
    all_chars = set()
    all_labels = set(['O'])
    for data in train_data + dev_data + test_data:
        all_chars.update(data['chars'])
        all_labels.update(data['labels'])
    
    # 字符词汇表
    processor.char_vocab = {'<PAD>': 0, '<UNK>': 1}
    processor.idx2char = {0: '<PAD>', 1: '<UNK>'}
    for i, char in enumerate(sorted(all_chars)):
        idx = len(processor.char_vocab)
        processor.char_vocab[char] = idx
        processor.idx2char[idx] = char
    
    # 标签词汇表
    processor.label_vocab = {'<PAD>': 0}
    processor.idx2label = {0: '<PAD>'}
    for label in sorted(all_labels):
        if label not in processor.label_vocab:
            idx = len(processor.label_vocab)
            processor.label_vocab[label] = idx
            processor.idx2label[idx] = label
    
    print(f"  字符数量: {len(processor.char_vocab)}")
    print(f"  标签数量: {len(processor.label_vocab)}")
    
    # 加载模型
    print("\n[4/4] 加载模型并评估...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(args.model_dir, 'best_model.pt')
    
    model = load_flat_model(model_path, model_config, processor, device)
    
    # 使用基线评估函数
    print("\n使用基线模型相同的评估函数计算指标...")
    metrics = evaluate_with_baseline_function(model, processor, test_data, device)
    
    print("\n" + "="*70)
    print("【实验】测试结果（与基线一致的评估方式）")
    print("="*70)
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"  Gold:      {metrics['n_gold']} 个实体")
    print(f"  Pred:      {metrics['n_pred']} 个预测")
    print(f"  TP:        {metrics['n_true_positive']} 个正确")
    print("="*70)
    
    # 打印各类别指标
    if 'per_type_scores' in metrics:
        print("\n各实体类型指标:")
        print("-" * 50)
        for etype, score in sorted(metrics['per_type_scores'].items()):
            print(f"  {etype:20s}: P={score['precision']:.4f} R={score['recall']:.4f} F1={score['f1']:.4f}")
    
    # 对比原评估结果
    original_metrics = results['test_metrics']
    print(f"\n原评估结果（FLAT 自带评估函数）:")
    print(f"  Precision: {original_metrics['precision']:.4f} ({original_metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {original_metrics['recall']:.4f} ({original_metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {original_metrics['f1']:.4f} ({original_metrics['f1']*100:.2f}%)")
    
    print(f"\n\n差异（基线评估 - 原评估）:")
    print(f"  ΔPrecision: {(metrics['precision'] - original_metrics['precision'])*100:+.2f}%")
    print(f"  ΔRecall:    {(metrics['recall'] - original_metrics['recall'])*100:+.2f}%")
    print(f"  ΔF1:        {(metrics['f1'] - original_metrics['f1'])*100:+.2f}%")
    
    # 保存对比结果
    save_metrics = {
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'n_gold': metrics['n_gold'],
        'n_pred': metrics['n_pred'],
        'n_true_positive': metrics['n_true_positive']
    }
    comparison = {
        'model_dir': args.model_dir,
        'baseline_evaluation': save_metrics,
        'original_evaluation': original_metrics,
        'difference': {
            'precision': metrics['precision'] - original_metrics['precision'],
            'recall': metrics['recall'] - original_metrics['recall'],
            'f1': metrics['f1'] - original_metrics['f1']
        }
    }
    
    output_path = os.path.join(args.model_dir, 'baseline_evaluation_comparison.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 对比结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
