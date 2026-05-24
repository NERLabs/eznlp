#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-006-GTRNNER 训练脚本 V2
真正集成RoPE和TriAffine模块

消融模式:
  - baseline: BERT + CRF (无任何增强)
  - no_rope: BERT + BiLSTM + TriAffine + 专家词典 + CRF (无RoPE)
  - no_triaffine: BERT + BiLSTM + RoPE + CRF (无TriAffine)
  - full: 完整模型 (BERT + RoPE + BiLSTM + TriAffine + 专家词典 + CRF)
"""

import os
import sys
import json
import logging
import argparse
import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np

# 导入模型
from model_v2 import GTRNNERModelV2, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ============== 数据处理 ==============
def load_bmes_data(filepath):
    """加载BMES格式数据"""
    sentences = []
    labels = []
    current_tokens = []
    current_labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current_tokens.append(parts[0])
                    current_labels.append(parts[1])
    
    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)
    
    return sentences, labels


def load_dict(dict_path):
    """加载专家词典"""
    word2type = {}
    word2freq = {}
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                word = parts[0]
                label = parts[1]
                freq = int(parts[2]) if len(parts) > 2 else 1
                word2type[word] = label
                word2freq[word] = freq
    
    return word2type, word2freq


def extract_dict_features(tokens, word2type, max_len=10):
    """提取词典特征
    
    返回:
        dict_match: 每个token是否在词典中
        dict_type: 每个token对应的实体类型
    """
    n = len(tokens)
    dict_match = [0] * n
    dict_type = [0] * n
    
    # 构建token到位置的映射
    for i in range(n):
        for l in range(1, min(max_len + 1, n - i + 1)):
            span = ''.join(tokens[i:i+l])
            if span in word2type:
                # 标记span内的所有token
                for j in range(i, i+l):
                    dict_match[j] = 1
                    # 使用span的类型
                    type_str = word2type[span]
                    # 简化：提取类型前缀 (如 B-PAR -> PAR)
                    if '-' in type_str:
                        dict_type[j] = type_str.split('-')[-1]
    
    return dict_match, dict_type


class NERDataset(Dataset):
    """NER数据集"""
    
    def __init__(self, sentences, labels, tokenizer, word2type, label2id, type2id, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.word2type = word2type
        self.label2id = label2id
        self.type2id = type2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labels = self.labels[idx]
        
        # 提取词典特征
        dict_match, dict_type = extract_dict_features(tokens, self.word2type)
        
        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 对齐标签
        word_ids = encoding.word_ids()
        label_ids = []
        dict_match_ids = []
        dict_type_ids = []
        
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # PyTorch忽略的标签
                dict_match_ids.append(0)
                dict_type_ids.append(0)
            else:
                label_str = labels[word_id] if word_id < len(labels) else 'O'
                label_ids.append(self.label2id.get(label_str, self.label2id['O']))
                dict_match_ids.append(dict_match[word_id] if word_id < len(dict_match) else 0)
                
                type_str = dict_type[word_id] if word_id < len(dict_type) else 0
                if isinstance(type_str, str):
                    dict_type_ids.append(self.type2id.get(type_str, 0))
                else:
                    dict_type_ids.append(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'dict_match_ids': torch.tensor(dict_match_ids, dtype=torch.long),
            'dict_type_ids': torch.tensor(dict_type_ids, dtype=torch.long)
        }


def build_label_vocab(all_labels):
    """构建标签词表"""
    label_set = set()
    for labels in all_labels:
        label_set.update(labels)
    
    label2id = {label: i for i, label in enumerate(sorted(label_set))}
    # 添加特殊标签
    if 'O' not in label2id:
        label2id['O'] = len(label2id)
    
    return label2id


def build_type_vocab():
    """构建实体类型词表"""
    types = ['O', 'AGR', 'CUL', 'DIS', 'DRU', 'EQU', 'FER', 'LOC', 'NUT', 'PAR', 'PER', 'PES', 'PRO', 'TAX', 'WED']
    type2id = {t: i for i, t in enumerate(types)}
    return type2id


def collate_fn(batch):
    """批处理函数"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'dict_match_ids': torch.stack([item['dict_match_ids'] for item in batch]),
        'dict_type_ids': torch.stack([item['dict_type_ids'] for item in batch])
    }


# ============== 评估函数 ==============
def extract_entities(labels, tokens=None):
    """从标签序列提取实体"""
    entities = []
    current_entity = None
    
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = [i, i, entity_type]
        elif label.startswith('M-') or label.startswith('I-'):
            if current_entity:
                current_entity[1] = i
        elif label.startswith('E-'):
            if current_entity:
                current_entity[1] = i
                entities.append(current_entity)
                current_entity = None
        elif label.startswith('S-'):
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]
            entities.append([i, i, entity_type])
            current_entity = None
        else:  # O
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities


def compute_f1(pred_labels_list, gold_labels_list):
    """计算F1分数"""
    tp, fp, fn = 0, 0, 0
    
    for pred_labels, gold_labels in zip(pred_labels_list, gold_labels_list):
        pred_entities = set(tuple(e) for e in extract_entities(pred_labels))
        gold_entities = set(tuple(e) for e in extract_entities(gold_labels))
        
        tp += len(pred_entities & gold_entities)
        fp += len(pred_entities - gold_entities)
        fn += len(gold_entities - pred_entities)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


# ============== 训练函数 ==============
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        dict_match_ids = batch['dict_match_ids'].to(device)
        dict_type_ids = batch['dict_type_ids'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dict_match_ids=dict_match_ids,
            dict_type_ids=dict_type_ids,
            labels=labels
        )
        
        loss = outputs['loss']
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, id2label):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_pred_labels = []
    all_gold_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            dict_match_ids = batch['dict_match_ids'].to(device)
            dict_type_ids = batch['dict_type_ids'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_match_ids=dict_match_ids,
                dict_type_ids=dict_type_ids,
                labels=labels
            )
            
            total_loss += outputs['loss'].item()
            
            # 解码
            # 简化：使用发射分数的argmax
            emit_scores = model.crf.emit(outputs['hidden'])
            pred_ids = emit_scores.argmax(dim=-1).cpu().tolist()
            gold_ids = labels.cpu().tolist()
            
            for pred_seq, gold_seq, mask in zip(pred_ids, gold_ids, attention_mask.cpu().tolist()):
                pred_labels = [id2label[p] for p, m in zip(pred_seq, mask) if m == 1]
                gold_labels = [id2label[g] for g, m in zip(gold_seq, mask) if m == 1 and g != -100]
                all_pred_labels.append(pred_labels[:len(gold_labels)])
                all_gold_labels.append(gold_labels)
    
    precision, recall, f1 = compute_f1(all_pred_labels, all_gold_labels)
    
    return total_loss / len(dataloader), precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/shiwenlong/NERlabs/eznlp/datasets/raw/RedJujube')
    parser.add_argument('--bert_path', type=str, default='/home/shiwenlong/NERlabs/eznlp/assets/transformers/hfl/chinese-macbert-base')
    parser.add_argument('--dict_path', type=str, default='/home/shiwenlong/NERlabs/eznlp/datasets/raw/RedJujube/expert_lexicon_auto.txt')
    parser.add_argument('--save_dir', type=str, default='/home/shiwenlong/NERlabs/eznlp/experiments/EXP-006-GTRNNER/results_v2')
    parser.add_argument('--ablation', type=str, default='full', choices=['baseline', 'no_rope', 'no_triaffine', 'full'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.ablation}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(save_dir, 'train.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info(f"EXP-006-GTRNNER V2 - 消融模式: {args.ablation}")
    logger.info("=" * 70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    # 加载数据
    logger.info("加载数据...")
    train_sents, train_labels = load_bmes_data(os.path.join(args.data_dir, 'redjujube_train.bmes'))
    dev_sents, dev_labels = load_bmes_data(os.path.join(args.data_dir, 'redjujube_dev.bmes'))
    test_sents, test_labels = load_bmes_data(os.path.join(args.data_dir, 'redjujube_test.bmes'))
    logger.info(f"训练集: {len(train_sents)}, 验证集: {len(dev_sents)}, 测试集: {len(test_sents)}")
    
    # 构建词表
    all_labels = train_labels + dev_labels + test_labels
    label2id = build_label_vocab(all_labels)
    id2label = {v: k for k, v in label2id.items()}
    type2id = build_type_vocab()
    logger.info(f"标签数: {len(label2id)}, 类型数: {len(type2id)}")
    
    # 加载词典
    word2type, word2freq = load_dict(args.dict_path)
    logger.info(f"词典大小: {len(word2type)}")
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    
    # 创建数据集
    train_dataset = NERDataset(train_sents, train_labels, tokenizer, word2type, label2id, type2id)
    dev_dataset = NERDataset(dev_sents, dev_labels, tokenizer, word2type, label2id, type2id)
    test_dataset = NERDataset(test_sents, test_labels, tokenizer, word2type, label2id, type2id)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # 根据消融模式配置模型
    use_rope = args.ablation != 'baseline' and args.ablation != 'no_rope'
    use_triaffine = args.ablation != 'baseline' and args.ablation != 'no_triaffine'
    use_dict = args.ablation != 'baseline' and args.ablation != 'no_triaffine'
    
    logger.info(f"配置: RoPE={use_rope}, TriAffine={use_triaffine}, Dict={use_dict}")
    
    # 创建模型
    model = GTRNNERModelV2(
        bert_path=args.bert_path,
        hidden_dim=args.hidden_dim,
        num_labels=len(label2id),
        dropout=args.dropout,
        use_rope=use_rope,
        use_triaffine=use_triaffine,
        use_dict=use_dict
    ).to(device)
    
    logger.info(f"参数量: {count_parameters(model):,}")
    
    # 优化器
    bert_params = list(model.bert.parameters())
    other_params = [p for n, p in model.named_parameters() if 'bert' not in n]
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': args.lr},
        {'params': other_params, 'lr': args.lr * 50}
    ], weight_decay=0.01)
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps)
    
    # 训练
    best_f1 = 0
    best_epoch = 0
    patience = 0
    max_patience = 5
    
    logger.info("开始训练...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        dev_loss, dev_prec, dev_rec, dev_f1 = evaluate(model, dev_loader, device, id2label)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Dev P/R/F1: {dev_prec:.4f}/{dev_rec:.4f}/{dev_f1:.4f}")
        
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch + 1
            patience = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            logger.info(f"✅ 保存最佳模型 (F1={dev_f1:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                logger.info(f"早停: {max_patience} 轮无提升")
                break
    
    # 测试
    logger.info("\n测试集评估...")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_loss, test_prec, test_rec, test_f1 = evaluate(model, test_loader, device, id2label)
    
    logger.info(f"测试集 P/R/F1: {test_prec:.4f}/{test_rec:.4f}/{test_f1:.4f}")
    logger.info(f"最佳验证F1: {best_f1:.4f} (epoch {best_epoch})")
    
    # 保存结果
    results = {
        'ablation': args.ablation,
        'config': {
            'use_rope': use_rope,
            'use_triaffine': use_triaffine,
            'use_dict': use_dict
        },
        'best_dev_f1': best_f1,
        'best_epoch': best_epoch,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"结果保存至: {save_dir}")
    
    # 移除文件处理器
    logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()
