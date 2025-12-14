#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【实验】完整 FLAT 模型训练脚本 - RedJujube 数据集

使用自己构建的 FLAT 模型组件进行 NER 训练

使用方式:
    python train_flat_complete.py --data_dir _2DATA/RedJujube --word_file assets/vectors/ctb.50d.vec
"""

import argparse
import os
import sys
import datetime
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入 FLAT 组件
from _4MODELS.models.flat_data_processor import FLATDataProcessor, FLATDataset, load_word_list
from _4MODELS.models.flat_extractor import FLATModel, FLATModelWithBERT


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='完整 FLAT 模型训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='_2DATA/RedJujube',
                        help='数据目录')
    parser.add_argument('--word_file', type=str, default='assets/vectors/ctb.50d.vec',
                        help='词表文件路径（用于构建 Trie）')
    parser.add_argument('--save_dir', type=str, default='cache/flat_complete',
                        help='模型保存目录')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='隐藏层维度')
    parser.add_argument('--embed_size', type=int, default=50,
                        help='嵌入维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Transformer 层数')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--ff_size', type=int, default=-1,
                        help='前馈网络维度（-1 表示 hidden_size * 4）')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='最大序列长度')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout 率')
    parser.add_argument('--four_pos_fusion', type=str, default='ff',
                        choices=['ff', 'attn', 'gate', 'ff_linear', 'ff_two'],
                        help='四位置融合方式')
    parser.add_argument('--use_bert', action='store_true',
                        help='是否使用 BERT 嵌入')
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese',
                        help='BERT 模型名称')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Warmup 步数')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='梯度裁剪')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--eval_every', type=int, default=200,
                        help='每 N 步评估一次')
    
    return parser.parse_args()


def load_bmes_data(file_path):
    """加载 BMES 格式的 NER 数据"""
    data = []
    chars = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if chars:
                    data.append({'chars': chars, 'labels': labels})
                    chars = []
                    labels = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    chars.append(parts[0])
                    labels.append(parts[1])
    
    if chars:
        data.append({'chars': chars, 'labels': labels})
    
    return data


def evaluate(model, dataloader, device, bert_model=None, bert_tokenizer=None, use_bert=False, idx2label=None):
    """评估模型（使用实体级别 F1）"""
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            lattice = batch['lattice'].to(device)
            pos_s = batch['pos_s'].to(device)
            pos_e = batch['pos_e'].to(device)
            seq_len = batch['seq_len'].to(device)
            lex_num = batch['lex_num'].to(device)
            target = batch['target'].to(device)
            
            # 提取 BERT embeddings
            bert_embed = None
            if use_bert and bert_model is not None:
                chars_list = batch['chars']
                batch_size = len(chars_list)
                max_char_len = max(len(chars) for chars in chars_list)
                bert_embeds = []
                
                for chars in chars_list:
                    text = ''.join(chars)
                    inputs = bert_tokenizer(
                        text,
                        return_tensors='pt',
                        add_special_tokens=False,
                        truncation=True,
                        max_length=len(chars)
                    ).to(device)
                    outputs = bert_model(**inputs)
                    char_embeds = outputs.last_hidden_state[0]
                    
                    if len(char_embeds) != len(chars):
                        aligned = []
                        token_per_char = len(char_embeds) / len(chars)
                        for i in range(len(chars)):
                            start_idx = int(i * token_per_char)
                            end_idx = int((i + 1) * token_per_char)
                            if end_idx > start_idx:
                                aligned.append(char_embeds[start_idx:end_idx].mean(0))
                            else:
                                aligned.append(char_embeds[start_idx])
                        char_embeds = torch.stack(aligned)
                    
                    bert_embeds.append(char_embeds[:len(chars)])
                
                bert_embed = torch.zeros(batch_size, max_char_len, 768, device=device)
                for i, emb in enumerate(bert_embeds):
                    bert_embed[i, :len(emb)] = emb
            
            # 计算损失
            model.train()
            output = model(lattice, seq_len, lex_num, pos_s, pos_e, target, bert_embed=bert_embed)
            total_loss += output['loss'].item()
            num_batches += 1
            
            # 获取预测
            model.eval()
            output = model(lattice, seq_len, lex_num, pos_s, pos_e, bert_embed=bert_embed)
            preds = output['pred']
            
            # 收集预测和目标
            for i, (pred, length) in enumerate(zip(preds, seq_len)):
                all_preds.append(pred[:length.item()])
                all_targets.append(target[i, :length.item()].cpu().tolist())
    
    # 计算 F1（实体级别）
    tp, fp, fn = 0, 0, 0
    for pred, gold in zip(all_preds, all_targets):
        pred_entities = extract_entities(pred, idx2label)
        gold_entities = extract_entities(gold, idx2label)
        
        for entity in pred_entities:
            if entity in gold_entities:
                tp += 1
            else:
                fp += 1
        
        for entity in gold_entities:
            if entity not in pred_entities:
                fn += 1
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    return {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def extract_entities(labels, idx2label=None):
    """从 BMES 标签中提取实体（实体级别评估）
    
    Args:
        labels: 标签 ID 列表
        idx2label: ID 到标签字符串的映射
        
    Returns:
        set of (start, end, entity_type) tuples
    """
    entities = []
    current_entity = None
    current_start = None
    
    for i, label_id in enumerate(labels):
        # 转换为标签字符串
        if idx2label is not None:
            label = idx2label.get(label_id, 'O')
        else:
            label = str(label_id)
        
        if label == 'O' or label == '<PAD>' or label_id == 0:
            if current_entity:
                entities.append((current_start, i, current_entity))
                current_entity = None
        elif label.startswith('B-'):
            if current_entity:
                entities.append((current_start, i, current_entity))
            current_entity = label[2:]
            current_start = i
        elif label.startswith('M-'):
            if current_entity != label[2:]:
                if current_entity:
                    entities.append((current_start, i, current_entity))
                current_entity = None
        elif label.startswith('E-'):
            if current_entity == label[2:]:
                entities.append((current_start, i + 1, current_entity))
            current_entity = None
        elif label.startswith('S-'):
            if current_entity:
                entities.append((current_start, i, current_entity))
            entities.append((i, i + 1, label[2:]))
            current_entity = None
    
    if current_entity:
        entities.append((current_start, len(labels), current_entity))
    
    return set(entities)


def train(args):
    """训练主函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{args.save_dir}/flat_{timestamp}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("【实验】完整 FLAT 模型训练 - RedJujube 数据集")
    print("="*70)
    
    # 加载词表
    print("\n[1/6] 加载词表...")
    if os.path.exists(args.word_file):
        word_list = load_word_list(args.word_file)
    else:
        # 使用默认词表（从 softlexicon 文件加载）
        softlex_file = os.path.join(args.data_dir, 'softlexicon_train.txt')
        if os.path.exists(softlex_file):
            word_list = load_word_list(softlex_file)
        else:
            print("⚠️  未找到词表文件，使用空词表")
            word_list = []
    
    # 初始化 BERT （如果需要）
    bert_model = None
    bert_tokenizer = None
    if args.use_bert:
        print("\n[2/6] 初始化 BERT...")
        from transformers import BertModel, BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        bert_model = BertModel.from_pretrained(args.bert_model)
        bert_model.eval()
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        bert_model.to(device)
        print(f"  BERT 模型: {args.bert_model}")
        print(f"  设备: {device}")
    
    # 创建数据处理器
    processor = FLATDataProcessor(word_list, max_seq_len=args.max_seq_len)
    
    # 加载数据
    print(f"\n[{3 if args.use_bert else 2}/6] 加载数据...")
    train_file = os.path.join(args.data_dir, 'redjujube_train.bmes')
    dev_file = os.path.join(args.data_dir, 'redjujube_dev.bmes')
    test_file = os.path.join(args.data_dir, 'redjujube_test.bmes')
    
    train_data = load_bmes_data(train_file)
    dev_data = load_bmes_data(dev_file)
    test_data = load_bmes_data(test_file)
    
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(dev_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    # 构建词汇表
    print(f"\n[{4 if args.use_bert else 3}/6] 构建词汇表...")
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
    
    # 处理数据
    def process_data(data_list):
        processed = []
        for data in data_list:
            result = processor.process_sentence(data['chars'], data['labels'])
            # 转换标签为索引
            result['target'] = [processor.label_vocab.get(l, 0) for l in data['labels'][:result['seq_len']]]
            # 转换字符为索引
            result['lattice_ids'] = []
            for item in result['lattice']:
                if len(item) == 1:
                    result['lattice_ids'].append(processor.char_vocab.get(item, 1))
                else:
                    # 对于词汇，使用第一个字符的索引（简化处理）
                    result['lattice_ids'].append(processor.char_vocab.get(item[0], 1))
            processed.append(result)
        return processed
    
    train_processed = process_data(train_data)
    dev_processed = process_data(dev_data)
    test_processed = process_data(test_data)
    
    # 创建数据集
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    def collate_fn(batch):
        batch_size = len(batch)
        max_seq_len = max(d['seq_len'] for d in batch)
        max_lattice_len = max(d['seq_len'] + d['lex_num'] for d in batch)
        
        lattice_ids = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        pos_s = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        pos_e = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        seq_len = torch.zeros(batch_size, dtype=torch.long)
        lex_num = torch.zeros(batch_size, dtype=torch.long)
        target = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        for i, d in enumerate(batch):
            cur_len = d['seq_len'] + d['lex_num']
            lattice_ids[i, :cur_len] = torch.tensor(d['lattice_ids'][:cur_len])
            pos_s[i, :cur_len] = torch.tensor(d['pos_s'][:cur_len])
            pos_e[i, :cur_len] = torch.tensor(d['pos_e'][:cur_len])
            seq_len[i] = d['seq_len']
            lex_num[i] = d['lex_num']
            target[i, :d['seq_len']] = torch.tensor(d['target'])
        
        return {
            'lattice': lattice_ids,
            'pos_s': pos_s,
            'pos_e': pos_e,
            'seq_len': seq_len,
            'lex_num': lex_num,
            'target': target,
            'chars': [d['chars'] for d in batch],  # 用于 BERT
        }
    
    train_dataset = SimpleDataset(train_processed)
    dev_dataset = SimpleDataset(dev_processed)
    test_dataset = SimpleDataset(test_processed)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    print(f"\n[{5 if args.use_bert else 4}/6] 创建模型...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = FLATModel(
        vocab_size=len(processor.char_vocab),
        label_size=len(processor.label_vocab),
        hidden_size=args.hidden_size,
        embed_size=args.embed_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_size=args.ff_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        use_bigram=False,
        use_bert=args.use_bert,
        bert_hidden_size=768 if args.use_bert else 0,
        four_pos_fusion=args.four_pos_fusion,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  设备: {device}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    total_steps = len(train_loader) * args.num_epochs
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return max(0.1, 1 - (step - args.warmup_steps) / (total_steps - args.warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练
    print(f"\n[6/6] 开始训练...")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()
    
    best_f1 = 0
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            lattice = batch['lattice'].to(device)
            pos_s = batch['pos_s'].to(device)
            pos_e = batch['pos_e'].to(device)
            seq_len = batch['seq_len'].to(device)
            lex_num = batch['lex_num'].to(device)
            target = batch['target'].to(device)
            
            # 提取 BERT embeddings
            bert_embed = None
            if args.use_bert and bert_model is not None:
                chars_list = batch['chars']  # List[List[str]]
                batch_size = len(chars_list)
                max_char_len = max(len(chars) for chars in chars_list)
                
                # 逐个处理每个句子，确保长度对齐
                bert_embeds = []
                with torch.no_grad():
                    for chars in chars_list:
                        text = ''.join(chars)
                        # tokenize 不添加特殊token
                        inputs = bert_tokenizer(
                            text,
                            return_tensors='pt',
                            add_special_tokens=False,
                            truncation=True,
                            max_length=len(chars)
                        ).to(device)
                        outputs = bert_model(**inputs)
                        char_embeds = outputs.last_hidden_state[0]  # [seq_len, 768]
                        
                        # 处理中文分词：BERT可能将一个汉字split成多个subword
                        # 简单策略：取每个字符对应的第一个token
                        if len(char_embeds) != len(chars):
                            # 使用平均池化对齐
                            aligned = []
                            token_per_char = len(char_embeds) / len(chars)
                            for i in range(len(chars)):
                                start_idx = int(i * token_per_char)
                                end_idx = int((i + 1) * token_per_char)
                                if end_idx > start_idx:
                                    aligned.append(char_embeds[start_idx:end_idx].mean(0))
                                else:
                                    aligned.append(char_embeds[start_idx])
                            char_embeds = torch.stack(aligned)
                        
                        bert_embeds.append(char_embeds[:len(chars)])  # 确保长度正确
                
                # Padding 到 batch 中最大长度
                bert_embed = torch.zeros(batch_size, max_char_len, 768, device=device)
                for i, emb in enumerate(bert_embeds):
                    bert_embed[i, :len(emb)] = emb
            
            optimizer.zero_grad()
            output = model(lattice, seq_len, lex_num, pos_s, pos_e, target, bert_embed=bert_embed)
            loss = output['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.6f}"})
            
            # 定期评估
            if global_step % args.eval_every == 0:
                dev_metrics = evaluate(model, dev_loader, device, bert_model, bert_tokenizer, args.use_bert, processor.idx2label)
                print(f"\n  Step {global_step}: Dev Loss={dev_metrics['loss']:.4f}, "
                      f"P={dev_metrics['precision']:.2%}, R={dev_metrics['recall']:.2%}, "
                      f"F1={dev_metrics['f1']:.2%}")
                
                if dev_metrics['f1'] > best_f1:
                    best_f1 = dev_metrics['f1']
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                    print(f"  ✅ 保存最佳模型 (F1={best_f1:.2%})")
                
                model.train()
        
        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1} 完成: 平均损失={avg_loss:.4f}")
    
    # 最终测试
    print("\n" + "="*70)
    print("最终测试")
    print("="*70)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, device, bert_model, bert_tokenizer, args.use_bert, processor.idx2label)
    
    print(f"\n📊 测试结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.2%}")
    print(f"  Recall: {test_metrics['recall']:.2%}")
    print(f"  F1: {test_metrics['f1']:.2%}")
    
    # 保存结果
    results = {
        'model_config': {
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'dropout': args.dropout,
        },
        'test_metrics': test_metrics,
        'best_dev_f1': best_f1,
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 结果已保存到: {save_dir}")
    
    return test_metrics


if __name__ == '__main__':
    args = parse_args()
    train(args)
