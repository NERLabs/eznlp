#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯净的 BERT + BiLSTM + CRF 基线模型训练脚本
无字符嵌入、无词典特征、无FGM/EMA增强
"""

import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from eznlp.dataset import Dataset
from eznlp.wrapper import Batch
from eznlp.model.bert_like import BertLikeConfig
from eznlp.nn.modules import CRF
from eznlp.io.conll import ConllIO


class PureBertBiLSTMCRF(nn.Module):
    """纯净的 BERT + BiLSTM + CRF 模型"""
    
    def __init__(self, bert_name: str, num_tags: int, hid_dim: int = 256, dropout: float = 0.5, pad_idx: int = 0):
        super().__init__()
        
        # 1. BERT 编码器
        self.bert_config = BertLikeConfig(
            arch="BERT",
            bert_name=bert_name,
            bert_max_length=512,
            truncation=True,
        )
        self.bert = self.bert_config.build()
        
        # 2. BiLSTM 层
        self.lstm = nn.LSTM(
            input_size=768,  # BERT hidden size
            hidden_size=hid_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        
        # 3. 发射层
        self.hid2tag = nn.Linear(hid_dim * 2, num_tags)
        self.num_tags = num_tags
        self.pad_idx = pad_idx
        
        # 4. CRF 层
        self.crf = CRF(tag_dim=num_tags, pad_idx=pad_idx, batch_first=True)
    
    def forward(self, tokens, attention_mask, tags=None):
        """前向传播"""
        # BERT 编码
        bert_output = self.bert(tokens, attention_mask=attention_mask, return_dict=True)
        bert_hidden = bert_output.last_hidden_state  # (batch, seq_len, 768)
        
        # BiLSTM
        lstm_out, _ = self.lstm(bert_hidden)  # (batch, seq_len, hid*2)
        lstm_out = self.dropout(lstm_out)
        
        # 发射分数
        emissions = self.hid2tag(lstm_out)  # (batch, seq_len, num_tags)
        
        if tags is not None:
            # 训练模式：返回负对数似然损失
            loss = -self.crf(emissions, tags, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            # 推理模式：返回最优路径
            return self.crf.decode(emissions, mask=attention_mask.bool())


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="纯净 BERT + BiLSTM + CRF 训练")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--save_dir", type=str, required=True, help="保存目录")
    parser.add_argument("--bert_arch", type=str, default="hfl/chinese-macbert-base", help="BERT模型")
    parser.add_argument("--hid_dim", type=int, default=256, help="LSTM隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout率")
    parser.add_argument("--num_epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-3, help="学习率")
    parser.add_argument("--finetune_lr", type=float, default=2e-5, help="BERT微调学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="梯度裁剪")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--scheme", type=str, default="BMES", help="标注方案")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度")
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"加载数据: {args.data_dir}")
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme=args.scheme,
        encoding="utf-8",
        token_sep="",
        pad_token="",
    )
    train_data = io.read(os.path.join(args.data_dir, "redjujube_train.bmes"))
    dev_data = io.read(os.path.join(args.data_dir, "redjujube_dev.bmes"))
    test_data = io.read(os.path.join(args.data_dir, "redjujube_test.bmes"))
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
    
    # 构建数据集
    train_dataset = Dataset(train_data)
    dev_dataset = Dataset(dev_data)
    test_dataset = Dataset(test_data)
    
    # 标签映射
    tag2idx = train_dataset.tag2idx
    idx2tag = {v: k for k, v in tag2idx.items()}
    num_tags = len(tag2idx)
    pad_idx = tag2idx.get('<pad>', tag2idx.get('O', 0))
    print(f"标签数量: {num_tags}, PAD idx: {pad_idx}")
    
    # 构建模型
    print("构建模型...")
    model = PureBertBiLSTMCRF(
        bert_name=args.bert_arch,
        num_tags=num_tags,
        hid_dim=args.hid_dim,
        dropout=args.dropout,
        pad_idx=pad_idx,
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 创建优化器 (BERT使用较小学习率)
    bert_params = list(model.bert.parameters())
    other_params = [p for n, p in model.named_parameters() if 'bert' not in n]
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': args.finetune_lr, 'weight_decay': args.weight_decay},
        {'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
    ])
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    best_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.num_epochs} =====")
        
        # 训练
        model.train()
        train_loss = 0
        pbar = tqdm(train_dataset.loader(batch_size=args.batch_size), desc="Training")
        
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            loss = model(batch.tokens['input_ids'], batch.tokens['attention_mask'], batch.chars)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_dataset.loader(batch_size=args.batch_size))
        print(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dev_dataset.loader(batch_size=args.batch_size), desc="Evaluating"):
                batch = batch.to(device)
                preds = model(batch.tokens['input_ids'], batch.tokens['attention_mask'])
                
                # 收集预测和标签
                for i, pred_seq in enumerate(preds):
                    seq_len = batch.tokens['attention_mask'][i].sum().item()
                    all_preds.extend(pred_seq[:seq_len])
                    all_labels.extend(batch.chars[i].cpu().tolist()[:seq_len])
        
        # 计算 F1
        from sklearn.metrics import classification_report, f1_score
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"Val Loss: -, Val F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✓ 保存最佳模型 (F1: {best_f1:.4f})")
    
    # 加载最佳模型进行测试
    print(f"\n===== 测试 (Epoch {best_epoch}) =====")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataset.loader(batch_size=args.batch_size), desc="Testing"):
            batch = batch.to(device)
            preds = model(batch.tokens['input_ids'], batch.tokens['attention_mask'])
            
            for i, pred_seq in enumerate(preds):
                seq_len = batch.tokens['attention_mask'][i].sum().item()
                all_preds.extend(pred_seq[:seq_len])
                all_labels.extend(batch.chars[i].cpu().tolist()[:seq_len])
    
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Test F1: {test_f1:.4f}")
    
    # 保存结果
    results = {
        'best_epoch': best_epoch,
        'best_f1': best_f1,
        'test_f1': test_f1,
        'idx2tag': idx2tag,
    }
    torch.save(results, os.path.join(args.save_dir, 'results.pth'))
    
    print(f"\n训练完成！最佳验证F1: {best_f1:.4f}, 测试F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
