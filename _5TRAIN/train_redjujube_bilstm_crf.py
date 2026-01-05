#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube NER 训练脚本 - 仅BiLSTM-CRF（无BERT）

支持功能：
- BiLSTM编码器
- CRF解码器
- 无预训练BERT模型
"""

import argparse
import os
import sys
import logging
import datetime
import json
import torch
import numpy as np
import transformers

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from eznlp.io import ConllIO
from eznlp.model import (
    EncoderConfig,
    ExtractorConfig,
)
from eznlp.model.decoder import SequenceTaggingDecoderConfig
from eznlp.dataset import Dataset
from eznlp.training import Trainer
from eznlp.config import ConfigDict


def setup_logger(save_dir):
    """设置日志器"""
    os.makedirs(save_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f'{save_dir}/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_redjujube_data(data_dir, max_seq_len=450):
    """加载 RedJujube 数据集，截断过长序列"""
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="<pad>"
    )
    
    train_data = io.read(f"{data_dir}/redjujube_train.bmes")
    dev_data = io.read(f"{data_dir}/redjujube_dev.bmes")
    test_data = io.read(f"{data_dir}/redjujube_test.bmes")
    
    # 截断过长序列（BERT最大512，预留空间给[CLS][SEP]和特殊标记）
    def truncate_sample(sample):
        tokens = sample["tokens"]
        if len(tokens) > max_seq_len:
            # 截断 tokens 和 chunks
            sample["tokens"] = tokens[:max_seq_len]
            # 截断完成后需要过滤超出范围的 chunks
            new_chunks = []
            for chunk in sample["chunks"]:
                if chunk[1] < max_seq_len:  # start < max_seq_len
                    if chunk[2] <= max_seq_len:  # end <= max_seq_len
                        new_chunks.append(chunk)
                    else:
                        new_chunks.append((chunk[0], chunk[1], max_seq_len))
            sample["chunks"] = new_chunks
        return sample
    
    truncated_count = sum(1 for s in train_data if len(s["tokens"]) > max_seq_len)
    train_data = [truncate_sample(s) for s in train_data]
    dev_data = [truncate_sample(s) for s in dev_data]
    test_data = [truncate_sample(s) for s in test_data]
    
    if truncated_count > 0:
        print(f"⚠️  截断了 {truncated_count} 个训练样本（长度>{max_seq_len}）")
    
    return train_data, dev_data, test_data


def build_bilstm_crf_config(args):
    """构建 BiLSTM-CRF 模型配置"""
    
    encoder_config = EncoderConfig(
        arch="LSTM",
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        in_drop_rates=(args.dropout, 0.0, 0.0)
    )
    
    decoder_config = SequenceTaggingDecoderConfig(
        scheme="BMES",
        use_crf=True,
        in_drop_rates=(args.dropout,)
    )
    
    # 仅使用BiLSTM和CRF，不使用BERT
    config = ExtractorConfig(
        encoder=encoder_config,
        decoder=decoder_config
    )
    
    return config


def build_optimizer_and_scheduler(model, num_train_batches, args):
    """构建优化器和调度器"""
    # 分组参数：预训练模型用小学习率，其他用大学习率
    param_groups = []
    if hasattr(model, 'pretrained_parameters'):
        pretrained_params = list(model.pretrained_parameters())
        pretrained_param_ids = {id(p) for p in pretrained_params}
        other_params = [p for p in model.parameters() if id(p) not in pretrained_param_ids]
        
        param_groups.append({
            'params': pretrained_params,
            'lr': args.finetune_lr
        })
        param_groups.append({
            'params': other_params,
            'lr': args.lr
        })
    else:
        param_groups.append({
            'params': model.parameters(),
            'lr': args.lr
        })
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # 线性衰减 + Warmup
    num_warmup_epochs = max(2, args.num_epochs // 5)
    num_warmup_steps = num_train_batches * num_warmup_epochs
    num_total_steps = num_train_batches * args.num_epochs
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_total_steps - current_step) / float(max(1, num_total_steps - num_warmup_steps))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_model(config, train_data, dev_data, test_data, args, logger, save_dir, model_name="BiLSTM-CRF"):
    """训练模型"""
    logger.info(f"\n{'='*70}")
    logger.info(f"开始训练: {model_name}")
    logger.info(f"{'='*70}\n")
    
    # 构建数据集
    logger.info("构建数据集...")
    train_set = Dataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims(dev_data, test_data)
    
    dev_set = Dataset(dev_data, config, training=False)
    test_set = Dataset(test_data, config, training=False)
    
    logger.info(train_set.summary)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_set.collate
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dev_set.collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_set.collate
    )
    
    # 实例化模型
    logger.info("实例化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.instantiate().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    # 构建优化器和调度器
    logger.info("构建优化器和调度器...")
    optimizer, scheduler = build_optimizer_and_scheduler(model, len(train_loader), args)
    
    # 创建训练器
    trainer = Trainer(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        schedule_by_step=True,
        num_grad_acc_steps=args.num_grad_acc_steps,
        device=device,
        grad_clip=args.grad_clip,
        use_amp=args.use_amp
    )
    
    # 保存回调
    best_model_path = f"{save_dir}/best_model.pt"
    def save_callback(model):
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"✅ 保存最佳模型到: {best_model_path}")
    
    # 开始训练
    logger.info(f"\n开始训练 {args.num_epochs} 个 epoch...")
    logger.info(f"设备: {device}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"微调学习率: {args.finetune_lr}\n")
    
    trainer.train_steps(
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_epochs=args.num_epochs,
        disp_every_steps=args.disp_every_steps,
        eval_every_steps=args.eval_every_steps,
        save_callback=save_callback,
        save_by_loss=False  # 按 F1 保存
    )
    
    # 加载最佳模型并在测试集上评估
    logger.info(f"\n加载最佳模型: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    
    logger.info("在测试集上评估...")
    test_loss, *test_metrics = trainer.eval_epoch(test_loader)
    
    logger.info(f"\n{'='*70}")
    logger.info("测试集结果:")
    logger.info(f"  Loss: {test_loss:.4f}")
    if test_metrics:
        for i, metric in enumerate(test_metrics):
            logger.info(f"  Metric {i}: {metric:.4f}")
    logger.info(f"{'='*70}\n")
    
    # 保存结果
    results = {
        'model_type': model_name,
        'test_loss': float(test_loss),
        'test_metrics': [float(m) for m in test_metrics] if test_metrics else [],
        'total_params': total_params,
        'trainable_params': trainable_params,
        'args': vars(args)
    }
    
    with open(f"{save_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='RedJujube NER: BiLSTM-CRF')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='_2DATA/RedJujube',
                        help='数据目录')
    parser.add_argument('--save_dir', type=str, default='cache/redjujube_bilstm_crf',
                        help='保存目录')
    
    # 模型参数
    parser.add_argument('--hid_dim', type=int, default=256,
                        help='LSTM 隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM 层数')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout 率')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--finetune_lr', type=float, default=1e-5,
                        help='微调学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='梯度裁剪')
    parser.add_argument('--num_grad_acc_steps', type=int, default=1,
                        help='梯度累积步数')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')
    
    # 显示和评估参数
    parser.add_argument('--disp_every_steps', type=int, default=50,
                        help='每N步显示一次')
    parser.add_argument('--eval_every_steps', type=int, default=200,
                        help='每N步评估一次')
    
    # 实验参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 加载数据
    print(f"\n{'='*70}")
    print("加载 RedJujube 数据集...")
    print(f"{'='*70}\n")
    
    train_data, dev_data, test_data = load_redjujube_data(args.data_dir, max_seq_len=450)
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(dev_data)} 条")
    print(f"测试集: {len(test_data)} 条\n")
    
    # 创建保存目录
    bilstm_crf_dir = f"{args.save_dir}/bilstm_crf_{timestamp}"
    logger = setup_logger(bilstm_crf_dir)
    
    logger.info("="*70)
    logger.info("实验: BiLSTM-CRF (纯BiLSTM + CRF，无BERT)")
    logger.info("="*70)
    
    config = build_bilstm_crf_config(args)
    results = train_model(config, train_data, dev_data, test_data, args, logger, bilstm_crf_dir, model_name="BiLSTM-CRF")
    
    print(f"\n{'='*70}")
    print("实验完成!")
    print(f"{'='*70}\n")
    print(f"结果保存在: {bilstm_crf_dir}")
    print(f"测试集 F1: {results['test_metrics'][0] if results['test_metrics'] else 0.0:.4f}")


if __name__ == "__main__":
    main()