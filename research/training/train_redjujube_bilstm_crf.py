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
import datetime
import numpy as np
import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eznlp.io import ConllIO
from eznlp.model import (
    EncoderConfig,
    ExtractorConfig,
    SequenceTaggingDecoderConfig,
)
from eznlp.token import LexiconTokenizer

from redjujube_trainer import (
    RedJujubeTrainerConfig,
    LoggerManager,
    RedJujubeNERTrainer,
)


def load_redjujube_data(data_dir):
    """加载 RedJujube 数据集"""
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


def build_bilstm_crf_config(args):
    """构建 BiLSTM-CRF 组合配置"""
    
    encoder_config = EncoderConfig(
        arch="LSTM",
        in_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        in_drop_rates=(args.dropout, 0.0, 0.0),
        hid_drop_rate=args.dropout,
    )

    decoder_config = SequenceTaggingDecoderConfig(
        scheme="BMES",
        use_crf=True,
        in_drop_rates=(args.dropout,),
    )

    config = ExtractorConfig(
        encoder=encoder_config,
        decoder=decoder_config,
    )
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="RedJujube NER - 仅BiLSTM-CRF")
    
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--save_dir", type=str, required=True, help="保存目录")
    
    parser.add_argument("--emb_dim", type=int, default=100, help="词嵌入维度")
    parser.add_argument("--hid_dim", type=int, default=256, help="LSTM隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM层数")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout率")
    
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--no_tensorboard", dest="use_tensorboard", action="store_false")
    
    return parser.parse_args()


def truncate_long_sequences(datasets, max_char_len: int = 510):
    """
    将过长样本截断到 max_char_len，避免超过 BERT 的 512 上限。
    只保留完全落在截断范围内的实体 span。
    """
    num_truncated = 0
    for data in datasets:
        for entry in data:
            tokens = entry["tokens"]
            if len(tokens) > max_char_len:
                num_truncated += 1
                # 截断 token 序列
                entry["tokens"] = tokens[:max_char_len]
                # 调整实体标注：丢掉越界的 span
                chunks = entry.get("chunks", [])
                new_chunks = []
                for label, start, end in chunks:
                    if end <= max_char_len:
                        new_chunks.append((label, start, end))
                entry["chunks"] = new_chunks
    return num_truncated


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = "bilstm_crf"
    save_dir = os.path.join(args.save_dir, f"{model_name}_{timestamp}")
    
    logger = LoggerManager.setup_logger(save_dir)
    
    logger.info("=" * 70)
    logger.info("RedJujube NER - 仅BiLSTM-CRF")
    logger.info("=" * 70)
    logger.info(f"保存目录: {save_dir}")
    logger.info(f"词嵌入维度: {args.emb_dim}")
    logger.info(f"LSTM隐藏层维度: {args.hid_dim}")
    logger.info(f"LSTM层数: {args.num_layers}")
    logger.info(f"Dropout率: {args.dropout}")
    
    train_data, dev_data, test_data = load_redjujube_data(args.data_dir)
    logger.info(f"加载数据: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")

    # 对过长样本做统一截断
    num_trunc = truncate_long_sequences(
        (train_data, dev_data, test_data), max_char_len=510
    )
    if num_trunc > 0:
        logger.info(f"发现并截断过长样本: {num_trunc} 条 (长度 > 510)")
    
    logger.info("✅ 数据预处理完成")
    
    model_config = build_bilstm_crf_config(args)
    
    train_config = RedJujubeTrainerConfig(args, save_dir, model_name)
    trainer = RedJujubeNERTrainer(train_config, logger)
    
    results = trainer.train(
        model_config=model_config,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data,
        use_expert_dict=False  # 不使用专家词典
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("训练完成！")
    logger.info(f"最终测试 F1: {results.get('test_f1', 0):.4f}")
    logger.info(f"结果保存在: {save_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()