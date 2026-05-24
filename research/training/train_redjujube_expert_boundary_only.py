#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube NER 训练脚本 - 仅边界选择（无专家词典）

支持功能：
- BoundarySelection: 边界选择解码器 (支持边界平滑)
- 无专家词典模块，仅使用边界平滑技巧
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
    BertLikeConfig,
    EncoderConfig,
    ExtractorConfig,
    BoundarySelectionDecoderConfig,
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


def build_boundary_only_config(args):
    """构建仅边界选择配置（无专家词典）"""
    bert_model = transformers.AutoModel.from_pretrained(args.bert_arch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_arch)

    bert_config = BertLikeConfig(
        tokenizer=tokenizer,
        bert_like=bert_model,
        freeze=False,
        mix_layers="top",
        bert_max_length=512,
        truncation=True,
    )

    reduction_config = EncoderConfig(
        arch=args.red_arch,
        hid_dim=args.red_dim,
        num_layers=args.red_num_layers,
        in_drop_rates=(0.0, 0.0, 0.0),
        hid_drop_rate=0.0,
    )

    decoder_config = BoundarySelectionDecoderConfig(
        reduction=reduction_config,
        size_emb_dim=args.size_emb_dim,
        multilabel=args.multilabel,
        conf_thresh=args.conf_thresh,
        fl_gamma=args.fl_gamma,
        sl_epsilon=args.sl_epsilon,
        neg_sampling_rate=args.neg_sampling_rate,
        neg_sampling_power_decay=args.neg_sampling_power_decay,
        neg_sampling_surr_rate=args.neg_sampling_surr_rate,
        neg_sampling_surr_size=args.neg_sampling_surr_size,
        sb_epsilon=args.sb_epsilon,
        sb_size=args.sb_size,
        sb_adj_factor=args.sb_adj_factor,
    )

    config = ExtractorConfig(
        bert_like=bert_config,
        nested_ohots=None,  # 不使用专家词典
        intermediate2=None,
        decoder=decoder_config,
    )
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="RedJujube NER - 仅边界选择（无专家词典）")
    
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--save_dir", type=str, required=True, help="保存目录")
    
    parser.add_argument("--bert_arch", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    
    parser.add_argument("--red_arch", type=str, default="FFN", help="边界降维架构")
    parser.add_argument("--red_dim", type=int, default=150, help="边界降维隐藏维度")
    parser.add_argument("--red_num_layers", type=int, default=1, help="边界降维层数")
    parser.add_argument("--size_emb_dim", type=int, default=25, help="跨度大小嵌入维度")
    
    parser.add_argument("--multilabel", action="store_true", default=False, help="多标签模式")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--fl_gamma", type=float, default=0.0, help="Focal Loss gamma")
    parser.add_argument("--sl_epsilon", type=float, default=0.0, help="标签平滑 epsilon")
    
    parser.add_argument("--neg_sampling_rate", type=float, default=1.0, help="负采样率")
    parser.add_argument("--neg_sampling_power_decay", type=float, default=0.0, help="负采样衰减")
    parser.add_argument("--neg_sampling_surr_rate", type=float, default=0.0, help="周围负采样率")
    parser.add_argument("--neg_sampling_surr_size", type=int, default=5, help="周围采样窗口")
    
    parser.add_argument("--sb_epsilon", type=float, default=0.1, help="边界平滑 epsilon")
    parser.add_argument("--sb_size", type=int, default=2, help="边界平滑窗口大小")
    parser.add_argument("--sb_adj_factor", type=float, default=1.0, help="边界平滑调整因子")
    
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
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
    model_name = "boundary_only"  # 修改模型名称
    save_dir = os.path.join(args.save_dir, f"{model_name}_{timestamp}")
    
    logger = LoggerManager.setup_logger(save_dir)
    
    logger.info("=" * 70)
    logger.info("RedJujube NER - 仅边界选择（无专家词典）")
    logger.info("=" * 70)
    logger.info(f"边界平滑: sb_epsilon={args.sb_epsilon}, sb_size={args.sb_size}")
    logger.info(f"保存目录: {save_dir}")
    
    train_data, dev_data, test_data = load_redjujube_data(args.data_dir)
    logger.info(f"加载数据: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")

    # 对过长样本做统一截断，避免超过 BERT 的 512 上限
    num_trunc = truncate_long_sequences(
        (train_data, dev_data, test_data), max_char_len=510
    )
    if num_trunc > 0:
        logger.info(f"发现并截断过长样本: {num_trunc} 条 (长度 > 510)")
    
    # 不需要处理专家词典特征，直接构建模型配置
    logger.info("✅ 数据预处理完成")
    
    model_config = build_boundary_only_config(args)
    
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