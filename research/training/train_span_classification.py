#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube NER 训练脚本 - Span-based 分类解码

使用 Span Classification 解码器进行 NER，替代传统的序列标注方法。
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
    SpanClassificationDecoderConfig,
)

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


def build_span_classification_config(args):
    """构建 Span Classification 配置"""
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

    decoder_config = SpanClassificationDecoderConfig(
        agg_mode=args.agg_mode,
        max_span_size=args.max_span_size,
        size_emb_dim=args.size_emb_dim,
        multilabel=args.multilabel,
        conf_thresh=args.conf_thresh,
        fl_gamma=args.fl_gamma,
        sl_epsilon=args.sl_epsilon,
        neg_sampling_rate=args.neg_sampling_rate,
        neg_sampling_power_decay=args.neg_sampling_power_decay,
        neg_sampling_surr_rate=args.neg_sampling_surr_rate,
        neg_sampling_surr_size=args.neg_sampling_surr_size,
        nested_sampling_rate=args.nested_sampling_rate,
        sb_epsilon=args.sb_epsilon,
        sb_size=args.sb_size,
        sb_adj_factor=args.sb_adj_factor,
        in_drop_rates=(args.dropout, 0.0, 0.0),
    )

    config = ExtractorConfig(
        bert_like=bert_config,
        nested_ohots=None,
        intermediate2=None,
        decoder=decoder_config,
    )
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="RedJujube NER - Span Classification 解码")
    
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--save_dir", type=str, required=True, help="保存目录")
    
    parser.add_argument("--bert_arch", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Span Classification 参数
    parser.add_argument("--agg_mode", type=str, default="max_pooling", 
                        choices=["max_pooling", "mean_pooling", "first_last"],
                        help="跨度聚合方式")
    parser.add_argument("--max_span_size", type=int, default=10, help="最大跨度大小")
    parser.add_argument("--size_emb_dim", type=int, default=25, help="跨度大小嵌入维度")
    
    parser.add_argument("--multilabel", action="store_true", default=False, help="多标签模式")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--fl_gamma", type=float, default=0.0, help="Focal Loss gamma")
    parser.add_argument("--sl_epsilon", type=float, default=0.0, help="标签平滑 epsilon")
    
    parser.add_argument("--neg_sampling_rate", type=float, default=1.0, help="负采样率")
    parser.add_argument("--neg_sampling_power_decay", type=float, default=0.0, help="负采样衰减")
    parser.add_argument("--neg_sampling_surr_rate", type=float, default=0.0, help="周围负采样率")
    parser.add_argument("--neg_sampling_surr_size", type=int, default=5, help="周围采样窗口")
    parser.add_argument("--nested_sampling_rate", type=float, default=1.0, help="嵌套跨度采样率")
    
    parser.add_argument("--sb_epsilon", type=float, default=0.0, help="边界平滑 epsilon")
    parser.add_argument("--sb_size", type=int, default=1, help="边界平滑窗口大小")
    parser.add_argument("--sb_adj_factor", type=float, default=1.0, help="边界平滑调整因子")
    
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tensorboard", dest="use_tensorboard", action="store_false",
                        help="禁用 TensorBoard")
    
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
                entry["tokens"] = tokens[:max_char_len]
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
    model_name = "span_classification"
    save_dir = os.path.join(args.save_dir, f"{model_name}_{timestamp}")
    
    logger = LoggerManager.setup_logger(save_dir)
    
    logger.info("=" * 70)
    logger.info("RedJujube NER - Span Classification 解码")
    logger.info("=" * 70)
    logger.info(f"聚合方式: {args.agg_mode}")
    logger.info(f"最大跨度: {args.max_span_size}")
    logger.info(f"保存目录: {save_dir}")
    
    train_data, dev_data, test_data = load_redjujube_data(args.data_dir)
    logger.info(f"加载数据: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")

    num_trunc = truncate_long_sequences(
        (train_data, dev_data, test_data), max_char_len=510
    )
    if num_trunc > 0:
        logger.info(f"截断过长样本: {num_trunc} 条 (max_len=510)")
    
    model_config = build_span_classification_config(args)
    
    train_config = RedJujubeTrainerConfig(args, save_dir, model_name)
    trainer = RedJujubeNERTrainer(train_config, logger)
    
    results = trainer.train(
        model_config=model_config,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data,
        use_expert_dict=False
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("训练完成！")
    logger.info(f"最终测试 F1: {results.get('test_f1', 0):.4f}")
    logger.info(f"结果保存在: {save_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
