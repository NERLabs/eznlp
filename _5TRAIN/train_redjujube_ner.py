#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube NER 训练入口脚本 - 支持带类型的专家词典

使用 redjujube_trainer.py 中的训练器类
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
    ExpertDictConfig,
    EncoderConfig,
    ExtractorConfig,
)
from eznlp.model.decoder import SequenceTaggingDecoderConfig
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


def load_expert_lexicon(dict_path, with_type=False):
    """加载专家词典
    
    Args:
        dict_path: 词典路径
        with_type: 是否带类型（word\ttype 格式）
    """
    lexicon = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            word = parts[0]
            if word:
                if with_type and len(parts) >= 2 and parts[1]:
                    word = f"{word}_{parts[1]}"
                lexicon.append(word)
    return lexicon


def build_expert_dict_config(args):
    """构建模型配置"""
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

    nested_ohots = {}
    if args.expert_dict_path is not None:
        expert_dict_config = ExpertDictConfig(
            emb_dim=args.expert_dict_dim,
            agg_mode="wtd_mean_pooling",
            use_channel_attention=getattr(args, "use_channel_attention", False),
            channel_attn_heads=getattr(args, "channel_attn_heads", 4),
            channel_attn_dropout=getattr(args, "channel_attn_dropout", 0.1),
            channel_attn_version=getattr(args, "channel_attn_version", "v1"),
        )
        nested_ohots["expert_dict"] = expert_dict_config

    encoder_config = EncoderConfig(
        arch="LSTM",
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        in_drop_rates=(args.dropout, 0.0, 0.0),
    )

    decoder_config = SequenceTaggingDecoderConfig(
        scheme="BMES",
        use_crf=True,
        in_drop_rates=(args.dropout,),
    )

    config = ExtractorConfig(
        bert_like=bert_config,
        ohot_configs={},  # 显式设为空，去掉多余的 OneHotEmbedder (ohots)
        nested_ohots=nested_ohots,
        encoder=encoder_config,
        decoder=decoder_config,
    )
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="RedJujube NER 训练 - 支持带类型专家词典及基准模型")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--expert_dict_path", type=str, default=None, help="专家词典路径 (可选)")
    parser.add_argument("--save_dir", type=str, required=True, help="保存目录")
    parser.add_argument("--with_type", action="store_true", default=False, help="使用带类型词典")
    parser.add_argument("--bert_arch", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--expert_dict_dim", type=int, default=50)
    
    # 通道注意力参数
    parser.add_argument("--use_channel_attention", action="store_true", default=False, 
                        help="是否使用BMES通道间注意力")
    parser.add_argument("--channel_attn_version", type=str, default="v1", choices=["v1", "v2"],
                        help="通道注意力版本: v1=多头注意力, v2=简化单头")
    parser.add_argument("--channel_attn_heads", type=int, default=4,
                        help="通道注意力头数(仅v1)")
    parser.add_argument("--channel_attn_dropout", type=float, default=0.1,
                        help="通道注意力Dropout率")
    
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tensorboard", action="store_true", default=True, help="启用 TensorBoard")
    parser.add_argument("--no_tensorboard", dest="use_tensorboard", action="store_false", help="禁用 TensorBoard")
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if args.expert_dict_path is not None:
        model_name = "expert_dict_typed" if args.with_type else "expert_dict"
    else:
        model_name = "bert_bilstm_crf"
        
    save_dir = os.path.join(args.save_dir, f"{model_name}_{timestamp}")
    
    logger = LoggerManager.setup_logger(save_dir)
    
    logger.info("=" * 70)
    logger.info(f"RedJujube NER - {model_name.upper()} 训练")
    logger.info("=" * 70)
    if args.expert_dict_path is not None:
        logger.info(f"带类型词典: {args.with_type}")
    logger.info(f"保存目录: {save_dir}")

    train_data, dev_data, test_data = load_redjujube_data(args.data_dir)
    logger.info(f"加载数据: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")

    if args.expert_dict_path is not None:
        lexicon = load_expert_lexicon(args.expert_dict_path, with_type=args.with_type)
        logger.info(f"加载专家词典: {len(lexicon)} 个词 (with_type={args.with_type})")
        if args.with_type:
            logger.info(f"词典示例: {lexicon[:5]}")

        tokenizer = LexiconTokenizer(lexicon, max_len=10)
        for data in (train_data, dev_data, test_data):
            for entry in data:
                entry["tokens"].build_expert_dict_tags(tokenizer.tokenize)
        logger.info("✅ 专家词典特征添加完成")
    else:
        logger.info("ℹ️ 未提供专家词典，运行 BERT + BiLSTM + CRF 基准模型")

    model_config = build_expert_dict_config(args)
    
    train_config = RedJujubeTrainerConfig(args, save_dir, model_name)
    trainer = RedJujubeNERTrainer(train_config, logger)
    
    results = trainer.train(
        model_config=model_config,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data,
        use_expert_dict=(args.expert_dict_path is not None)
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("训练完成！")
    logger.info(f"结果保存在: {save_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()