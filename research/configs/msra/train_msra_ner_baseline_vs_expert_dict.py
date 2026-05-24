#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSRA 数据集 NER 训练脚本 - Baseline vs +ExpertDict 对比实验（自动词典）

对比两个模型：
1. Baseline: MacBERT + BiLSTM + CRF
2. +ExpertDict: MacBERT + BiLSTM + CRF + 专家词典特征（从 MSRA 训练集自动抽取）
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
from eznlp.dataset import Dataset
from eznlp.training import Trainer


def setup_logger(save_dir):
    """设置日志器"""
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{save_dir}/training.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_msra_data(train_path, dev_path, test_path):
    """加载 MSRA 数据集（BMES 字符级标注）"""
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="<pad>",
    )

    train_data = io.read(train_path)
    dev_data = io.read(dev_path)
    test_data = io.read(test_path)
    return train_data, dev_data, test_data


def load_expert_lexicon(dict_path):
    """加载专家词典"""
    lexicon = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().split("\t")[0]
            if word:
                lexicon.append(word)
    return lexicon


def build_baseline_config(args):
    """构建 Baseline 模型配置（无专家词典）"""
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

    encoder_config = EncoderConfig(
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
        encoder=encoder_config,
        decoder=decoder_config,
    )
    return config


def build_expert_dict_config(args):
    """构建 +ExpertDict 模型配置"""
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

    expert_dict_config = ExpertDictConfig(
        emb_dim=args.expert_dict_dim,
        agg_mode="wtd_mean_pooling",
    )

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
        nested_ohots={"expert_dict": expert_dict_config},
        encoder=encoder_config,
        decoder=decoder_config,
    )
    return config


def build_optimizer_and_scheduler(model, num_train_batches, args):
    """构建优化器和调度器"""
    param_groups = []
    if hasattr(model, "pretrained_parameters"):
        pretrained_params = list(model.pretrained_parameters())
        pretrained_param_ids = {id(p) for p in pretrained_params}
        other_params = [p for p in model.parameters() if id(p) not in pretrained_param_ids]

        param_groups.append({"params": pretrained_params, "lr": args.finetune_lr})
        param_groups.append({"params": other_params, "lr": args.lr})
    else:
        param_groups.append({"params": model.parameters(), "lr": args.lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    num_warmup_epochs = max(2, args.num_epochs // 5)
    num_warmup_steps = num_train_batches * num_warmup_epochs
    num_total_steps = num_train_batches * args.num_epochs

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_total_steps - current_step)
            / float(max(1, num_total_steps - num_warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_model(config, train_data, dev_data, test_data, args, logger, save_dir, use_expert_dict=False):
    """训练模型"""
    logger.info("\n" + "=" * 70)
    logger.info("开始训练: %s" % ("Baseline" if not use_expert_dict else "+ExpertDict"))
    logger.info("=" * 70 + "\n")

    logger.info("构建数据集...")
    train_set = Dataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims(dev_data, test_data)

    if use_expert_dict and getattr(config, "nested_ohots", None) is not None:
        if "expert_dict" in config.nested_ohots:
            logger.info("构建专家词典词频统计...")
            config.nested_ohots["expert_dict"].build_freqs(train_data, dev_data, test_data)

    dev_set = Dataset(dev_data, config, training=False)
    test_set = Dataset(test_data, config, training=False)

    logger.info(train_set.summary)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_set.collate,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dev_set.collate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_set.collate,
    )

    logger.info("实例化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.instantiate().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    logger.info("构建优化器和调度器...")
    optimizer, scheduler = build_optimizer_and_scheduler(model, len(train_loader), args)

    trainer = Trainer(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        schedule_by_step=True,
        num_grad_acc_steps=args.num_grad_acc_steps,
        device=device,
        grad_clip=args.grad_clip,
        use_amp=args.use_amp,
    )

    best_model_path = f"{save_dir}/best_model.pt"

    def save_callback(model):
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"✅ 保存最佳模型到: {best_model_path}")

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
        save_by_loss=False,
    )

    logger.info(f"\n加载最佳模型: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))

    logger.info("在测试集上评估...")
    test_loss, *test_metrics = trainer.eval_epoch(test_loader)

    logger.info("\n" + "=" * 70)
    logger.info("测试集结果:")
    logger.info(f"  Loss: {test_loss:.4f}")
    if test_metrics:
        for i, metric in enumerate(test_metrics):
            logger.info(f"  Metric {i}: {metric:.4f}")
    logger.info("=" * 70 + "\n")

    results = {
        "model_type": "ExpertDict" if use_expert_dict else "Baseline",
        "test_loss": float(test_loss),
        "test_metrics": [float(m) for m in test_metrics] if test_metrics else [],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "args": vars(args),
    }

    with open(f"{save_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main():
    parser = argparse.ArgumentParser(description="MSRA NER: Baseline vs +ExpertDict (自动词典)")

    # 数据参数
    parser.add_argument("--train_path", type=str, default="data/MSRA/train.char.bmes", help="训练数据路径")
    parser.add_argument("--dev_path", type=str, default="data/MSRA/dev.char.bmes", help="验证数据路径")
    parser.add_argument("--test_path", type=str, default="data/MSRA/test.char.bmes", help="测试数据路径")
    parser.add_argument(
        "--expert_dict_path",
        type=str,
        default="data/MSRA/expert_lexicon_auto.txt",
        help="专家词典路径（从训练集自动抽取）",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="cache/msra_ner_comparison_auto_lexicon",
        help="保存目录",
    )

    # 模型参数
    parser.add_argument(
        "--bert_arch",
        type=str,
        default="hfl/chinese-macbert-base",
        help="BERT 模型架构",
    )
    parser.add_argument("--hid_dim", type=int, default=256, help="LSTM 隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=1, help="LSTM 层数")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout 率")
    parser.add_argument("--expert_dict_dim", type=int, default=50, help="专家词典特征维度")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-3, help="学习率")
    parser.add_argument("--finetune_lr", type=float, default=2e-5, help="BERT 微调学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="梯度裁剪")
    parser.add_argument("--num_grad_acc_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练")

    # 显示和评估参数
    parser.add_argument("--disp_every_steps", type=int, default=50, help="每N步显示一次")
    parser.add_argument("--eval_every_steps", type=int, default=200, help="每N步评估一次")

    # 实验参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir_baseline = os.path.join(args.save_dir, f"baseline_{timestamp}")
    save_dir_expert = os.path.join(args.save_dir, f"expert_dict_{timestamp}")
    os.makedirs(save_dir_baseline, exist_ok=True)
    os.makedirs(save_dir_expert, exist_ok=True)

    print("\n" + "=" * 70)
    print("加载 MSRA 数据集...")
    print("=" * 70 + "\n")

    train_data, dev_data, test_data = load_msra_data(
        args.train_path, args.dev_path, args.test_path
    )

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(dev_data)} 条")
    print(f"测试集: {len(test_data)} 条\n")

    # Baseline
    logger_baseline = setup_logger(save_dir_baseline)
    config_baseline = build_baseline_config(args)
    baseline_results = train_model(
        config_baseline,
        train_data,
        dev_data,
        test_data,
        args,
        logger_baseline,
        save_dir_baseline,
        use_expert_dict=False,
    )

    # +ExpertDict
    logger_expert = setup_logger(save_dir_expert)
    expert_lexicon = load_expert_lexicon(args.expert_dict_path)
    logger_expert.info(f"加载专家词典: {args.expert_dict_path} (大小: {len(expert_lexicon)})")

    tokenizer = LexiconTokenizer(expert_lexicon, max_len=10)
    for data in (train_data, dev_data, test_data):
        for entry in data:
            entry["tokens"].build_expert_dict_tags(tokenizer.tokenize)

    config_expert = build_expert_dict_config(args)
    expert_results = train_model(
        config_expert,
        train_data,
        dev_data,
        test_data,
        args,
        logger_expert,
        save_dir_expert,
        use_expert_dict=True,
    )

    comparison = {
        "baseline": baseline_results,
        "expert_dict": expert_results,
    }
    with open(os.path.join(args.save_dir, f"comparison_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("MSRA NER 对比实验完成！结果保存在:")
    print(f"  - Baseline:     {save_dir_baseline}")
    print(f"  - +ExpertDict:  {save_dir_expert}")
    print(f"  - Comparison:   {os.path.join(args.save_dir, f'comparison_{timestamp}.json')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
