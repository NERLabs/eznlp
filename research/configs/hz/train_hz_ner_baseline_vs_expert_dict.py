#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HZ 数据集 NER 训练脚本 - Baseline vs +ExpertDict 对比实验

对比两个模型：
1. Baseline: MacBERT + BiLSTM + CRF
2. +ExpertDict: MacBERT + BiLSTM + CRF + 专家词典特征
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
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from eznlp.io import ConllIO
from eznlp.model import (
    BertLikeConfig,
    ExpertDictConfig,
    EncoderConfig,
    ExtractorConfig,
    SoftLexiconConfig,
)
from eznlp.model.decoder import SequenceTaggingDecoderConfig
from eznlp.token import LexiconTokenizer
from eznlp.dataset import Dataset
from eznlp.metrics import precision_recall_f1_report
from eznlp.training import Trainer
from eznlp.config import ConfigDict
from research.tools.utils import load_vectors


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


def load_expert_lexicon(dict_path):
    """加载专家词典"""
    lexicon = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                lexicon.append(word)
    return lexicon


def load_hz_data(data_dir):
    """加载 NER 数据集（支持HZ和RedJujube）"""
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="<pad>"
    )
    
    # 根据目录名称自动选择文件名前缀
    if "RedJujube" in data_dir or "redjujube" in data_dir.lower():
        train_file = f"{data_dir}/redjujube_train.bmes"
        dev_file = f"{data_dir}/redjujube_dev.bmes"
        test_file = f"{data_dir}/redjujube_test.bmes"
    else:
        train_file = f"{data_dir}/hz_train.bmes"
        dev_file = f"{data_dir}/hz_dev.bmes"
        test_file = f"{data_dir}/hz_test.bmes"
    
    train_data = io.read(train_file)
    dev_data = io.read(dev_file)
    test_data = io.read(test_file)
    
    return train_data, dev_data, test_data


def build_baseline_config(args):
    """构建 Baseline 模型配置（无专家词典）"""
    # 加载 BERT 模型和 tokenizer
    bert_model = transformers.AutoModel.from_pretrained(args.bert_arch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_arch)
    
    bert_config = BertLikeConfig(
        tokenizer=tokenizer,
        bert_like=bert_model,
        freeze=False,
        mix_layers="top"
    )
    
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
    
    config = ExtractorConfig(
        bert_like=bert_config,
        encoder=encoder_config,
        decoder=decoder_config
    )
    
    return config


def build_expert_dict_config(args):
    """构建 +ExpertDict 模型配置"""
    # 加载 BERT 模型和 tokenizer
    bert_model = transformers.AutoModel.from_pretrained(args.bert_arch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_arch)
    
    bert_config = BertLikeConfig(
        tokenizer=tokenizer,
        bert_like=bert_model,
        freeze=False,
        mix_layers="top"
    )
    
    expert_dict_config = ExpertDictConfig(
        emb_dim=args.expert_dict_dim,
        agg_mode="wtd_mean_pooling"
    )
    
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
    
    config = ExtractorConfig(
        bert_like=bert_config,
        nested_ohots={
            "expert_dict": expert_dict_config
        },
        encoder=encoder_config,
        decoder=decoder_config
    )
    
    return config


def build_softlexicon_config(args, vectors):
    """构建 SoftLexicon 模型配置"""
    # 加载 BERT 模型和 tokenizer
    bert_model = transformers.AutoModel.from_pretrained(args.bert_arch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_arch)
    
    bert_config = BertLikeConfig(
        tokenizer=tokenizer,
        bert_like=bert_model,
        freeze=False,
        mix_layers="top"
    )
    
    softlexicon_config = SoftLexiconConfig(
        vectors=vectors,
        emb_dim=50,
        agg_mode="wtd_mean_pooling",
    )
    
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
    
    config = ExtractorConfig(
        bert_like=bert_config,
        nested_ohots={
            "softlexicon": softlexicon_config
        },
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


def train_model(config, train_data, dev_data, test_data, args, logger, save_dir, model_name="Baseline", use_expert_dict=False):
    """训练模型"""
    logger.info(f"\n{'='*70}")
    logger.info(f"开始训练: {model_name}")
    logger.info(f"{'='*70}\n")
    
    # 构建数据集
    logger.info("构建数据集...")
    train_set = Dataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims(dev_data, test_data)
    
    # 如果使用专家词典，构建词频统计
    if use_expert_dict and "expert_dict" in config.nested_ohots:
        logger.info("构建专家词典词频统计...")
        config.nested_ohots["expert_dict"].build_freqs(train_data, dev_data, test_data)
    
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

    test_preds = trainer.predict(test_set, batch_size=args.batch_size)
    test_gold = [entry["chunks"] for entry in test_set.data]
    _, ave_scores = precision_recall_f1_report(
        test_gold, test_preds, macro_over="types"
    )
    micro_scores = {
        key: float(value) if key in {"precision", "recall", "f1"} else int(value)
        for key, value in ave_scores["micro"].items()
    }
    macro_scores = {
        key: float(value)
        for key, value in ave_scores["macro"].items()
    }
    
    logger.info(f"\n{'='*70}")
    logger.info("测试集结果:")
    logger.info(f"  Loss: {test_loss:.4f}")
    if test_metrics:
        for i, metric in enumerate(test_metrics):
            logger.info(f"  Metric {i}: {metric:.4f}")
    logger.info(
        "  Micro P/R/F1: "
        f"{micro_scores['precision']:.4f}/"
        f"{micro_scores['recall']:.4f}/"
        f"{micro_scores['f1']:.4f}"
    )
    logger.info(f"{'='*70}\n")
    
    # 保存结果
    results = {
        'model_type': model_name,
        'test_loss': float(test_loss),
        'test_metrics': [float(m) for m in test_metrics] if test_metrics else [],
        'test_precision': micro_scores['precision'],
        'test_recall': micro_scores['recall'],
        'test_f1': micro_scores['f1'],
        'test_micro': micro_scores,
        'test_macro': macro_scores,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'args': vars(args)
    }
    
    with open(f"{save_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='HZ NER: Baseline vs +ExpertDict')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/HZ',
                        help='数据目录')
    parser.add_argument('--expert_dict_path', type=str, default='data/HZ/expert_lexicon.txt',
                        help='专家词典路径')
    parser.add_argument('--save_dir', type=str, default='cache/hz_ner_comparison',
                        help='保存目录')
    
    # 模型参数
    parser.add_argument('--bert_arch', type=str, default='hfl/chinese-macbert-base',
                        help='BERT 模型架构')
    parser.add_argument('--bert_drop_rate', type=float, default=0.2,
                        help='BERT dropout率')
    parser.add_argument('--hid_dim', type=int, default=256,
                        help='LSTM 隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='LSTM 层数')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout 率')
    parser.add_argument('--expert_dict_dim', type=int, default=50,
                        help='专家词典特征维度')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='学习率')
    parser.add_argument('--finetune_lr', type=float, default=2e-5,
                        help='BERT 微调学习率')
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
    parser.add_argument('--run_baseline', action='store_true',
                        help='运行 Baseline 实验')
    parser.add_argument('--run_expert_dict', action='store_true',
                        help='运行 +ExpertDict 实验')
    parser.add_argument('--run_softlexicon', action='store_true',
                        help='运行 SoftLexicon 实验')
    parser.add_argument('--run_softlexicon_trainlex', action='store_true',
                        help='运行 SoftLexicon (训练集词表) 实验')
    parser.add_argument('--softlex_train_path', type=str, default='data/HZ/softlexicon_train.txt',
                        help='SoftLexicon 训练集词表路径')
    parser.add_argument('--run_both', action='store_true',
                        help='运行 Baseline 和 +ExpertDict 两个实验')
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
    print("加载 HZ 数据集...")
    print(f"{'='*70}\n")
    
    train_data, dev_data, test_data = load_hz_data(args.data_dir)
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(dev_data)} 条")
    print(f"测试集: {len(test_data)} 条\n")
    
    results_summary = {}
    
    # 运行 Baseline 实验
    if args.run_baseline or args.run_both:
        baseline_dir = f"{args.save_dir}/baseline_{timestamp}"
        logger = setup_logger(baseline_dir)
        
        logger.info("="*70)
        logger.info("实验 1: Baseline (MacBERT + BiLSTM + CRF)")
        logger.info("="*70)
        
        config = build_baseline_config(args)
        results = train_model(config, train_data, dev_data, test_data, args, logger, baseline_dir, model_name="Baseline", use_expert_dict=False)
        results_summary['baseline'] = results
    
    # 运行 +ExpertDict 实验
    if args.run_expert_dict or args.run_both:
        # 加载专家词典
        print(f"\n{'='*70}")
        print("加载专家词典...")
        print(f"{'='*70}\n")
        
        expert_lexicon = load_expert_lexicon(args.expert_dict_path)
        expert_tokenizer = LexiconTokenizer(expert_lexicon, max_len=10)
        print(f"专家词典大小: {len(expert_lexicon)} 个词\n")
        
        # 为数据添加专家词典匹配
        print("为数据添加专家词典匹配特征...")
        for entry in train_data:
            entry["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
        for entry in dev_data:
            entry["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
        for entry in test_data:
            entry["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
        print("✅ 完成\n")
        
        expert_dict_dir = f"{args.save_dir}/expert_dict_{timestamp}"
        logger = setup_logger(expert_dict_dir)
        
        logger.info("="*70)
        logger.info("实验 2: +ExpertDict (MacBERT + BiLSTM + CRF + 专家词典)")
        logger.info("="*70)
        
        config = build_expert_dict_config(args)
        results = train_model(config, train_data, dev_data, test_data, args, logger, expert_dict_dir, model_name="+ExpertDict", use_expert_dict=True)
        results_summary['expert_dict'] = results
    
    # 运行 SoftLexicon 实验
    if args.run_softlexicon:
        print(f"\n{'='*70}")
        print("加载 SoftLexicon 词典向量...")
        print(f"{'='*70}\n")

        vectors = load_vectors("chinese", 50)
        if vectors is None:
            raise ValueError("无法加载中文 50 维词向量，请检查 assets/vectors 下是否存在相应文件。")

        tokenizer = LexiconTokenizer(vectors.itos)

        print("为数据添加 SoftLexicon 特征...")
        for data in (train_data, dev_data, test_data):
            for entry in data:
                entry["tokens"].build_softwords(tokenizer.tokenize)
                entry["tokens"].build_softlexicons(tokenizer.tokenize)
        print("✅ 完成\n")

        softlexicon_dir = f"{args.save_dir}/softlexicon_{timestamp}"
        logger = setup_logger(softlexicon_dir)

        logger.info("="*70)
        logger.info("实验 3: SoftLexicon (MacBERT + BiLSTM + CRF + SoftLexicon)")
        logger.info("="*70)

        config = build_softlexicon_config(args, vectors)
        results = train_model(config, train_data, dev_data, test_data, args, logger, softlexicon_dir, model_name="SoftLexicon", use_expert_dict=False)
        results_summary['softlexicon'] = results
    
    # 运行 SoftLexicon (训练集词表) 实验
    if args.run_softlexicon_trainlex:
        print(f"\n{'='*70}")
        print("加载 SoftLexicon 词典向量（用于初始化）...")
        print(f"{'='*70}\n")

        vectors = load_vectors("chinese", 50)
        if vectors is None:
            raise ValueError("无法加载中文 50 维词向量，请检查 assets/vectors 下是否存在相应文件。")

        print(f"\n{'='*70}")
        print("从训练集加载 SoftLexicon 候选词表...")
        print(f"{'='*70}\n")
        
        # 加载训练集词表
        train_lexicon = load_expert_lexicon(args.softlex_train_path)
        print(f"训练集词表大小: {len(train_lexicon):,} 个词\n")
        
        tokenizer = LexiconTokenizer(train_lexicon, max_len=10)

        print("为数据添加 SoftLexicon (训练集词表) 特征...")
        for data in (train_data, dev_data, test_data):
            for entry in data:
                entry["tokens"].build_softwords(tokenizer.tokenize)
                entry["tokens"].build_softlexicons(tokenizer.tokenize)
        print("✅ 完成\n")

        softlexicon_trainlex_dir = f"{args.save_dir}/softlexicon_trainlex_{timestamp}"
        logger = setup_logger(softlexicon_trainlex_dir)

        logger.info("="*70)
        logger.info("实验 4: SoftLexicon-TrainLex (MacBERT + BiLSTM + CRF + SoftLexicon[训练集词表])")
        logger.info("="*70)

        config = build_softlexicon_config(args, vectors)
        results = train_model(config, train_data, dev_data, test_data, args, logger, softlexicon_trainlex_dir, model_name="SoftLexicon-TrainLex", use_expert_dict=False)
        results_summary['softlexicon_trainlex'] = results
    
    # 打印对比结果
    if args.run_both and len(results_summary) == 2:
        print(f"\n{'='*70}")
        print("对比结果总结")
        print(f"{'='*70}\n")
        
        baseline_res = results_summary['baseline']
        expert_dict_res = results_summary['expert_dict']
        
        print(f"{'模型':<20} {'测试Loss':<12} {'测试F1':<12} {'参数量':<15}")
        print(f"{'-'*70}")
        print(f"{'Baseline':<20} {baseline_res['test_loss']:<12.4f} {baseline_res['test_metrics'][0] if baseline_res['test_metrics'] else 0.0:<12.4f} {baseline_res['total_params']:<15,}")
        print(f"{'+ExpertDict':<20} {expert_dict_res['test_loss']:<12.4f} {expert_dict_res['test_metrics'][0] if expert_dict_res['test_metrics'] else 0.0:<12.4f} {expert_dict_res['total_params']:<15,}")
        
        if baseline_res['test_metrics'] and expert_dict_res['test_metrics']:
            improvement = (expert_dict_res['test_metrics'][0] - baseline_res['test_metrics'][0]) * 100
            print(f"\n📊 F1 提升: {improvement:+.2f}%")
        
        print(f"{'='*70}\n")
        
        # 保存对比结果
        comparison_file = f"{args.save_dir}/comparison_{timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"💾 对比结果已保存到: {comparison_file}\n")


if __name__ == "__main__":
    main()
