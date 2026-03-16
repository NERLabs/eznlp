# -*- coding: utf-8 -*-
"""
EXP-004-epfd-ner 简化训练脚本
直接使用 eznlp 框架和 Trainer
"""

import os
import sys
import argparse
import logging
import datetime

import torch
import numpy as np
import transformers

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '_8TOOL'))

from eznlp.io import ConllIO
from eznlp.config import ConfigDict
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

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_expert_lexicon(path):
    """加载专家词典"""
    lexicon = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lexicon.append(line)
    return lexicon


def build_config(args, bert_model, tokenizer, use_expert_dict=False):
    """构建模型配置"""
    
    # BERT 配置
    bert_config = BertLikeConfig(
        tokenizer=tokenizer,
        bert_like=bert_model,
        freeze=False,
        mix_layers="top",
        bert_max_length=512,
        truncation=True,
    )
    
    # 词典配置
    nested_ohots = {}
    if use_expert_dict:
        expert_dict_config = ExpertDictConfig(
            emb_dim=args.expert_dict_dim,
            agg_mode="wtd_mean_pooling",
        )
        nested_ohots["expert_dict"] = expert_dict_config
    
    # 编码器配置 (标准 BiLSTM)
    encoder_config = EncoderConfig(
        arch="LSTM",
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        in_drop_rates=(args.dropout, 0.0, 0.0),
        shortcut=args.shortcut,  # 残差连接
        shortcut_mode=args.shortcut_mode,  # 残差连接方式
        use_srg=args.use_srg,  # Self-rectified Gate
        srg_hid_dim=args.srg_hid_dim,
        srg_dropout=args.srg_dropout,
    )
    
    # 解码器配置 (BMES + CRF)
    decoder_config = SequenceTaggingDecoderConfig(
        scheme="BMES",
        use_crf=True,
        in_drop_rates=(args.dropout,),
    )
    
    # 完整模型配置
    config = ExtractorConfig(
        bert_like=bert_config,
        ohots=None,  # 显式设为 None，跳过默认的 OneHotConfig
        nested_ohots=nested_ohots,
        encoder=encoder_config,
        decoder=decoder_config,
    )
    
    return config


def main():
    parser = argparse.ArgumentParser(description='EPFD-NER 简化训练')
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--expert_dict_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--bert_arch", type=str, default="hfl/chinese-macbert-base")
    
    # 模型参数
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--expert_dict_dim", type=int, default=50)
    parser.add_argument("--shortcut", action="store_true", help="使用残差连接")
    parser.add_argument("--shortcut_mode", type=str, default="concat", choices=["concat", "add"],
                        help="残差连接方式: concat (eznlp默认) 或 add (EPFD论文方式)")
    parser.add_argument("--use_srg", action="store_true", help="使用 Self-rectified Gate")
    parser.add_argument("--srg_hid_dim", type=int, default=128, help="SRG 隐藏层维度")
    parser.add_argument("--srg_dropout", type=float, default=0.2, help="SRG dropout")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.use_srg:
        model_name = f"srg_{args.shortcut_mode}"
    elif args.shortcut:
        model_name = f"shortcut_{args.shortcut_mode}"
    else:
        model_name = "baseline"
    save_dir = os.path.join(args.save_dir, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(save_dir, "train.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info(f"EPFD-NER 简化训练 (eznlp 标准框架) - {model_name}")
    logger.info(f"shortcut={args.shortcut}, shortcut_mode={args.shortcut_mode}, use_srg={args.use_srg}")
    logger.info("=" * 70)
    logger.info(f"保存目录: {save_dir}")
    
    # 加载数据
    logger.info("加载数据...")
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="<pad>",
    )
    train_data = io.read(os.path.join(args.data_dir, "redjujube_train.bmes"))
    dev_data = io.read(os.path.join(args.data_dir, "redjujube_dev.bmes"))
    test_data = io.read(os.path.join(args.data_dir, "redjujube_test.bmes"))
    logger.info(f"训练集: {len(train_data)}, 验证集: {len(dev_data)}, 测试集: {len(test_data)}")
    
    # 加载专家词典
    use_expert_dict = args.expert_dict_path is not None
    if use_expert_dict:
        logger.info(f"加载专家词典: {args.expert_dict_path}")
        expert_lexicon = load_expert_lexicon(args.expert_dict_path)
        expert_tokenizer = LexiconTokenizer(expert_lexicon, max_len=10)
        logger.info(f"专家词典大小: {len(expert_lexicon)} 个词")
        
        # 添加词典特征
        for data in train_data:
            data["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
        for data in dev_data:
            data["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
        for data in test_data:
            data["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
        logger.info("✅ 专家词典特征添加完成")
    
    # 加载 BERT
    logger.info(f"加载 BERT: {args.bert_arch}")
    bert_model = transformers.AutoModel.from_pretrained(args.bert_arch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_arch)
    
    # 构建模型配置
    config = build_config(args, bert_model, tokenizer, use_expert_dict)
    
    # 构建数据集
    logger.info("构建数据集...")
    train_dataset = Dataset(train_data, config, training=True)
    train_dataset.build_vocabs_and_dims(dev_data, test_data)
    
    # 如果使用专家词典，构建词频统计
    if use_expert_dict and hasattr(config, 'nested_ohots') and config.nested_ohots:
        if "expert_dict" in config.nested_ohots:
            config.nested_ohots["expert_dict"].build_freqs(train_data, dev_data, test_data)
    
    dev_dataset = Dataset(dev_data, config, training=False)
    test_dataset = Dataset(test_data, config, training=False)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 实例化模型
    logger.info("实例化模型...")
    model = config.instantiate().to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    # 构建优化器
    logger.info("构建优化器和调度器...")
    
    # 分层学习率
    bert_params = list(model.bert_like.parameters())
    other_params = [p for n, p in model.named_parameters() if 'bert_like' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': args.finetune_lr},
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    # 使用 eznlp 的 Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        grad_clip=args.grad_clip,
    )
    
    # 创建 DataLoader
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        collate_fn=dev_dataset.collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=test_dataset.collate,
    )
    
    # 训练
    logger.info("=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)
    
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(args.num_epochs):
        logger.info(f"\n===== Epoch {epoch+1}/{args.num_epochs} =====")
        
        # 训练一个 epoch
        train_result = trainer.train_epoch(train_loader)
        if isinstance(train_result, tuple):
            train_loss = train_result[0]
            train_f1 = train_result[1] if len(train_result) > 1 else 0.0
        else:
            train_loss = train_result
            train_f1 = 0.0
        logger.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        
        # 验证
        eval_result = trainer.eval_epoch(dev_loader)
        if isinstance(eval_result, tuple):
            dev_loss = eval_result[0]
            dev_f1 = eval_result[1] if len(eval_result) > 1 else 0.0
        else:
            dev_loss = eval_result
            dev_f1 = 0.0
        
        logger.info(f"Dev Loss: {dev_loss:.4f}, Dev F1: {dev_f1:.4f}")
        
        # 保存最佳模型
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            patience_counter = 0
            model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model, model_path)
            logger.info(f"✅ 保存最佳模型 (epoch {epoch}, F1={dev_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"早停: {early_stop_patience} 轮无提升")
                break
    
    # 测试
    logger.info("\n" + "=" * 70)
    logger.info("测试集评估")
    logger.info("=" * 70)
    
    # 加载最佳模型
    model_path = os.path.join(save_dir, "best_model.pth")
    logger.info(f"加载最佳模型: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        grad_clip=args.grad_clip,
    )
    
    test_result = trainer.eval_epoch(test_loader)
    if isinstance(test_result, tuple):
        test_loss = test_result[0]
        test_f1 = test_result[1] if len(test_result) > 1 else 0.0
    else:
        test_loss = test_result
        test_f1 = 0.0
    
    logger.info(f"最优验证 F1: {best_f1:.4f} (epoch {best_epoch})")
    logger.info(f"测试 F1: {test_f1:.4f}")
    logger.info(f"结果保存在: {save_dir}")
    
    # 保存结果
    import json
    results = {
        "best_dev_f1": best_f1,
        "best_epoch": best_epoch,
        "test_f1": test_f1,
        "args": vars(args),
    }
    with open(os.path.join(save_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return test_f1


if __name__ == "__main__":
    main()
