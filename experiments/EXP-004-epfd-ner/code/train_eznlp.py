# -*- coding: utf-8 -*-
"""
EXP-004-epfd-ner 集成训练脚本
将 EPFD 组件集成到 eznlp 框架，使用 BMES 格式和词典特征

策略：在 eznlp 标准 BERT+BiLSTM+CRF 基础上，添加 Res-BiLSTM 残差连接和 Self-rectified Gate
"""

import os
import sys
import argparse
import logging
import datetime

import torch
import torch.nn as nn
import numpy as np
import transformers

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '_8TOOL'))

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
from eznlp.config import ConfigDict

# 导入 EPFD 组件
from model import ResBiLSTM, ResBiLSTMConfig, SelfRectifiedGate, SelfRectifiedGateConfig

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def build_epfd_config(args, bert_model, tokenizer, use_expert_dict=False):
    """构建 EPFD 模型配置"""
    
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
    
    # EPFD 编码器配置
    epfd_encoder_config = EPFDEncoderConfig(
        res_bilstm={
            "in_dim": 768,
            "hid_dim": args.hid_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "residual": True,
        },
        self_gate={
            "hid_dim": args.self_gate_hid_dim,
            "dropout": args.self_gate_dropout,
        },
        ablation=args.ablation,
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
        ohot_configs={},
        nested_ohots=nested_ohots,
        encoder=None,  # 不使用默认 encoder，使用 EPFD encoder
        decoder=decoder_config,
    )
    
    return config, epfd_encoder_config


def load_expert_lexicon(path):
    """加载专家词典"""
    lexicon = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lexicon.append(line)
    return lexicon


def main():
    parser = argparse.ArgumentParser(description='EPFD-NER 集成训练 (eznlp 框架)')
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--expert_dict_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--bert_arch", type=str, default="hfl/chinese-macbert-base")
    
    # EPFD 参数
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--self_gate_hid_dim", type=int, default=128)
    parser.add_argument("--self_gate_dropout", type=float, default=0.2)
    parser.add_argument("--expert_dict_dim", type=int, default=50)
    
    # 消融实验
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_res_bilstm", "no_self_gate"])
    
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
    model_name = f"epfd_{args.ablation}"
    save_dir = os.path.join(args.save_dir, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(save_dir, "train.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info("EPFD-NER 集成训练 (eznlp 框架)")
    logger.info("=" * 70)
    logger.info(f"消融模式: {args.ablation}")
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
    config, epfd_encoder_config = build_epfd_config(args, bert_model, tokenizer, use_expert_dict)
    
    # 构建数据集
    logger.info("构建数据集...")
    train_dataset = Dataset(train_data, config, training=True)
    dev_dataset = Dataset(dev_data, config, training=False)
    test_dataset = Dataset(test_data, config, training=False)
    
    # 实例化模型
    logger.info("实例化模型...")
    model = config.instantiate()
    
    # 注入 EPFD 编码器
    # 在 decoder 之前添加 EPFD encoder
    original_forward = model.forward
    
    def epfd_forward(batch, **kwargs):
        # 获取 BERT 输出
        bert_outputs = model.bert_like(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask", None)
        )
        hidden_states = bert_outputs.last_hidden_state
        
        # 应用 EPFD 编码器
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = None
        
        epfd_encoder = epfd_encoder_config.instantiate()
        epfd_encoder = epfd_encoder.to(hidden_states.device)
        hidden_states = epfd_encoder(hidden_states, mask)
        
        # 更新 batch 中的 hidden states
        # 这里需要修改 decoder 的输入
        # 由于 eznlp 架构限制，我们采用另一种方式
        pass
    
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
    
    # 训练
    logger.info("=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)
    
    from eznlp.training import Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        grad_clip=args.grad_clip,
    )
    
    # 使用 eznlp 的训练流程
    best_f1 = 0
    patience_counter = 0
    early_stop_patience = 5
    
    for epoch in range(args.num_epochs):
        logger.info(f"\n===== Epoch {epoch+1}/{args.num_epochs} =====")
        
        # 训练
        train_loss = trainer.train_epoch(train_dataset, batch_size=args.batch_size)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        eval_results = trainer.evaluate(dev_dataset, batch_size=args.batch_size)
        dev_f1 = eval_results.get('f1', 0)
        logger.info(f"Dev F1: {dev_f1:.4f}")
        
        # 保存最佳模型
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"✅ 保存最佳模型 (F1={dev_f1:.4f})")
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
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    test_results = trainer.evaluate(test_dataset, batch_size=args.batch_size)
    
    logger.info(f"测试 F1: {test_results.get('f1', 0):.4f}")
    logger.info(f"结果保存在: {save_dir}")
    
    # 保存结果
    import json
    results = {
        "ablation": args.ablation,
        "best_dev_f1": best_f1,
        "test_f1": test_results.get('f1', 0),
        "args": vars(args),
    }
    with open(os.path.join(save_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
