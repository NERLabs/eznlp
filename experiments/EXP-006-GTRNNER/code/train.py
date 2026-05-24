#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-006-GTRNNER 训练脚本
将 GTR-NNER 的相对位置编码和三仿射融合迁移到扁平NER

消融模式:
  - baseline: 仅 BERT + CRF (复用EXP-005基线)
  - no_rope: BERT + BiLSTM + 三仿射融合 + CRF (无RoPE)
  - no_triaffine: BERT + BiLSTM + RoPE + CRF (无三仿射)
  - full: 完整模型 (BERT + RoPE + BiLSTM + 三仿射融合 + CRF)
"""

import os
import sys
import argparse
import logging
import datetime
import json
import yaml

import torch
import numpy as np

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

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


def load_config(config_path):
    """加载 YAML 配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_expert_lexicon(path):
    """加载专家词典"""
    lexicon = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lexicon.append(line)
    return lexicon


def build_config_from_yaml(config, ablation_mode, bert_model, tokenizer, expert_dict_path=None):
    """从 YAML 配置构建模型配置
    
    现在正确集成RoPE和TriAffine到eznlp框架的EncoderConfig中
    返回: (model_config, use_rope, use_triaffine, use_expert_dict)
    """
    
    model_cfg = config.get('model', {})
    encoder_cfg = model_cfg.get('encoder', {})
    bilstm_cfg = model_cfg.get('bilstm', {})
    rope_cfg = model_cfg.get('rope', {})
    triaffine_cfg = model_cfg.get('triaffine', {})
    
    # 根据消融模式调整配置
    use_rope = rope_cfg.get('enabled', True)
    use_triaffine = triaffine_cfg.get('enabled', True)
    use_bilstm = True  # 默认启用BiLSTM
    use_expert_dict = expert_dict_path is not None  # 默认根据是否提供词典路径决定
    
    # 消融模式配置 - 使用明确命名
    if ablation_mode == "BERT-CRF":
        # 基线：仅BERT+CRF
        use_bilstm = False
        use_rope = False
        use_triaffine = False
        use_expert_dict = False
    elif ablation_mode == "BERT-BiLSTM-CRF":
        # 加BiLSTM
        use_rope = False
        use_triaffine = False
        use_expert_dict = False
    elif ablation_mode == "BERT-BiLSTM-RoPE-CRF":
        # 加RoPE
        use_triaffine = False
        use_expert_dict = False
    elif ablation_mode == "BERT-BiLSTM-TriAffine-CRF":
        # 加TriAffine（不含ExpertDict）
        use_rope = False
        use_expert_dict = False
    elif ablation_mode == "BERT-BiLSTM-RoPE-TriAffine-CRF":
        # RoPE + TriAffine组合
        use_expert_dict = False
    elif ablation_mode == "BERT-BiLSTM-RoPE-TriAffine-ExpertDict-CRF":
        # 完整配置
        pass  # 保持所有默认True
    else:
        # 兼容旧命名
        if ablation_mode == "baseline":
            use_bilstm = False
            use_rope = False
            use_triaffine = False
            use_expert_dict = False
        elif ablation_mode == "no_rope":
            use_rope = False
        elif ablation_mode == "no_triaffine":
            use_triaffine = False
            use_expert_dict = False
        elif ablation_mode == "full":
            pass  # 保持所有默认True
    
    # BERT 配置
    bert_config = BertLikeConfig(
        tokenizer=tokenizer,
        bert_like=bert_model,
        freeze=encoder_cfg.get('freeze', False),
        mix_layers="top",
        bert_max_length=512,
        truncation=True,
    )
    
    # 词典配置
    nested_ohots = None
    if use_expert_dict and use_triaffine:
        expert_dict_config = ExpertDictConfig(
            emb_dim=triaffine_cfg.get('dict_dim', 64),
            agg_mode="wtd_mean_pooling",
        )
        nested_ohots = {"expert_dict": expert_dict_config}
    
    # 编码器配置 - 根据配置选择是否使用BiLSTM
    if not use_bilstm:
        # BERT-CRF: 完全跳过 encoder，使用 identity
        encoder_config = EncoderConfig(
            arch="identity",
            in_drop_rates=(0.0, 0.0, 0.0),
            hid_drop_rate=0.0,
        )
    else:
        encoder_config = EncoderConfig(
            arch="LSTM",
            hid_dim=bilstm_cfg.get('hid_dim', 256),
            num_layers=bilstm_cfg.get('num_layers', 1),
            in_drop_rates=(bilstm_cfg.get('dropout', 0.3), 0.0, 0.0),
            shortcut=True,  # 残差连接 (EXP-004/005验证有效)
            shortcut_mode="concat",
            # RoPE 参数 - 集成到eznlp框架
            use_rope=use_rope,
            rope_base=rope_cfg.get('base', 10000),
            rope_max_seq_len=rope_cfg.get('max_seq_len', 512),
            # TriAffine 参数 - 集成到eznlp框架
            use_triaffine=use_triaffine,
            triaffine_hid_dim=triaffine_cfg.get('hid_dim', 128),
        )
    
    # 解码器配置
    decoder_config = SequenceTaggingDecoderConfig(
        scheme="BMES",
        use_crf=True,
        in_drop_rates=(bilstm_cfg.get('dropout', 0.3),),
    )
    
    # 完整模型配置
    config = ExtractorConfig(
        bert_like=bert_config,
        ohots=None,
        nested_ohots=nested_ohots,
        encoder=encoder_config,
        decoder=decoder_config,
    )
    
    return config, use_rope, use_triaffine, use_expert_dict


def run_experiment(ablation_mode, config, args, bert_model, tokenizer):
    """运行单个消融实验
    
    消融模式命名规范：明确列出所有启用的模块
    - BERT-CRF: 基线
    - BERT-BiLSTM-CRF: 加BiLSTM
    - BERT-BiLSTM-RoPE-CRF: 加RoPE
    - BERT-BiLSTM-TriAffine-CRF: 加TriAffine
    - BERT-BiLSTM-RoPE-TriAffine-CRF: 组合效果
    - BERT-BiLSTM-RoPE-TriAffine-ExpertDict-CRF: 完整配置
    """
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{ablation_mode}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(save_dir, "train.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    # 打印实验信息头
    logger.info("=" * 70)
    logger.info(f"EXP-006-GTRNNER - 模型配置: {ablation_mode}")
    logger.info("=" * 70)
    
    # 打印运行命令
    import sys
    run_cmd = f"python {' '.join(sys.argv)}"
    logger.info(f"运行命令: {run_cmd}")
    
    # 打印模型结构说明
    logger.info("-" * 70)
    logger.info("模型结构:")
    model_components = ablation_mode.split("-")
    for i, comp in enumerate(model_components):
        indent = "  " * i
        if comp == "BERT":
            logger.info(f"{indent}└── BERT Encoder (hfl/chinese-macbert-base)")
        elif comp == "BiLSTM":
            logger.info(f"{indent}└── BiLSTM Encoder (hid=256, shortcut=concat)")
        elif comp == "RoPE":
            logger.info(f"{indent}└── RoPE (Rotary Position Embedding)")
        elif comp == "TriAffine":
            logger.info(f"{indent}└── TriAffine Attention Fusion")
        elif comp == "ExpertDict":
            logger.info(f"{indent}└── Expert Dictionary Embedding")
        elif comp == "CRF":
            logger.info(f"{indent}└── CRF Decoder (BMES scheme)")
    logger.info("-" * 70)
    
    logger.info(f"保存目录: {save_dir}")
    
    # 加载数据
    logger.info("加载数据...")
    data_cfg = config.get('data', {})
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
    
    # 构建模型配置（包含use_expert_dict设置）
    model_config, use_rope, use_triaffine, use_expert_dict = build_config_from_yaml(
        config, ablation_mode, bert_model, tokenizer, args.expert_dict_path
    )
    
    # 加载专家词典（根据配置决定是否加载）
    if use_expert_dict:
        logger.info(f"加载专家词典: {args.expert_dict_path}")
        expert_lexicon = load_expert_lexicon(args.expert_dict_path)
        expert_tokenizer = LexiconTokenizer(expert_lexicon, max_len=10)
        logger.info(f"专家词典大小: {len(expert_lexicon)} 个词")
        
        for data in train_data:
            data["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
        for data in dev_data:
            data["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
        for data in test_data:
            data["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
        logger.info("✅ 专家词典特征添加完成")
    
    logger.info(f"配置: RoPE={use_rope}, TriAffine={use_triaffine}, ExpertDict={use_expert_dict}")
    
    # 构建数据集
    logger.info("构建数据集...")
    train_dataset = Dataset(train_data, model_config, training=True)
    train_dataset.build_vocabs_and_dims(dev_data, test_data)
    
    if use_expert_dict and hasattr(model_config, 'nested_ohots') and model_config.nested_ohots:
        if "expert_dict" in model_config.nested_ohots:
            model_config.nested_ohots["expert_dict"].build_freqs(train_data, dev_data, test_data)
    
    dev_dataset = Dataset(dev_data, model_config, training=False)
    test_dataset = Dataset(test_data, model_config, training=False)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 模型
    logger.info("实例化模型...")
    model = model_config.instantiate().to(device)
    
    # 打印模型结构
    logger.info("-" * 70)
    logger.info("模型结构:")
    logger.info(model)
    logger.info("-" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"总参数量: {total_params:,}")
    
    # 优化器
    training_cfg = config.get('training', {})
    bert_params = list(model.bert_like.parameters())
    other_params = [p for n, p in model.named_parameters() if 'bert_like' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': training_cfg.get('learning_rate', {}).get('bert', 2e-5)},
        {'params': other_params, 'lr': training_cfg.get('learning_rate', {}).get('other', 1e-3)}
    ], weight_decay=training_cfg.get('weight_decay', 0.01))
    
    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        grad_clip=training_cfg.get('max_grad_norm', 1.0),
    )
    
    # DataLoader
    from torch.utils.data import DataLoader
    batch_size = training_cfg.get('batch_size', 16)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=dev_dataset.collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate)
    
    # 训练
    logger.info("开始训练...")
    best_dev_f1 = 0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = config.get('evaluation', {}).get('early_stop', 5)
    num_epochs = training_cfg.get('epochs', 30)
    
    for epoch in range(num_epochs):
        logger.info(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        
        train_result = trainer.train_epoch(train_loader)
        train_loss = train_result[0] if isinstance(train_result, tuple) else train_result
        train_f1 = train_result[1] if isinstance(train_result, tuple) and len(train_result) > 1 else 0.0
        logger.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        
        eval_result = trainer.eval_epoch(dev_loader)
        dev_loss = eval_result[0] if isinstance(eval_result, tuple) else eval_result
        dev_f1 = eval_result[1] if isinstance(eval_result, tuple) and len(eval_result) > 1 else 0.0
        logger.info(f"Dev Loss: {dev_loss:.4f}, Dev F1: {dev_f1:.4f}")
        
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model, os.path.join(save_dir, "best_model.pth"))
            logger.info(f"✅ 保存最佳模型 (epoch {epoch}, F1={dev_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"早停: {early_stop_patience} 轮无提升")
                break
    
    # 测试
    logger.info("\n测试集评估...")
    model = torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device, weights_only=False)
    trainer = Trainer(model=model, optimizer=optimizer, device=device, grad_clip=1.0)
    
    test_result = trainer.eval_epoch(test_loader)
    test_metrics = {
        'f1': test_result[1] if isinstance(test_result, tuple) and len(test_result) > 1 else 0.0
    }
    test_f1 = test_metrics['f1']
    
    logger.info(f"最优验证 F1: {best_dev_f1:.4f} (epoch {best_epoch})")
    logger.info(f"测试 F1: {test_f1:.4f}")
    
    # 保存结果
    results = {
        "ablation_mode": ablation_mode,
        "best_dev_f1": best_dev_f1,
        "test_metrics": test_metrics,
        "config": {
            "use_rope": use_rope,
            "use_triaffine": use_triaffine,
            "use_expert_dict": use_expert_dict,
        },
        "total_params": total_params,
    }
    
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"最佳验证F1: {best_dev_f1:.4f}")
    logger.info(f"测试集指标: {test_metrics}")
    logger.info(f"结果保存至: {results_path}")
    
    # 移除文件处理器
    logger.removeHandler(file_handler)
    
    return best_dev_f1, test_metrics


def main():
    parser = argparse.ArgumentParser(description="EXP-006-GTRNNER 训练脚本")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="/home/shiwenlong/NERlabs/eznlp/datasets/raw/RedJujube", help="数据目录")
    parser.add_argument("--save_dir", type=str, default="/home/shiwenlong/NERlabs/eznlp/experiments/EXP-006-GTRNNER/results", help="保存目录")
    parser.add_argument("--bert_model", type=str, default="/home/shiwenlong/NERlabs/eznlp/assets/transformers/hfl/chinese-macbert-base", help="BERT模型路径")
    parser.add_argument("--expert_dict_path", type=str, default="/home/shiwenlong/NERlabs/eznlp/datasets/raw/RedJujube/expert_lexicon_auto.txt", help="专家词典路径")
    parser.add_argument("--ablation", type=str, default="BERT-BiLSTM-RoPE-TriAffine-ExpertDict-CRF", 
                        choices=[
                            # 新命名规范（推荐）
                            "BERT-CRF",
                            "BERT-BiLSTM-CRF", 
                            "BERT-BiLSTM-RoPE-CRF",
                            "BERT-BiLSTM-TriAffine-CRF",
                            "BERT-BiLSTM-RoPE-TriAffine-CRF",
                            "BERT-BiLSTM-RoPE-TriAffine-ExpertDict-CRF",
                            # 兼容旧命名
                            "baseline", "no_rope", "no_triaffine", "full"
                        ],
                        help="消融模式（推荐使用新命名规范）")
    parser.add_argument("--run_all", action="store_true", help="运行所有消融实验")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载BERT
    logger.info(f"加载BERT模型: {args.bert_model}")
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    bert_model = BertModel.from_pretrained(args.bert_model)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.run_all:
        # 运行所有消融实验（使用新命名规范）
        modes = [
            "BERT-CRF",
            "BERT-BiLSTM-CRF",
            "BERT-BiLSTM-RoPE-CRF",
            "BERT-BiLSTM-TriAffine-CRF",
            "BERT-BiLSTM-RoPE-TriAffine-CRF",
            "BERT-BiLSTM-RoPE-TriAffine-ExpertDict-CRF",
        ]
        all_results = {}
        
        for mode in modes:
            logger.info(f"\n{'='*70}")
            logger.info(f"运行消融实验: {mode}")
            logger.info(f"{'='*70}\n")
            
            dev_f1, test_metrics = run_experiment(mode, config, args, bert_model, tokenizer)
            all_results[mode] = {
                "dev_f1": dev_f1,
                "test_metrics": test_metrics
            }
        
        # 保存汇总结果
        summary_path = os.path.join(args.save_dir, "all_results.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n所有实验结果保存至: {summary_path}")
        
    else:
        # 运行单个实验
        run_experiment(args.ablation, config, args, bert_model, tokenizer)


if __name__ == "__main__":
    main()
