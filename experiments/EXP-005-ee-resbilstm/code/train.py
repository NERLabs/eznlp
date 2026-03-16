# -*- coding: utf-8 -*-
"""
EXP-005-ee-resbilstm 训练脚本
支持完整消融实验，复用 eznlp 框架

消融模式:
  - baseline: 仅 BERT + CRF
  - no_residual: BERT + BiLSTM (无残差) + CRF
  - no_srg: BERT + Res-BiLSTM + CRF
  - no_dict: BERT + Res-BiLSTM + SRG + CRF (无词典)
  - full: 完整模型
"""

import os
import sys
import argparse
import logging
import datetime
import yaml

import torch
import numpy as np
import transformers

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


def build_config_from_yaml(config, ablation_mode, bert_model, tokenizer, use_expert_dict=False):
    """从 YAML 配置构建模型配置"""
    
    model_cfg = config.get('model', {})
    encoder_cfg = model_cfg.get('encoder', {})
    res_bilstm_cfg = model_cfg.get('res_bilstm', {})
    self_gate_cfg = model_cfg.get('self_gate', {})
    entity_cfg = model_cfg.get('entity_feature', {})
    
    # 根据消融模式调整配置
    use_residual = res_bilstm_cfg.get('residual', True)
    use_srg = self_gate_cfg.get('enabled', True)
    use_entity = entity_cfg.get('enabled', True)
    
    if ablation_mode == "baseline":
        use_residual = False
        use_srg = False
        use_entity = False
        use_expert_dict = False
    elif ablation_mode == "no_residual":
        use_residual = False
        use_srg = False
        use_entity = False
    elif ablation_mode == "no_srg":
        use_srg = False
    elif ablation_mode == "no_dict" or ablation_mode == "no_entity_feature":
        use_entity = False
        use_expert_dict = False
    # full: 保持所有配置
    
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
    if use_expert_dict and use_entity:
        expert_dict_config = ExpertDictConfig(
            emb_dim=entity_cfg.get('dict_embed_size', 64),
            agg_mode="wtd_mean_pooling",
        )
        nested_ohots = {"expert_dict": expert_dict_config}
    
    # 编码器配置
    if ablation_mode == "baseline":
        # baseline: 完全跳过 encoder，使用 identity
        encoder_config = EncoderConfig(
            arch="identity",
            in_drop_rates=(0.0, 0.0, 0.0),
            hid_drop_rate=0.0,
        )
    else:
        encoder_config = EncoderConfig(
            arch="LSTM",
            hid_dim=res_bilstm_cfg.get('hidden_size', 256),
            num_layers=res_bilstm_cfg.get('num_layers', 1),
            in_drop_rates=(res_bilstm_cfg.get('dropout', 0.3), 0.0, 0.0),
            shortcut=use_residual,  # 残差连接
            shortcut_mode="concat",  # EXP-005 使用 concat
            use_srg=use_srg,  # Self-rectified Gate
            srg_hid_dim=self_gate_cfg.get('hidden_size', 128),
            srg_dropout=self_gate_cfg.get('dropout', 0.2),
        )
    
    # 解码器配置
    decoder_config = SequenceTaggingDecoderConfig(
        scheme="BMES",
        use_crf=True,
        in_drop_rates=(res_bilstm_cfg.get('dropout', 0.3),),
    )
    
    # 完整模型配置
    config = ExtractorConfig(
        bert_like=bert_config,
        ohots=None,
        nested_ohots=nested_ohots,
        encoder=encoder_config,
        decoder=decoder_config,
    )
    
    return config


def run_experiment(ablation_mode, config, args, bert_model, tokenizer):
    """运行单个消融实验"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{ablation_mode}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(save_dir, "train.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info(f"EXP-005-ee-resbilstm - 消融模式: {ablation_mode}")
    logger.info("=" * 70)
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
    
    # 加载专家词典
    use_expert_dict = args.expert_dict_path is not None
    if ablation_mode in ["baseline", "no_residual", "no_dict", "no_entity_feature"]:
        use_expert_dict = False
    
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
    
    # 构建模型配置
    model_config = build_config_from_yaml(
        config, ablation_mode, bert_model, tokenizer, use_expert_dict
    )
    
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
    logger.info("=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)
    
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = training_cfg.get('early_stop', 10)
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
        
        if dev_f1 > best_f1:
            best_f1 = dev_f1
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
    logger.info("\n" + "=" * 70)
    logger.info("测试集评估")
    logger.info("=" * 70)
    
    model = torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device, weights_only=False)
    trainer = Trainer(model=model, optimizer=optimizer, device=device, grad_clip=1.0)
    
    test_result = trainer.eval_epoch(test_loader)
    test_f1 = test_result[1] if isinstance(test_result, tuple) and len(test_result) > 1 else 0.0
    
    logger.info(f"最优验证 F1: {best_f1:.4f} (epoch {best_epoch})")
    logger.info(f"测试 F1: {test_f1:.4f}")
    
    # 保存结果
    import json
    results = {
        "ablation_mode": ablation_mode,
        "best_dev_f1": best_f1,
        "best_epoch": best_epoch,
        "test_f1": test_f1,
    }
    with open(os.path.join(save_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.removeHandler(file_handler)
    return test_f1


def main():
    parser = argparse.ArgumentParser(description='EXP-005-ee-resbilstm 训练')
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--data_dir", type=str, default="_2DATA/RedJujube")
    parser.add_argument("--expert_dict_path", type=str, default="_2DATA/RedJujube/expert_lexicon_auto_min1.txt")
    parser.add_argument("--save_dir", type=str, default="cache/exp005_ablation")
    parser.add_argument("--bert_arch", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["baseline", "no_residual", "no_srg", "no_dict", "full", "all"],
                        help="消融模式，'all' 运行所有消融实验")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        # 默认配置
        config = {
            'model': {
                'encoder': {'freeze': False},
                'res_bilstm': {'hidden_size': 256, 'num_layers': 1, 'dropout': 0.3, 'residual': True},
                'self_gate': {'enabled': True, 'hidden_size': 128, 'dropout': 0.2},
                'entity_feature': {'enabled': True, 'dict_embed_size': 64},
            },
            'training': {
                'batch_size': 16, 'epochs': 30,
                'learning_rate': {'bert': 2e-5, 'other': 1e-3},
                'weight_decay': 0.01, 'early_stop': 10,
            }
        }
    
    # 加载 BERT
    logger.info(f"加载 BERT: {args.bert_arch}")
    bert_model = transformers.AutoModel.from_pretrained(args.bert_arch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_arch)
    
    # 运行实验
    if args.ablation == "all":
        modes = ["baseline", "no_residual", "no_srg", "no_dict", "full"]
        results = {}
        for mode in modes:
            logger.info(f"\n{'='*70}")
            logger.info(f"运行消融实验: {mode}")
            logger.info(f"{'='*70}")
            results[mode] = run_experiment(mode, config, args, bert_model, tokenizer)
        
        # 汇总结果
        logger.info("\n" + "=" * 70)
        logger.info("消融实验汇总")
        logger.info("=" * 70)
        for mode, f1 in results.items():
            logger.info(f"{mode}: Test F1 = {f1:.4f}")
    else:
        run_experiment(args.ablation, config, args, bert_model, tokenizer)


if __name__ == "__main__":
    main()
