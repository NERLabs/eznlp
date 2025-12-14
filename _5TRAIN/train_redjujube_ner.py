#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube 数据集 NER 训练入口

功能：
- 配置和调度实验运行
- 调用模块化的模型构建器、数据加载器和训练器
- 支持多种模型类型的对比实验

使用方式：
    python train_redjujube_ner.py --run_baseline --data_dir _2DATA/RedJujube
"""

import argparse
import os
import sys
import datetime
import json
import torch
import numpy as np

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '_8TOOL'))

# 导入重构后的模块
from _4MODELS.models.redjujube_model_builder import RedJujubeModelFactory as ModelFactory
from _5TRAIN.redjujube_data_loader import RedJujubeDataLoader, DataPreparationPipeline
from _5TRAIN.redjujube_trainer import (
    RedJujubeTrainerConfig,
    LoggerManager,
    RedJujubeNERTrainer
)
from utils import load_vectors


def run_experiment(model_type, args, vectors=None):
    """运行单个实验
    
    Args:
        model_type: 模型类型字符串
        args: 命令行参数
        vectors: 词向量对象（可选）
        
    Returns:
        dict: 实验结果
    """
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{args.save_dir}/{model_type}_{timestamp}"
    
    # 设置日志
    logger = LoggerManager.setup_logger(save_dir)
    model_display_name = ModelFactory.get_model_display_name(model_type)
    
    logger.info("="*70)
    logger.info(f"实验: {model_display_name}")
    logger.info("="*70)
    
    # 准备数据
    logger.info("加载和准备数据...")
    data_loader = RedJujubeDataLoader(args.data_dir)
    data_pipeline = DataPreparationPipeline(data_loader)
    
    # 根据模型类型准备数据
    if model_type in ['expert_dict_auto', 'expert_dict_manual']:
        dict_path = args.expert_dict_auto_path if 'auto' in model_type else args.expert_dict_path
        train_data, dev_data, test_data = data_pipeline.prepare(
            model_type, 
            expert_dict_path=dict_path
        )
    elif model_type in ['softlexicon', 'softlexicon_trainlex']:
        if vectors is None:
            raise ValueError(f"模型 {model_type} 需要 vectors 参数")
        
        # 如果是 trainlex 版本，从文件加载词表
        if model_type == 'softlexicon_trainlex':
            train_lexicon = data_loader.load_lexicon(args.softlex_train_path)
            softlex_lexicon = train_lexicon
        else:
            softlex_lexicon = vectors.itos
        
        train_data, dev_data, test_data = data_pipeline.prepare(
            model_type,
            softlex_lexicon=softlex_lexicon
        )
    elif model_type.startswith('fusion_'):
        if vectors is None:
            raise ValueError(f"融合模型需要 vectors 参数")
        
        train_lexicon = data_loader.load_lexicon(args.softlex_train_path)
        train_data, dev_data, test_data = data_pipeline.prepare(
            model_type,
            expert_dict_path=args.expert_dict_auto_path,
            softlex_lexicon=train_lexicon
        )
    else:
        # baseline
        train_data, dev_data, test_data = data_pipeline.prepare(model_type)
    
    logger.info(f"训练集: {len(train_data)} 条")
    logger.info(f"验证集: {len(dev_data)} 条")
    logger.info(f"测试集: {len(test_data)} 条")
    
    # 构建模型配置
    logger.info("构建模型配置...")
    model_config = ModelFactory.create_model_config(model_type, args, vectors)
    
    # 创建训练器并训练
    train_config = RedJujubeTrainerConfig(args, save_dir, model_display_name)
    trainer = RedJujubeNERTrainer(train_config, logger)
    
    use_expert_dict = 'expert' in model_type or 'fusion' in model_type
    results = trainer.train(model_config, train_data, dev_data, test_data, use_expert_dict)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='RedJujube NER: 模块化对比实验')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='_2DATA/RedJujube',
                        help='数据目录')
    parser.add_argument('--expert_dict_path', type=str, default='_2DATA/RedJujube/expert_lexicon.txt',
                        help='手动专家词典路径')
    parser.add_argument('--expert_dict_auto_path', type=str, default='_2DATA/RedJujube/expert_lexicon_auto.txt',
                        help='自动专家词典路径')
    parser.add_argument('--softlex_train_path', type=str, default='_2DATA/RedJujube/softlexicon_train.txt',
                        help='SoftLexicon 训练集词表路径')
    parser.add_argument('--save_dir', type=str, default='cache/redjujube_ner_comparison',
                        help='保存目录')
    
    # 模型参数
    parser.add_argument('--bert_arch', type=str, default='hfl/chinese-macbert-base',
                        help='BERT 模型架构')
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
                        help='每 N 步显示一次')
    parser.add_argument('--eval_every_steps', type=int, default=200,
                        help='每 N 步评估一次')
    
    # 实验参数
    parser.add_argument('--run_baseline', action='store_true',
                        help='运行 Baseline 实验')
    parser.add_argument('--run_expert_dict', action='store_true',
                        help='运行 +ExpertDict (手动) 实验')
    parser.add_argument('--run_expert_dict_auto', action='store_true',
                        help='运行 +ExpertDict (自动) 实验')
    parser.add_argument('--run_softlexicon', action='store_true',
                        help='运行 SoftLexicon 实验')
    parser.add_argument('--run_softlexicon_trainlex', action='store_true',
                        help='运行 SoftLexicon (训练集词表) 实验')
    parser.add_argument('--run_softlexicon_expert_concat', action='store_true',
                        help='运行 Soft+Expert 融合（方案A：直接拼接）')
    parser.add_argument('--run_softlexicon_expert_weighted', action='store_true',
                        help='运行 Soft+Expert 融合（方案B：加权求和）')
    parser.add_argument('--run_softlexicon_expert_gated', action='store_true',
                        help='运行 Soft+Expert 融合（方案C：门控机制）')
    parser.add_argument('--run_softlexicon_expert_attention', action='store_true',
                        help='运行 Soft+Expert 融合（方案D：注意力融合）')
    parser.add_argument('--run_all', action='store_true',
                        help='运行所有实验')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"\n{'='*70}")
    print("RedJujube NER 对比实验 - 模块化版本")
    print(f"{'='*70}\n")
    
    # 记录结果
    results_summary = {}
    
    # 加载向量（如果需要）
    vectors = None
    if (args.run_softlexicon or args.run_softlexicon_trainlex or 
        args.run_softlexicon_expert_concat or args.run_softlexicon_expert_weighted or
        args.run_softlexicon_expert_gated or args.run_softlexicon_expert_attention or
        args.run_all):
        print("\n加载词向量...")
        vectors = load_vectors("chinese", 50)
        if vectors is None:
            raise ValueError("无法加载中文 50 维词向量，请检查 assets/vectors 下是否存在相应文件。")
        print("✅ 词向量加载成功\n")
    
    # 执行实验
    if args.run_baseline or args.run_all:
        results_summary['baseline'] = run_experiment('baseline', args)
    
    if args.run_expert_dict or args.run_all:
        results_summary['expert_dict_manual'] = run_experiment('expert_dict_manual', args)
    
    if args.run_expert_dict_auto or args.run_all:
        results_summary['expert_dict_auto'] = run_experiment('expert_dict_auto', args)
    
    if args.run_softlexicon or args.run_all:
        results_summary['softlexicon'] = run_experiment('softlexicon', args, vectors)
    
    if args.run_softlexicon_trainlex:
        results_summary['softlexicon_trainlex'] = run_experiment('softlexicon_trainlex', args, vectors)
    
    if args.run_softlexicon_expert_concat:
        results_summary['fusion_concat'] = run_experiment('fusion_concat', args, vectors)
    
    if args.run_softlexicon_expert_weighted:
        results_summary['fusion_weighted'] = run_experiment('fusion_weighted', args, vectors)
    
    if args.run_softlexicon_expert_gated:
        results_summary['fusion_gated'] = run_experiment('fusion_gated', args, vectors)
    
    if args.run_softlexicon_expert_attention:
        results_summary['fusion_attention'] = run_experiment('fusion_attention', args, vectors)
    
    # 打印对比结果
    if len(results_summary) >= 2:
        print(f"\n{'='*70}")
        print("对比结果总结")
        print(f"{'='*70}\n")
        
        header = f"{'模型':<30} {'测试Loss':<15} {'测试F1':<15} {'参数量':<15}"
        separator = '-' * 75
        print(header)
        print(separator)
        
        for model_type, result in results_summary.items():
            model_name = ModelFactory.get_model_display_name(model_type)
            test_loss = result['test_loss']
            test_f1 = result['test_metrics'][0] if result['test_metrics'] else 0.0
            total_params = result['total_params']
            print(f"{model_name:<30} {test_loss:<15.4f} {test_f1:<15.4f} {total_params:<15,}")
        
        print(f"\n{'='*70}\n")
        
        # 保存对比结果
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        comparison_file = f"{args.save_dir}/comparison_{timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"💾 对比结果已保存到: {comparison_file}\n")


if __name__ == "__main__":
    main()
