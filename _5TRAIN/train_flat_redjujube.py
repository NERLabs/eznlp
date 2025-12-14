#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLAT+BERT 模型在 RedJujube 数据集上的训练脚本

功能：
- 使用自己构建的 FLAT 模型架构
- 结合 BERT 预训练模型
- 在 RedJujube 数据集上进行 NER 任务训练

使用方式：
    python train_flat_redjujube.py --data_dir _2DATA/RedJujube --gpu 0
"""

import argparse
import os
import sys
import datetime
import json
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入 FLAT 模型构建器
from _4MODELS.models.flat_model_builder import FLATModelFactory
from _5TRAIN.redjujube_data_loader import RedJujubeDataLoader, DataPreparationPipeline
from _5TRAIN.redjujube_trainer import (
    RedJujubeTrainerConfig,
    LoggerManager,
    RedJujubeNERTrainer
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FLAT+BERT 在 RedJujube 数据集上的训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='_2DATA/RedJujube',
                        help='数据目录')
    parser.add_argument('--save_dir', type=str, default='cache/flat_redjujube',
                        help='模型保存目录')
    
    # FLAT 模型参数
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='隐藏层维度（必须能被 num_heads 整除）')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Transformer 层数')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='多头注意力头数')
    parser.add_argument('--ff_size', type=int, default=2048,
                        help='前馈网络隐藏层维度')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='最大序列长度')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout 率')
    parser.add_argument('--four_pos_fusion', type=str, default='ff',
                        choices=['ff', 'attn', 'gate'],
                        help='四位置融合方式')
    parser.add_argument('--learnable_position', action='store_true',
                        help='是否使用可学习的位置编码')
    parser.add_argument('--layer_preprocess', type=str, default='n',
                        help='层前处理序列')
    parser.add_argument('--layer_postprocess', type=str, default='dan',
                        help='层后处理序列')
    
    # BERT 参数
    parser.add_argument('--bert_arch', type=str, default='hfl/chinese-macbert-base',
                        help='BERT 模型架构')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='是否冻结 BERT 参数')
    parser.add_argument('--mix_layers', type=str, default='top',
                        help='BERT 层融合方式')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小（FLAT 显存占用较大，建议 8-16）')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--finetune_lr', type=float, default=2e-5,
                        help='BERT 微调学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='梯度裁剪')
    parser.add_argument('--num_grad_acc_steps', type=int, default=1,
                        help='梯度累积步数')
    parser.add_argument('--warmup_steps', type=float, default=0.1,
                        help='Warmup 步数（比例）')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')
    
    # 显示和评估参数
    parser.add_argument('--disp_every_steps', type=int, default=50,
                        help='每 N 步显示一次')
    parser.add_argument('--eval_every_steps', type=int, default=200,
                        help='每 N 步评估一次')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU 设备 ID')
    parser.add_argument('--model_type', type=str, default='flat_bert',
                        choices=['flat_baseline', 'flat_bert'],
                        help='模型类型（是否使用 BERT）')
    
    # 用于兼容性的额外参数
    parser.add_argument('--hid_dim', type=int, default=256,
                        help='（兼容性参数，不使用）')
    parser.add_argument('--num_layers_lstm', type=int, default=1,
                        help='（兼容性参数，不使用）')
    
    return parser.parse_args()


def setup_environment(args):
    """设置环境"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置 GPU
    if args.gpu >= 0 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"✓ 使用 GPU: {args.gpu}")
    else:
        print("✓ 使用 CPU")
    
    # 创建保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{args.save_dir}/{args.model_type}_{timestamp}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    return save_dir


def validate_model_params(args):
    """验证模型参数"""
    # 检查 hidden_size 必须能被 num_heads 整除
    if args.hidden_size % args.num_heads != 0:
        raise ValueError(
            f"hidden_size ({args.hidden_size}) 必须能被 num_heads ({args.num_heads}) 整除！\n"
            f"建议配置:\n"
            f"  - hidden_size=512, num_heads=8 (每头 64 维)\n"
            f"  - hidden_size=768, num_heads=12 (每头 64 维)\n"
            f"  - hidden_size=256, num_heads=4 (每头 64 维)"
        )
    
    print(f"\n✓ 模型参数验证通过:")
    print(f"  - hidden_size: {args.hidden_size}")
    print(f"  - num_heads: {args.num_heads}")
    print(f"  - 每个注意力头维度: {args.hidden_size // args.num_heads}")


def load_data(args, logger):
    """加载数据"""
    logger.info("="*70)
    logger.info("加载 RedJujube 数据集")
    logger.info("="*70)
    
    data_loader = RedJujubeDataLoader(args.data_dir)
    data_pipeline = DataPreparationPipeline(data_loader)
    
    # 准备数据（Baseline 模式，不使用词典特征）
    train_data, dev_data, test_data = data_pipeline.prepare('baseline')
    
    logger.info(f"✓ 训练集: {len(train_data)} 条")
    logger.info(f"✓ 验证集: {len(dev_data)} 条")
    logger.info(f"✓ 测试集: {len(test_data)} 条")
    
    return train_data, dev_data, test_data


def build_model(args, logger):
    """构建 FLAT 模型"""
    logger.info("="*70)
    logger.info("构建 FLAT+BERT 模型")
    logger.info("="*70)
    
    # 使用 FLAT 模型工厂创建模型配置
    model_config = FLATModelFactory.create_model_config(args.model_type, args)
    
    logger.info(f"✓ 模型类型: {args.model_type}")
    logger.info(f"✓ 使用 BERT: {hasattr(model_config, 'bert_like') and model_config.bert_like is not None}")
    logger.info(f"✓ BERT 架构: {args.bert_arch}")
    logger.info(f"✓ FLAT 层数: {args.num_layers}")
    logger.info(f"✓ 隐藏维度: {args.hidden_size}")
    logger.info(f"✓ 注意力头数: {args.num_heads}")
    logger.info(f"✓ 前馈网络维度: {args.ff_size}")
    logger.info(f"✓ 四位置融合: {args.four_pos_fusion}")
    
    return model_config


def train_model(model_config, train_data, dev_data, test_data, args, save_dir, logger):
    """训练模型"""
    logger.info("="*70)
    logger.info("开始训练")
    logger.info("="*70)
    
    # 创建训练器配置
    model_display_name = f"FLAT+BERT (层数={args.num_layers}, dim={args.hidden_size})"
    train_config = RedJujubeTrainerConfig(args, save_dir, model_display_name)
    
    # 创建训练器
    trainer = RedJujubeNERTrainer(train_config, logger)
    
    # 开始训练
    results = trainer.train(model_config, train_data, dev_data, test_data, use_expert_dict=False)
    
    return results


def save_results(results, save_dir, args):
    """保存实验结果"""
    result_file = os.path.join(save_dir, 'experiment_results.json')
    
    results_dict = {
        'model_type': args.model_type,
        'model_config': {
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'ff_size': args.ff_size,
            'max_seq_len': args.max_seq_len,
            'dropout': args.dropout,
            'four_pos_fusion': args.four_pos_fusion,
            'bert_arch': args.bert_arch,
        },
        'training_config': {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'finetune_lr': args.finetune_lr,
        },
        'results': results
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 实验结果已保存到: {result_file}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 打印标题
    print("\n" + "="*70)
    print("FLAT+BERT 模型 - RedJujube 数据集训练")
    print("="*70)
    print(f"\n📊 模型配置:")
    print(f"  - 模型类型: {args.model_type}")
    print(f"  - BERT: {args.bert_arch}")
    print(f"  - FLAT 层数: {args.num_layers}")
    print(f"  - 隐藏维度: {args.hidden_size}")
    print(f"  - 注意力头数: {args.num_heads}")
    print(f"  - 前馈维度: {args.ff_size}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 学习率: {args.lr}")
    print()
    
    # 验证参数
    validate_model_params(args)
    
    # 设置环境
    save_dir = setup_environment(args)
    print(f"\n💾 模型保存目录: {save_dir}\n")
    
    # 设置日志
    logger = LoggerManager.setup_logger(save_dir)
    
    try:
        # 加载数据
        train_data, dev_data, test_data = load_data(args, logger)
        
        # 构建模型
        model_config = build_model(args, logger)
        
        # 训练模型
        results = train_model(model_config, train_data, dev_data, test_data, 
                             args, save_dir, logger)
        
        # 保存结果
        save_results(results, save_dir, args)
        
        # 打印最终结果
        print("\n" + "="*70)
        print("训练完成！")
        print("="*70)
        print(f"\n📈 最终结果:")
        print(f"  - 测试集 Loss: {results['test_loss']:.4f}")
        print(f"  - 测试集 F1: {results['test_metrics'][0]:.4f}")
        print(f"  - 模型参数量: {results['total_params']:,}")
        print(f"  - 保存目录: {save_dir}")
        print()
        
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
        print("\n⚠️  训练已中断")
    except Exception as e:
        logger.error(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
