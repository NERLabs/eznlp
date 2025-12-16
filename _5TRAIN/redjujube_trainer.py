#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube NER 训练器

功能：
- 封装训练流程和训练循环
- 优化器和学习率调度器构建
- 日志管理和结果保存
- 模型评估和检查点保存
"""

import os
import sys
import logging
import json
import torch
from eznlp.dataset import Dataset
from eznlp.training import Trainer


class RedJujubeTrainerConfig:
    """训练器配置类"""
    
    def __init__(self, args, save_dir, model_name):
        """初始化训练器配置
        
        Args:
            args: 命令行参数对象
            save_dir: 保存目录
            model_name: 模型名称
        """
        self.args = args
        self.save_dir = save_dir
        self.model_name = model_name
        
        # 训练参数
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.finetune_lr = args.finetune_lr
        self.weight_decay = args.weight_decay
        self.grad_clip = args.grad_clip
        self.num_grad_acc_steps = getattr(args, 'num_grad_acc_steps', 1)
        self.use_amp = getattr(args, 'use_amp', False)
        
        # 显示和评估参数
        self.disp_every_steps = getattr(args, 'disp_every_steps', 50)
        self.eval_every_steps = getattr(args, 'eval_every_steps', 200)
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LoggerManager:
    """日志管理器"""
    
    @staticmethod
    def setup_logger(save_dir):
        """设置日志器
        
        Args:
            save_dir: 保存目录
            
        Returns:
            logging.Logger: 日志器对象
        """
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


class OptimizerBuilder:
    """优化器构建器"""
    
    @staticmethod
    def build_optimizer_and_scheduler(model, num_train_batches, config):
        """构建优化器和调度器
        
        Args:
            model: PyTorch 模型
            num_train_batches: 每个 epoch 的批次数
            config: RedJujubeTrainerConfig 对象
            
        Returns:
            tuple: (optimizer, scheduler)
        """
        # 分组参数：预训练模型用小学习率，其他用大学习率
        param_groups = []
        if hasattr(model, 'pretrained_parameters'):
            pretrained_params = list(model.pretrained_parameters())
            pretrained_param_ids = {id(p) for p in pretrained_params}
            other_params = [p for p in model.parameters() if id(p) not in pretrained_param_ids]
            
            param_groups.append({
                'params': pretrained_params,
                'lr': config.finetune_lr
            })
            param_groups.append({
                'params': other_params,
                'lr': config.lr
            })
        else:
            param_groups.append({
                'params': model.parameters(),
                'lr': config.lr
            })
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
        
        # 线性衰减 + Warmup
        num_warmup_epochs = max(2, config.num_epochs // 5)
        num_warmup_steps = num_train_batches * num_warmup_epochs
        num_total_steps = num_train_batches * config.num_epochs
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_total_steps - current_step) / float(max(1, num_total_steps - num_warmup_steps))
            )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return optimizer, scheduler


class DatasetBuilder:
    """数据集构建器"""
    
    @staticmethod
    def build_datasets(train_data, dev_data, test_data, config, use_expert_dict=False):
        """构建训练、验证和测试数据集
        
        Args:
            train_data: 训练数据
            dev_data: 验证数据
            test_data: 测试数据
            config: 模型配置对象
            use_expert_dict: 是否使用专家词典
            
        Returns:
            tuple: (train_set, dev_set, test_set)
        """
        # 构建训练集
        train_set = Dataset(train_data, config, training=True)
        train_set.build_vocabs_and_dims(dev_data, test_data)
        
        # 如果使用专家词典，构建词频统计
        if use_expert_dict and hasattr(config, 'nested_ohots') and config.nested_ohots:
            if "expert_dict" in config.nested_ohots:
                config.nested_ohots["expert_dict"].build_freqs(train_data, dev_data, test_data)
        
        # 如果使用SoftLexicon，构建词频统计
        if hasattr(config, 'nested_ohots') and config.nested_ohots:
            if "softlexicon" in config.nested_ohots:
                config.nested_ohots["softlexicon"].build_freqs(train_data, dev_data, test_data)
        
        # 构建验证集和测试集
        dev_set = Dataset(dev_data, config, training=False)
        test_set = Dataset(test_data, config, training=False)
        
        return train_set, dev_set, test_set


class RedJujubeNERTrainer:
    """RedJujube NER 训练器"""
    
    def __init__(self, train_config, logger):
        """初始化训练器
        
        Args:
            train_config: RedJujubeTrainerConfig 对象
            logger: 日志器对象
        """
        self.config = train_config
        self.logger = logger
    
    def train(self, model_config, train_data, dev_data, test_data, use_expert_dict=False):
        """训练模型
        
        Args:
            model_config: 模型配置对象
            train_data: 训练数据
            dev_data: 验证数据
            test_data: 测试数据
            use_expert_dict: 是否使用专家词典
            
        Returns:
            dict: 训练结果字典
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"开始训练: {self.config.model_name}")
        self.logger.info(f"{'='*70}\n")
        
        # 1. 构建数据集
        self.logger.info("构建数据集...")
        train_set, dev_set, test_set = DatasetBuilder.build_datasets(
            train_data, dev_data, test_data, model_config, use_expert_dict
        )
        self.logger.info(train_set.summary)
        
        # 2. 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=train_set.collate
        )
        dev_loader = torch.utils.data.DataLoader(
            dev_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=dev_set.collate
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=test_set.collate
        )
        
        # 3. 实例化模型
        self.logger.info("实例化模型...")
        model = model_config.instantiate().to(self.config.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"总参数量: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
        
        # 4. 构建优化器和调度器
        self.logger.info("构建优化器和调度器...")
        optimizer, scheduler = OptimizerBuilder.build_optimizer_and_scheduler(
            model, len(train_loader), self.config
        )
        
        # 5. 创建训练器
        trainer = Trainer(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            schedule_by_step=True,
            num_grad_acc_steps=self.config.num_grad_acc_steps,
            device=self.config.device,
            grad_clip=self.config.grad_clip,
            use_amp=self.config.use_amp
        )
        
        # 6. 保存模型配置（用于测试和部署）
        torch.save(model_config, f"{self.config.save_dir}/{model_config.name}-config.pth")
        self.logger.info(f"💾 已保存配置: {self.config.save_dir}/{model_config.name}-config.pth")
        
        # 7. 保存回调
        def save_callback(model):
            # 保存完整模型（推荐用于测试和部署）
            model_path = f"{self.config.save_dir}/{model_config.name}.pth"
            torch.save(model, model_path)
            self.logger.info(f"✅ 保存最佳模型到: {model_path}")
        
        # 8. 开始训练
        self.logger.info(f"\n开始训练 {self.config.num_epochs} 个 epoch...")
        self.logger.info(f"设备: {self.config.device}")
        self.logger.info(f"批次大小: {self.config.batch_size}")
        self.logger.info(f"学习率: {self.config.lr}")
        self.logger.info(f"微调学习率: {self.config.finetune_lr}\n")
        
        trainer.train_steps(
            train_loader=train_loader,
            dev_loader=dev_loader,
            num_epochs=self.config.num_epochs,
            disp_every_steps=self.config.disp_every_steps,
            eval_every_steps=self.config.eval_every_steps,
            save_callback=save_callback,
            save_by_loss=False  # 按 F1 保存
        )
        
        # 9. 加载最佳模型并在测试集上评估
        best_model_path = f"{self.config.save_dir}/{model_config.name}.pth"
        self.logger.info(f"\n加载最佳模型: {best_model_path}")
        model = torch.load(best_model_path, map_location=self.config.device, weights_only=False)
        
        self.logger.info("在测试集上评估...")
        test_loss, *test_metrics = trainer.eval_epoch(test_loader)
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("测试集结果:")
        self.logger.info(f"  Loss: {test_loss:.4f}")
        if test_metrics:
            for i, metric in enumerate(test_metrics):
                self.logger.info(f"  Metric {i}: {metric:.4f}")
        self.logger.info(f"{'='*70}\n")
        
        # 10. 保存结果
        results = {
            'model_type': self.config.model_name,
            'test_loss': float(test_loss),
            'test_metrics': [float(m) for m in test_metrics] if test_metrics else [],
            'total_params': total_params,
            'trainable_params': trainable_params,
            'args': vars(self.config.args)
        }
        
        with open(f"{self.config.save_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
