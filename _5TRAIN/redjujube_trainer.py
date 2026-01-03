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
from torch.utils.tensorboard import SummaryWriter
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
        
        # TensorBoard 配置
        self.use_tensorboard = getattr(args, 'use_tensorboard', True)
        self.tensorboard_dir = os.path.join(save_dir, 'tensorboard')
        
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
        self.config = train_config
        self.logger = logger
        self.writer = None
        
        # 初始化 TensorBoard
        if self.config.use_tensorboard:
            tb_log_dir = os.path.join(self.config.save_dir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_log_dir)
            self.logger.info(f"📊 TensorBoard 日志目录: {tb_log_dir}")
    
    def _log_tensorboard(self, tag, value, step):
        """记录到 TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def _log_tensorboard_metrics(self, prefix, metrics_dict, step):
        """记录多个指标到 TensorBoard"""
        if self.writer is not None:
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{prefix}/{key}", value, step)
    
    def _close_tensorboard(self):
        """关闭 TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()
    
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
        
        # 6. 保存模型配置
        torch.save(model_config, f"{self.config.save_dir}/{model_config.name}-config.pth")
        self.logger.info(f"💾 已保存配置: {self.config.save_dir}/{model_config.name}-config.pth")
        
        # 7. 记录模型结构到 TensorBoard
        if self.writer is not None:
            try:
                sample_batch = next(iter(train_loader))
                sample_batch = sample_batch.to(self.config.device)
                
                # 包装一个简化的 forward，只接受 batch，返回 loss
                class ModelWrapper(torch.nn.Module):
                    def __init__(self, base_model):
                        super().__init__()
                        self.base_model = base_model
                    
                    def forward(self, batch):
                        losses, _ = self.base_model(batch, return_states=True)
                        return losses
                
                wrapped_model = ModelWrapper(model)
                self.writer.add_graph(wrapped_model, sample_batch)
                self.logger.info("📊 已记录模型结构到 TensorBoard")
            except Exception as e:
                self.logger.warning(f"记录模型图失败: {e}")
        
        # 8. 训练循环变量
        best_dev_f1 = 0.0
        best_epoch = 0
        global_step = 0
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.logger.info(f"\n===== Epoch {epoch}/{self.config.num_epochs} =====")
            
            # 手动训练循环（支持 step 级别的 TensorBoard 记录）
            model.train()
            epoch_losses = []
            
            for batch in train_loader:
                batch = batch.to(self.config.device)
                
                # 前向传播
                with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                    losses, states = model(batch, return_states=True)
                    loss = losses.mean()
                
                # 反向传播
                loss.backward()
                
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                epoch_losses.append(loss.item())
                
                # 记录到 TensorBoard
                if self.writer:
                    self.writer.add_scalar("train/loss", loss.item(), global_step)
                    current_lr = scheduler.get_last_lr()[0]
                    self.writer.add_scalar("train/lr", current_lr, global_step)
            
            train_loss = sum(epoch_losses) / len(epoch_losses)
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            if self.writer:
                self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            
            # 验证
            eval_result = trainer.eval_epoch(dev_loader)
            if isinstance(eval_result, tuple):
                dev_loss = eval_result[0]
                dev_f1 = eval_result[1] if len(eval_result) > 1 else 0.0
            else:
                dev_loss = eval_result
                dev_f1 = 0.0
            
            self.logger.info(f"Dev Loss: {dev_loss:.4f}, Dev F1: {dev_f1:.4f}")
            if self.writer:
                self.writer.add_scalar("dev/loss", dev_loss, epoch)
                self.writer.add_scalar("dev/f1", dev_f1, epoch)
            
            # 保存最佳模型
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch = epoch
                model_path = f"{self.config.save_dir}/{model_config.name}.pth"
                torch.save(model, model_path)
                self.logger.info(f"✅ 保存最佳模型 (epoch {epoch}, F1={dev_f1:.4f})")
        
        # 9. 加载最佳模型并在测试集上评估
        self.logger.info(f"\n最优验证 F1: {best_dev_f1:.4f} (epoch {best_epoch})")
        best_model_path = f"{self.config.save_dir}/{model_config.name}.pth"
        self.logger.info(f"加载最佳模型: {best_model_path}")
        model = torch.load(best_model_path, map_location=self.config.device, weights_only=False)
        
        # 重新构建 trainer 用于评估
        trainer = Trainer(
            model,
            device=self.config.device,
            use_amp=self.config.use_amp
        )
        
        self.logger.info("在测试集上评估...")
        test_result = trainer.eval_epoch(test_loader)
        if isinstance(test_result, tuple):
            test_loss = test_result[0]
            test_f1 = test_result[1] if len(test_result) > 1 else 0.0
        else:
            test_loss = test_result
            test_f1 = 0.0
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("测试集结果:")
        self.logger.info(f"  Loss: {test_loss:.4f}")
        self.logger.info(f"  F1: {test_f1:.4f}")
        self.logger.info(f"{'='*70}\n")
        
        # 记录测试结果到 TensorBoard
        if self.writer:
            self.writer.add_scalar("test/loss", test_loss, 0)
            self.writer.add_scalar("test/f1", test_f1, 0)
        
        # 保存结果
        results = {
            'model_type': self.config.model_name,
            'best_dev_f1': float(best_dev_f1),
            'best_epoch': best_epoch,
            'test_loss': float(test_loss),
            'test_f1': float(test_f1),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'args': vars(self.config.args)
        }
        
        with open(f"{self.config.save_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 关闭 TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return results
