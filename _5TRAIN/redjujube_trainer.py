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
from eznlp.utils.transition import ChunksTagsTranslator  # 新增：BMES chunks→tags 转换工具


class FGM:
    """FGM 对抗训练 - Fast Gradient Method
    
    在嵌入层施加梯度方向的微小扰动，提升模型鲁棒性。
    """
    
    def __init__(self, model, epsilon=1.0, emb_name='embeddings'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
    
    def attack(self):
        """对嵌入层施加对抗扰动"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = self.epsilon * param.grad / norm
                        param.data.add_(r_at)
    
    def restore(self):
        """恢复嵌入层原始参数"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class EMA:
    """EMA 指数移动平均
    
    对模型权重做指数移动平均，平滑训练过程，减少过拟合。
    """
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()
    
    def _register(self):
        """注册所有可训练参数的影子副本"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新影子参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
    
    def apply_shadow(self):
        """将影子参数应用到模型（评估时使用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数（评估后使用）"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


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
        
        # FGM 对抗训练参数
        self.use_fgm = getattr(args, 'use_fgm', True)
        self.fgm_epsilon = getattr(args, 'fgm_epsilon', 1.0)
        
        # EMA 参数
        self.use_ema = getattr(args, 'use_ema', True)
        self.ema_decay = getattr(args, 'ema_decay', 0.999)
        
        # BMES Aux Loss 参数（默认禁用）
        self.bmes_aux_lambda = getattr(args, 'bmes_aux_lambda', 0.0)
        self.bmes_label_aux_lambda = getattr(args, 'bmes_label_aux_lambda', 0.0)
        
        # R-Drop 正则化参数
        self.use_rdrop = getattr(args, 'use_rdrop', False)
        self.rdrop_alpha = getattr(args, 'rdrop_alpha', 0.5)
        
        # 显示和评估参数
        self.disp_every_steps = getattr(args, 'disp_every_steps', 50)
        self.eval_every_steps = getattr(args, 'eval_every_steps', 200)

        # 每 epoch 在测试集上评估（用于诊断过拟合，默认关闭）
        self.eval_test_each_epoch = getattr(args, 'eval_test_each_epoch', False)
        
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
        
        # 打印模型结构
        self.logger.info("\n" + "=" * 70)
        self.logger.info("模型结构:")
        self.logger.info("=" * 70)
        self.logger.info(model)
        self.logger.info("=" * 70 + "\n")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"总参数量: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
        
        # === BMES 注意力 aux loss 的先验矩阵与权重 ===
        # 通道顺序: 0=B, 1=M, 2=E, 3=S
        bmes_prior = torch.zeros(4, 4, device=self.config.device)
        # B↔E 强, M↔B/E 中等, S 自关联
        bmes_prior[0, 2] = 1.0
        bmes_prior[2, 0] = 1.0
        bmes_prior[1, 0] = 0.5
        bmes_prior[1, 2] = 0.5
        bmes_prior[0, 1] = 0.5
        bmes_prior[2, 1] = 0.5
        bmes_prior[3, 3] = 1.0
        # 行归一化成概率分布
        bmes_prior = bmes_prior / bmes_prior.sum(dim=-1, keepdim=True)
        # 全局先验型 aux loss 权重（从配置读取）
        bmes_aux_lambda = self.config.bmes_aux_lambda

        # === Label-aware BMES aux loss 配置 ===
        # chunks -> BMES token 标签
        bmes_translator = ChunksTagsTranslator(scheme="BMES", sep="-", breaking_for_types=True)
        # BMES 标签前缀到通道下标
        bmes_tag2idx = {"B": 0, "M": 1, "E": 2, "S": 3}
        # label-aware aux loss 权重（从配置读取）
        bmes_label_aux_lambda = self.config.bmes_label_aux_lambda
        
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

        # 每 epoch 指标记录（用于诊断过拟合）
        epoch_metrics = []
        best_test_f1_seen = 0.0
        best_test_f1_epoch = 0
        
        # 初始化 FGM 对抗训练
        fgm = None
        if self.config.use_fgm:
            fgm = FGM(model, epsilon=self.config.fgm_epsilon)
            self.logger.info(f"✅ FGM 对抗训练已启用 (epsilon={self.config.fgm_epsilon})")
        
        # 初始化 EMA
        ema = None
        if self.config.use_ema:
            ema = EMA(model, decay=self.config.ema_decay)
            self.logger.info(f"✅ EMA 已启用 (decay={self.config.ema_decay})")
        
        # 初始化 R-Drop 正则化
        if self.config.use_rdrop:
            self.logger.info(f"✅ R-Drop 正则化已启用 (alpha={self.config.rdrop_alpha})")
        
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
                    main_loss = losses.mean()
                    
                    # ===== R-Drop 正则化：两次前向传播取 loss 平均 =====
                    rdrop_loss = 0.0
                    if self.config.use_rdrop:
                        # 第二次前向传播（不同的 dropout mask）
                        losses2, states2 = model(batch, return_states=True)
                        loss2 = losses2.mean()
                        # 两次 loss 取平均（简化版 R-Drop，不需要 KL 散度）
                        main_loss = (main_loss + loss2) / 2.0
                        rdrop_loss = loss2.item()  # 用于日志记录
                    
                    # ===== BMES 通道注意力 aux loss：全局先验 + label-aware =====
                    aux_prior_loss = 0.0
                    aux_label_loss = 0.0

                    if "expert_bmes_attn_weights" in states:
                        # attn: (batch, step, num_heads, 4, 4)
                        attn = states["expert_bmes_attn_weights"]

                        # ---------- 3.1 全局先验型约束（保持你之前的实现） ----------
                        attn_mean_global = attn.mean(dim=(0, 1, 2))  # (4, 4)
                        attn_mean_global = attn_mean_global / (
                            attn_mean_global.sum(dim=-1, keepdim=True) + 1e-8
                        )
                        aux_prior_loss = ((attn_mean_global - bmes_prior) ** 2).mean()

                        # ---------- 3.2 label-aware 约束 ----------
                        # 先在 head 维做平均：每个 token 一个 4x4 矩阵
                        # attn_token: (batch, step, 4, 4)
                        attn_token = attn.mean(dim=2)

                        per_token_losses = []

                        # 逐样本，根据 gold chunks 生成 BMES 标签，再对相应 token 的通道行加约束
                        batch_size = attn_token.size(0)
                        for i in range(batch_size):
                            seq_len_i = batch.seq_lens[i].item()

                            # Boundaries 对象里有 gold chunks
                            if not hasattr(batch, "boundaries_objs"):
                                continue
                            boundaries_obj = batch.boundaries_objs[i]
                            chunks = getattr(boundaries_obj, "chunks", None)
                            if chunks is None:
                                continue

                            # gold BMES 标签序列（长度 = seq_len_i）
                            tags = bmes_translator.chunks2tags(chunks, seq_len_i)

                            for t, tag in enumerate(tags):
                                # 跳过非实体位置
                                if tag == "O" or tag == "<pad>":
                                    continue

                                # 取标签前缀：B/M/E/S
                                if "-" in tag:
                                    tag_prefix = tag.split("-", maxsplit=1)[0]
                                else:
                                    tag_prefix = tag

                                ch_idx = bmes_tag2idx.get(tag_prefix, None)
                                if ch_idx is None:
                                    continue

                                # 该 token 在对应通道的注意力行分布: (4,)
                                pred_row = attn_token[i, t, ch_idx]
                                # 目标分布：用先验矩阵的对应行
                                target_row = bmes_prior[ch_idx]

                                per_token_losses.append(
                                    ((pred_row - target_row) ** 2).mean()
                                )

                        if len(per_token_losses) > 0:
                            aux_label_loss = torch.stack(per_token_losses).mean()

                    # 合并两个 aux loss
                    loss = (
                        main_loss
                        + bmes_aux_lambda * aux_prior_loss
                        + bmes_label_aux_lambda * aux_label_loss
                    )
                # ===== 反向传播和优化保持不变 =====
                
                # 反向传播
                loss.backward()
                
                # FGM 对抗训练
                if fgm is not None:
                    fgm.attack()
                    with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                        losses_adv, _ = model(batch, return_states=True)
                        loss_adv = losses_adv.mean()
                    loss_adv.backward()
                    fgm.restore()
                
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # EMA 更新
                if ema is not None:
                    ema.update()
                
                global_step += 1
                epoch_losses.append(loss.item())
                
                # 记录到 TensorBoard
                if self.writer:
                    self.writer.add_scalar("train/loss", loss.item(), global_step)
                    self.writer.add_scalar("train/bmes_aux_prior", float(aux_prior_loss), global_step)
                    self.writer.add_scalar("train/bmes_aux_label", float(aux_label_loss), global_step)
                    if self.config.use_rdrop:
                        self.writer.add_scalar("train/rdrop_loss2", float(rdrop_loss), global_step)
                    current_lr = scheduler.get_last_lr()[0]
                    self.writer.add_scalar("train/lr", current_lr, global_step)
            
            train_loss = sum(epoch_losses) / len(epoch_losses)
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            if self.writer:
                self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            
            # 验证
            # 评估时使用 EMA 权重
            if ema is not None:
                ema.apply_shadow()
            
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
                # 如果使用 EMA，此时模型已应用 EMA 权重，直接保存
                torch.save(model, model_path)
                self.logger.info(f"✅ 保存最佳模型 (epoch {epoch}, F1={dev_f1:.4f})")

            # 每 epoch 在测试集上评估（仅诊断用，不影响选模型）
            epoch_test_loss = None
            epoch_test_f1 = None
            if self.config.eval_test_each_epoch:
                test_eval = trainer.eval_epoch(test_loader)
                if isinstance(test_eval, tuple):
                    epoch_test_loss = float(test_eval[0])
                    epoch_test_f1 = float(test_eval[1]) if len(test_eval) > 1 else 0.0
                else:
                    epoch_test_loss = float(test_eval)
                    epoch_test_f1 = 0.0
                self.logger.info(
                    f"📊 [诊断] Test Loss: {epoch_test_loss:.4f}, Test F1: {epoch_test_f1:.4f}"
                )
                if self.writer:
                    self.writer.add_scalar("test_each_epoch/loss", epoch_test_loss, epoch)
                    self.writer.add_scalar("test_each_epoch/f1", epoch_test_f1, epoch)
                if epoch_test_f1 > best_test_f1_seen:
                    best_test_f1_seen = epoch_test_f1
                    best_test_f1_epoch = epoch

            # 记录每 epoch 指标
            epoch_metrics.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "dev_loss": float(dev_loss),
                "dev_f1": float(dev_f1),
                "test_loss": epoch_test_loss,
                "test_f1": epoch_test_f1,
            })

            # 恢复原始权重继续训练
            if ema is not None:
                ema.restore()
        
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

        # 注入每 epoch 诊断指标（如启用了 eval_test_each_epoch）
        if epoch_metrics:
            results['epoch_metrics'] = epoch_metrics
        if self.config.eval_test_each_epoch:
            results['best_test_f1_seen'] = float(best_test_f1_seen)
            results['best_test_f1_epoch'] = int(best_test_f1_epoch)
            self.logger.info(
                f"\n📊 [诊断] 训练过程中观测到的最高 Test F1: {best_test_f1_seen:.4f} (epoch {best_test_f1_epoch})"
            )
        
        with open(f"{self.config.save_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 关闭 TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return results
