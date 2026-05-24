#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lattice Transformer 辅助工具模块

提供 FLAT 模型使用的辅助组件，包括：
- 层处理模块（Layer Process）：灵活的层前/后处理序列
- 位置感知前馈网络（Positionwise Feed-Forward）
- 自适应 Dropout
- 绝对位置编码（Start-End 感知）

参考：Li et al. "FLAT: Chinese NER Using Flat-Lattice Transformer" (ACL 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


class AdaptiveDropout(nn.Module):
    """自适应 Dropout
    
    FLAT 项目中的自定义 Dropout 实现，支持更灵活的训练控制
    """
    
    def __init__(self, p: float):
        """初始化
        
        Args:
            p: Dropout 概率
        """
        super().__init__()
        assert 0 <= p <= 1, "Dropout 概率必须在 [0, 1] 范围内"
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            
        Returns:
            Dropout 后的张量
        """
        if self.training and self.p > 0.001:
            mask = torch.rand(x.size(), device=x.device)
            mask = mask.lt(self.p)
            x = x.masked_fill(mask, 0) / (1 - self.p)
        return x


class LayerProcess(nn.Module):
    """层处理模块
    
    灵活的层前/后处理序列，支持：
    - 'a': Add (残差连接)
    - 'd': Dropout
    - 'n': LayerNorm
    
    示例：
        - 'dan': Dropout -> Add -> LayerNorm
        - 'n': 仅 LayerNorm
        - 'ad': Add -> Dropout
    """
    
    def __init__(
        self, 
        process_sequence: str, 
        hidden_size: int, 
        dropout: float = 0.0,
        use_adaptive_dropout: bool = True
    ):
        """初始化
        
        Args:
            process_sequence: 处理序列，如 'dan', 'n', 'ad'
            hidden_size: 隐藏层维度
            dropout: Dropout 概率
            use_adaptive_dropout: 是否使用自适应 Dropout
        """
        super().__init__()
        self.process_sequence = process_sequence.lower()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        
        if 'd' in self.process_sequence:
            if use_adaptive_dropout:
                self.dropout = AdaptiveDropout(dropout)
            else:
                self.dropout = nn.Dropout(dropout)
        
        if 'n' in self.process_sequence:
            self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            residual: 残差张量（用于 'a' 操作），如果为 None 则使用输入本身
            
        Returns:
            处理后的张量
        """
        if residual is None:
            residual = x
        
        output = x
        for op in self.process_sequence:
            if op == 'a':
                output = output + residual
            elif op == 'd':
                output = self.dropout(output)
            elif op == 'n':
                output = self.layer_norm(output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """位置感知前馈网络
    
    标准的 Transformer 前馈网络，支持多层配置和灵活的激活函数
    """
    
    def __init__(
        self, 
        layer_sizes: List[int],
        dropout: float = 0.1,
        dropout_2: Optional[float] = None,
        activation: str = 'relu',
        use_adaptive_dropout: bool = True
    ):
        """初始化
        
        Args:
            layer_sizes: 各层维度列表，如 [512, 2048, 512]
            dropout: 第一个 Dropout 概率
            dropout_2: 第二个 Dropout 概率（可选）
            activation: 激活函数类型 ('relu', 'gelu', 'leaky')
            use_adaptive_dropout: 是否使用自适应 Dropout
        """
        super().__init__()
        
        assert len(layer_sizes) >= 2, "至少需要两层"
        
        self.num_layers = len(layer_sizes) - 1
        
        # 创建线性层
        for i in range(self.num_layers):
            setattr(self, f'w{i}', nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        # Dropout
        if use_adaptive_dropout:
            self.dropout = AdaptiveDropout(dropout)
            self.dropout_2 = AdaptiveDropout(dropout_2 if dropout_2 is not None else dropout)
        else:
            self.dropout = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout_2 if dropout_2 is not None else dropout)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            
        Returns:
            前馈网络输出 [batch, seq_len, hidden_size]
        """
        output = x
        for i in range(self.num_layers):
            # 第一层之后添加激活函数
            if i != 0:
                output = self.activation(output)
            
            # 线性变换
            w = getattr(self, f'w{i}')
            output = w(output)
            
            # Dropout（在不同位置）
            if i == 0:
                output = self.dropout(output)
            elif i == 1:
                output = self.dropout_2(output)
        
        return output


class AbsolutePositionEmbedding(nn.Module):
    """绝对位置编码
    
    标准的正弦位置编码，支持：
    - 可学习 / 固定位置编码
    - 位置归一化
    - 加法 / 拼接融合
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_len: int = 5000,
        learnable: bool = False,
        pos_norm: bool = False,
        fusion_mode: str = 'add'
    ):
        """初始化
        
        Args:
            hidden_size: 隐藏层维度
            max_len: 最大序列长度
            learnable: 是否可学习
            pos_norm: 是否对位置编码归一化
            fusion_mode: 融合模式 ('add' 或 'concat')
        """
        super().__init__()
        
        assert fusion_mode in ['add', 'concat'], "fusion_mode 必须是 'add' 或 'concat'"
        
        self.hidden_size = hidden_size
        self.pos_norm = pos_norm
        self.fusion_mode = fusion_mode
        
        # 生成正弦位置编码
        pe = self._get_sinusoidal_encoding(max_len, hidden_size)
        
        # 位置归一化
        if pos_norm:
            pe_sum = pe.sum(dim=-1, keepdim=True)
            with torch.no_grad():
                pe = pe / pe_sum
        
        pe = pe.unsqueeze(0)  # [1, max_len, hidden_size]
        self.pe = nn.Parameter(pe, requires_grad=learnable)
        
        # 拼接模式需要投影层
        if fusion_mode == 'concat':
            self.proj = nn.Linear(hidden_size * 2, hidden_size)
    
    @staticmethod
    def _get_sinusoidal_encoding(num_positions: int, hidden_size: int) -> torch.Tensor:
        """生成正弦位置编码
        
        Args:
            num_positions: 位置数量
            hidden_size: 隐藏层维度
            
        Returns:
            位置编码张量 [num_positions, hidden_size]
        """
        half_dim = hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_positions, -1)
        
        if hidden_size % 2 == 1:
            # 零填充
            emb = torch.cat([emb, torch.zeros(num_positions, 1)], dim=1)
        
        return emb
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            
        Returns:
            添加位置编码后的张量
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        if self.fusion_mode == 'add':
            return x + self.pe[:, :seq_len]
        else:  # concat
            pe_expanded = self.pe[:, :seq_len].expand(batch_size, -1, -1)
            x_with_pe = torch.cat([x, pe_expanded], dim=-1)
            return self.proj(x_with_pe)


class StartEndPositionEmbedding(nn.Module):
    """Start-End 位置编码
    
    为 Lattice 结构设计的双位置编码（开始位置和结束位置）
    支持多种融合策略
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_len: int = 5000,
        learnable: bool = False,
        pos_norm: bool = False,
        fusion_mode: str = 'add'
    ):
        """初始化
        
        Args:
            hidden_size: 隐藏层维度
            max_len: 最大序列长度
            learnable: 是否可学习
            pos_norm: 是否归一化
            fusion_mode: 融合模式
                - 'add': 直接相加
                - 'concat': 拼接后线性变换
                - 'nonlinear_add': 非线性变换后相加
                - 'nonlinear_concat': 非线性变换后拼接
        """
        super().__init__()
        
        assert fusion_mode in [
            'add', 'concat', 'nonlinear_add', 'nonlinear_concat',
            'add_nonlinear', 'concat_nonlinear'
        ], f"不支持的融合模式: {fusion_mode}"
        
        self.hidden_size = hidden_size
        self.fusion_mode = fusion_mode
        self.pos_norm = pos_norm
        
        # 生成两个独立的位置编码
        pe = AbsolutePositionEmbedding._get_sinusoidal_encoding(max_len, hidden_size)
        
        if pos_norm:
            pe_sum = pe.sum(dim=-1, keepdim=True)
            with torch.no_grad():
                pe = pe / pe_sum
        
        self.pe_s = nn.Parameter(pe.clone(), requires_grad=learnable)
        self.pe_e = nn.Parameter(pe.clone(), requires_grad=learnable)
        
        # 根据融合模式创建变换层
        if fusion_mode == 'concat':
            self.proj = nn.Linear(hidden_size * 3, hidden_size)
        elif fusion_mode == 'nonlinear_concat':
            self.pos_proj = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            self.proj = nn.Linear(hidden_size * 2, hidden_size)
        elif fusion_mode == 'nonlinear_add':
            self.pos_proj = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
        elif fusion_mode == 'add_nonlinear':
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
        elif fusion_mode == 'concat_nonlinear':
            self.proj = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        pos_s: torch.Tensor, 
        pos_e: torch.Tensor
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            pos_s: 开始位置索引 [batch, seq_len]
            pos_e: 结束位置索引 [batch, seq_len]
            
        Returns:
            添加位置编码后的张量
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 获取开始和结束位置编码
        pe_s = self.pe_s[pos_s.view(-1)].view(batch_size, seq_len, -1)
        pe_e = self.pe_e[pos_e.view(-1)].view(batch_size, seq_len, -1)
        
        # 根据融合模式处理
        if self.fusion_mode == 'add':
            return x + pe_s + pe_e
        elif self.fusion_mode == 'concat':
            return self.proj(torch.cat([x, pe_s, pe_e], dim=-1))
        elif self.fusion_mode == 'nonlinear_add':
            pos = self.pos_proj(torch.cat([pe_s, pe_e], dim=-1))
            return x + pos
        elif self.fusion_mode == 'nonlinear_concat':
            pos = self.pos_proj(torch.cat([pe_s, pe_e], dim=-1))
            return self.proj(torch.cat([x, pos], dim=-1))
        elif self.fusion_mode == 'add_nonlinear':
            return self.proj(x + pe_s + pe_e)
        elif self.fusion_mode == 'concat_nonlinear':
            return self.proj(torch.cat([x, pe_s, pe_e], dim=-1))
