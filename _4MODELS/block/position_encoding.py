#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位置编码模块

提供各种位置编码方式，包括：
- 正弦位置编码（Sinusoidal Position Encoding）
- 相对位置编码（Relative Position Encoding）
- 绝对位置编码（Absolute Position Encoding）
- 四位置融合编码（Four Position Fusion，用于 Lattice 结构）
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal


class SinusoidalPositionEncoding(nn.Module):
    """正弦位置编码
    
    标准的 Transformer 位置编码实现
    """
    
    def __init__(self, max_len: int, d_model: int, padding_idx: Optional[int] = None):
        """初始化
        
        Args:
            max_len: 最大序列长度
            d_model: 模型维度
            padding_idx: padding 位置索引
        """
        super().__init__()
        self.d_model = d_model
        
        pe = self._create_sinusoidal_embedding(max_len, d_model)
        if padding_idx is not None:
            pe[padding_idx, :] = 0
        
        self.register_buffer('pe', pe)
    
    @staticmethod
    def _create_sinusoidal_embedding(num_positions: int, embedding_dim: int) -> torch.Tensor:
        """创建正弦位置嵌入
        
        Args:
            num_positions: 位置数量
            embedding_dim: 嵌入维度
            
        Returns:
            位置嵌入矩阵 [num_positions, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_positions, 1)], dim=1)
        
        return emb
    
    def forward(self, positions: torch.LongTensor) -> torch.Tensor:
        """前向传播
        
        Args:
            positions: 位置索引 [batch, seq_len]
            
        Returns:
            位置编码 [batch, seq_len, d_model]
        """
        return self.pe[positions]


class RelativePositionEncoding(nn.Module):
    """相对位置编码
    
    用于 Transformer 的相对位置编码，支持 FLAT 模型
    """
    
    def __init__(
        self, 
        max_len: int, 
        d_model: int, 
        learnable: bool = False,
        symmetric: bool = True
    ):
        """初始化
        
        Args:
            max_len: 最大相对距离
            d_model: 模型维度
            learnable: 是否可学习
            symmetric: 是否对称初始化（-max_len 到 max_len）
        """
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        num_embeddings = 2 * max_len + 1
        pe = self._create_relative_embedding(num_embeddings, d_model, symmetric, max_len)
        
        self.pe = nn.Parameter(pe, requires_grad=learnable)
    
    @staticmethod
    def _create_relative_embedding(
        num_embeddings: int, 
        embedding_dim: int, 
        symmetric: bool,
        max_len: int
    ) -> torch.Tensor:
        """创建相对位置嵌入"""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        
        if symmetric:
            # 从 -max_len 到 max_len
            positions = torch.arange(-max_len, max_len + 1, dtype=torch.float)
        else:
            # 从 0 到 2*max_len
            positions = torch.arange(num_embeddings, dtype=torch.float)
        
        emb = positions.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        
        return emb
    
    def forward(self, rel_positions: torch.LongTensor) -> torch.Tensor:
        """前向传播
        
        Args:
            rel_positions: 相对位置 [batch, seq_len, seq_len]
            
        Returns:
            相对位置编码 [batch, seq_len, seq_len, d_model]
        """
        batch, seq_len, _ = rel_positions.size()
        # 将相对位置映射到 [0, 2*max_len]
        rel_pos_shifted = rel_positions + self.max_len
        rel_pos_shifted = torch.clamp(rel_pos_shifted, 0, 2 * self.max_len)
        
        pe = self.pe[rel_pos_shifted.view(-1)].view(batch, seq_len, seq_len, -1)
        return pe


class FourPositionFusion(nn.Module):
    """四位置融合模块
    
    用于 Lattice 结构的四种相对位置编码融合：
    - SS (Start-Start): 开始位置到开始位置
    - SE (Start-End): 开始位置到结束位置
    - ES (End-Start): 结束位置到开始位置
    - EE (End-End): 结束位置到结束位置
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_len: int,
        fusion_mode: Literal['concat', 'ff', 'attn', 'gate'] = 'ff',
        learnable: bool = False,
        shared: bool = True
    ):
        """初始化
        
        Args:
            hidden_size: 隐藏层维度
            max_len: 最大序列长度
            fusion_mode: 融合方式 ('concat', 'ff', 'attn', 'gate')
            learnable: 位置编码是否可学习
            shared: 四个位置编码是否共享参数
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.fusion_mode = fusion_mode
        self.shared = shared
        
        # 创建四个位置编码
        pe = RelativePositionEncoding._create_relative_embedding(
            2 * max_len + 1, hidden_size, symmetric=True, max_len=max_len
        )
        
        if shared:
            self.pe_ss = nn.Parameter(pe, requires_grad=learnable)
            self.pe_se = self.pe_ss
            self.pe_es = self.pe_ss
            self.pe_ee = self.pe_ss
        else:
            self.pe_ss = nn.Parameter(pe.clone(), requires_grad=learnable)
            self.pe_se = nn.Parameter(pe.clone(), requires_grad=learnable)
            self.pe_es = nn.Parameter(pe.clone(), requires_grad=learnable)
            self.pe_ee = nn.Parameter(pe.clone(), requires_grad=learnable)
        
        # 融合层
        if fusion_mode == 'concat' or fusion_mode == 'ff':
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size),
                nn.ReLU()
            )
        elif fusion_mode == 'attn':
            self.attn_score = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, 4),
                nn.Softmax(dim=-1)
            )
            self.w_r = nn.Linear(hidden_size, hidden_size)
        elif fusion_mode == 'gate':
            self.gate_score = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, 4 * hidden_size)
            )
            self.w_r = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, pos_s: torch.Tensor, pos_e: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            pos_s: 开始位置 [batch, seq_len]
            pos_e: 结束位置 [batch, seq_len]
            
        Returns:
            融合后的位置编码 [batch, seq_len, seq_len, hidden_size]
        """
        batch, seq_len = pos_s.size()
        
        # 计算四种相对位置
        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)  # [batch, seq_len, seq_len]
        pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)
        
        # 获取四个位置编码
        pe_ss = self.pe_ss[(pos_ss + self.max_len).view(-1)].view(batch, seq_len, seq_len, -1)
        pe_se = self.pe_se[(pos_se + self.max_len).view(-1)].view(batch, seq_len, seq_len, -1)
        pe_es = self.pe_es[(pos_es + self.max_len).view(-1)].view(batch, seq_len, seq_len, -1)
        pe_ee = self.pe_ee[(pos_ee + self.max_len).view(-1)].view(batch, seq_len, seq_len, -1)
        
        # 融合
        if self.fusion_mode in ['concat', 'ff']:
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            output = self.fusion_layer(pe_4)
        
        elif self.fusion_mode == 'attn':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            attn_weights = self.attn_score(pe_4)  # [batch, seq_len, seq_len, 4]
            pe_4_reshaped = self.w_r(pe_4.view(batch, seq_len, seq_len, 4, self.hidden_size))
            output = (attn_weights.unsqueeze(-1) * pe_4_reshaped).sum(dim=-2)
        
        elif self.fusion_mode == 'gate':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            gate_weights = self.gate_score(pe_4).view(batch, seq_len, seq_len, 4, self.hidden_size)
            gate_weights = torch.softmax(gate_weights, dim=-2)
            pe_4_reshaped = self.w_r(pe_4.view(batch, seq_len, seq_len, 4, self.hidden_size))
            output = (gate_weights * pe_4_reshaped).sum(dim=-2)
        
        return output
