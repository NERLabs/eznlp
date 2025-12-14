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
    """四位置融合编码
    
    FLAT 的核心创新，融合四种相对位置编码：
    - SS: start-to-start
    - SE: start-to-end
    - ES: end-to-start
    - EE: end-to-end
    
    优化版本支持：
    - use_unique: 使用 torch.unique() 去重，减少显存占用
    - use_scalar: 使用标量替代向量，大幅减少显存
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_len: int,
        fusion_mode: Literal['concat', 'ff', 'attn', 'gate', 'scalar'] = 'ff',
        learnable: bool = False,
        shared: bool = True,
        use_unique: bool = True,
        use_scalar: bool = False,
        scalar_dim: int = 4
    ):
        """初始化
        
        Args:
            hidden_size: 隐藏层维度
            max_len: 最大序列长度
            fusion_mode: 融合方式 ('ff', 'attn', 'gate', 'concat', 'scalar')
            learnable: 是否可学习
            shared: 四个位置编码是否共享参数
            use_unique: 是否使用 unique 优化来减少显存
            use_scalar: 是否使用标量位置编码（类似 T5）
            scalar_dim: 标量位置编码维度
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.fusion_mode = fusion_mode
        self.shared = shared
        self.use_unique = use_unique
        self.use_scalar = use_scalar
        self.scalar_dim = scalar_dim
        
        if use_scalar:
            # 使用标量位置编码（类似 T5，大幅减少显存）
            self.pe_ss = nn.Parameter(torch.zeros(2 * max_len + 1, scalar_dim), requires_grad=True)
            self.pe_se = nn.Parameter(torch.zeros(2 * max_len + 1, scalar_dim), requires_grad=True)
            self.pe_es = nn.Parameter(torch.zeros(2 * max_len + 1, scalar_dim), requires_grad=True)
            self.pe_ee = nn.Parameter(torch.zeros(2 * max_len + 1, scalar_dim), requires_grad=True)
            nn.init.xavier_uniform_(self.pe_ss)
            nn.init.xavier_uniform_(self.pe_se)
            nn.init.xavier_uniform_(self.pe_es)
            nn.init.xavier_uniform_(self.pe_ee)
            self.scalar_proj = nn.Linear(scalar_dim * 4, hidden_size)
        else:
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
        """前向传播"""
        batch, seq_len = pos_s.size()
        
        # 计算四种相对位置
        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)
        
        if self.use_scalar:
            # 标量位置编码：直接处理，显存占用极小
            pe_ss = self.pe_ss[(pos_ss + self.max_len).clamp(0, 2*self.max_len)]
            pe_se = self.pe_se[(pos_se + self.max_len).clamp(0, 2*self.max_len)]
            pe_es = self.pe_es[(pos_es + self.max_len).clamp(0, 2*self.max_len)]
            pe_ee = self.pe_ee[(pos_ee + self.max_len).clamp(0, 2*self.max_len)]
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            return self.scalar_proj(pe_4)
        
        if self.use_unique:
            return self._forward_unique(pos_ss, pos_se, pos_es, pos_ee, batch, seq_len)
        else:
            return self._forward_standard(pos_ss, pos_se, pos_es, pos_ee, batch, seq_len)
    
    def _forward_unique(self, pos_ss, pos_se, pos_es, pos_ee, batch, seq_len):
        """使用 unique 优化的前向传播"""
        pos_4 = torch.stack([pos_ss, pos_se, pos_es, pos_ee], dim=-1)
        pos_4_flat = pos_4.view(-1, 4)
        
        unique_pos, inverse_indices = torch.unique(pos_4_flat, dim=0, return_inverse=True)
        
        unique_pe_ss = self.pe_ss[(unique_pos[:, 0] + self.max_len).clamp(0, 2*self.max_len)]
        unique_pe_se = self.pe_se[(unique_pos[:, 1] + self.max_len).clamp(0, 2*self.max_len)]
        unique_pe_es = self.pe_es[(unique_pos[:, 2] + self.max_len).clamp(0, 2*self.max_len)]
        unique_pe_ee = self.pe_ee[(unique_pos[:, 3] + self.max_len).clamp(0, 2*self.max_len)]
        
        unique_pe_4 = torch.cat([unique_pe_ss, unique_pe_se, unique_pe_es, unique_pe_ee], dim=-1)
        
        if hasattr(self, 'fusion_layer'):
            unique_output = self.fusion_layer(unique_pe_4)
        else:
            unique_output = unique_pe_4[:, :self.hidden_size]
        
        output = unique_output[inverse_indices].view(batch, seq_len, seq_len, -1)
        return output
    
    def _forward_standard(self, pos_ss, pos_se, pos_es, pos_ee, batch, seq_len):
        """标准前向传播（无优化）"""
        pe_ss = self.pe_ss[(pos_ss + self.max_len).view(-1)].view(batch, seq_len, seq_len, -1)
        pe_se = self.pe_se[(pos_se + self.max_len).view(-1)].view(batch, seq_len, seq_len, -1)
        pe_es = self.pe_es[(pos_es + self.max_len).view(-1)].view(batch, seq_len, seq_len, -1)
        pe_ee = self.pe_ee[(pos_ee + self.max_len).view(-1)].view(batch, seq_len, seq_len, -1)
        
        if self.fusion_mode in ['concat', 'ff']:
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            output = self.fusion_layer(pe_4)
        elif self.fusion_mode == 'attn':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            attn_weights = self.attn_score(pe_4)
            pe_4_reshaped = self.w_r(pe_4.view(batch, seq_len, seq_len, 4, self.hidden_size))
            output = (attn_weights.unsqueeze(-1) * pe_4_reshaped).sum(dim=-2)
        elif self.fusion_mode == 'gate':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            gate_weights = self.gate_score(pe_4).view(batch, seq_len, seq_len, 4, self.hidden_size)
            gate_weights = torch.softmax(gate_weights, dim=-2)
            pe_4_reshaped = self.w_r(pe_4.view(batch, seq_len, seq_len, 4, self.hidden_size))
            output = (gate_weights * pe_4_reshaped).sum(dim=-2)
        
        return output
