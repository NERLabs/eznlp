#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lattice 注意力机制模块

实现 FLAT (Flat-Lattice-Transformer) 的核心注意力机制，支持：
- 相对位置编码的多头注意力
- Lattice 结构的四位置融合注意力
- 标准多头注意力的改进版本

参考：Li et al. "FLAT: Chinese NER Using Flat-Lattice Transformer" (ACL 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttentionWithRelativePosition(nn.Module):
    """带相对位置编码的多头注意力
    
    Transformer-XL 风格的相对位置注意力，用于 FLAT 模型
    核心思想：在注意力计算中引入相对位置信息
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        scaled: bool = True,
        use_projection: bool = True
    ):
        """初始化
        
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout 比率
            scaled: 是否使用缩放点积注意力
            use_projection: 是否对 Q/K/V/R 使用线性投影
        """
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = hidden_size // num_heads
        self.scaled = scaled
        self.use_projection = use_projection
        
        # Q/K/V/R 投影层
        if use_projection:
            self.w_q = nn.Linear(hidden_size, hidden_size)
            self.w_k = nn.Linear(hidden_size, hidden_size)
            self.w_v = nn.Linear(hidden_size, hidden_size)
            self.w_r = nn.Linear(hidden_size, hidden_size)  # 相对位置投影
        
        # 最终输出投影
        self.w_out = nn.Linear(hidden_size, hidden_size)
        
        # Transformer-XL 风格的可学习参数
        self.u = nn.Parameter(torch.Tensor(num_heads, self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(num_heads, self.per_head_size))
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rel_pos_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        计算公式（Transformer-XL）：
        Attention = softmax((Q + u)K^T + (Q + v)R^T) V
        
        其中 R 是相对位置编码，u 和 v 是可学习参数
        
        Args:
            query: Query 张量 [batch, seq_len, hidden_size]
            key: Key 张量 [batch, seq_len, hidden_size]
            value: Value 张量 [batch, seq_len, hidden_size]
            rel_pos_embedding: 相对位置编码 [batch, seq_len, seq_len, hidden_size]
            mask: 注意力掩码 [batch, seq_len] (True 表示有效位置)
            
        Returns:
            注意力输出 [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = query.size()
        
        # 线性投影
        if self.use_projection:
            query = self.w_q(query)
            key = self.w_k(key)
            value = self.w_v(value)
            rel_pos_embedding = self.w_r(rel_pos_embedding)
        
        # 重塑为多头格式
        # [batch, seq_len, num_heads, per_head_size]
        query = query.view(batch, seq_len, self.num_heads, self.per_head_size)
        key = key.view(batch, seq_len, self.num_heads, self.per_head_size)
        value = value.view(batch, seq_len, self.num_heads, self.per_head_size)
        rel_pos_embedding = rel_pos_embedding.view(
            batch, seq_len, seq_len, self.num_heads, self.per_head_size
        )
        
        # 转置为 [batch, num_heads, seq_len, per_head_size]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # 计算注意力分数（分为四个部分）
        # A: 内容-内容注意力 (Q + u) @ K^T
        key_t = key.transpose(-1, -2)  # [batch, num_heads, per_head_size, seq_len]
        A = torch.matmul(query, key_t)  # [batch, num_heads, seq_len, seq_len]
        
        # B: 内容-位置注意力 Q @ R^T
        # rel_pos_embedding: [batch, num_heads, seq_len, per_head_size, seq_len]
        rel_pos_embedding_t = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        query_expanded = query.unsqueeze(-2)  # [batch, num_heads, seq_len, 1, per_head_size]
        B = torch.matmul(query_expanded, rel_pos_embedding_t).squeeze(-2)
        
        # C: 全局内容偏置 u @ K^T
        u_expanded = self.u.unsqueeze(0).unsqueeze(-2)  # [1, num_heads, 1, per_head_size]
        C = torch.matmul(u_expanded, key_t)  # [1, num_heads, 1, seq_len]
        
        # D: 全局位置偏置 v @ R^T
        rel_pos_for_d = rel_pos_embedding.unsqueeze(-2)
        # [batch, seq_len, seq_len, num_heads, 1, per_head_size]
        v_expanded = self.v.unsqueeze(-1)  # [num_heads, per_head_size, 1]
        D = torch.matmul(rel_pos_for_d, v_expanded).squeeze(-1).squeeze(-1)
        # [batch, seq_len, seq_len, num_heads] -> [batch, num_heads, seq_len, seq_len]
        D = D.permute(0, 3, 1, 2)
        
        # 合并注意力分数
        attn_score = A + B + C + D
        
        # 缩放
        if self.scaled:
            attn_score = attn_score / math.sqrt(self.per_head_size)
        
        # 应用掩码
        if mask is not None:
            # mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn_score = attn_score.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, value)
        # [batch, num_heads, seq_len, per_head_size]
        
        # 转回原始维度
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, self.hidden_size)
        
        # 输出投影
        output = self.w_out(output)
        
        return output


class LatticeSelfAttention(nn.Module):
    """Lattice 自注意力机制
    
    FLAT 的核心组件：结合四位置融合的自注意力
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_len: int,
        dropout: float = 0.1,
        four_pos_fusion_mode: str = 'ff'
    ):
        """初始化
        
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            max_len: 最大序列长度
            dropout: Dropout 比率
            four_pos_fusion_mode: 四位置融合模式 ('ff', 'attn', 'gate')
        """
        super().__init__()
        
        from .position_encoding import FourPositionFusion
        
        # 四位置融合模块
        self.four_pos_fusion = FourPositionFusion(
            hidden_size=hidden_size,
            max_len=max_len,
            fusion_mode=four_pos_fusion_mode,
            learnable=True,
            shared=True
        )
        
        # 带相对位置的多头注意力
        self.attention = MultiHeadAttentionWithRelativePosition(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            scaled=True,
            use_projection=True
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_s: torch.Tensor,
        pos_e: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch, seq_len, hidden_size]
            pos_s: 开始位置 [batch, seq_len]
            pos_e: 结束位置 [batch, seq_len]
            mask: 序列掩码 [batch, seq_len]
            
        Returns:
            注意力输出 [batch, seq_len, hidden_size]
        """
        # 计算相对位置编码（四位置融合）
        rel_pos_embedding = self.four_pos_fusion(pos_s, pos_e)
        
        # 应用注意力
        output = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            rel_pos_embedding=rel_pos_embedding,
            mask=mask
        )
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层（FLAT 风格）
    
    结合 Lattice 注意力和前馈网络
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_len: int,
        ff_size: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-12
    ):
        """初始化
        
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            max_len: 最大序列长度
            ff_size: 前馈网络隐藏层维度（默认为 hidden_size * 4）
            dropout: Dropout 比率
            activation: 激活函数 ('relu', 'gelu')
            layer_norm_eps: LayerNorm epsilon
        """
        super().__init__()
        
        if ff_size is None:
            ff_size = hidden_size * 4
        
        # Lattice 自注意力
        self.self_attn = LatticeSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout
        )
        
        # 前馈网络
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_s: torch.Tensor,
        pos_e: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            hidden_states: 输入隐藏状态 [batch, seq_len, hidden_size]
            pos_s: 开始位置 [batch, seq_len]
            pos_e: 结束位置 [batch, seq_len]
            mask: 序列掩码 [batch, seq_len]
            
        Returns:
            输出隐藏状态 [batch, seq_len, hidden_size]
        """
        # 自注意力 + 残差连接
        attn_output = self.self_attn(hidden_states, pos_s, pos_e, mask)
        hidden_states = self.norm1(hidden_states + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states + ff_output)
        
        return hidden_states
