#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旋转位置编码 (Rotary Position Embedding - RoPE)

用于扁平NER的边界识别增强。

参考:
- Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2024)
- 余肖生等. "GTR-NNER" (2025)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码
    
    通过旋转矩阵实现相对位置编码，使位置表示仅与相对位置差值有关。
    适用于扁平NER的边界识别增强。
    
    Args:
        hidden_dim: 隐藏层维度
        max_seq_len: 最大序列长度
        base: 旋转角度基数
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        max_seq_len: int = 512,
        base: int = 10000
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码缓存
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """预计算位置编码缓存"""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, :, :])
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """应用旋转位置编码
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
            offset: 位置偏移量
            
        Returns:
            应用旋转位置编码后的张量
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        if offset + seq_len > self.max_seq_len:
            self._build_cache(offset + seq_len + 100)
        
        cos = self.cos_cached[:, offset:offset+seq_len, :]
        sin = self.sin_cached[:, offset:offset+seq_len, :]
        
        return self._apply_rotary_emb(x, cos, sin)
    
    def _apply_rotary_emb(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:
        """应用旋转位置编码的核心操作"""
        # 将hidden_dim分成两半进行旋转
        x1, x2 = x[..., :self.hidden_dim//2], x[..., self.hidden_dim//2:]
        
        # 旋转: [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.cat([
            x1 * cos[..., :self.hidden_dim//2] - x2 * sin[..., :self.hidden_dim//2],
            x1 * sin[..., self.hidden_dim//2:] + x2 * cos[..., self.hidden_dim//2:]
        ], dim=-1)
        
        return rotated


class RoPEEnhancedEncoder(nn.Module):
    """RoPE增强的编码器
    
    将旋转位置编码集成到BERT输出上，增强边界感知能力。
    
    Args:
        hidden_dim: 隐藏层维度
        max_seq_len: 最大序列长度
        dropout: dropout比率
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.rotary_emb = RotaryPositionEmbedding(hidden_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: BERT输出 [batch, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch, seq_len]
            
        Returns:
            增强后的表示 [batch, seq_len, hidden_dim]
        """
        # 应用旋转位置编码
        x_rotated = self.rotary_emb(x)
        
        # 残差连接 + 可学习缩放
        output = x + self.scale * x_rotated
        
        return self.dropout(output)


class RelativePositionConv(nn.Module):
    """相对位置卷积模块
    
    结合旋转位置编码和1D卷积，提取相对位置特征。
    参考GTR-NNER论文中的位置特征提取方式。
    
    Args:
        hidden_dim: 隐藏层维度
        kernel_size: 卷积核大小
        dropout: dropout比率
    """
    
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.rotary_emb = RotaryPositionEmbedding(hidden_dim)
        
        self.conv1d = nn.Conv1d(
            hidden_dim, 
            hidden_dim, 
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征 [batch, seq_len, hidden_dim]
            
        Returns:
            相对位置特征 [batch, seq_len, hidden_dim]
        """
        # 旋转位置编码
        x_rotated = self.rotary_emb(x)
        
        # 1D卷积提取局部位置特征
        x_conv = x_rotated.transpose(1, 2)  # [batch, hidden, seq]
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [batch, seq, hidden]
        
        # 激活 + LayerNorm
        x_conv = self.activation(x_conv)
        x_conv = self.layer_norm(x_conv)
        
        # 残差连接
        output = x + self.dropout(x_conv)
        
        return output


if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    seq_len = 128
    hidden_dim = 768
    
    print("测试旋转位置编码模块...")
    
    # 测试RoPE
    rope = RotaryPositionEmbedding(hidden_dim)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    x_rotated = rope(x)
    print(f"RoPE: {x.shape} -> {x_rotated.shape}")
    
    # 测试RoPE增强编码器
    encoder = RoPEEnhancedEncoder(hidden_dim)
    output = encoder(x)
    print(f"RoPE Encoder: {x.shape} -> {output.shape}")
    
    # 测试相对位置卷积
    conv_module = RelativePositionConv(hidden_dim)
    output = conv_module(x)
    print(f"RelativePositionConv: {x.shape} -> {output.shape}")
    
    print(f"\n参数量统计:")
    print(f"  RoPE: {sum(p.numel() for p in rope.parameters()):,}")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Conv: {sum(p.numel() for p in conv_module.parameters()):,}")
