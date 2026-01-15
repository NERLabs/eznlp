#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMES通道间注意力模块

为ExpertDict的四通道(B/M/E/S)引入通道间注意力机制,
灵感来自FLAT的四位置融合注意力(SS/SE/ES/EE)

核心思想:
- B(Begin)通道关注E(End)通道 → 建模实体完整性  
- M(Middle)通道关注B/E通道 → 强化边界一致性
- S(Single)通道独立处理 → 单字实体特殊处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class BMESChannelAttention(nn.Module):
    """BMES通道间注意力机制
    
    对ExpertDict的四通道(B/M/E/S)应用通道间注意力:
    - 通道维度: 4 (B, M, E, S)
    - 每个通道嵌入维度: emb_dim (默认50)
    - 输出: 经过通道交互增强后的嵌入
    
    参数量: O(emb_dim^2 * num_heads) ≈ 50^2 * 4 = 10K (轻量级)
    """
    
    def __init__(
        self,
        emb_dim: int = 50,
        num_channels: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_channel_pos: bool = True
    ):
        """初始化
        
        Args:
            emb_dim: 每个通道的嵌入维度
            num_channels: 通道数量 (B/M/E/S = 4)
            num_heads: 注意力头数（需满足 emb_dim % num_heads == 0；
                       理想情况下可与通道数相等，但不是强制）
            dropout: Dropout比率
            use_channel_pos: 是否使用通道位置编码
        """
        super().__init__()
        
        assert num_channels == 4, "BMES通道固定为4"
        assert emb_dim % num_heads == 0, "emb_dim必须能被num_heads整除"
        
        self.emb_dim = emb_dim
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.per_head_dim = emb_dim // num_heads
        self.use_channel_pos = use_channel_pos
        
        # Q/K/V投影层 (通道间共享)
        self.w_q = nn.Linear(emb_dim, emb_dim)
        self.w_k = nn.Linear(emb_dim, emb_dim)
        self.w_v = nn.Linear(emb_dim, emb_dim)
        self.w_out = nn.Linear(emb_dim, emb_dim)
        
        # 通道位置编码 (可学习)
        if use_channel_pos:
            self.channel_pos_emb = nn.Parameter(
                torch.zeros(num_channels, emb_dim)
            )
            nn.init.xavier_uniform_(self.channel_pos_emb)
        
        # 通道关系偏置矩阵 (4x4, 建模B-E, M-B, M-E等关系)
        self.channel_bias = nn.Parameter(
            torch.zeros(num_heads, num_channels, num_channels)
        )
        self._init_channel_bias()
        
        # 残差缩放因子：控制BMES注意力对输出的影响强度
        # 初始化为0.5，既不过强也不过弱
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
    
    def _init_channel_bias(self):
        """初始化静态通道偏置矩阵
        
        根据BMES语义设置先验偏置:
        - B→E: 强关联 (实体起止)
        - M→B, M→E: 中等关联 (边界感知)
        - S→S: 自关联 (单字实体)
        """
        with torch.no_grad():
            self.channel_bias.zero_()
            for h in range(self.num_heads):
                # B(0) → E(2): 强关联
                self.channel_bias[h, 0, 2] = 1.0
                self.channel_bias[h, 2, 0] = 1.0
                
                # M(1) → B(0), M(1) → E(2): 中等关联
                self.channel_bias[h, 1, 0] = 0.5
                self.channel_bias[h, 1, 2] = 0.5
                self.channel_bias[h, 0, 1] = 0.5
                self.channel_bias[h, 2, 1] = 0.5
                
                # S(3) → S(3): 自关联
                self.channel_bias[h, 3, 3] = 1.0
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch*seq_len, num_channels, emb_dim]
               从NestedOneHotEmbedder重塑得到
            mask: 通道掩码 [batch*seq_len, num_channels]
                  True表示有效通道 (有词典匹配)
        
        Returns:
            output: 注意力增强后的嵌入 [batch*seq_len, num_channels, emb_dim]
        """
        batch_seq_len, num_channels, emb_dim = x.size()
        assert num_channels == self.num_channels
        
        # 添加通道位置编码
        if self.use_channel_pos:
            x = x + self.channel_pos_emb.unsqueeze(0)
        
        # 线性投影: [batch*seq_len, num_channels, emb_dim]
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)
        
        # 重塑为多头: [batch*seq_len, num_channels, num_heads, per_head_dim]
        query = query.view(batch_seq_len, num_channels, self.num_heads, self.per_head_dim)
        key = key.view(batch_seq_len, num_channels, self.num_heads, self.per_head_dim)
        value = value.view(batch_seq_len, num_channels, self.num_heads, self.per_head_dim)
        
        # 转置: [batch*seq_len, num_heads, num_channels, per_head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # 计算注意力分数: Q @ K^T
        # [batch*seq_len, num_heads, num_channels, num_channels]
        attn_score = torch.matmul(query, key.transpose(-1, -2))
        attn_score = attn_score / math.sqrt(self.per_head_dim)
        
        # 添加静态通道关系偏置 (结构先验)
        attn_score = attn_score + self.channel_bias.unsqueeze(0)
        
        # （如果你有动态偏置，这里继续加；否则保持现状）
        # dyn_bias = self._build_dynamic_channel_bias()
        # attn_score = attn_score + dyn_bias.unsqueeze(0).unsqueeze(0)
        
        # 应用掩码 (如果某通道无词典匹配，将其注意力分数置为极小值)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            attn_score = attn_score.masked_fill(~mask_expanded, -1e9)
        
        # Softmax 前做数值裁剪，提升稳定性
        attn_score = torch.clamp(attn_score, min=-10, max=10)
        attn_weights = F.softmax(attn_score, dim=-1)
        
        # 如仍出现 NaN（极端情况下），退回为均匀分布
        if torch.isnan(attn_weights).any():
            uniform_weights = torch.ones_like(attn_weights) / self.num_channels
            nan_mask = torch.isnan(attn_weights)
            attn_weights = torch.where(nan_mask, uniform_weights, attn_weights)
        
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和: [batch*seq_len, num_heads, num_channels, per_head_dim]
        output = torch.matmul(attn_weights, value)
        
        # 转回: [batch*seq_len, num_channels, num_heads, per_head_dim]
        output = output.transpose(1, 2).contiguous()
        
        # 合并多头: [batch*seq_len, num_channels, emb_dim]
        output = output.view(batch_seq_len, num_channels, emb_dim)
        
        # 输出投影
        output = self.w_out(output)
        output = self.dropout(output)
        
        # 残差连接 + 可学习缩放因子，再做LayerNorm
        # output = x + alpha * output
        output = self.layer_norm(x + self.alpha * output)
        
        return output


class BMESChannelAttentionV2(nn.Module):
    """BMES通道注意力 v2 - 简化版
    
    使用更轻量的实现:
    - 单头注意力 (减少参数)
    - 无位置编码 (通道顺序固定)
    - 直接应用通道关系矩阵
    
    适合快速验证实验
    """
    
    def __init__(
        self,
        emb_dim: int = 50,
        num_channels: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.num_channels = num_channels
        
        # 简化为单头注意力
        self.w_q = nn.Linear(emb_dim, emb_dim)
        self.w_k = nn.Linear(emb_dim, emb_dim)
        self.w_v = nn.Linear(emb_dim, emb_dim)
        
        # 通道关系矩阵 (4x4) - 使用Xavier初始化代替单位矩阵
        self.channel_relation = nn.Parameter(
            torch.zeros(num_channels, num_channels)
        )
        nn.init.xavier_uniform_(self.channel_relation)
        
        self.dropout = nn.Dropout(dropout)
        # 增加eps提高数值稳定性
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-6)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """前向传播
        
        Args:
            x: [batch*seq_len, num_channels, emb_dim]
            mask: [batch*seq_len, num_channels]
        
        Returns:
            [batch*seq_len, num_channels, emb_dim]
        """
        batch_seq_len, num_channels, emb_dim = x.size()
        
        # Q/K/V投影
        query = self.w_q(x)  # [batch*seq_len, 4, emb_dim]
        key = self.w_k(x)
        value = self.w_v(x)
        
        # 注意力分数: [batch*seq_len, 4, 4]
        # 使用缩放因子防止数值过大
        scale = math.sqrt(emb_dim)
        attn_score = torch.matmul(query, key.transpose(-1, -2)) / scale
        
        # 加入可学习的通道关系矩阵（限制幅度防止梯度爆炸）
        attn_score = attn_score + torch.tanh(self.channel_relation).unsqueeze(0)
        
        # 掩码
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            # 使用-1e9而非-inf，更加数值稳定
            attn_score = attn_score.masked_fill(~mask_expanded, -1e9)
        
        # Softmax（加入数值裁剪）
        attn_score = torch.clamp(attn_score, min=-10, max=10)
        attn_weights = F.softmax(attn_score, dim=-1)
        
        # 检查NaN
        if torch.isnan(attn_weights).any():
            # 如果出现NaN，使用均匀分布
            attn_weights = torch.ones_like(attn_weights) / num_channels
        
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, value)
        output = self.dropout(output)
        
        # 残差 + LayerNorm
        output = self.layer_norm(x + output)
        
        return output
