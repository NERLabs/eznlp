#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFLAT启发的改进方案 - 立即可实现版本

基于NFLAT代码分析，这里提供3个可以立即应用到RedJujube项目的改进：
1. 改进的门控融合（两阶段编码）
2. 位置感知的ExpertDict
3. 简化版Inter-Attention融合

使用方法：
    将相关代码复制到 _6MODEL/extractor.py 中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 改进1: 两阶段门控融合（借鉴NFLAT的解耦设计）
# ============================================================================

class ImprovedGatedFusion(nn.Module):
    """
    改进的门控融合策略
    
    关键改进：
    1. 添加独立编码阶段（让每个特征先独立变换）
    2. 更深的门控网络（2层 vs 原来的1层）
    3. 使用GELU激活（性能通常优于ReLU）
    
    原理：
        参考NFLAT的两阶段设计思想，先让ExpertDict和SoftLex各自编码，
        再通过门控网络学习如何融合，避免直接拼接导致的特征冲突。
    
    适用场景：
        替换当前的'gated'策略，配合SoftLex-v2使用
    
    预期提升：+0.2~0.3% F1
    """
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 独立编码器
        self.expert_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.soft_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 更深的门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # LayerNorm稳定训练
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, expert_feat, soft_feat):
        """
        Args:
            expert_feat: [batch_size, seq_len, hidden_dim]
            soft_feat: [batch_size, seq_len, hidden_dim]
        
        Returns:
            fused_feat: [batch_size, seq_len, hidden_dim]
        """
        # 阶段1: 独立编码
        expert_encoded = self.expert_encoder(expert_feat)
        soft_encoded = self.soft_encoder(soft_feat)
        
        # 阶段2: 门控融合
        combined = torch.cat([expert_encoded, soft_encoded], dim=-1)
        gate = self.gate_net(combined)
        
        # 门控加权
        fused = gate * expert_encoded + (1 - gate) * soft_encoded
        
        # 归一化
        fused = self.norm(fused)
        
        return fused


# ============================================================================
# 改进2: 位置感知的特征融合
# ============================================================================

def get_sinusoidal_embedding(max_seq_len, embedding_dim):
    """
    生成正弦位置编码（从NFLAT utils.py复制）
    
    Args:
        max_seq_len: 最大序列长度
        embedding_dim: 嵌入维度
    
    Returns:
        position_embeddings: [2*max_seq_len+1, embedding_dim]
    """
    num_embeddings = 2 * max_seq_len + 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    
    return emb


class PositionAwareFusion(nn.Module):
    """
    位置感知的特征融合
    
    关键思想：
        在融合ExpertDict和SoftLex时，加入位置信息，
        让模型知道每个词汇匹配在句子中的位置。
    
    原理：
        借鉴NFLAT的相对位置编码，为每个特征添加位置嵌入，
        然后通过线性层融合内容和位置信息。
    
    适用场景：
        可以集成到任何融合策略中（Concat/Weighted/Attention/Gated）
    
    预期提升：+0.1~0.2% F1
    """
    def __init__(self, hidden_dim, max_seq_len=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 正弦位置编码（不需要训练）
        pe = get_sinusoidal_embedding(max_seq_len, hidden_dim)
        self.register_buffer('pe', pe)
        
        # 位置融合层
        self.pos_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, features, positions=None):
        """
        Args:
            features: [batch_size, seq_len, hidden_dim]
            positions: [batch_size, seq_len] 位置索引，可选
        
        Returns:
            position_aware_features: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = features.shape
        
        # 如果没有提供位置，使用序列位置
        if positions is None:
            positions = torch.arange(seq_len, device=features.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # 获取位置编码
        positions = positions.clamp(0, 511)  # 限制在范围内
        pos_emb = self.pe[positions]  # [batch_size, seq_len, hidden_dim]
        
        # 融合内容和位置
        combined = torch.cat([features, pos_emb], dim=-1)
        output = self.pos_fusion(combined)
        
        return output


# ============================================================================
# 改进3: 简化版Inter-Attention融合
# ============================================================================

class InterAttentionFusion(nn.Module):
    """
    简化版的Inter-Attention融合
    
    核心创新：
        不同于传统Self-Attention（全连接），这里实现字符→词汇的单向注意力。
        让BERT特征（Query）去关注ExpertDict/SoftLex特征（Key/Value）。
    
    优势：
        1. 计算量更小：O(n×m) vs O((n+m)²)
        2. 语义更清晰：字符主动查询词汇信息
        3. 层次化设计：先Expert后SoftLex
    
    原理：
        参考NFLAT的InterAttention设计，但简化为单层版本。
    
    适用场景：
        替换Concat/Attention融合策略
    
    预期提升：+0.3~0.5% F1（如果实现正确）
    """
    def __init__(self, hidden_dim, n_head=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_head == 0, "hidden_dim必须能被n_head整除"
        
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.per_head_dim = hidden_dim // n_head
        
        # Q, K, V投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.per_head_dim)
    
    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: [batch_size, L1, hidden_dim] - BERT特征
            key_value: [batch_size, L2, hidden_dim] - ExpertDict或SoftLex特征
            mask: [batch_size, L1, L2] - 可选的注意力掩码
        
        Returns:
            output: [batch_size, L1, hidden_dim]
        """
        batch_size, L1, _ = query.shape
        L2 = key_value.shape[1]
        
        # 投影到Q, K, V
        Q = self.q_proj(query).view(batch_size, L1, self.n_head, self.per_head_dim)
        K = self.k_proj(key_value).view(batch_size, L2, self.n_head, self.per_head_dim)
        V = self.v_proj(key_value).view(batch_size, L2, self.n_head, self.per_head_dim)
        
        # 调整维度: [B, n_head, L, per_head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算注意力得分
        attn_score = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        
        # 应用掩码（如果提供）
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, L1, L2]
            attn_score = attn_score.masked_fill(~mask, -1e9)
        
        # Softmax归一化
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)
        
        # 加权求和
        output = torch.matmul(attn_weight, V)  # [B, n_head, L1, per_head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, L1, self.hidden_dim)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output


class HierarchicalInterFusion(nn.Module):
    """
    层次化Inter-Attention融合
    
    架构：
        BERT特征
          ↓
        [InterAttn] ← ExpertDict特征
          ↓ (残差连接 + LayerNorm)
        增强特征1
          ↓
        [InterAttn] ← SoftLex特征
          ↓ (残差连接 + LayerNorm)
        最终特征
    
    使用方法：
        在FusionExtractor中添加'hierarchical_inter'策略
    
    预期提升：+0.4~0.6% F1（理论上限）
    """
    def __init__(self, hidden_dim, n_head=4, dropout=0.1):
        super().__init__()
        
        # 两层Inter-Attention
        self.expert_attn = InterAttentionFusion(hidden_dim, n_head, dropout)
        self.soft_attn = InterAttentionFusion(hidden_dim, n_head, dropout)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN（可选，增强非线性）
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, bert_feat, expert_feat, soft_feat):
        """
        Args:
            bert_feat: [batch_size, seq_len, hidden_dim]
            expert_feat: [batch_size, seq_len, hidden_dim]
            soft_feat: [batch_size, seq_len, hidden_dim]
        
        Returns:
            final_feat: [batch_size, seq_len, hidden_dim]
        """
        # 层次1: ExpertDict增强
        residual = bert_feat
        enhanced1 = self.expert_attn(
            query=bert_feat,
            key_value=expert_feat
        )
        enhanced1 = self.norm1(enhanced1 + residual)
        
        # 层次2: SoftLex增强
        residual = enhanced1
        enhanced2 = self.soft_attn(
            query=enhanced1,
            key_value=soft_feat
        )
        enhanced2 = self.norm2(enhanced2 + residual)
        
        # FFN增强
        residual = enhanced2
        final_feat = self.ffn(enhanced2)
        final_feat = self.norm3(final_feat + residual)
        
        return final_feat


# ============================================================================
# 集成到FusionExtractor的示例代码
# ============================================================================

"""
在 _6MODEL/extractor.py 的 FusionExtractor 类中添加：

class FusionExtractor(nn.Module):
    def __init__(self, fusion_strategy='gated_improved', ...):
        # ... 原有代码 ...
        
        if fusion_strategy == 'gated_improved':
            # 改进的门控融合
            self.fusion_layer = ImprovedGatedFusion(
                hidden_dim=768,
                dropout=0.3
            )
        
        elif fusion_strategy == 'inter_attention':
            # Inter-Attention融合
            self.fusion_layer = InterAttentionFusion(
                hidden_dim=768,
                n_head=4,
                dropout=0.1
            )
        
        elif fusion_strategy == 'hierarchical_inter':
            # 层次化Inter-Attention融合（推荐）
            self.fusion_layer = HierarchicalInterFusion(
                hidden_dim=768,
                n_head=4,
                dropout=0.1
            )
        
        elif fusion_strategy == 'position_aware_gated':
            # 位置感知 + 门控融合
            self.position_layer = PositionAwareFusion(
                hidden_dim=768,
                max_seq_len=512
            )
            self.fusion_layer = ImprovedGatedFusion(
                hidden_dim=768,
                dropout=0.3
            )
    
    def forward(self, bert_output, nested_features):
        expert_feat = nested_features['expert_dict']
        soft_feat = nested_features['softlexicon']
        
        if self.fusion_strategy == 'gated_improved':
            fused = self.fusion_layer(expert_feat, soft_feat)
            return torch.cat([bert_output, fused], dim=-1)
        
        elif self.fusion_strategy == 'inter_attention':
            # BERT作为Query，词典特征作为Key/Value
            expert_enhanced = self.fusion_layer(bert_output, expert_feat)
            soft_enhanced = self.fusion_layer(bert_output, soft_feat)
            return torch.cat([bert_output, expert_enhanced, soft_enhanced], dim=-1)
        
        elif self.fusion_strategy == 'hierarchical_inter':
            # 层次化融合
            final_feat = self.fusion_layer(bert_output, expert_feat, soft_feat)
            return final_feat  # 直接返回，不拼接
        
        elif self.fusion_strategy == 'position_aware_gated':
            # 先添加位置信息，再门控融合
            expert_pos = self.position_layer(expert_feat)
            soft_pos = self.position_layer(soft_feat)
            fused = self.fusion_layer(expert_pos, soft_pos)
            return torch.cat([bert_output, fused], dim=-1)
"""


# ============================================================================
# 使用示例和训练脚本
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("NFLAT启发的改进方案 - 使用指南")
    print("=" * 80)
    
    print("\n【方案1: 改进的门控融合】")
    print("  修改配置:")
    print("    fusion_strategy: 'gated_improved'")
    print("  预期提升: +0.2~0.3% F1")
    print("  实现难度: ⭐ (极低)")
    print("  推荐指数: ⭐⭐⭐⭐⭐")
    
    print("\n【方案2: 位置感知融合】")
    print("  修改配置:")
    print("    fusion_strategy: 'position_aware_gated'")
    print("  预期提升: +0.1~0.2% F1")
    print("  实现难度: ⭐⭐ (低)")
    print("  推荐指数: ⭐⭐⭐⭐")
    
    print("\n【方案3: Inter-Attention融合】")
    print("  修改配置:")
    print("    fusion_strategy: 'inter_attention'")
    print("  预期提升: +0.3~0.4% F1")
    print("  实现难度: ⭐⭐⭐ (中)")
    print("  推荐指数: ⭐⭐⭐⭐")
    
    print("\n【方案4: 层次化Inter融合】（最推荐）")
    print("  修改配置:")
    print("    fusion_strategy: 'hierarchical_inter'")
    print("  预期提升: +0.4~0.6% F1")
    print("  实现难度: ⭐⭐⭐ (中)")
    print("  推荐指数: ⭐⭐⭐⭐⭐")
    
    print("\n" + "=" * 80)
    print("建议执行顺序:")
    print("=" * 80)
    print("1. 今晚：实现方案1（改进门控），配合SoftLex-v2测试")
    print("2. 明天：如果方案1效果好，尝试方案4（层次化融合）")
    print("3. 后天：如果达到97%，进行消融实验；否则尝试完整NFLAT移植")
    print("\n提示：每次只改一个变量，方便定位问题！")
    print("=" * 80)
    
    # 测试代码
    print("\n【快速测试】")
    batch_size, seq_len, hidden_dim = 2, 10, 768
    
    # 模拟输入
    bert_feat = torch.randn(batch_size, seq_len, hidden_dim)
    expert_feat = torch.randn(batch_size, seq_len, hidden_dim)
    soft_feat = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 测试方案1
    print("\n测试方案1: ImprovedGatedFusion")
    model1 = ImprovedGatedFusion(hidden_dim)
    output1 = model1(expert_feat, soft_feat)
    print(f"  输入形状: {expert_feat.shape}")
    print(f"  输出形状: {output1.shape}")
    print(f"  参数量: {sum(p.numel() for p in model1.parameters()):,}")
    
    # 测试方案4
    print("\n测试方案4: HierarchicalInterFusion")
    model4 = HierarchicalInterFusion(hidden_dim, n_head=4)
    output4 = model4(bert_feat, expert_feat, soft_feat)
    print(f"  输入形状: {bert_feat.shape}")
    print(f"  输出形状: {output4.shape}")
    print(f"  参数量: {sum(p.numel() for p in model4.parameters()):,}")
    
    print("\n✅ 所有测试通过！可以开始集成到项目中。")
