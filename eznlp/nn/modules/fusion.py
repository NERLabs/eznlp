# -*- coding: utf-8 -*-
"""
特征融合模块

实现多种特征融合策略,用于联合SoftLexicon和ExpertDict特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFeatureFusion(nn.Module):
    """加权求和融合层
    
    通过可学习的权重对多个特征进行加权求和融合
    
    Args:
        num_features: 特征数量
        feature_dims: 各特征维度列表,如[768, 200, 50]
        aligned_dim: 对齐后的目标维度,如768
    """
    
    def __init__(self, num_features, feature_dims, aligned_dim):
        super().__init__()
        self.num_features = num_features
        self.feature_dims = feature_dims
        self.aligned_dim = aligned_dim
        
        # 为每个特征创建线性对齐层
        self.align_layers = nn.ModuleList([
            nn.Linear(dim, aligned_dim) if dim != aligned_dim else nn.Identity()
            for dim in feature_dims
        ])
        
        # 可学习的融合权重(未归一化)
        self.fusion_weights = nn.Parameter(torch.ones(num_features))
    
    def forward(self, features):
        """前向传播
        
        Args:
            features: 特征列表,每个形状为(batch, seq_len, dim_i)
        
        Returns:
            fused: 融合后的特征,(batch, seq_len, aligned_dim)
        """
        # 对齐特征维度
        aligned_features = []
        for feat, align_layer in zip(features, self.align_layers):
            aligned_features.append(align_layer(feat))
        
        # 堆叠特征: (batch, seq_len, num_features, aligned_dim)
        stacked = torch.stack(aligned_features, dim=2)
        
        # 归一化权重
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # 加权求和: (batch, seq_len, aligned_dim)
        weights_expanded = weights.view(1, 1, self.num_features, 1)
        fused = (stacked * weights_expanded).sum(dim=2)
        
        return fused


class GatedFeatureFusion(nn.Module):
    """门控融合层
    
    使用门控机制动态学习每个位置的特征权重
    
    Args:
        num_features: 特征数量
        feature_dims: 各特征维度列表
        aligned_dim: 对齐后的目标维度
        gate_hidden_dim: 门控网络隐藏层维度(默认为aligned_dim)
    """
    
    def __init__(self, num_features, feature_dims, aligned_dim, gate_hidden_dim=None):
        super().__init__()
        self.num_features = num_features
        self.feature_dims = feature_dims
        self.aligned_dim = aligned_dim
        self.gate_hidden_dim = gate_hidden_dim or aligned_dim
        
        # 特征对齐层
        self.align_layers = nn.ModuleList([
            nn.Linear(dim, aligned_dim) if dim != aligned_dim else nn.Identity()
            for dim in feature_dims
        ])
        
        # 门控网络: 输入所有对齐后的特征,输出每个特征的权重
        concat_dim = aligned_dim * num_features
        self.gate_network = nn.Sequential(
            nn.Linear(concat_dim, self.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.gate_hidden_dim, num_features)
        )
    
    def forward(self, features):
        """前向传播
        
        Args:
            features: 特征列表,每个形状为(batch, seq_len, dim_i)
        
        Returns:
            fused: 融合后的特征,(batch, seq_len, aligned_dim)
        """
        # 对齐特征维度
        aligned_features = []
        for feat, align_layer in zip(features, self.align_layers):
            aligned_features.append(align_layer(feat))
        
        # 拼接所有特征用于门控计算
        concat_features = torch.cat(aligned_features, dim=-1)  # (batch, seq_len, concat_dim)
        
        # 计算门控值
        gate_scores = self.gate_network(concat_features)  # (batch, seq_len, num_features)
        gates = F.softmax(gate_scores, dim=-1)  # 归一化
        
        # 堆叠特征
        stacked = torch.stack(aligned_features, dim=2)  # (batch, seq_len, num_features, aligned_dim)
        
        # 门控加权
        gates_expanded = gates.unsqueeze(-1)  # (batch, seq_len, num_features, 1)
        fused = (stacked * gates_expanded).sum(dim=2)  # (batch, seq_len, aligned_dim)
        
        return fused


class AttentionFeatureFusion(nn.Module):
    """注意力融合层
    
    使用多头注意力机制融合多个特征
    
    Args:
        num_features: 特征数量
        feature_dims: 各特征维度列表
        aligned_dim: 对齐后的目标维度
        num_heads: 注意力头数
        dropout: dropout率
    """
    
    def __init__(self, num_features, feature_dims, aligned_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.feature_dims = feature_dims
        self.aligned_dim = aligned_dim
        self.num_heads = num_heads
        
        # 特征对齐层
        self.align_layers = nn.ModuleList([
            nn.Linear(dim, aligned_dim) if dim != aligned_dim else nn.Identity()
            for dim in feature_dims
        ])
        
        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=aligned_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Norm 和 残差连接
        self.layer_norm = nn.LayerNorm(aligned_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features):
        """前向传播
        
        Args:
            features: 特征列表,每个形状为(batch, seq_len, dim_i)
                     第一个特征作为Query(通常是BERT特征)
        
        Returns:
            fused: 融合后的特征,(batch, seq_len, aligned_dim)
        """
        # 对齐特征维度
        aligned_features = []
        for feat, align_layer in zip(features, self.align_layers):
            aligned_features.append(align_layer(feat))
        
        # 第一个特征(BERT)作为Query
        query = aligned_features[0]
        
        # 堆叠所有特征作为Key和Value
        key_value = torch.stack(aligned_features, dim=2)  # (batch, seq_len, num_features, aligned_dim)
        batch_size, seq_len, num_feat, _ = key_value.shape
        
        # 重塑为 (batch * seq_len, num_features, aligned_dim) 用于多头注意力
        key_value = key_value.reshape(batch_size * seq_len, num_feat, self.aligned_dim)
        query_reshaped = query.reshape(batch_size * seq_len, 1, self.aligned_dim)
        
        # 多头注意力
        attn_output, _ = self.multihead_attn(
            query=query_reshaped,
            key=key_value,
            value=key_value
        )
        
        # 重塑回 (batch, seq_len, aligned_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.aligned_dim)
        
        # 残差连接 + Layer Norm
        fused = self.layer_norm(query + self.dropout(attn_output))
        
        return fused
