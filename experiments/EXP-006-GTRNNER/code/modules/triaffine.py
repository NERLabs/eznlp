#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三仿射词典融合模块

使用三仿射注意力机制融合词典特征、边界特征和类型特征。
适用于扁平NER的词典信息增强。

参考:
- 余肖生等. "GTR-NNER" (2025)
- Dozat & Manning. "Deep biaffine attention for neural dependency parsing" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class TriAffineDictFusion(nn.Module):
    """三仿射词典融合模块
    
    使用三仿射注意力融合：
    1. 词典特征 (dict_feature): 来自专家词典匹配
    2. 边界特征 (boundary_feature): 来自BiLSTM编码
    3. 类型特征 (type_feature): 可学习的实体类型嵌入
    
    Args:
        hidden_dim: 隐藏层维度
        dict_dim: 词典特征维度
        num_dict_types: 词典类型数量
        dropout: dropout比率
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dict_dim: int = 64,
        num_dict_types: int = 10,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dict_dim = dict_dim
        self.num_dict_types = num_dict_types
        
        # 特征投影层
        self.dict_proj = nn.Linear(dict_dim, hidden_dim)
        self.boundary_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 类型嵌入
        self.type_embedding = nn.Embedding(num_dict_types + 1, hidden_dim)  # +1 for unknown
        
        # 三仿射权重张量
        # TriAffine(h_dict, h_boundary, h_type) = W @ h_dict @ h_boundary @ h_type
        self.tri_weight = nn.Parameter(
            torch.zeros(hidden_dim, hidden_dim, hidden_dim)
        )
        nn.init.xavier_uniform_(self.tri_weight)
        
        # 输出层
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        dict_feature: torch.Tensor,
        boundary_feature: torch.Tensor,
        dict_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            dict_feature: 词典特征 [batch, seq_len, dict_dim]
            boundary_feature: 边界特征 [batch, seq_len, hidden_dim]
            dict_type_ids: 词典类型ID [batch, seq_len]，0表示无匹配
            
        Returns:
            融合后的特征 [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = boundary_feature.size()
        
        # 特征投影
        h_dict = self.dict_proj(dict_feature)  # [batch, seq, hidden]
        h_boundary = self.boundary_proj(boundary_feature)  # [batch, seq, hidden]
        
        # 类型嵌入
        if dict_type_ids is None:
            dict_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=boundary_feature.device)
        h_type = self.type_embedding(dict_type_ids)  # [batch, seq, hidden]
        
        # 三仿射计算 (简化版，避免高维张量运算)
        # 原始: output = W @ h_dict @ h_boundary @ h_type
        # 简化: output = sum(W_i * h_dict * h_boundary * h_type)
        
        # 方法1: 逐元素乘法后求和
        tri_output = (h_dict * h_boundary * h_type).sum(dim=-1, keepdim=True)  # [batch, seq, 1]
        
        # 方法2: 使用双线性形式近似
        # tri_output = h_dict @ W @ h_boundary.T，然后与h_type结合
        # 这里使用更简单的实现
        
        # 将三仿射结果广播到hidden_dim
        tri_output = tri_output.expand(-1, -1, self.hidden_dim)  # [batch, seq, hidden]
        
        # 结合原始特征
        combined = h_dict + h_boundary + h_type + tri_output
        
        # 输出投影
        output = self.output_proj(combined)
        output = self.dropout(output)
        
        # 残差连接 + LayerNorm
        output = self.layer_norm(boundary_feature + output)
        
        return output


class SimplifiedTriAffine(nn.Module):
    """简化版三仿射融合
    
    使用eznlp框架已有的TriAffineFusor实现三仿射融合。
    
    Args:
        hidden_dim: 隐藏层维度
        dropout: dropout比率
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        try:
            from eznlp.nn.modules import TriAffineFusor
            self.tri_affine = TriAffineFusor(hidden_dim, hidden_dim)
        except ImportError:
            # 如果eznlp不可用，使用自定义实现
            self.tri_affine = None
            self.tri_weight = nn.Parameter(
                torch.zeros(hidden_dim, hidden_dim, hidden_dim)
            )
            nn.init.xavier_uniform_(self.tri_weight)
        
        # 特征投影
        self.proj_dict = nn.Linear(hidden_dim, hidden_dim)
        self.proj_boundary = nn.Linear(hidden_dim, hidden_dim)
        self.proj_type = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        h_dict: torch.Tensor,
        h_boundary: torch.Tensor,
        h_type: torch.Tensor
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            h_dict: 词典特征 [batch, seq_len, hidden_dim]
            h_boundary: 边界特征 [batch, seq_len, hidden_dim]
            h_type: 类型特征 [batch, seq_len, hidden_dim]
            
        Returns:
            融合特征 [batch, seq_len, hidden_dim]
        """
        # 投影
        h_dict = self.proj_dict(h_dict)
        h_boundary = self.proj_boundary(h_boundary)
        h_type = self.proj_type(h_type)
        
        if self.tri_affine is not None:
            # 使用eznlp的TriAffineFusor
            tri_out = self.tri_affine(h_dict, h_boundary, h_type)  # [batch, seq, seq, hidden]
            # 取对角线元素
            tri_out = tri_out.diagonal(dim1=1, dim2=2).transpose(1, 2)
        else:
            # 自定义实现
            tri_out = self._custom_tri_affine(h_dict, h_boundary, h_type)
        
        # 输出
        output = self.output_proj(tri_out)
        output = self.dropout(output)
        output = self.layer_norm(h_boundary + output)
        
        return output
    
    def _custom_tri_affine(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        h3: torch.Tensor
    ) -> torch.Tensor:
        """自定义三仿射计算"""
        # 简化实现：逐元素乘法
        return h1 * h2 * h3


class DictFeatureExtractor(nn.Module):
    """词典特征提取器
    
    从专家词典中提取特征，包括：
    1. 匹配类型嵌入
    2. 匹配位置信息
    3. 匹配频率信息
    
    Args:
        dict_path: 词典文件路径
        dict_dim: 词典特征维度
        num_types: 实体类型数量
    """
    
    def __init__(
        self,
        dict_path: str = None,
        dict_dim: int = 64,
        num_types: int = 10
    ):
        super().__init__()
        self.dict_dim = dict_dim
        self.num_types = num_types
        
        # 类型嵌入
        self.type_embedding = nn.Embedding(num_types + 1, dict_dim // 2)
        
        # 位置嵌入 (相对位置)
        self.position_embedding = nn.Embedding(32, dict_dim // 4)  # 最大跨度32
        
        # 频率嵌入
        self.freq_embedding = nn.Embedding(100, dict_dim // 4)  # 最大频率100
        
        # 输出投影
        self.output_proj = nn.Linear(dict_dim, dict_dim)
        
        # 加载词典
        self.dict_data = {}
        if dict_path:
            self._load_dict(dict_path)
    
    def _load_dict(self, dict_path: str):
        """加载词典"""
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        word = parts[0]
                        label = parts[1]
                        freq = int(parts[2]) if len(parts) > 2 else 1
                        self.dict_data[word] = (label, freq)
            print(f"加载词典: {len(self.dict_data)} 条目")
        except Exception as e:
            print(f"词典加载失败: {e}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        tokens: List[List[str]],
        type2id: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取词典特征
        
        Args:
            input_ids: token IDs [batch, seq_len]
            tokens: token列表 [batch][seq]
            type2id: 类型到ID的映射
            
        Returns:
            dict_feature: 词典特征 [batch, seq_len, dict_dim]
            dict_type_ids: 词典类型ID [batch, seq_len]
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # 初始化输出
        dict_features = torch.zeros(batch_size, seq_len, self.dict_dim, device=device)
        dict_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        # 逐样本处理
        for b in range(batch_size):
            for i, token in enumerate(tokens[b]):
                if token in self.dict_data:
                    label, freq = self.dict_data[token]
                    type_id = type2id.get(label, 0)
                    
                    # 类型嵌入
                    type_emb = self.type_embedding(torch.tensor([type_id], device=device))
                    
                    # 频率嵌入
                    freq_id = min(freq, 99)
                    freq_emb = self.freq_embedding(torch.tensor([freq_id], device=device))
                    
                    # 组合特征
                    dict_features[b, i] = torch.cat([type_emb.squeeze(0), freq_emb.squeeze(0)], dim=-1)
                    dict_type_ids[b, i] = type_id
        
        # 输出投影
        dict_features = self.output_proj(dict_features)
        
        return dict_features, dict_type_ids


if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    seq_len = 32
    hidden_dim = 256
    dict_dim = 64
    
    print("测试三仿射词典融合模块...")
    
    # 测试TriAffineDictFusion
    fusion = TriAffineDictFusion(hidden_dim, dict_dim)
    dict_feat = torch.randn(batch_size, seq_len, dict_dim)
    boundary_feat = torch.randn(batch_size, seq_len, hidden_dim)
    type_ids = torch.randint(0, 10, (batch_size, seq_len))
    
    output = fusion(dict_feat, boundary_feat, type_ids)
    print(f"TriAffineDictFusion: {boundary_feat.shape} -> {output.shape}")
    
    # 测试SimplifiedTriAffine
    simple_fusion = SimplifiedTriAffine(hidden_dim)
    h_dict = torch.randn(batch_size, seq_len, hidden_dim)
    h_boundary = torch.randn(batch_size, seq_len, hidden_dim)
    h_type = torch.randn(batch_size, seq_len, hidden_dim)
    
    output = simple_fusion(h_dict, h_boundary, h_type)
    print(f"SimplifiedTriAffine: {h_boundary.shape} -> {output.shape}")
    
    print(f"\n参数量统计:")
    print(f"  TriAffineDictFusion: {sum(p.numel() for p in fusion.parameters()):,}")
    print(f"  SimplifiedTriAffine: {sum(p.numel() for p in simple_fusion.parameters()):,}")
