# -*- coding: utf-8 -*-
"""
融合特征提取器

支持多种特征融合策略的Extractor实现
"""

import torch
from ...config import ConfigDict
from ...wrapper import Batch
from ..encoder import EncoderConfig
from ..embedder import OneHotConfig
from ..decoder import SequenceTaggingDecoderConfig
from ..nested_embedder import SoftLexiconConfig
from .extractor import ExtractorConfig, Extractor
from ...nn.modules.fusion import (
    WeightedFeatureFusion,
    GatedFeatureFusion, 
    AttentionFeatureFusion
)


class FusionExtractorConfig(ExtractorConfig):
    """融合特征提取器配置
    
    扩展ExtractorConfig,支持指定融合策略
    
    Args:
        fusion_strategy: 融合策略,'weighted'/'gated'/'attention'
        fusion_params: 融合层参数字典
        其他参数同ExtractorConfig
    """
    
    def __init__(self, fusion_strategy="concat", fusion_params=None, **kwargs):
        super().__init__(**kwargs)
        self.fusion_strategy = fusion_strategy
        self.fusion_params = fusion_params or {}
    
    def build_vocabs_and_dims(self, *partitions):
        """构建词表和维度
        
        重写父类方法，确保融合模式下decoder.in_dim设置正确
        """
        # 先调用父类方法构建基础词表和维度
        super().build_vocabs_and_dims(*partitions)
        
        # 如果使用融合策略，需要重新设置decoder.in_dim
        if self.fusion_strategy != "concat":
            # 融合后的输出维度等于BERT维度（aligned_dim）
            if self.bert_like is not None:
                aligned_dim = 768  # MacBERT hidden size
                self.decoder.in_dim = aligned_dim
    
    def instantiate(self):
        """实例化融合提取器"""
        return FusionExtractor(self)


class FusionExtractor(Extractor):
    """融合特征提取器
    
    支持多种融合策略的NER模型
    """
    
    def __init__(self, config: FusionExtractorConfig):
        super().__init__(config)
        
        # 如果使用非concat融合策略,创建融合层
        if config.fusion_strategy != "concat":
            self._build_fusion_layer(config)
    
    def _build_fusion_layer(self, config):
        """构建融合层"""
        # 获取各特征维度 - 从config而不是实例化的模块获取
        feature_dims = []
        
        # BERT特征维度
        if hasattr(self, "bert_like"):
            # BertLikeEmbedder的bert_like属性是实际的transformers模型
            bert_dim = self.bert_like.bert_like.config.hidden_size
            feature_dims.append(bert_dim)
        
        # SoftLexicon特征维度 - 从config获取
        if config.nested_ohots is not None and "softlexicon" in config.nested_ohots:
            soft_dim = config.nested_ohots["softlexicon"].out_dim
            feature_dims.append(soft_dim)
        
        # ExpertDict特征维度 - 从config获取
        if config.nested_ohots is not None and "expert_dict" in config.nested_ohots:
            expert_dim = config.nested_ohots["expert_dict"].out_dim
            feature_dims.append(expert_dim)
        
        # 目标对齐维度(使用BERT维度)
        aligned_dim = feature_dims[0] if feature_dims else 768
        num_features = len(feature_dims)
        
        # 根据策略创建融合层
        if config.fusion_strategy == "weighted":
            self.fusion_layer = WeightedFeatureFusion(
                num_features=num_features,
                feature_dims=feature_dims,
                aligned_dim=aligned_dim
            )
        elif config.fusion_strategy == "gated":
            gate_hidden_dim = config.fusion_params.get("gate_hidden_dim", aligned_dim)
            self.fusion_layer = GatedFeatureFusion(
                num_features=num_features,
                feature_dims=feature_dims,
                aligned_dim=aligned_dim,
                gate_hidden_dim=gate_hidden_dim
            )
        elif config.fusion_strategy == "attention":
            num_heads = config.fusion_params.get("num_heads", 8)
            dropout = config.fusion_params.get("dropout", 0.1)
            self.fusion_layer = AttentionFeatureFusion(
                num_features=num_features,
                feature_dims=feature_dims,
                aligned_dim=aligned_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {config.fusion_strategy}")
    
    def _get_full_embedded(self, batch: Batch):
        """获取全部嵌入特征(保持原有行为)"""
        return super()._get_full_embedded(batch)
    
    def _get_full_hidden(self, batch: Batch):
        """获取全部隐藏层特征
        
        如果使用融合策略,则分别提取BERT、SoftLex、Expert特征再融合
        否则使用原始拼接方式
        """
        if not hasattr(self, "fusion_layer"):
            # 使用原始拼接方式
            return super()._get_full_hidden(batch)
        
        # 收集各特征 - 分别处理
        features = []
        
        # BERT特征
        if hasattr(self, "bert_like"):
            bert_hidden = self.bert_like(**batch.bert_like)
            features.append(bert_hidden)
        
        # SoftLexicon特征
        if hasattr(self, "nested_ohots") and "softlexicon" in self.nested_ohots:
            soft_hidden = self.nested_ohots["softlexicon"](
                **batch.nested_ohots["softlexicon"],
                seq_lens=batch.seq_lens
            )
            features.append(soft_hidden)
        
        # ExpertDict特征
        if hasattr(self, "nested_ohots") and "expert_dict" in self.nested_ohots:
            expert_hidden = self.nested_ohots["expert_dict"](
                **batch.nested_ohots["expert_dict"],
                seq_lens=batch.seq_lens
            )
            features.append(expert_hidden)
        
        # 使用融合层融合特征
        fused = self.fusion_layer(features)
        
        # 通过intermediate2编码器
        if hasattr(self, "intermediate2"):
            full_hidden = self.intermediate2(fused, batch.mask)
        else:
            full_hidden = fused
        
        return full_hidden
