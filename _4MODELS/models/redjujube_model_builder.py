#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube NER 模型构建器

功能：
- 封装 RedJujube 数据集所有模型配置的构建逻辑
- 支持多种模型架构（Baseline, SoftLexicon, ExpertDict, Fusion等）
- 提供统一的模型构建接口

设计模式：
- 工厂模式：根据参数动态创建不同的模型配置
- 组合模式：组合通用组件构建完整模型
"""

from eznlp.model import ExtractorConfig
from eznlp.model.model import FusionExtractorConfig

# 导入通用组件构建器
from _4MODELS.block import EncoderBuilder, DecoderBuilder, EmbeddingBuilder


class RedJujubeModelBuilder:
    """RedJujube NER 模型构建器
    
    组合通用组件构建器，提供针对 RedJujube 数据集的模型配置
    """
    
    def __init__(self, args):
        """初始化模型构建器
        
        Args:
            args: 命令行参数对象，包含所有模型超参数
        """
        self.args = args
        
        # 初始化组件构建器
        self.encoder_builder = EncoderBuilder()
        self.decoder_builder = DecoderBuilder()
        self.embedding_builder = EmbeddingBuilder()
    
    def _get_bert_config(self):
        """获取 BERT 配置"""
        return self.embedding_builder.build_from_args(self.args)
    
    def _get_encoder_config(self):
        """获取编码器配置"""
        return self.encoder_builder.build_from_args(self.args, arch="LSTM")
    
    def _get_decoder_config(self):
        """获取解码器配置"""
        return self.decoder_builder.build_from_args(self.args, use_crf=True, scheme="BMES")
    
    def _get_softlexicon_config(self, vectors):
        """获取 SoftLexicon 配置"""
        return self.embedding_builder.build_softlexicon_config(vectors, emb_dim=50)
    
    def _get_expert_dict_config(self):
        """获取 ExpertDict 配置"""
        emb_dim = getattr(self.args, 'expert_dict_dim', 50)
        return self.embedding_builder.build_expert_dict_config(emb_dim)
    
    def build_baseline(self):
        """构建 Baseline 模型配置（MacBERT + BiLSTM + CRF）
        
        Returns:
            ExtractorConfig: Baseline 模型配置
        """
        return ExtractorConfig(
            bert_like=self._get_bert_config(),
            encoder=self._get_encoder_config(),
            decoder=self._get_decoder_config()
        )
    
    def build_expert_dict(self):
        """构建 +ExpertDict 模型配置
        
        Returns:
            ExtractorConfig: ExpertDict 模型配置
        """
        return ExtractorConfig(
            bert_like=self._get_bert_config(),
            nested_ohots={
                "expert_dict": self._get_expert_dict_config()
            },
            encoder=self._get_encoder_config(),
            decoder=self._get_decoder_config()
        )
    
    def build_softlexicon(self, vectors):
        """构建 SoftLexicon 模型配置
        
        Args:
            vectors: 词向量对象
            
        Returns:
            ExtractorConfig: SoftLexicon 模型配置
        """
        return ExtractorConfig(
            bert_like=self._get_bert_config(),
            nested_ohots={
                "softlexicon": self._get_softlexicon_config(vectors)
            },
            encoder=self._get_encoder_config(),
            decoder=self._get_decoder_config()
        )
    
    def build_fusion_concat(self, vectors):
        """构建 Soft+Expert 融合模型配置（方案A：直接拼接）
        
        Args:
            vectors: 词向量对象
            
        Returns:
            ExtractorConfig: 融合模型配置（拼接）
        """
        return ExtractorConfig(
            bert_like=self._get_bert_config(),
            nested_ohots={
                "softlexicon": self._get_softlexicon_config(vectors),
                "expert_dict": self._get_expert_dict_config()
            },
            encoder=self._get_encoder_config(),
            decoder=self._get_decoder_config()
        )
    
    def build_fusion_weighted(self, vectors):
        """构建 Soft+Expert 融合模型配置（方案B：加权求和）
        
        Args:
            vectors: 词向量对象
            
        Returns:
            FusionExtractorConfig: 融合模型配置（加权）
        """
        return FusionExtractorConfig(
            bert_like=self._get_bert_config(),
            nested_ohots={
                "softlexicon": self._get_softlexicon_config(vectors),
                "expert_dict": self._get_expert_dict_config()
            },
            intermediate2=None,
            decoder=self._get_decoder_config(),
            fusion_strategy="weighted"
        )
    
    def build_fusion_gated(self, vectors):
        """构建 Soft+Expert 融合模型配置（方案C：门控机制）
        
        Args:
            vectors: 词向量对象
            
        Returns:
            FusionExtractorConfig: 融合模型配置（门控）
        """
        return FusionExtractorConfig(
            bert_like=self._get_bert_config(),
            nested_ohots={
                "softlexicon": self._get_softlexicon_config(vectors),
                "expert_dict": self._get_expert_dict_config()
            },
            intermediate2=None,
            decoder=self._get_decoder_config(),
            fusion_strategy="gated",
            fusion_params={"gate_hidden_dim": 768}
        )
    
    def build_fusion_attention(self, vectors):
        """构建 Soft+Expert 融合模型配置（方案D：注意力融合）
        
        Args:
            vectors: 词向量对象
            
        Returns:
            FusionExtractorConfig: 融合模型配置（注意力）
        """
        return FusionExtractorConfig(
            bert_like=self._get_bert_config(),
            nested_ohots={
                "softlexicon": self._get_softlexicon_config(vectors),
                "expert_dict": self._get_expert_dict_config()
            },
            intermediate2=None,
            decoder=self._get_decoder_config(),
            fusion_strategy="attention",
            fusion_params={"num_heads": 8, "dropout": 0.1}
        )


class RedJujubeModelFactory:
    """RedJujube 模型工厂类
    
    根据模型类型字符串动态创建对应的模型配置
    """
    
    # 模型类型映射表
    MODEL_TYPES = {
        'baseline': 'build_baseline',
        'expert_dict': 'build_expert_dict',
        'expert_dict_auto': 'build_expert_dict',
        'expert_dict_manual': 'build_expert_dict',
        'softlexicon': 'build_softlexicon',
        'softlexicon_trainlex': 'build_softlexicon',
        'fusion_concat': 'build_fusion_concat',
        'fusion_weighted': 'build_fusion_weighted',
        'fusion_gated': 'build_fusion_gated',
        'fusion_attention': 'build_fusion_attention',
    }
    
    @staticmethod
    def create_model_config(model_type, args, vectors=None):
        """创建模型配置
        
        Args:
            model_type: 模型类型字符串
            args: 命令行参数对象
            vectors: 词向量对象（可选，仅 SoftLexicon 相关模型需要）
            
        Returns:
            模型配置对象
            
        Raises:
            ValueError: 不支持的模型类型
        """
        if model_type not in RedJujubeModelFactory.MODEL_TYPES:
            raise ValueError(
                f"不支持的模型类型: {model_type}. "
                f"支持的类型: {list(RedJujubeModelFactory.MODEL_TYPES.keys())}"
            )
        
        builder = RedJujubeModelBuilder(args)
        method_name = RedJujubeModelFactory.MODEL_TYPES[model_type]
        method = getattr(builder, method_name)
        
        # 如果方法需要 vectors 参数
        if method_name in ['build_softlexicon', 'build_fusion_concat', 
                          'build_fusion_weighted', 'build_fusion_gated', 
                          'build_fusion_attention']:
            if vectors is None:
                raise ValueError(f"模型类型 {model_type} 需要提供 vectors 参数")
            return method(vectors)
        else:
            return method()
    
    @staticmethod
    def get_model_display_name(model_type):
        """获取模型的显示名称
        
        Args:
            model_type: 模型类型字符串
            
        Returns:
            str: 模型的显示名称
        """
        display_names = {
            'baseline': 'Baseline (MacBERT + BiLSTM + CRF)',
            'expert_dict': '+ExpertDict',
            'expert_dict_auto': '+ExpertDict (自动)',
            'expert_dict_manual': '+ExpertDict (手动)',
            'softlexicon': 'SoftLexicon',
            'softlexicon_trainlex': 'SoftLexicon (训练集词表)',
            'fusion_concat': 'Soft+Expert (方案A: 直接拼接)',
            'fusion_weighted': 'Soft+Expert (方案B: 加权求和)',
            'fusion_gated': 'Soft+Expert (方案C: 门控机制)',
            'fusion_attention': 'Soft+Expert (方案D: 注意力融合)',
        }
        return display_names.get(model_type, model_type)


# 为了向后兼容，保留旧的类名
ModelBuilder = RedJujubeModelBuilder
ModelFactory = RedJujubeModelFactory
