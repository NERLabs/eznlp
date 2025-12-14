#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入层配置构建器

提供各种嵌入层配置的构建方法，支持：
- BERT/预训练模型
- SoftLexicon
- ExpertDict
"""

import transformers
from eznlp.model import (
    BertLikeConfig,
    SoftLexiconConfig,
    ExpertDictConfig,
)


class EmbeddingBuilder:
    """嵌入层配置构建器
    
    封装各种嵌入层配置的构建逻辑
    """
    
    def __init__(self):
        """初始化构建器"""
        self._bert_model_cache = {}
        self._tokenizer_cache = {}
    
    def build_bert_config(self, bert_arch, freeze=False, mix_layers="top"):
        """构建 BERT 配置（带缓存）
        
        Args:
            bert_arch: BERT 模型架构名称
            freeze: 是否冻结 BERT 参数
            mix_layers: 层融合方式
            
        Returns:
            BertLikeConfig: BERT 配置
        """
        # 使用缓存避免重复加载
        if bert_arch not in self._bert_model_cache:
            self._bert_model_cache[bert_arch] = transformers.AutoModel.from_pretrained(bert_arch)
            self._tokenizer_cache[bert_arch] = transformers.AutoTokenizer.from_pretrained(bert_arch)
        
        return BertLikeConfig(
            tokenizer=self._tokenizer_cache[bert_arch],
            bert_like=self._bert_model_cache[bert_arch],
            freeze=freeze,
            mix_layers=mix_layers
        )
    
    @staticmethod
    def build_softlexicon_config(vectors, emb_dim=50, agg_mode="wtd_mean_pooling"):
        """构建 SoftLexicon 配置
        
        Args:
            vectors: 词向量对象
            emb_dim: 嵌入维度
            agg_mode: 聚合模式
            
        Returns:
            SoftLexiconConfig: SoftLexicon 配置
        """
        return SoftLexiconConfig(
            vectors=vectors,
            emb_dim=emb_dim,
            agg_mode=agg_mode,
        )
    
    @staticmethod
    def build_expert_dict_config(emb_dim=50, agg_mode="wtd_mean_pooling"):
        """构建 ExpertDict 配置
        
        Args:
            emb_dim: 嵌入维度
            agg_mode: 聚合模式
            
        Returns:
            ExpertDictConfig: ExpertDict 配置
        """
        return ExpertDictConfig(
            emb_dim=emb_dim,
            agg_mode=agg_mode
        )
    
    def build_from_args(self, args):
        """从参数对象构建 BERT 配置
        
        Args:
            args: 命令行参数对象
            
        Returns:
            BertLikeConfig: BERT 配置
        """
        bert_arch = getattr(args, 'bert_arch', 'hfl/chinese-macbert-base')
        freeze = getattr(args, 'freeze_bert', False)
        mix_layers = getattr(args, 'mix_layers', 'top')
        
        return self.build_bert_config(bert_arch, freeze, mix_layers)
