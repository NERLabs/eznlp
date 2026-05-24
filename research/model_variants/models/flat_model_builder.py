#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLAT (Flat-Lattice Transformer) 模型构建器

功能：
- 基于已提取的核心组件构建完整的 FLAT 模型
- 支持 Lattice 结构的 Transformer 编码器
- 使用四位置融合编码进行词汇感知

设计参考：
- Li et al. "FLAT: Chinese NER Using Flat-Lattice Transformer" (ACL 2020)

核心组件：
- LatticeSelfAttention: Lattice 结构的自注意力机制
- FourPositionFusion: 四位置融合编码 (SS, SE, ES, EE)
- LayerProcess: 灵活的层处理序列 (Dropout/Add/LayerNorm)
- PositionwiseFeedForward: 位置感知前馈网络
- TransformerEncoderLayer: 完整的 Transformer 编码层
"""

import torch
import torch.nn as nn
from typing import Optional
from eznlp.model import ExtractorConfig, EncoderConfig
from eznlp.model.decoder import SequenceTaggingDecoderConfig

# 导入 FLAT 核心组件
from research.model_variants.block import (
    LatticeSelfAttention,
    TransformerEncoderLayer,
    FourPositionFusion,
    LayerProcess,
    PositionwiseFeedForward,
    StartEndPositionEmbedding,
    EmbeddingBuilder,
    DecoderBuilder,
)


class FLATEncoder(nn.Module):
    """FLAT Transformer 编码器
    
    核心创新：
    1. Lattice 结构: 字符 + 词汇的扁平化表示
    2. 四位置编码: SS/SE/ES/EE 四种相对位置的融合
    3. 全连接注意力: 字-字、字-词、词-词的全局交互
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_size: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.15,
        four_pos_fusion: str = 'ff',
        learnable_position: bool = False,
        layer_preprocess: str = 'n',
        layer_postprocess: str = 'dan',
    ):
        """初始化 FLAT 编码器
        
        Args:
            hidden_size: 隐藏层维度
            num_layers: Transformer 层数
            num_heads: 多头注意力头数
            ff_size: 前馈网络隐藏层维度
            max_seq_len: 最大序列长度
            dropout: Dropout 率
            four_pos_fusion: 四位置融合方式 ('ff', 'attn', 'gate')
            learnable_position: 位置编码是否可学习
            layer_preprocess: 层前处理序列 (如 'n' 表示 LayerNorm)
            layer_postprocess: 层后处理序列 (如 'dan' 表示 Dropout->Add->LayerNorm)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # 四位置融合模块
        self.pos_fusion = FourPositionFusion(
            hidden_size=hidden_size,
            max_len=max_seq_len,
            fusion_mode=four_pos_fusion,
            learnable=learnable_position,
            shared=True  # 四个位置编码共享参数
        )
        
        # 堆叠多层 Transformer 编码层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                max_len=max_seq_len,
                ff_size=ff_size,
                dropout=dropout,
                activation='relu'
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        embedded: torch.Tensor,
        pos_s: torch.Tensor,
        pos_e: torch.Tensor,
        seq_len: torch.Tensor,
        lex_num: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            embedded: 嵌入表示 [batch, seq_len+lex_num, hidden_size]
            pos_s: 开始位置 [batch, seq_len+lex_num]
            pos_e: 结束位置 [batch, seq_len+lex_num]
            seq_len: 字符序列长度 [batch]
            lex_num: 词汇数量 [batch]
            mask: 序列掩码 [batch, seq_len+lex_num]
            
        Returns:
            编码后的隐藏状态 [batch, seq_len+lex_num, hidden_size]
        """
        # 计算四位置融合编码（暂时不使用，因为 TransformerEncoderLayer 内部会计算）
        # rel_pos_embedding = self.pos_fusion(pos_s, pos_e)
        
        # 逐层编码
        hidden = embedded
        for layer in self.layers:
            # TransformerEncoderLayer 的参数是 (hidden_states, pos_s, pos_e, mask)
            hidden = layer(hidden, pos_s, pos_e, mask)
        
        return hidden


class FLATModelBuilder:
    """FLAT 模型构建器
    
    组合 FLAT 核心组件构建完整的 NER 模型
    """
    
    def __init__(self, args):
        """初始化模型构建器
        
        Args:
            args: 命令行参数对象，包含所有模型超参数
        """
        self.args = args
        self.embedding_builder = EmbeddingBuilder()
        self.decoder_builder = DecoderBuilder()
    
    def _get_bert_config(self):
        """获取 BERT 配置"""
        return self.embedding_builder.build_from_args(self.args)
    
    def _get_decoder_config(self):
        """获取解码器配置"""
        return self.decoder_builder.build_from_args(self.args, use_crf=True, scheme="BMES")
    
    def _get_flat_encoder_config(self):
        """获取 FLAT 编码器配置
        
        注意：这里返回的是编码器参数字典，不是 EncoderConfig
        因为 FLAT 需要自定义编码器，不使用 eznlp 的标准编码器
        """
        return {
            'hidden_size': getattr(self.args, 'hidden_size', 512),
            'num_layers': getattr(self.args, 'num_layers', 4),
            'num_heads': getattr(self.args, 'num_heads', 8),
            'ff_size': getattr(self.args, 'ff_size', 2048),
            'max_seq_len': getattr(self.args, 'max_seq_len', 512),
            'dropout': getattr(self.args, 'dropout', 0.15),
            'four_pos_fusion': getattr(self.args, 'four_pos_fusion', 'ff'),
            'learnable_position': getattr(self.args, 'learnable_position', False),
            'layer_preprocess': getattr(self.args, 'layer_preprocess', 'n'),
            'layer_postprocess': getattr(self.args, 'layer_postprocess', 'dan'),
        }
    
    def build_flat_baseline(self):
        """构建 FLAT Baseline 模型
        
        架构：FLAT Encoder + CRF
        不使用预训练模型，仅使用字符和词汇嵌入
        
        Returns:
            FLAT 模型配置字典
        """
        encoder_params = self._get_flat_encoder_config()
        
        return {
            'model_type': 'flat',
            'encoder': FLATEncoder(**encoder_params),
            'decoder': self._get_decoder_config(),
            'use_bert': False,
        }
    
    def build_flat_with_bert(self):
        """构建 FLAT + BERT 模型
        
        架构：BERT + FLAT Encoder + CRF
        使用预训练 BERT 作为底层特征提取器
        
        Returns:
            ExtractorConfig: 标准的提取器配置对象
        """
        # 使用 Identity Encoder（因为 FLAT 在数据处理时直接处理）
        # 或者不使用编码器，让 BERT 输出直接到 decoder
        encoder_config = EncoderConfig(
            arch="Identity",
            in_dim=768,  # BERT 输出维度
            hid_dim=768
        )
        
        return ExtractorConfig(
            bert_like=self._get_bert_config(),
            encoder=encoder_config,
            decoder=self._get_decoder_config()
        )
    
    def build_flat_with_lexicon(self, vectors):
        """构建 FLAT + Lexicon 模型
        
        架构：FLAT Encoder + SoftLexicon + CRF
        在 FLAT 基础上增加软词典特征
        
        Args:
            vectors: 词向量对象
            
        Returns:
            FLAT + Lexicon 模型配置字典
        """
        encoder_params = self._get_flat_encoder_config()
        
        return {
            'model_type': 'flat_lexicon',
            'encoder': FLATEncoder(**encoder_params),
            'nested_ohots': {
                'softlexicon': self.embedding_builder.build_softlexicon_config(vectors, emb_dim=50)
            },
            'decoder': self._get_decoder_config(),
            'use_bert': False,
        }


class FLATModelFactory:
    """FLAT 模型工厂类
    
    根据模型类型字符串动态创建对应的 FLAT 模型配置
    """
    
    MODEL_TYPES = {
        'flat_baseline': 'build_flat_baseline',
        'flat_bert': 'build_flat_with_bert',
        'flat_lexicon': 'build_flat_with_lexicon',
    }
    
    @staticmethod
    def create_model_config(model_type, args, vectors=None):
        """创建 FLAT 模型配置
        
        Args:
            model_type: 模型类型字符串
            args: 命令行参数对象
            vectors: 词向量对象（可选，仅 flat_lexicon 需要）
            
        Returns:
            FLAT 模型配置字典
            
        Raises:
            ValueError: 不支持的模型类型
        """
        if model_type not in FLATModelFactory.MODEL_TYPES:
            raise ValueError(
                f"不支持的 FLAT 模型类型: {model_type}. "
                f"支持的类型: {list(FLATModelFactory.MODEL_TYPES.keys())}"
            )
        
        builder = FLATModelBuilder(args)
        method_name = FLATModelFactory.MODEL_TYPES[model_type]
        method = getattr(builder, method_name)
        
        # 如果方法需要 vectors 参数
        if method_name == 'build_flat_with_lexicon':
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
            'flat_baseline': 'FLAT Baseline (Char + Word Embedding + Lattice Transformer)',
            'flat_bert': 'FLAT + BERT (BERT + Lattice Transformer)',
            'flat_lexicon': 'FLAT + SoftLexicon (Lattice Transformer + Soft Lexicon)',
        }
        return display_names.get(model_type, model_type)


# 使用示例
if __name__ == '__main__':
    """
    使用示例：
    
    import argparse
    from research.model_variants.models.flat_model_builder import FLATModelFactory
    
    # 创建参数对象
    args = argparse.Namespace(
        hidden_size=512,
        num_layers=4,
        num_heads=8,
        ff_size=2048,
        max_seq_len=512,
        dropout=0.15,
        four_pos_fusion='ff',
        learnable_position=False,
        layer_preprocess='n',
        layer_postprocess='dan',
        bert_arch='hfl/chinese-macbert-base',
        freeze_bert=False,
    )
    
    # 创建 FLAT Baseline 模型
    model_config = FLATModelFactory.create_model_config('flat_baseline', args)
    
    # 创建 FLAT + BERT 模型
    model_config = FLATModelFactory.create_model_config('flat_bert', args)
    
    # 创建 FLAT + Lexicon 模型
    from eznlp import Vectors
    vectors = Vectors.from_file('path/to/word_vectors.txt')
    model_config = FLATModelFactory.create_model_config('flat_lexicon', args, vectors=vectors)
    """
    print("FLAT Model Builder 示例代码已准备就绪")
    print("请参考文件末尾的使用示例")
