#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型通用组件模块

提供可复用的模型组件构建器，包括：
- 编码器配置构建器
- 解码器配置构建器
- 嵌入层配置构建器
- 位置编码模块
- Lattice 结构模块
- Lattice 注意力机制
- Lattice 辅助工具 ⭐ 新增
"""

from .encoder_builder import EncoderBuilder
from .decoder_builder import DecoderBuilder
from .embedding_builder import EmbeddingBuilder
from .position_encoding import (
    SinusoidalPositionEncoding,
    RelativePositionEncoding,
    FourPositionFusion
)
from .lattice_modules import MultiInputLSTMCell, LatticeLSTMEncoder
from .lattice_attention import (
    MultiHeadAttentionWithRelativePosition,
    LatticeSelfAttention,
    TransformerEncoderLayer
)
from .lattice_utils import (
    AdaptiveDropout,
    LayerProcess,
    PositionwiseFeedForward,
    AbsolutePositionEmbedding,
    StartEndPositionEmbedding
)

__all__ = [
    # 构建器
    'EncoderBuilder',
    'DecoderBuilder',
    'EmbeddingBuilder',
    # 位置编码
    'SinusoidalPositionEncoding',
    'RelativePositionEncoding',
    'FourPositionFusion',
    # Lattice 模块
    'MultiInputLSTMCell',
    'LatticeLSTMEncoder',
    # Lattice 注意力
    'MultiHeadAttentionWithRelativePosition',
    'LatticeSelfAttention',
    'TransformerEncoderLayer',
    # Lattice 辅助工具
    'AdaptiveDropout',
    'LayerProcess',
    'PositionwiseFeedForward',
    'AbsolutePositionEmbedding',
    'StartEndPositionEmbedding',
]
