# -*- coding: utf-8 -*-
"""
EXP-006-GTRNNER 自定义模块

包含:
- rope: 旋转位置编码 (RoPE)
- triaffine: 三仿射词典融合
"""

from .rope import RotaryPositionEmbedding, RoPEEnhancedEncoder, RelativePositionConv
from .triaffine import TriAffineDictFusion, SimplifiedTriAffine, DictFeatureExtractor

__all__ = [
    'RotaryPositionEmbedding',
    'RoPEEnhancedEncoder',
    'RelativePositionConv',
    'TriAffineDictFusion',
    'SimplifiedTriAffine',
    'DictFeatureExtractor',
]
