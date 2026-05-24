#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型组件和模型定义模块

目录结构：
- block/: 通用模型组件（编码器、解码器、嵌入层等）
- models/: 完整模型定义（针对具体数据集的模型构建器）
- model/: 其他模型文件（如 FLAT 等）
"""

# 导出通用组件
from .block import EncoderBuilder, DecoderBuilder, EmbeddingBuilder

# 导出完整模型构建器
from .models import RedJujubeModelBuilder, RedJujubeModelFactory

__all__ = [
    # 通用组件
    'EncoderBuilder',
    'DecoderBuilder',
    'EmbeddingBuilder',
    # 完整模型
    'RedJujubeModelBuilder',
    'RedJujubeModelFactory',
]
