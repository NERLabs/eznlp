#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解码器配置构建器

提供各种解码器配置的构建方法，支持：
- CRF 解码器
- Softmax 解码器
"""

from eznlp.model.decoder import SequenceTaggingDecoderConfig


class DecoderBuilder:
    """解码器配置构建器
    
    封装解码器配置的构建逻辑，提供统一接口
    """
    
    @staticmethod
    def build_crf_decoder(scheme="BMES", dropout=0.5):
        """构建 CRF 解码器配置
        
        Args:
            scheme: 标注方案 ("BMES", "BIO" 等)
            dropout: Dropout 率
            
        Returns:
            SequenceTaggingDecoderConfig: CRF 解码器配置
        """
        return SequenceTaggingDecoderConfig(
            scheme=scheme,
            use_crf=True,
            in_drop_rates=(dropout,)
        )
    
    @staticmethod
    def build_softmax_decoder(scheme="BMES", dropout=0.5):
        """构建 Softmax 解码器配置
        
        Args:
            scheme: 标注方案 ("BMES", "BIO" 等)
            dropout: Dropout 率
            
        Returns:
            SequenceTaggingDecoderConfig: Softmax 解码器配置
        """
        return SequenceTaggingDecoderConfig(
            scheme=scheme,
            use_crf=False,
            in_drop_rates=(dropout,)
        )
    
    @staticmethod
    def build_from_args(args, use_crf=True, scheme="BMES"):
        """从参数对象构建解码器配置
        
        Args:
            args: 命令行参数对象
            use_crf: 是否使用 CRF
            scheme: 标注方案
            
        Returns:
            SequenceTaggingDecoderConfig: 解码器配置
        """
        dropout = getattr(args, 'dropout', 0.5)
        
        if use_crf:
            return DecoderBuilder.build_crf_decoder(scheme, dropout)
        else:
            return DecoderBuilder.build_softmax_decoder(scheme, dropout)
