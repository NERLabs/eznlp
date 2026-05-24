#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码器配置构建器

提供各种编码器配置的构建方法，支持：
- LSTM 编码器
- GRU 编码器
- Transformer 编码器
"""

from eznlp.model import EncoderConfig


class EncoderBuilder:
    """编码器配置构建器
    
    封装编码器配置的构建逻辑，提供统一接口
    """
    
    @staticmethod
    def build_lstm_encoder(hid_dim, num_layers=1, dropout=0.5, bidirectional=True):
        """构建 LSTM 编码器配置
        
        Args:
            hid_dim: 隐藏层维度
            num_layers: LSTM 层数
            dropout: Dropout 率
            bidirectional: 是否双向
            
        Returns:
            EncoderConfig: LSTM 编码器配置
        """
        return EncoderConfig(
            arch="LSTM",
            hid_dim=hid_dim,
            num_layers=num_layers,
            in_drop_rates=(dropout, 0.0, 0.0),
            bidirectional=bidirectional
        )
    
    @staticmethod
    def build_gru_encoder(hid_dim, num_layers=1, dropout=0.5, bidirectional=True):
        """构建 GRU 编码器配置
        
        Args:
            hid_dim: 隐藏层维度
            num_layers: GRU 层数
            dropout: Dropout 率
            bidirectional: 是否双向
            
        Returns:
            EncoderConfig: GRU 编码器配置
        """
        return EncoderConfig(
            arch="GRU",
            hid_dim=hid_dim,
            num_layers=num_layers,
            in_drop_rates=(dropout, 0.0, 0.0),
            bidirectional=bidirectional
        )
    
    @staticmethod
    def build_from_args(args, arch="LSTM"):
        """从参数对象构建编码器配置
        
        Args:
            args: 命令行参数对象
            arch: 编码器架构类型 ("LSTM" 或 "GRU")
            
        Returns:
            EncoderConfig: 编码器配置
        """
        hid_dim = getattr(args, 'hid_dim', 256)
        num_layers = getattr(args, 'num_layers', 1)
        dropout = getattr(args, 'dropout', 0.5)
        
        if arch.upper() == "LSTM":
            return EncoderBuilder.build_lstm_encoder(hid_dim, num_layers, dropout)
        elif arch.upper() == "GRU":
            return EncoderBuilder.build_gru_encoder(hid_dim, num_layers, dropout)
        else:
            raise ValueError(f"不支持的编码器架构: {arch}")
