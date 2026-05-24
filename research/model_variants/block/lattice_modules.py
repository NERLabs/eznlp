#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lattice 结构模块

提供 Lattice 结构的核心组件，用于处理词汇增强的序列：
- MultiInputLSTMCell: 多输入 LSTM 单元（处理字符和词汇）
- LatticeLSTM: Lattice LSTM 编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MultiInputLSTMCell(nn.Module):
    """多输入 LSTM Cell
    
    支持同时处理字符序列和匹配的词汇信息，是 Lattice LSTM 的核心组件
    """
    
    def __init__(self, input_size: int, hidden_size: int, use_bias: bool = True):
        """初始化
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            use_bias: 是否使用偏置
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        
        # 标准 LSTM 参数
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        
        # Alpha 门控参数（用于词汇信息）
        self.alpha_weight_ih = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self.alpha_weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('alpha_bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数（使用正交初始化）"""
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.alpha_weight_ih.data)
        
        # 隐藏层权重使用单位矩阵
        weight_hh_data = torch.eye(self.hidden_size).repeat(1, 3)
        self.weight_hh.data.copy_(weight_hh_data)
        
        alpha_weight_hh_data = torch.eye(self.hidden_size)
        self.alpha_weight_hh.data.copy_(alpha_weight_hh_data)
        
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)
            nn.init.constant_(self.alpha_bias.data, 0)
    
    def forward(
        self, 
        input_: torch.Tensor, 
        c_input: List[torch.Tensor], 
        hx: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            input_: 当前字符输入 [1, input_size]
            c_input: 匹配词汇的 cell 状态列表，每个 [1, hidden_size]
            hx: 前一时刻隐藏状态 (h_0, c_0)
            
        Returns:
            (h_1, c_1): 新的隐藏状态和 cell 状态
        """
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        
        # 标准 LSTM 计算
        if self.use_bias:
            bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())
            wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        else:
            wh_b = torch.mm(h_0, self.weight_hh)
        
        wi = torch.mm(input_, self.weight_ih)
        i, o, g = torch.split(wh_b + wi, split_size=self.hidden_size, dim=1)
        
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_num = len(c_input)
        
        if c_num == 0:
            # 没有词汇匹配，标准 LSTM
            f = 1 - i
            c_1 = f * c_0 + i * g
            h_1 = o * torch.tanh(c_1)
        else:
            # 有词汇匹配，使用 alpha 门控融合
            c_input_var = torch.cat(c_input, 0)  # [c_num, hidden_size]
            
            if self.use_bias:
                alpha_bias_batch = self.alpha_bias.unsqueeze(0).expand(batch_size, *self.alpha_bias.size())
                alpha_wi = torch.addmm(alpha_bias_batch, input_, self.alpha_weight_ih)
            else:
                alpha_wi = torch.mm(input_, self.alpha_weight_ih)
            
            alpha_wh = torch.mm(h_0, self.alpha_weight_hh)
            
            # 计算 alpha 权重
            alpha = torch.sigmoid(alpha_wi + alpha_wh).expand(c_num, self.hidden_size)
            
            # 计算遗忘门
            _, _ , fg = torch.split(wh_b.expand(c_num, 3 * self.hidden_size), 
                                    split_size=self.hidden_size, dim=1)
            fg = torch.tanh(fg)
            
            # 融合词汇信息
            c_1 = (i * g).expand(c_num, self.hidden_size) + \
                  (alpha * fg).expand(c_num, self.hidden_size) * c_input_var
            c_1 = torch.mean(c_1, 0).view(batch_size, self.hidden_size)
            
            h_1 = o * torch.tanh(c_1)
        
        return h_1, c_1


class LatticeLSTMEncoder(nn.Module):
    """Lattice LSTM 编码器
    
    将 MultiInputLSTMCell 组合成完整的编码器
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0
    ):
        """初始化
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            num_layers: 层数
            bidirectional: 是否双向
            dropout: Dropout 比率
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # 创建前向和后向的 LSTM Cells
        self.forward_cells = nn.ModuleList([
            MultiInputLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        if bidirectional:
            self.backward_cells = nn.ModuleList([
                MultiInputLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ])
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        char_inputs: torch.Tensor,
        word_inputs: List[List[torch.Tensor]],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            char_inputs: 字符输入 [batch, seq_len, input_size]
            word_inputs: 词汇输入列表，word_inputs[i][j] 是第 i 个样本第 j 个位置匹配的词汇
            mask: 序列掩码 [batch, seq_len]
            
        Returns:
            编码后的隐藏状态 [batch, seq_len, hidden_size * (2 if bidirectional else 1)]
        """
        # 注意：完整实现较复杂，这里提供接口框架
        # 实际使用时需要根据具体的 lattice 结构实现
        raise NotImplementedError(
            "LatticeLSTMEncoder 需要根据具体的 lattice 结构实现。"
            "这里提供 MultiInputLSTMCell 作为核心组件。"
        )
