#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-006-GTRNNER 模型实现
将 GTR-NNER 论文中的相对位置编码和三仿射注意力迁移到扁平NER任务

核心组件:
1. RoPE: 旋转位置编码，增强边界感知
2. TriAffine: 三仿射词典融合，改进词典特征融合方式
3. Res-BiLSTM: 来自EXP-005验证有效的残差BiLSTM

参考论文: 
- 余肖生等. GTR-NNER (2025)
- Zhang et al. EPFD (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Dict, Tuple, List
import yaml

# 导入自定义模块
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from modules.rope import RotaryPositionEmbedding, RelativePositionConv
from modules.triaffine import TriAffineDictFusion, DictFeatureExtractor


class RoPEConfig:
    """旋转位置编码配置"""
    def __init__(self, **kwargs):
        self.enabled = kwargs.pop("enabled", True)
        self.hidden_dim = kwargs.pop("hidden_dim", 768)
        self.max_seq_len = kwargs.pop("max_seq_len", 512)
        self.base = kwargs.pop("base", 10000)
        self.dropout = kwargs.pop("dropout", 0.1)
    
    @property
    def out_dim(self):
        return self.hidden_dim


class TriAffineConfig:
    """三仿射融合配置"""
    def __init__(self, **kwargs):
        self.enabled = kwargs.pop("enabled", True)
        self.hidden_dim = kwargs.pop("hidden_dim", 256)
        self.dict_dim = kwargs.pop("dict_dim", 64)
        self.num_dict_types = kwargs.pop("num_dict_types", 10)
        self.dropout = kwargs.pop("dropout", 0.3)
    
    @property
    def out_dim(self):
        return self.hidden_dim


class ResBiLSTM(nn.Module):
    """残差BiLSTM模块"""
    def __init__(self, in_dim=768, hid_dim=256, num_layers=1, dropout=0.3, residual=True):
        super().__init__()
        self.hid_dim = hid_dim
        self.residual = residual
        
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hid_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers <= 1 else dropout
        )
        self.dropout = nn.Dropout(dropout)
    
    @property
    def out_dim(self):
        return self.hid_dim * 2 if self.residual else self.hid_dim
    
    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, padding_value=0)
        else:
            lstm_out, _ = self.lstm(x)
        
        lstm_out = self.dropout(lstm_out)
        
        if self.residual:
            return torch.cat([lstm_out, x], dim=-1)
        return lstm_out


class GTRNNERModel(nn.Module):
    """GTR-NNER技术迁移模型
    
    将相对位置编码和三仿射融合应用于扁平NER。
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # 从配置中提取参数
        model_config = config.get('model', {})
        encoder_config = model_config.get('encoder', {})
        bilstm_config = model_config.get('bilstm', {})
        rope_config = model_config.get('rope', {})
        triaffine_config = model_config.get('triaffine', {})
        dict_config = model_config.get('expert_dict', {})
        
        # 实验模式
        self.ablation = config.get('ablation', {}).get('mode', 'full')
        
        # BERT编码器
        bert_path = encoder_config.get('pretrained', 'hfl/chinese-macbert-base')
        try:
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(bert_path)
            self.bert_hidden = self.bert.config.hidden_size
        except Exception as e:
            print(f"BERT加载失败: {e}")
            self.bert = None
            self.bert_hidden = 768
        
        # 相对位置编码 (RoPE)
        self.use_rope = rope_config.get('enabled', True) and self.ablation not in ['baseline', 'no_rope']
        if self.use_rope:
            self.rope = RotaryPositionEmbedding(
                hidden_dim=self.bert_hidden,
                max_seq_len=rope_config.get('max_seq_len', 512),
                base=rope_config.get('base', 10000)
            )
            self.rope_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # BiLSTM
        self.bilstm = ResBiLSTM(
            in_dim=self.bert_hidden,
            hid_dim=bilstm_config.get('hid_dim', 256),
            num_layers=bilstm_config.get('num_layers', 1),
            dropout=bilstm_config.get('dropout', 0.3),
            residual=True
        )
        
        # 当前特征维度
        current_dim = self.bilstm.out_dim
        
        # 三仿射词典融合
        self.use_triaffine = triaffine_config.get('enabled', True) and self.ablation not in ['baseline', 'no_triaffine']
        if self.use_triaffine:
            self.triaffine = TriAffineDictFusion(
                hidden_dim=current_dim,
                dict_dim=triaffine_config.get('dict_dim', 64),
                num_dict_types=triaffine_config.get('num_dict_types', 10),
                dropout=triaffine_config.get('dropout', 0.3)
            )
        
        # 词典特征提取器
        self.use_dict = dict_config.get('enabled', True) and self.ablation != 'baseline'
        if self.use_dict:
            self.dict_extractor = DictFeatureExtractor(
                dict_path=dict_config.get('dict_path'),
                dict_dim=dict_config.get('dict_dim', 64),
                num_types=14  # RedJujube有14种实体类型
            )
        
        # CRF解码器
        decoder_config = model_config.get('decoder', {})
        if decoder_config.get('type') == 'crf':
            try:
                from eznlp.nn.modules import CRF
                self.num_labels = 14 * 4 + 1  # BIOES: 14 types * 4 + O
                self.crf = CRF(self.num_labels, pad_id=0)
                self.use_crf = True
            except ImportError:
                self.use_crf = False
                self.classifier = nn.Linear(current_dim, 14 * 4 + 1)
        else:
            self.use_crf = False
            self.classifier = nn.Linear(current_dim, 14 * 4 + 1)
        
        # 输出投影
        self.output_proj = nn.Linear(current_dim, current_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tokens: Optional[List[List[str]]] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            labels: [batch, seq_len]
            tokens: 用于词典匹配的token列表
            
        Returns:
            dict with 'logits', 'loss' (if labels provided)
        """
        batch_size, seq_len = input_ids.size()
        
        # BERT编码
        if self.bert is not None:
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            hidden_states = bert_outputs.last_hidden_state
        else:
            # 如果BERT不可用，使用随机初始化
            hidden_states = torch.randn(batch_size, seq_len, self.bert_hidden, device=input_ids.device)
        
        # 相对位置编码
        if self.use_rope:
            rope_out = self.rope(hidden_states)
            hidden_states = hidden_states + self.rope_scale * rope_out
        
        # BiLSTM编码
        mask = attention_mask.bool()
        hidden_states = self.bilstm(hidden_states, mask)
        
        # 词典特征提取和融合
        if self.use_dict and self.use_triaffine and tokens is not None:
            dict_feat, dict_type_ids = self.dict_extractor(input_ids, tokens, {})
            hidden_states = self.triaffine(dict_feat, hidden_states, dict_type_ids)
        
        # 输出投影
        hidden_states = self.output_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 解码
        if self.use_crf:
            logits = self.crf(hidden_states, mask)
        else:
            logits = self.classifier(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            if self.use_crf:
                loss = -self.crf(logits, labels, mask)
            else:
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), ignore_index=-1)
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config: dict) -> GTRNNERModel:
    """构建模型"""
    return GTRNNERModel(config)


if __name__ == "__main__":
    # 测试代码
    print("测试 GTRNNERModel...")
    
    # 加载配置
    config_path = __file__.rsplit('/', 1)[0] + '/config.yaml'
    config = load_config(config_path)
    
    # 构建模型
    model = build_model(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = model(input_ids, attention_mask)
    print(f"\n输出:")
    print(f"  logits shape: {outputs['logits'].shape}")
