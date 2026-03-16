# -*- coding: utf-8 -*-
"""
EXP-004-epfd-ner 模型实现
将 EPFD 论文中的 Res-BiLSTM + Self-rectified Gate 迁移到 NER 任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from eznlp.config import Config
from eznlp.nn.init import reinit_lstm_, reinit_layer_


class ResBiLSTMConfig(Config):
    """Res-BiLSTM 配置"""
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop("in_dim", 768)  # BERT hidden size
        self.hid_dim = kwargs.pop("hid_dim", 256)
        self.num_layers = kwargs.pop("num_layers", 1)
        self.dropout = kwargs.pop("dropout", 0.3)
        self.residual = kwargs.pop("residual", True)  # 残差连接
        super().__init__(**kwargs)
    
    @property
    def out_dim(self):
        # 如果使用残差连接，输出维度 = hid_dim + in_dim
        return self.hid_dim + self.in_dim if self.residual else self.hid_dim
    
    def instantiate(self):
        return ResBiLSTM(self)


class ResBiLSTM(nn.Module):
    """
    残差 BiLSTM 模块
    参考: Zhang et al. 2025 - EPFD
    
    o_i = l_i + v_i  (残差连接)
    """
    def __init__(self, config: ResBiLSTMConfig):
        super().__init__()
        self.config = config
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=config.in_dim,
            hidden_size=config.hid_dim // 2,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if config.num_layers <= 1 else config.dropout
        )
        reinit_lstm_(self.lstm)
        
        self.dropout = nn.Dropout(config.dropout)
        self.residual = config.residual
    
    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        """
        Args:
            x: (batch, seq_len, in_dim) - BERT 输出
            mask: (batch, seq_len) - 有效位置掩码
        Returns:
            (batch, seq_len, out_dim)
        """
        # 处理变长序列
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, padding_value=0)
        else:
            lstm_out, _ = self.lstm(x)
        
        lstm_out = self.dropout(lstm_out)
        
        # 残差连接: o_i = l_i + v_i
        if self.residual:
            return torch.cat([lstm_out, x], dim=-1)
        else:
            return lstm_out


class SelfRectifiedGateConfig(Config):
    """Self-rectified Gate 配置"""
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop("in_dim", 1024)  # Res-BiLSTM 输出维度
        self.hid_dim = kwargs.pop("hid_dim", 128)
        self.dropout = kwargs.pop("dropout", 0.2)
        super().__init__(**kwargs)
    
    @property
    def out_dim(self):
        return self.in_dim  # 门控不改变维度
    
    def instantiate(self):
        return SelfRectifiedGate(self)


class SelfRectifiedGate(nn.Module):
    """
    自校正门模块
    参考: Zhang et al. 2025 - EPFD
    
    g = σ(W_4 σ(W_3 c + b_3) + b_4)
    corrected = g * hidden_states
    
    从句子本身生成门控值，用于校正句子表示
    """
    def __init__(self, config: SelfRectifiedGateConfig):
        super().__init__()
        self.config = config
        
        # 门控网络: 两层全连接
        self.gate_fc1 = nn.Linear(config.in_dim, config.hid_dim)
        self.gate_fc2 = nn.Linear(config.hid_dim, config.in_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        reinit_layer_(self.gate_fc1, "linear")
        reinit_layer_(self.gate_fc2, "linear")
    
    def forward(self, hidden_states: torch.Tensor, mask: torch.BoolTensor = None):
        """
        Args:
            hidden_states: (batch, seq_len, in_dim)
            mask: (batch, seq_len)
        Returns:
            corrected: (batch, seq_len, in_dim) - 校正后的隐状态
            gate: (batch, seq_len, in_dim) - 门控值
        """
        # 生成门控向量
        gate = F.relu(self.gate_fc1(hidden_states))
        gate = self.dropout(gate)
        gate = torch.sigmoid(self.gate_fc2(gate))
        
        # 校正隐状态
        corrected = gate * hidden_states
        
        return corrected, gate


class EntityFeatureConfig(Config):
    """实体特征增强配置"""
    def __init__(self, **kwargs):
        self.word_dim = kwargs.pop("word_dim", 768)
        self.char_dim = kwargs.pop("char_dim", 50)
        self.dict_dim = kwargs.pop("dict_dim", 64)  # 词典类型嵌入维度
        self.num_dict_types = kwargs.pop("num_dict_types", 10)  # 词典类型数量
        self.dropout = kwargs.pop("dropout", 0.3)
        super().__init__(**kwargs)
    
    @property
    def out_dim(self):
        return self.word_dim + self.dict_dim
    
    def instantiate(self):
        return EntityFeatureEnhancer(self)


class EntityFeatureEnhancer(nn.Module):
    """
    实体特征增强模块
    参考: Zhang et al. 2025 - EPFD
    
    e_entity = e_ep ⊙ e_type'  (Hadamard product)
    """
    def __init__(self, config: EntityFeatureConfig):
        super().__init__()
        self.config = config
        
        # 词典类型嵌入
        self.dict_type_embed = nn.Embedding(config.num_dict_types + 1, config.dict_dim, padding_idx=0)
        
        # 类型特征变换 (参考论文中的 MLP)
        self.type_fc = nn.Sequential(
            nn.Linear(config.dict_dim, config.dict_dim),
            nn.ReLU(),
            nn.Linear(config.dict_dim, config.dict_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, word_embeds: torch.Tensor, dict_type_ids: torch.Tensor = None):
        """
        Args:
            word_embeds: (batch, seq_len, word_dim)
            dict_type_ids: (batch, seq_len) - 词典匹配类型 ID
        Returns:
            (batch, seq_len, word_dim + dict_dim)
        """
        batch_size, seq_len, _ = word_embeds.shape
        
        if dict_type_ids is not None:
            # 获取词典类型嵌入
            dict_embeds = self.dict_type_embed(dict_type_ids)  # (batch, seq_len, dict_dim)
            
            # 类型特征变换
            type_features = self.type_fc(dict_embeds)
            
            # 与词嵌入拼接
            enhanced = torch.cat([word_embeds, type_features], dim=-1)
        else:
            # 无词典特征时，使用零向量
            zeros = torch.zeros(batch_size, seq_len, self.config.dict_dim, device=word_embeds.device)
            enhanced = torch.cat([word_embeds, zeros], dim=-1)
        
        return self.dropout(enhanced)


class EPFDNERConfig(Config):
    """EPFD-NER 完整模型配置"""
    def __init__(self, **kwargs):
        # BERT 配置
        self.bert_pretrained = kwargs.pop("bert_pretrained", None)
        self.bert_freeze = kwargs.pop("bert_freeze", False)
        
        # Res-BiLSTM
        self.res_bilstm = ResBiLSTMConfig(**kwargs.pop("res_bilstm", {}))
        
        # Self-rectified Gate
        self.self_gate = SelfRectifiedGateConfig(**kwargs.pop("self_gate", {}))
        
        # Entity Feature
        self.entity_feature = EntityFeatureConfig(**kwargs.pop("entity_feature", {}))
        
        # CRF 解码器
        self.num_labels = kwargs.pop("num_labels", 10)
        
        # 消融模式
        self.ablation = kwargs.pop("ablation", "full")  # full, no_res_bilstm, no_self_gate, no_entity_feature
        
        super().__init__(**kwargs)
    
    def instantiate(self):
        return EPFDNERModel(self)


class EPFDNERModel(nn.Module):
    """
    EPFD-NER 完整模型
    
    架构: BERT -> Res-BiLSTM -> Self-rectified Gate -> CRF
    """
    def __init__(self, config: EPFDNERConfig):
        super().__init__()
        self.config = config
        self.ablation = config.ablation
        
        # BERT 编码器 (延迟加载，由训练脚本注入)
        self.bert = None
        
        # Res-BiLSTM
        if "no_res_bilstm" not in self.ablation:
            self.res_bilstm = ResBiLSTM(config.res_bilstm)
            lstm_out_dim = config.res_bilstm.out_dim
        else:
            self.res_bilstm = None
            lstm_out_dim = 768  # BERT hidden size
        
        # Self-rectified Gate
        if "no_self_gate" not in self.ablation:
            config.self_gate.in_dim = lstm_out_dim
            self.self_gate = SelfRectifiedGate(config.self_gate)
        
        # Entity Feature Enhancer
        if "no_entity_feature" not in self.ablation:
            self.entity_feature = EntityFeatureEnhancer(config.entity_feature)
        
        # 分类层
        self.classifier = nn.Linear(lstm_out_dim, config.num_labels)
        
        # CRF (使用 eznlp 内置)
        self.crf = None  # 延迟初始化
    
    def set_bert(self, bert_model):
        """设置 BERT 模型"""
        self.bert = bert_model
        if self.config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def set_crf(self, crf_module):
        """设置 CRF 解码器"""
        self.crf = crf_module
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                dict_type_ids: torch.Tensor = None,
                labels: torch.Tensor = None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            dict_type_ids: (batch, seq_len) - 词典匹配类型
            labels: (batch, seq_len) - 标签 ID
        Returns:
            如果 labels 不为 None，返回 loss
            否则返回预测结果
        """
        # 1. BERT 编码
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # 2. Res-BiLSTM
        if self.res_bilstm is not None:
            hidden_states = self.res_bilstm(hidden_states, attention_mask.bool())
        
        # 3. Self-rectified Gate
        if hasattr(self, 'self_gate') and self.self_gate is not None:
            hidden_states, gate = self.self_gate(hidden_states, attention_mask.bool())
        
        # 4. 分类
        logits = self.classifier(hidden_states)  # (batch, seq_len, num_labels)
        
        # 5. CRF 解码
        if self.crf is not None:
            # CRF 的 mask 语义：True 表示 padding 位置，False 表示有效位置
            # 我们的 attention_mask：True 表示有效位置，False 表示 padding
            # 所以需要取反
            crf_mask = ~attention_mask.bool()
            
            if labels is not None:
                # 训练模式：计算 loss
                # CRF.forward 返回的是 NLL (negative log-likelihood)，形状为 (batch,)
                nll = self.crf(logits, labels, mask=crf_mask)
                loss = nll.mean()  # 取平均得到标量 loss
                return {"loss": loss}
            else:
                # 推理模式：解码
                predictions = self.crf.decode(logits, mask=crf_mask)
                return {"predictions": predictions}
        else:
            # 无 CRF，使用 softmax
            if labels is not None:
                # 使用 cross entropy loss
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = F.cross_entropy(active_logits, active_labels)
                return {"loss": loss}
            else:
                predictions = logits.argmax(dim=-1)
                return {"predictions": predictions.tolist()}
    
    def get_gate_values(self, hidden_states: torch.Tensor, mask: torch.BoolTensor = None):
        """获取门控值（用于可视化分析）"""
        if hasattr(self, 'self_gate') and self.self_gate is not None:
            _, gate = self.self_gate(hidden_states, mask)
            return gate
        return None
