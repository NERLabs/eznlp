# -*- coding: utf-8 -*-
"""
EXP-005-ee-resbilstm 模型实现
将 EPFD 论文中的 Res-BiLSTM + Self-rectified Gate 完整迁移到 NER 任务
支持完整消融实验

参考论文: Zhang et al. 2025 - Improving distant supervised relation extraction 
         with entity enhanced res-BiLSTM and self-rectified gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Dict, Tuple, List

from eznlp.config import Config
from eznlp.nn.init import reinit_lstm_, reinit_layer_


class ResBiLSTMConfig(Config):
    """Res-BiLSTM 配置"""
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop("in_dim", 768)  # BERT hidden size
        self.hid_dim = kwargs.pop("hid_dim", 256)
        self.num_layers = kwargs.pop("num_layers", 1)
        self.dropout = kwargs.pop("dropout", 0.3)
        self.residual = kwargs.pop("residual", True)  # 残差连接开关
        self.enabled = kwargs.pop("enabled", True)
        super().__init__(**kwargs)
    
    @property
    def out_dim(self):
        if not self.enabled:
            return self.in_dim
        # 如果使用残差连接，输出维度 = hid_dim + in_dim
        return self.hid_dim + self.in_dim if self.residual else self.hid_dim
    
    def instantiate(self):
        return ResBiLSTM(self)


class ResBiLSTM(nn.Module):
    """
    残差 BiLSTM 模块
    参考: Zhang et al. 2025 - EPFD
    
    原论文: o_i = l_i + v_i (残差连接)
    本实现: 使用拼接方式保留完整特征
    """
    def __init__(self, config: ResBiLSTMConfig):
        super().__init__()
        self.config = config
        
        if not config.enabled:
            self.lstm = None
            return
        
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
    
    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_dim) - BERT 输出
            mask: (batch, seq_len) - 有效位置掩码
        Returns:
            (batch, seq_len, out_dim)
        """
        if self.lstm is None:
            return x
        
        # 处理变长序列
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, padding_value=0)
        else:
            lstm_out, _ = self.lstm(x)
        
        lstm_out = self.dropout(lstm_out)
        
        # 残差连接: 拼接而非相加（维度不同时更稳定）
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
        self.enabled = kwargs.pop("enabled", True)
        self.use_boundary = kwargs.pop("use_boundary", True)  # 是否使用边界位置特征
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
    
    原论文方法:
        g = σ(W_4 σ(W_3 c + b_3) + b_4)
        corrected = g * hidden_states
    
    其中 c 是从 Res-BiLSTM 提取的三处隐状态拼接：
    - 末态隐状态
    - 头实体位置隐状态
    - 尾实体位置隐状态
    
    NER 适配:
    - 使用全局池化替代固定位置提取
    - 可选使用词典匹配边界位置
    """
    def __init__(self, config: SelfRectifiedGateConfig):
        super().__init__()
        self.config = config
        
        if not config.enabled:
            self.gate_fc1 = None
            return
        
        # 门控网络: 两层全连接
        # 输入: 全局特征 (末态 + 平均池化 + 最大池化)
        gate_input_dim = config.in_dim * 3 if config.use_boundary else config.in_dim
        self.gate_fc1 = nn.Linear(config.in_dim, config.hid_dim)
        self.gate_fc2 = nn.Linear(config.hid_dim, config.in_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        reinit_layer_(self.gate_fc1, "linear")
        reinit_layer_(self.gate_fc2, "linear")
    
    def forward(self, hidden_states: torch.Tensor, 
                mask: torch.BoolTensor = None,
                boundary_positions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, in_dim)
            mask: (batch, seq_len)
            boundary_positions: (batch, 2) - 可选的边界位置 [start, end]
        Returns:
            corrected: (batch, seq_len, in_dim) - 校正后的隐状态
            gate: (batch, seq_len, in_dim) - 门控值
        """
        if self.gate_fc1 is None:
            return hidden_states, None
        
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
        self.dict_dim = kwargs.pop("dict_embed_size", 64)  # 词典嵌入维度
        self.type_dim = kwargs.pop("type_embed_size", 32)  # 类型嵌入维度
        self.num_dict_types = kwargs.pop("num_dict_types", 10)  # 词典类型数量
        self.dropout = kwargs.pop("dropout", 0.3)
        self.enabled = kwargs.pop("enabled", True)
        super().__init__(**kwargs)
    
    @property
    def out_dim(self):
        if not self.enabled:
            return self.word_dim
        return self.word_dim + self.dict_dim + self.type_dim
    
    def instantiate(self):
        return EntityFeatureEnhancer(self)


class EntityFeatureEnhancer(nn.Module):
    """
    实体特征增强模块
    参考: Zhang et al. 2025 - EPFD
    
    原论文方法:
        e_entity = e_ep ⊙ e_type'  (Hadamard product)
        其中 e_ep = e_h ⊙ e_t (实体对特征)
    
    NER 适配:
        - 词典匹配特征替代实体对嵌入
        - 词典类型嵌入替代外部知识库类型
    """
    def __init__(self, config: EntityFeatureConfig):
        super().__init__()
        self.config = config
        
        if not config.enabled:
            return
        
        # 词典类型嵌入
        self.dict_type_embed = nn.Embedding(
            config.num_dict_types + 1,  # +1 for padding
            config.type_dim, 
            padding_idx=0
        )
        
        # 词典匹配嵌入 (二值: 是否匹配词典)
        self.dict_match_embed = nn.Embedding(2, config.dict_dim)
        
        # 类型特征变换 (参考论文中的 MLP)
        self.type_fc = nn.Sequential(
            nn.Linear(config.type_dim, config.type_dim),
            nn.ReLU(),
            nn.Linear(config.type_dim, config.type_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, word_embeds: torch.Tensor, 
                dict_type_ids: torch.Tensor = None,
                dict_match_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            word_embeds: (batch, seq_len, word_dim)
            dict_type_ids: (batch, seq_len) - 词典匹配类型 ID
            dict_match_ids: (batch, seq_len) - 是否匹配词典 (0/1)
        Returns:
            (batch, seq_len, out_dim)
        """
        if not self.config.enabled:
            return word_embeds
        
        batch_size, seq_len, _ = word_embeds.shape
        device = word_embeds.device
        
        # 初始化特征
        type_features = torch.zeros(batch_size, seq_len, self.config.type_dim, device=device)
        dict_features = torch.zeros(batch_size, seq_len, self.config.dict_dim, device=device)
        
        if dict_type_ids is not None:
            # 获取词典类型嵌入
            type_embeds = self.dict_type_embed(dict_type_ids)
            type_features = self.type_fc(type_embeds)
        
        if dict_match_ids is not None:
            # 获取词典匹配嵌入
            dict_features = self.dict_match_embed(dict_match_ids)
        
        # 与词嵌入拼接
        enhanced = torch.cat([word_embeds, dict_features, type_features], dim=-1)
        
        return self.dropout(enhanced)


class EEPFDNERConfig(Config):
    """EE-ResBiLSTM-NER 完整模型配置"""
    def __init__(self, **kwargs):
        # BERT 配置
        self.bert_pretrained = kwargs.pop("bert_pretrained", None)
        self.bert_freeze = kwargs.pop("bert_freeze", False)
        self.bert_hidden_size = kwargs.pop("bert_hidden_size", 768)
        
        # Res-BiLSTM
        res_bilstm_config = kwargs.pop("res_bilstm", {})
        res_bilstm_config["in_dim"] = self.bert_hidden_size
        self.res_bilstm = ResBiLSTMConfig(**res_bilstm_config)
        
        # Self-rectified Gate
        self_gate_config = kwargs.pop("self_gate", {})
        self_gate_config["in_dim"] = self.res_bilstm.out_dim
        self.self_gate = SelfRectifiedGateConfig(**self_gate_config)
        
        # Entity Feature
        entity_feature_config = kwargs.pop("entity_feature", {})
        entity_feature_config["word_dim"] = self.res_bilstm.out_dim
        self.entity_feature = EntityFeatureConfig(**entity_feature_config)
        
        # CRF 解码器
        self.num_labels = kwargs.pop("num_labels", 10)
        
        # 消融模式
        self.ablation = kwargs.pop("ablation", "full")
        
        super().__init__(**kwargs)
        
        # 根据消融模式调整配置
        self._apply_ablation()
    
    def _apply_ablation(self):
        """根据消融模式调整配置"""
        if self.ablation == "baseline":
            # 仅 BERT + CRF
            self.res_bilstm.enabled = False
            self.self_gate.enabled = False
            self.entity_feature.enabled = False
        elif self.ablation == "no_residual":
            # 无残差连接
            self.res_bilstm.residual = False
            self.self_gate.enabled = False
            self.entity_feature.enabled = False
        elif self.ablation == "no_srg":
            # 无 Self-rectified Gate
            self.self_gate.enabled = False
        elif self.ablation == "no_dict" or self.ablation == "no_entity_feature":
            # 无词典特征
            self.entity_feature.enabled = False
    
    @property
    def hidden_size(self):
        """最终隐藏层维度"""
        if not self.res_bilstm.enabled:
            return self.bert_hidden_size
        if self.entity_feature.enabled:
            return self.entity_feature.out_dim
        return self.res_bilstm.out_dim
    
    def instantiate(self):
        return EEPFDNERModel(self)


class EEPFDNERModel(nn.Module):
    """
    EE-ResBiLSTM-NER 完整模型
    
    架构: BERT -> Res-BiLSTM -> Self-rectified Gate -> Entity Feature -> CRF
    
    消融模式:
        - baseline: BERT -> CRF
        - no_residual: BERT -> BiLSTM (无残差) -> CRF
        - no_srg: BERT -> Res-BiLSTM -> CRF
        - no_dict: BERT -> Res-BiLSTM -> SRG -> CRF
        - full: BERT -> Res-BiLSTM -> SRG -> Entity Feature -> CRF
    """
    def __init__(self, config: EEPFDNERConfig):
        super().__init__()
        self.config = config
        self.ablation = config.ablation
        
        # BERT 编码器 (延迟加载，由训练脚本注入)
        self.bert = None
        
        # Res-BiLSTM
        self.res_bilstm = ResBiLSTM(config.res_bilstm)
        
        # Self-rectified Gate
        self.self_gate = SelfRectifiedGate(config.self_gate)
        
        # Entity Feature Enhancer
        self.entity_feature = EntityFeatureEnhancer(config.entity_feature)
        
        # 分类层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # CRF (使用 eznlp 内置)
        self.crf = None  # 延迟初始化
        
        # 保存门控值用于分析
        self.last_gate_values = None
    
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
                dict_match_ids: torch.Tensor = None,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            dict_type_ids: (batch, seq_len) - 词典匹配类型
            dict_match_ids: (batch, seq_len) - 是否匹配词典
            labels: (batch, seq_len) - 标签 ID
        Returns:
            如果 labels 不为 None，返回 {"loss": loss}
            否则返回 {"predictions": predictions, "gate": gate}
        """
        # 1. BERT 编码
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # 2. Res-BiLSTM
        hidden_states = self.res_bilstm(hidden_states, attention_mask.bool())
        
        # 3. Self-rectified Gate
        hidden_states, gate = self.self_gate(hidden_states, attention_mask.bool())
        if gate is not None:
            self.last_gate_values = gate.detach()
        
        # 4. Entity Feature Enhancement
        hidden_states = self.entity_feature(hidden_states, dict_type_ids, dict_match_ids)
        
        # 5. 分类
        logits = self.classifier(hidden_states)  # (batch, seq_len, num_labels)
        
        # 6. CRF 解码
        if self.crf is not None:
            if labels is not None:
                # 训练模式：计算 loss
                loss = self.crf(logits, labels, mask=attention_mask.bool())
                return {"loss": -loss}  # CRF 返回的是 log likelihood，取负作为 loss
            else:
                # 推理模式：解码
                predictions = self.crf.decode(logits, mask=attention_mask.bool())
                return {"predictions": predictions, "gate": self.last_gate_values}
        else:
            # 无 CRF，使用 softmax
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1), 
                    ignore_index=-100
                )
                return {"loss": loss}
            else:
                predictions = logits.argmax(dim=-1)
                return {"predictions": predictions, "gate": self.last_gate_values}
    
    def get_gate_values(self) -> Optional[torch.Tensor]:
        """获取最近的门控值（用于可视化分析）"""
        return self.last_gate_values
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "ablation_mode": self.ablation,
            "res_bilstm_enabled": self.res_bilstm.config.enabled,
            "res_bilstm_residual": self.res_bilstm.config.residual,
            "self_gate_enabled": self.self_gate.config.enabled,
            "entity_feature_enabled": self.entity_feature.config.enabled,
            "hidden_size": self.config.hidden_size,
        }


def create_model(config: Dict, num_labels: int, device: torch.device) -> EEPFDNERModel:
    """
    工厂函数：创建模型
    
    Args:
        config: 配置字典
        num_labels: 标签数量
        device: 设备
    Returns:
        EEPFDNERModel 实例
    """
    model_config = config.get('model', {})
    ablation_mode = config.get('ablation', {}).get('mode', 'full')
    
    epfd_config = EEPFDNERConfig(
        bert_pretrained=model_config.get('encoder', {}).get('pretrained'),
        bert_freeze=model_config.get('encoder', {}).get('freeze', False),
        bert_hidden_size=model_config.get('encoder', {}).get('hidden_size', 768),
        res_bilstm=model_config.get('res_bilstm', {}),
        self_gate=model_config.get('self_gate', {}),
        entity_feature=model_config.get('entity_feature', {}),
        num_labels=num_labels,
        ablation=ablation_mode
    )
    
    model = epfd_config.instantiate()
    
    return model.to(device)
