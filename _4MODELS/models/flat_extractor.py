#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的 FLAT 模型实现

参考: projects/Flat-Lattice-Transformer-master/V1/models.py

核心架构:
1. Lattice Embedding: 字符 + 词汇的统一嵌入
2. Absolute SE Position Encoding: 基于开始/结束位置的绝对位置编码
3. FLAT Transformer Encoder: 支持四位置融合的 Transformer
4. CRF Decoder: 条件随机场解码
"""

import torch
import torch.nn as nn
import math
import copy
from typing import Optional, Dict, List, Tuple

# 导入已实现的组件
try:
    from _4MODELS.block.position_encoding import FourPositionFusion
    from _4MODELS.block.lattice_attention import LatticeSelfAttention, TransformerEncoderLayer
    from _4MODELS.block.lattice_utils import LayerProcess, PositionwiseFeedForward
except ImportError:
    from ..block.position_encoding import FourPositionFusion
    from ..block.lattice_attention import LatticeSelfAttention, TransformerEncoderLayer
    from ..block.lattice_utils import LayerProcess, PositionwiseFeedForward


def get_sinusoidal_embedding(max_seq_len: int, embedding_dim: int, rel_pos_init: int = 0):
    """生成正弦位置编码
    
    Args:
        max_seq_len: 最大序列长度
        embedding_dim: 嵌入维度
        rel_pos_init: 相对位置初始化方式，0 表示从 0 开始，1 表示从 -max_len 开始
        
    Returns:
        位置编码张量 [2*max_seq_len+1, embedding_dim]
    """
    num_embeddings = 2 * max_seq_len + 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len, max_seq_len + 1, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    
    return emb


class AbsoluteSEPositionEmbedding(nn.Module):
    """基于开始/结束位置的绝对位置编码
    
    支持多种融合方式：add, concat, nonlinear_concat 等
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_len: int = 512,
        fusion_func: str = 'add',
        learnable: bool = False,
        pos_norm: bool = False
    ):
        super().__init__()
        self.fusion_func = fusion_func
        self.hidden_size = hidden_size
        self.pos_norm = pos_norm
        
        # 生成位置编码
        pe = get_sinusoidal_embedding(max_len, hidden_size)
        pe_sum = pe.sum(dim=-1, keepdim=True)
        if pos_norm:
            with torch.no_grad():
                pe = pe / pe_sum
        
        # 开始位置和结束位置的编码
        self.pe_s = nn.Parameter(pe.clone(), requires_grad=learnable)
        self.pe_e = nn.Parameter(pe.clone(), requires_grad=learnable)
        
        # 不同融合方式的投影层
        if fusion_func == 'concat':
            self.proj = nn.Linear(hidden_size * 3, hidden_size)
        elif fusion_func == 'nonlinear_concat':
            self.pos_proj = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            self.proj = nn.Linear(hidden_size * 2, hidden_size)
        elif fusion_func == 'nonlinear_add':
            self.pos_proj = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
    
    def forward(self, inp: torch.Tensor, pos_s: torch.Tensor, pos_e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inp: 输入嵌入 [batch, seq_len, hidden_size]
            pos_s: 开始位置索引 [batch, seq_len]
            pos_e: 结束位置索引 [batch, seq_len]
            
        Returns:
            融合位置信息后的嵌入 [batch, seq_len, hidden_size]
        """
        batch = inp.size(0)
        max_len = inp.size(1)
        
        # 获取位置编码
        pe_s = self.pe_s.index_select(0, pos_s.view(-1)).view(batch, max_len, -1)
        pe_e = self.pe_e.index_select(0, pos_e.view(-1)).view(batch, max_len, -1)
        
        if self.fusion_func == 'add':
            output = inp + pe_s + pe_e
        elif self.fusion_func == 'concat':
            output = self.proj(torch.cat([inp, pe_s, pe_e], dim=-1))
        elif self.fusion_func == 'nonlinear_concat':
            pos = self.pos_proj(torch.cat([pe_s, pe_e], dim=-1))
            output = self.proj(torch.cat([inp, pos], dim=-1))
        elif self.fusion_func == 'nonlinear_add':
            pos = self.pos_proj(torch.cat([pe_s, pe_e], dim=-1))
            output = inp + pos
        else:
            output = inp + pe_s + pe_e
        
        return output


class FLATTransformerEncoder(nn.Module):
    """FLAT Transformer 编码器
    
    支持四位置相对位置编码的 Transformer 编码器
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_size: int = -1,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        four_pos_fusion: str = 'ff',
        learnable_position: bool = False,
        four_pos_shared: bool = True,
        pos_norm: bool = False,
        rel_pos_init: int = 0,
        ff_activate: str = 'relu',
        scaled: bool = True,
        attn_ff: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.four_pos_fusion = four_pos_fusion
        
        if ff_size == -1:
            ff_size = hidden_size * 4
        
        # 相对位置编码（四个方向：ss, se, es, ee）
        pe = get_sinusoidal_embedding(max_seq_len, hidden_size, rel_pos_init)
        pe_sum = pe.sum(dim=-1, keepdim=True)
        if pos_norm:
            with torch.no_grad():
                pe = pe / pe_sum
        
        self.pe = nn.Parameter(pe, requires_grad=learnable_position)
        
        if four_pos_shared:
            self.pe_ss = self.pe
            self.pe_se = self.pe
            self.pe_es = self.pe
            self.pe_ee = self.pe
        else:
            self.pe_ss = nn.Parameter(pe.clone(), requires_grad=learnable_position)
            self.pe_se = nn.Parameter(pe.clone(), requires_grad=learnable_position)
            self.pe_es = nn.Parameter(pe.clone(), requires_grad=learnable_position)
            self.pe_ee = nn.Parameter(pe.clone(), requires_grad=learnable_position)
        
        # 堆叠 Transformer 层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                max_len=max_seq_len,
                ff_size=ff_size,
                dropout=dropout,
                activation=ff_activate
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        hidden: torch.Tensor,
        seq_len: torch.Tensor,
        lex_num: torch.Tensor,
        pos_s: torch.Tensor,
        pos_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden: 输入隐藏状态 [batch, seq_len+lex_num, hidden_size]
            seq_len: 字符序列长度 [batch]
            lex_num: 词汇数量 [batch]
            pos_s: 开始位置 [batch, seq_len+lex_num]
            pos_e: 结束位置 [batch, seq_len+lex_num]
            
        Returns:
            编码后的隐藏状态 [batch, seq_len+lex_num, hidden_size]
        """
        batch_size = hidden.size(0)
        max_len = hidden.size(1)
        
        # 创建掩码
        total_len = seq_len + lex_num
        mask = torch.arange(max_len, device=hidden.device).unsqueeze(0) < total_len.unsqueeze(1)
        
        # 逐层编码
        for layer in self.layers:
            hidden = layer(hidden, pos_s, pos_e, mask)
        
        return hidden


class FLATModel(nn.Module):
    """完整的 FLAT 模型
    
    架构：Lattice Embedding + SE Position + FLAT Encoder + CRF
    """
    
    def __init__(
        self,
        vocab_size: int,
        label_size: int,
        hidden_size: int = 256,
        embed_size: int = 50,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_size: int = -1,
        max_seq_len: int = 512,
        dropout: float = 0.15,
        use_bigram: bool = True,
        bigram_vocab_size: int = 0,
        bigram_embed_size: int = 50,
        use_bert: bool = False,
        bert_hidden_size: int = 768,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        four_pos_fusion: str = 'ff',
        learnable_position: bool = False,
        abs_pos_fusion_func: str = 'concat',
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.use_bigram = use_bigram
        self.use_bert = use_bert
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.max_seq_len = max_seq_len
        
        # Lattice 嵌入（字符 + 词汇共享）
        self.lattice_embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Bigram 嵌入
        if use_bigram and bigram_vocab_size > 0:
            self.bigram_embed = nn.Embedding(bigram_vocab_size, bigram_embed_size, padding_idx=0)
            char_input_size = embed_size + bigram_embed_size
        else:
            self.bigram_embed = None
            char_input_size = embed_size
        
        # BERT 嵌入（可选）
        if use_bert:
            self.bert_proj = nn.Linear(bert_hidden_size, hidden_size)
            char_input_size += bert_hidden_size
        
        # 词汇输入维度
        lex_input_size = embed_size
        
        # 投影层
        self.char_proj = nn.Linear(char_input_size, hidden_size)
        self.lex_proj = nn.Linear(lex_input_size, hidden_size)
        
        # Dropout
        self.embed_dropout = nn.Dropout(dropout)
        
        # 绝对位置编码
        if use_abs_pos:
            self.abs_pos_encode = AbsoluteSEPositionEmbedding(
                hidden_size=hidden_size,
                max_len=max_seq_len,
                fusion_func=abs_pos_fusion_func,
                learnable=learnable_position
            )
        
        # FLAT Transformer 编码器
        self.encoder = FLATTransformerEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_size=ff_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            four_pos_fusion=four_pos_fusion,
            learnable_position=learnable_position,
        )
        
        # 输出层
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, label_size)
        
        # CRF
        from torchcrf import CRF
        self.crf = CRF(label_size, batch_first=True)
    
    def forward(
        self,
        lattice: torch.Tensor,
        seq_len: torch.Tensor,
        lex_num: torch.Tensor,
        pos_s: torch.Tensor,
        pos_e: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        bigrams: Optional[torch.Tensor] = None,
        bert_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            lattice: Lattice 序列索引 [batch, seq_len+lex_num]
            seq_len: 字符序列长度 [batch]
            lex_num: 词汇数量 [batch]
            pos_s: 开始位置 [batch, seq_len+lex_num]
            pos_e: 结束位置 [batch, seq_len+lex_num]
            target: 目标标签 [batch, seq_len]（可选）
            bigrams: Bigram 索引 [batch, seq_len]（可选）
            bert_embed: BERT 嵌入 [batch, seq_len, bert_hidden_size]（可选）
            
        Returns:
            训练时返回 {'loss': loss}，推理时返回 {'pred': pred}
        """
        batch_size = lattice.size(0)
        max_lattice_len = lattice.size(1)
        max_seq_len = seq_len.max().item()
        
        # 1. 获取 Lattice 嵌入
        raw_embed = self.lattice_embed(lattice)
        
        # 2. 处理字符嵌入（可能包含 bigram 和 BERT）
        if self.use_bigram and self.bigram_embed is not None and bigrams is not None:
            bigram_embed = self.bigram_embed(bigrams)
            # 扩展 bigram 到 lattice 长度（词汇部分填 0）
            bigram_pad = torch.zeros(
                batch_size, max_lattice_len - max_seq_len, bigram_embed.size(-1),
                device=bigram_embed.device
            )
            bigram_embed = torch.cat([bigram_embed, bigram_pad], dim=1)
            raw_embed_char = torch.cat([raw_embed, bigram_embed], dim=-1)
        else:
            raw_embed_char = raw_embed
        
        # BERT 嵌入
        if self.use_bert and bert_embed is not None:
            bert_pad = torch.zeros(
                batch_size, max_lattice_len - max_seq_len, bert_embed.size(-1),
                device=bert_embed.device
            )
            bert_embed = torch.cat([bert_embed, bert_pad], dim=1)
            raw_embed_char = torch.cat([raw_embed_char, bert_embed], dim=-1)
        
        # 3. 投影到隐藏维度
        raw_embed_char = self.embed_dropout(raw_embed_char)
        raw_embed = self.embed_dropout(raw_embed)
        
        embed_char = self.char_proj(raw_embed_char)
        embed_lex = self.lex_proj(raw_embed)
        
        # 4. 创建掩码并应用
        # 字符掩码
        char_mask = torch.arange(max_lattice_len, device=lattice.device).unsqueeze(0) < seq_len.unsqueeze(1)
        embed_char = embed_char.masked_fill(~char_mask.unsqueeze(-1), 0)
        
        # 词汇掩码（在字符之后，词汇数量之内）
        total_len = seq_len + lex_num
        lex_mask = (torch.arange(max_lattice_len, device=lattice.device).unsqueeze(0) >= seq_len.unsqueeze(1)) & \
                   (torch.arange(max_lattice_len, device=lattice.device).unsqueeze(0) < total_len.unsqueeze(1))
        embed_lex = embed_lex.masked_fill(~lex_mask.unsqueeze(-1), 0)
        
        # 5. 融合字符和词汇嵌入
        embedding = embed_char + embed_lex
        
        # 6. 添加绝对位置编码
        if self.use_abs_pos:
            embedding = self.abs_pos_encode(embedding, pos_s, pos_e)
        
        # 7. Transformer 编码
        encoded = self.encoder(embedding, seq_len, lex_num, pos_s, pos_e)
        
        # 8. 只取字符部分输出
        encoded = encoded[:, :max_seq_len, :]
        encoded = self.output_dropout(encoded)
        
        # 9. 输出层
        emissions = self.output(encoded)
        
        # 10. 创建字符序列掩码
        mask = torch.arange(max_seq_len, device=lattice.device).unsqueeze(0) < seq_len.unsqueeze(1)
        
        # 11. 训练或推理
        if self.training and target is not None:
            # CRF 负对数似然损失
            loss = -self.crf(emissions, target, mask=mask, reduction='mean')
            return {'loss': loss}
        else:
            # Viterbi 解码
            pred = self.crf.decode(emissions, mask=mask)
            return {'pred': pred}


class FLATModelWithBERT(nn.Module):
    """FLAT + BERT 模型
    
    使用 BERT 作为底层特征提取器，在其上加入 FLAT 编码器
    """
    
    def __init__(
        self,
        bert_model,
        label_size: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_size: int = -1,
        max_seq_len: int = 512,
        dropout: float = 0.15,
        four_pos_fusion: str = 'ff',
        learnable_position: bool = False,
        freeze_bert: bool = False,
    ):
        super().__init__()
        
        self.bert = bert_model
        self.hidden_size = hidden_size
        self.label_size = label_size
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        bert_hidden_size = self.bert.config.hidden_size
        
        # BERT 输出投影
        self.bert_proj = nn.Linear(bert_hidden_size, hidden_size)
        
        # 绝对位置编码
        self.abs_pos_encode = AbsoluteSEPositionEmbedding(
            hidden_size=hidden_size,
            max_len=max_seq_len,
            fusion_func='concat',
            learnable=learnable_position
        )
        
        # FLAT Transformer 编码器
        self.encoder = FLATTransformerEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_size=ff_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            four_pos_fusion=four_pos_fusion,
            learnable_position=learnable_position,
        )
        
        # 输出层
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, label_size)
        
        # CRF
        from torchcrf import CRF
        self.crf = CRF(label_size, batch_first=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_len: torch.Tensor,
        pos_s: torch.Tensor,
        pos_e: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        lex_num: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = input_ids.size(0)
        max_seq_len = input_ids.size(1)
        
        # 1. BERT 编码
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_hidden = bert_output.last_hidden_state
        
        # 2. 投影
        hidden = self.bert_proj(bert_hidden)
        
        # 3. 如果没有词汇信息，lex_num 为 0
        if lex_num is None:
            lex_num = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        
        # 4. 位置编码
        hidden = self.abs_pos_encode(hidden, pos_s, pos_e)
        
        # 5. FLAT 编码
        encoded = self.encoder(hidden, seq_len, lex_num, pos_s, pos_e)
        
        # 6. 输出
        encoded = self.output_dropout(encoded)
        emissions = self.output(encoded)
        
        # 7. 掩码
        mask = attention_mask.bool()
        
        # 8. 训练或推理
        if self.training and target is not None:
            loss = -self.crf(emissions, target, mask=mask, reduction='mean')
            return {'loss': loss}
        else:
            pred = self.crf.decode(emissions, mask=mask)
            return {'pred': pred}


# ============================================================================
# 改进版：FLAT + Inter-Attention (参考 NFLAT 设计)
# ============================================================================

class LatticeInterAttention(nn.Module):
    """
    Lattice 感知的 Inter-Attention 层
    
    核心创新：
    1. BERT 输出作为 Query，词汇信息作为 Key/Value
    2. 加入四种相对位置编码 (ss/se/es/ee)
    3. 单向注意力：字符 → 词汇，计算量 O(n×m)
    
    参考：NFLAT 的 InterFormer 设计
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_rel_pos: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_rel_pos = use_rel_pos
        self.max_seq_len = max_seq_len
        
        # Q/K/V 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # 四种相对位置编码
        if use_rel_pos:
            pe = get_sinusoidal_embedding(max_seq_len, self.head_dim)
            self.register_buffer('pe', pe)
            
            # 四种相对位置的投影
            self.pos_fusion = nn.Sequential(
                nn.Linear(self.head_dim * 4, self.head_dim),
                nn.ReLU(),
                nn.Linear(self.head_dim, 1)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
    
    def _get_rel_pos_bias(self, pos_s_q, pos_e_q, pos_s_k, pos_e_k):
        """计算四种相对位置的偏置分数"""
        batch_size = pos_s_q.size(0)
        len_q = pos_s_q.size(1)
        len_k = pos_s_k.size(1)
        
        # 计算四种相对位置
        # pos_s_q: [B, Lq], pos_s_k: [B, Lk]
        rel_ss = pos_s_q.unsqueeze(2) - pos_s_k.unsqueeze(1)  # [B, Lq, Lk]
        rel_se = pos_s_q.unsqueeze(2) - pos_e_k.unsqueeze(1)
        rel_es = pos_e_q.unsqueeze(2) - pos_s_k.unsqueeze(1)
        rel_ee = pos_e_q.unsqueeze(2) - pos_e_k.unsqueeze(1)
        
        # 限制范围并获取位置编码
        def get_pe(rel_pos):
            rel_pos = rel_pos.clamp(-self.max_seq_len, self.max_seq_len) + self.max_seq_len
            return self.pe[rel_pos.view(-1)].view(batch_size, len_q, len_k, self.head_dim)
        
        pe_ss = get_pe(rel_ss)
        pe_se = get_pe(rel_se)
        pe_es = get_pe(rel_es)
        pe_ee = get_pe(rel_ee)
        
        # 融合四种位置编码
        pos_concat = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)  # [B, Lq, Lk, head_dim*4]
        pos_bias = self.pos_fusion(pos_concat).squeeze(-1)  # [B, Lq, Lk]
        
        return pos_bias
    
    def forward(
        self,
        query: torch.Tensor,        # BERT 输出 [B, Lq, H]
        key_value: torch.Tensor,    # 词汇嵌入 [B, Lk, H]
        pos_s_q: torch.Tensor,      # Query 开始位置 [B, Lq]
        pos_e_q: torch.Tensor,      # Query 结束位置 [B, Lq]
        pos_s_k: torch.Tensor,      # Key 开始位置 [B, Lk]
        pos_e_k: torch.Tensor,      # Key 结束位置 [B, Lk]
        mask: Optional[torch.Tensor] = None,  # [B, Lq, Lk]
    ):
        """
        Inter-Attention: Query (BERT) 查询 Key/Value (词汇)
        """
        batch_size, len_q, _ = query.shape
        len_k = key_value.size(1)
        
        # 投影
        Q = self.q_proj(query).view(batch_size, len_q, self.num_heads, self.head_dim)
        K = self.k_proj(key_value).view(batch_size, len_k, self.num_heads, self.head_dim)
        V = self.v_proj(key_value).view(batch_size, len_k, self.num_heads, self.head_dim)
        
        # 调整维度: [B, num_heads, L, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算注意力分数
        attn_score = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # [B, H, Lq, Lk]
        
        # 添加相对位置偏置
        if self.use_rel_pos:
            pos_bias = self._get_rel_pos_bias(pos_s_q, pos_e_q, pos_s_k, pos_e_k)
            attn_score = attn_score + pos_bias.unsqueeze(1)  # [B, H, Lq, Lk]
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, Lq, Lk]
            attn_score = attn_score.masked_fill(~mask, -1e9)
        
        # Softmax
        attn_weight = torch.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)
        
        # 加权求和
        output = torch.matmul(attn_weight, V)  # [B, H, Lq, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, self.hidden_size)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output


class FLATWithInterAttention(nn.Module):
    """
    FLAT + Inter-Attention 模型 (参考 NFLAT 设计)
    
    核心改进：
    1. 不压缩 BERT 输出，保持 768 维
    2. 用 Inter-Attention 让 BERT 主动查询 Lattice 词汇信息
    3. 保留四种相对位置编码
    4. 残差连接 + LayerNorm 稳定训练
    
    架构：
        BERT (768维)
          ↓
        [InterAttn] ← Lattice 词汇 (768维)
          ↓ (残差 + LayerNorm)
        增强特征 (768维)
          ↓
        [FFN] (可选)
          ↓ (残差 + LayerNorm)
        最终特征 (768维)
          ↓
        CRF 解码
    """
    
    def __init__(
        self,
        vocab_size: int,
        label_size: int,
        hidden_size: int = 768,  # 保持与 BERT 一致
        embed_size: int = 50,
        num_heads: int = 8,
        num_inter_layers: int = 2,  # Inter-Attention 层数
        ff_size: int = -1,
        max_seq_len: int = 512,
        dropout: float = 0.15,
        use_bert: bool = True,
        bert_hidden_size: int = 768,
        use_ffn: bool = True,  # 是否使用 FFN
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.use_bert = use_bert
        self.use_ffn = use_ffn
        self.max_seq_len = max_seq_len
        
        # Lattice 词汇嵌入
        self.lattice_embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # 词汇嵌入投影到 hidden_size（与 BERT 维度对齐）
        self.lex_proj = nn.Linear(embed_size, hidden_size)
        
        # 如果不用 BERT，需要字符嵌入
        if not use_bert:
            self.char_embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.char_proj = nn.Linear(embed_size, hidden_size)
        
        # Dropout
        self.embed_dropout = nn.Dropout(dropout)
        
        # 多层 Inter-Attention
        self.inter_layers = nn.ModuleList([
            LatticeInterAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
                use_rel_pos=True,
            )
            for _ in range(num_inter_layers)
        ])
        
        # LayerNorm
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_inter_layers)
        ])
        
        # FFN (可选)
        if use_ffn:
            if ff_size == -1:
                ff_size = hidden_size * 4
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, ff_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_size, hidden_size),
                nn.Dropout(dropout)
            )
            self.ffn_norm = nn.LayerNorm(hidden_size)
        
        # 输出层
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, label_size)
        
        # CRF
        from torchcrf import CRF
        self.crf = CRF(label_size, batch_first=True)
    
    def forward(
        self,
        lattice: torch.Tensor,
        seq_len: torch.Tensor,
        lex_num: torch.Tensor,
        pos_s: torch.Tensor,
        pos_e: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        bert_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            lattice: Lattice 序列索引 [batch, seq_len+lex_num]
            seq_len: 字符序列长度 [batch]
            lex_num: 词汇数量 [batch]
            pos_s: 开始位置 [batch, seq_len+lex_num]
            pos_e: 结束位置 [batch, seq_len+lex_num]
            target: 目标标签 [batch, seq_len]（可选）
            bert_embed: BERT 嵌入 [batch, seq_len, bert_hidden_size]（可选）
        """
        batch_size = lattice.size(0)
        max_lattice_len = lattice.size(1)
        max_seq_len_batch = seq_len.max().item()
        max_lex_num = lex_num.max().item()
        
        # 1. 获取 Query (BERT 或字符嵌入)
        if self.use_bert and bert_embed is not None:
            # 直接用 BERT 输出，不压缩
            query = bert_embed  # [B, seq_len, 768]
        else:
            # 使用字符嵌入
            char_ids = lattice[:, :max_seq_len_batch]
            query = self.char_embed(char_ids)
            query = self.char_proj(query)
        
        query = self.embed_dropout(query)
        
        # 2. 获取 Key/Value (词汇嵌入)
        # 词汇在 lattice 中的位置是 [seq_len, seq_len+lex_num)
        # 为每个样本提取词汇部分
        lex_embed_list = []
        lex_pos_s_list = []
        lex_pos_e_list = []
        
        for i in range(batch_size):
            s = seq_len[i].item()
            e = s + lex_num[i].item()
            lex_ids = lattice[i, s:e]  # [lex_num_i]
            lex_emb = self.lattice_embed(lex_ids)  # [lex_num_i, embed_size]
            lex_emb = self.lex_proj(lex_emb)  # [lex_num_i, hidden_size]
            lex_embed_list.append(lex_emb)
            lex_pos_s_list.append(pos_s[i, s:e])
            lex_pos_e_list.append(pos_e[i, s:e])
        
        # Padding 到相同长度
        if max_lex_num > 0:
            lex_embed = torch.zeros(batch_size, max_lex_num, self.hidden_size, device=lattice.device)
            lex_pos_s = torch.zeros(batch_size, max_lex_num, dtype=torch.long, device=lattice.device)
            lex_pos_e = torch.zeros(batch_size, max_lex_num, dtype=torch.long, device=lattice.device)
            lex_mask = torch.zeros(batch_size, max_seq_len_batch, max_lex_num, dtype=torch.bool, device=lattice.device)
            
            for i in range(batch_size):
                n = lex_num[i].item()
                if n > 0:
                    lex_embed[i, :n] = lex_embed_list[i]
                    lex_pos_s[i, :n] = lex_pos_s_list[i]
                    lex_pos_e[i, :n] = lex_pos_e_list[i]
                    lex_mask[i, :seq_len[i].item(), :n] = True
        else:
            # 没有词汇匹配，创建空的 key_value
            lex_embed = torch.zeros(batch_size, 1, self.hidden_size, device=lattice.device)
            lex_pos_s = torch.zeros(batch_size, 1, dtype=torch.long, device=lattice.device)
            lex_pos_e = torch.zeros(batch_size, 1, dtype=torch.long, device=lattice.device)
            lex_mask = torch.zeros(batch_size, max_seq_len_batch, 1, dtype=torch.bool, device=lattice.device)
        
        lex_embed = self.embed_dropout(lex_embed)
        
        # 3. Query 的位置信息
        query_pos_s = pos_s[:, :max_seq_len_batch]
        query_pos_e = pos_e[:, :max_seq_len_batch]
        
        # 4. 多层 Inter-Attention
        hidden = query
        for inter_layer, norm in zip(self.inter_layers, self.layer_norms):
            residual = hidden
            attn_output = inter_layer(
                query=hidden,
                key_value=lex_embed,
                pos_s_q=query_pos_s,
                pos_e_q=query_pos_e,
                pos_s_k=lex_pos_s,
                pos_e_k=lex_pos_e,
                mask=lex_mask if max_lex_num > 0 else None
            )
            hidden = norm(attn_output + residual)
        
        # 5. FFN (可选)
        if self.use_ffn:
            residual = hidden
            hidden = self.ffn(hidden)
            hidden = self.ffn_norm(hidden + residual)
        
        # 6. 输出层
        hidden = self.output_dropout(hidden)
        emissions = self.output(hidden)
        
        # 7. 创建字符序列掩码
        mask = torch.arange(max_seq_len_batch, device=lattice.device).unsqueeze(0) < seq_len.unsqueeze(1)
        
        # 8. 训练或推理
        if self.training and target is not None:
            # 截断 target 到实际长度
            target = target[:, :max_seq_len_batch]
            loss = -self.crf(emissions, target, mask=mask, reduction='mean')
            return {'loss': loss}
        else:
            pred = self.crf.decode(emissions, mask=mask)
            return {'pred': pred}


if __name__ == '__main__':
    # 测试代码
    print("测试 FLAT 模型")
    
    # 创建模型
    model = FLATModel(
        vocab_size=5000,
        label_size=15,
        hidden_size=256,
        embed_size=50,
        num_heads=4,
        num_layers=2,
        max_seq_len=256,
        dropout=0.1,
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {total_params:,}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = torch.tensor([10, 8])
    lex_num = torch.tensor([3, 2])
    max_lattice_len = 15
    
    lattice = torch.randint(0, 5000, (batch_size, max_lattice_len))
    pos_s = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
    pos_e = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
    target = torch.randint(0, 15, (batch_size, 10))
    
    # 设置位置
    for i in range(batch_size):
        for j in range(seq_len[i].item()):
            pos_s[i, j] = j
            pos_e[i, j] = j
    
    model.train()
    output = model(lattice, seq_len, lex_num, pos_s, pos_e, target)
    print(f"✓ 训练模式 - Loss: {output['loss'].item():.4f}")
    
    model.eval()
    with torch.no_grad():
        output = model(lattice, seq_len, lex_num, pos_s, pos_e)
    print(f"✓ 推理模式 - 预测结果: {len(output['pred'])} 个序列")
    
    print("\n✓ FLAT 模型测试通过!")
