# NFLAT代码深度解析

## 📚 目录
1. [核心架构](#核心架构)
2. [InterFormer机制](#interformer机制)
3. [相对位置编码](#相对位置编码)
4. [可借鉴的核心代码](#可借鉴的核心代码)
5. [适配建议](#适配建议)

---

## 核心架构

### 整体设计思想

NFLAT采用**两阶段解耦架构**，将词典融合与上下文编码分离：

```
输入：字符 + 词汇（通过Lattice匹配）
  ↓
阶段1: TransformerEncoder - 字符自注意力（可选before=True）
  ↓
阶段2: InterFormer - 字词交互注意力（Inter-Attention）
  ↓
阶段3: TransformerEncoder - 字符自注意力（可选before=False）
  ↓
CRF解码
```

### NFLAT主类结构

**文件**: `models/NFLAT.py`

```python
class NFLAT(nn.Module):
    def __init__(self, ...):
        # 1. 字符和词汇嵌入层
        self.char_embed = char_embed  # 字符嵌入
        self.word_embed = word_embed  # 词汇嵌入
        self.bi_embed = bi_embed      # Bigram嵌入（可选）
        
        # 2. 投影层（统一维度）
        self.char_fc = nn.Linear(char_embed_size, hidden_size)
        self.word_fc = nn.Linear(word_embed_size, hidden_size)
        
        # 3. 字符级Transformer（标准Self-Attention）
        self.chars_transformer = TransformerEncoder(
            num_layers, hidden_size, n_head//is_less_head, ...
        )
        
        # 4. InterFormer（核心创新：Inter-Attention）
        self.informer = InterFormerEncoder(
            num_layers, hidden_size, n_head, max_seq_len, ...
        )
        
        # 5. CRF解码层
        self.out_fc = nn.Linear(hidden_size, len(tag_vocab))
        self.crf = get_crf_zero_init(len(tag_vocab))
```

**关键参数**:
- `before`: True表示先做字符自注意力，False表示后做
- `is_less_head`: 字符Transformer的头数缩减因子（节省内存）
- `four_pos_fusion`: 位置编码融合方式（'ff'/'ff_two'/'attn'/'gate'）

---

## InterFormer机制

### 核心创新：Inter-Attention

**与FLAT的区别**:
| 特性 | FLAT | NFLAT (InterFormer) |
|------|------|---------------------|
| 注意力类型 | Self-Attention（全连接） | Inter-Attention（字→词单向） |
| 计算复杂度 | O((n+m)²) | O(n×m) |
| 内存占用 | 高 | 节省50% |
| 词词交互 | 有 | 无（被砍掉） |
| 字字交互 | 通过Self-Attention | 独立的Transformer层 |

**设计理念**:
> 字与字之间的交互（Self-Attention）和字与词之间的交互（Inter-Attention）应该**解耦**。
> 词与词之间的交互是冗余的，可以砍掉。

### InterAttention代码详解

**文件**: `modules/infomer.py:91-221`

```python
class InterAttention(nn.Module):
    """
    核心：仅计算 字符→词汇 的注意力
    输入：
        chars: [batch, max_char_len, hidden_size]
        words: [batch, max_word_len, hidden_size]
        rel_pos_embedding: [batch, max_char_len, max_word_len, hidden_size]
    输出：
        chars_result: [batch, max_char_len, hidden_size]
    """
    
    def forward(self, chars, words, seq_len, lex_num, rel_pos_embedding, ...):
        # 1. 投影到Q, K, V
        chars_query = self.query_proj(chars)  # [B, C, H]
        words_key = self.key_proj(words)      # [B, W, H]
        words_value = self.value_proj(words)  # [B, W, H]
        rel_pos_embedding = self.w_r(rel_pos_embedding)  # [B, C, W, H]
        
        # 2. 多头切分
        chars_query = chars_query.view(batch, C, n_head, per_head_size)
        words_key = words_key.view(batch, W, n_head, per_head_size)
        words_value = words_value.view(batch, W, n_head, per_head_size)
        rel_pos_embedding = rel_pos_embedding.view(batch, C, W, n_head, per_head_size)
        
        # 3. 计算注意力得分（核心公式）
        # A_C: 内容项 (Content-based term)
        query_and_u = chars_query + self.u.unsqueeze(0).unsqueeze(-2)
        A_C = torch.matmul(query_and_u, words_key.transpose(-1, -2))
        
        # B_D: 位置项 (Position-based term)
        query_and_v = chars_query.view(B, n_head, C, 1, per_head_size) \
                      + self.v.view(1, n_head, 1, 1, per_head_size)
        rel_pos_embedding_permuted = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        B_D = torch.matmul(query_and_v, rel_pos_embedding_permuted).squeeze(-2)
        
        # 总得分 = 内容项 + 位置项
        attn_score_raw = A_C + B_D
        
        # 4. 掩码 + Softmax（只在词汇维度归一化）
        mask = char_lex_len_to_mask(seq_len, lex_num).unsqueeze(1)
        attn_score = F.softmax(
            attn_score_raw.masked_fill(~mask, -1e15),
            dim=self.softmax_axis  # 默认-1（词汇维度）
        )
        
        # 5. 加权求和
        value_weighted_sum = torch.matmul(attn_score, words_value)
        
        return value_weighted_sum.transpose(1, 2).reshape(batch, C, -1)
```

**关键点**:
1. **单向注意力**: Query来自字符，Key/Value来自词汇
2. **两部分得分**: 内容项A_C + 位置项B_D（类似Transformer-XL）
3. **可学习偏置**: `u`和`v`两个可学习参数
4. **掩码控制**: 通过`char_lex_len_to_mask`确保只关注有效词汇

---

## 相对位置编码

### Four_Pos_Fusion_Embedding

**文件**: `modules/rel_pos_embedding.py`

NFLAT使用**4种相对位置关系**来表示字符与词汇的位置交互：

```python
# 字符位置: [pos_s, pos_e)
# 词汇位置: [lex_s, lex_e)

pos_ss = pos_s - lex_s  # 字符开始 - 词汇开始
pos_se = pos_s - lex_e  # 字符开始 - 词汇结束
pos_es = pos_e - lex_s  # 字符结束 - 词汇开始
pos_ee = pos_e - lex_e  # 字符结束 - 词汇结束
```

**示例**: 字符"国"[2,3)，词汇"中国"[1,3)
- pos_ss = 2-1 = 1
- pos_se = 2-3 = -1
- pos_es = 3-1 = 2
- pos_ee = 3-3 = 0

### 四种融合策略

```python
class Four_Pos_Fusion_Embedding(nn.Module):
    def __init__(self, four_pos_fusion='ff_two', ...):
        if four_pos_fusion == 'ff':
            # 方案1: 全4个位置拼接 + FFN
            self.pos_fusion_forward = nn.Sequential(
                nn.Linear(hidden_size*4, hidden_size),
                nn.ReLU(inplace=True)
            )
        elif four_pos_fusion == 'ff_two':
            # 方案2: 只用2个位置（ss和ee）+ FFN
            self.pos_fusion_forward = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.ReLU(inplace=True)
            )
        elif four_pos_fusion == 'attn':
            # 方案3: 注意力加权融合
            self.pos_attn_score = nn.Sequential(
                nn.Linear(hidden_size*4, hidden_size*4),
                nn.ReLU(),
                nn.Linear(hidden_size*4, 4),
                nn.Softmax(dim=-1)
            )
        elif four_pos_fusion == 'gate':
            # 方案4: 门控融合
            self.pos_gate_score = nn.Sequential(
                nn.Linear(hidden_size*4, hidden_size*2),
                nn.ReLU(),
                nn.Linear(hidden_size*2, 4*hidden_size)
            )
```

**实验推荐**: 论文中'ff_two'性能最好，参数更少

---

## 可借鉴的核心代码

### 1. 字词分离的门控融合

**可以借鉴到你的项目**: `FusionExtractor`中的`gated`策略

```python
# 当前你的实现（需要改进）
class FusionExtractor(nn.Module):
    def __init__(self, fusion_strategy='gated'):
        if fusion_strategy == 'gated':
            # 简单门控
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
    
    def forward(self, expert_feat, soft_feat):
        combined = torch.cat([expert_feat, soft_feat], dim=-1)
        gate = self.gate(combined)
        return gate * expert_feat + (1 - gate) * soft_feat

# NFLAT的启发：分离计算 + 门控选择
class ImprovedGatedFusion(nn.Module):
    """
    参考NFLAT的两阶段设计：
    1. 先让每个特征独立编码
    2. 再通过门控选择性融合
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # 独立编码
        self.expert_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.soft_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, expert_feat, soft_feat):
        # 阶段1: 独立编码
        expert_encoded = self.expert_encoder(expert_feat)
        soft_encoded = self.soft_encoder(soft_feat)
        
        # 阶段2: 门控融合
        combined = torch.cat([expert_encoded, soft_encoded], dim=-1)
        gate = self.gate_net(combined)
        
        return gate * expert_encoded + (1 - gate) * soft_encoded
```

### 2. 位置感知的特征融合

**你的ExpertDict特征可以引入位置编码**:

```python
class ExpertDictWithPosition(nn.Module):
    """
    借鉴NFLAT的相对位置编码思想
    """
    def __init__(self, hidden_size, max_seq_len=512):
        super().__init__()
        # 正弦位置编码
        self.pe = get_embedding(max_seq_len, hidden_size)
        self.pos_fusion = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, expert_feat, match_positions):
        """
        expert_feat: [B, L, H]
        match_positions: [B, L] 匹配位置索引
        """
        batch, seq_len, _ = expert_feat.shape
        
        # 获取位置编码
        pos_emb = self.pe[match_positions.clamp(0, 511)]  # [B, L, H]
        
        # 融合内容和位置
        combined = torch.cat([expert_feat, pos_emb], dim=-1)
        return self.pos_fusion(combined)
```

### 3. 层次化注意力机制

**借鉴InterAttention的单向注意力设计**:

```python
class HierarchicalFusion(nn.Module):
    """
    层次1: ExpertDict特征 → 字符特征
    层次2: SoftLex特征 → 增强后的字符特征
    """
    def __init__(self, hidden_dim, n_head=4):
        super().__init__()
        self.expert_attn = InterAttentionLayer(hidden_dim, n_head)
        self.soft_attn = InterAttentionLayer(hidden_dim, n_head)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, char_feat, expert_feat, soft_feat):
        # 层次1: Expert增强
        residual = char_feat
        char_feat = self.expert_attn(
            query=char_feat,  # 字符作为Query
            key_value=expert_feat  # Expert作为K/V
        )
        char_feat = self.layer_norm1(char_feat + residual)
        
        # 层次2: SoftLex增强
        residual = char_feat
        char_feat = self.soft_attn(
            query=char_feat,
            key_value=soft_feat
        )
        char_feat = self.layer_norm2(char_feat + residual)
        
        return char_feat

class InterAttentionLayer(nn.Module):
    """简化版Inter-Attention"""
    def __init__(self, hidden_dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.per_head_dim = hidden_dim // n_head
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, query, key_value):
        """
        query: [B, L1, H]
        key_value: [B, L2, H]
        """
        B, L1, H = query.shape
        L2 = key_value.shape[1]
        
        Q = self.q_proj(query).view(B, L1, self.n_head, self.per_head_dim)
        K = self.k_proj(key_value).view(B, L2, self.n_head, self.per_head_dim)
        V = self.v_proj(key_value).view(B, L2, self.n_head, self.per_head_dim)
        
        Q = Q.transpose(1, 2)  # [B, n_head, L1, per_head_dim]
        K = K.transpose(1, 2).transpose(-1, -2)  # [B, n_head, per_head_dim, L2]
        V = V.transpose(1, 2)  # [B, n_head, L2, per_head_dim]
        
        # 计算注意力
        attn_score = torch.matmul(Q, K) / math.sqrt(self.per_head_dim)
        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)
        
        # 加权求和
        output = torch.matmul(attn_score, V)  # [B, n_head, L1, per_head_dim]
        output = output.transpose(1, 2).reshape(B, L1, H)
        
        return self.out_proj(output)
```

### 4. 掩码生成工具

**可以直接复用**:

```python
def char_lex_len_to_mask(char_len, lex_len):
    """
    生成字符-词汇的3维掩码
    char_len: [B] 每个样本的字符长度
    lex_len: [B] 每个样本的词汇数量
    返回: [B, max_char_len, max_lex_len] 掩码
    """
    batch_size = char_len.size(0)
    max_char_len = char_len.max().long()
    max_lex_len = lex_len.max().long()
    
    # 字符维度掩码
    broadcast_char_len = torch.arange(max_char_len).expand(batch_size, -1).to(char_len)
    char_mask = broadcast_char_len.lt(char_len.unsqueeze(1))
    
    # 词汇维度掩码
    broadcast_lex_len = torch.arange(max_lex_len).expand(batch_size, max_char_len, -1).to(char_len)
    mask = broadcast_lex_len.lt(lex_len.unsqueeze(1).unsqueeze(1)).masked_fill(~char_mask.unsqueeze(-1), False)
    
    return mask
```

---

## 适配建议

### 短期可实现（1-2天）

#### 方案1: 改进门控融合

**目标**: 提升当前Gated融合的性能

**修改文件**: `_6MODEL/extractor.py`

```python
# 在FusionExtractor类中添加
class FusionExtractor(nn.Module):
    def __init__(self, fusion_strategy='gated_improved'):
        if fusion_strategy == 'gated_improved':
            # 借鉴NFLAT的两阶段设计
            self.expert_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            )
            self.soft_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            )
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
    
    def forward(self, bert_output, nested_features):
        expert_feat = nested_features['expert_dict']
        soft_feat = nested_features['softlexicon']
        
        # 独立编码
        expert_encoded = self.expert_encoder(expert_feat)
        soft_encoded = self.soft_encoder(soft_feat)
        
        # 门控融合
        combined = torch.cat([expert_encoded, soft_encoded], dim=-1)
        gate = self.gate_net(combined)
        
        fused = gate * expert_encoded + (1 - gate) * soft_encoded
        return torch.cat([bert_output, fused], dim=-1)
```

**预期提升**: +0.2~0.3% F1

#### 方案2: 添加位置编码

**目标**: 让ExpertDict感知匹配位置

**修改文件**: `_6MODEL/extractor.py` 中的`ExpertDictExtractor`

```python
class ExpertDictExtractor(nn.Module):
    def __init__(self, add_position=True):
        if add_position:
            self.pe = nn.Parameter(
                get_embedding(512, hidden_size),
                requires_grad=False
            )
            self.pos_fusion = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, tokens, label_ids):
        # ... 原有匹配逻辑 ...
        
        if self.add_position:
            # 计算匹配位置的平均位置
            pos_indices = torch.arange(seq_len, device=tokens.device)
            pos_emb = self.pe[pos_indices.clamp(0, 511)]
            
            # 融合
            expert_output = torch.cat([expert_output, pos_emb], dim=-1)
            expert_output = self.pos_fusion(expert_output)
        
        return expert_output
```

**预期提升**: +0.1~0.2% F1

### 中期可探索（3-5天）

#### 方案3: 层次化融合架构

**目标**: 实现类似InterFormer的层次化设计

**新建文件**: `_6MODEL/hierarchical_fusion.py`

```python
class HierarchicalFusionExtractor(nn.Module):
    """
    两阶段融合：
    Stage1: BERT → Expert增强
    Stage2: Expert增强 → SoftLex增强
    """
    def __init__(self, bert_like, expert_config, soft_config, decoder):
        super().__init__()
        self.bert_extractor = BertLikeExtractor(**bert_like)
        self.expert_extractor = ExpertDictExtractor(**expert_config)
        self.soft_extractor = SoftLexiconExtractor(**soft_config)
        
        # 层次化注意力
        self.expert_attn = InterAttentionLayer(hidden_dim, n_head=4)
        self.soft_attn = InterAttentionLayer(hidden_dim, n_head=4)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.decoder = TransitionDecoder(**decoder)
    
    def forward(self, tokens, chars, words, ...):
        # Stage0: BERT编码
        bert_feat = self.bert_extractor(tokens)
        
        # Stage1: Expert增强
        expert_feat = self.expert_extractor(tokens, label_ids)
        bert_feat_enhanced = self.expert_attn(
            query=bert_feat,
            key_value=expert_feat
        )
        bert_feat_enhanced = self.norm1(bert_feat_enhanced + bert_feat)
        
        # Stage2: SoftLex增强
        soft_feat = self.soft_extractor(chars, words)
        final_feat = self.soft_attn(
            query=bert_feat_enhanced,
            key_value=soft_feat
        )
        final_feat = self.norm2(final_feat + bert_feat_enhanced)
        
        # 解码
        return self.decoder(final_feat)
```

**训练脚本**: 创建`run_hierarchical_fusion.sh`

**预期提升**: +0.3~0.5% F1（如果实现正确）

### 长期研究（1-2周）

#### 方案4: 完整的InterFormer移植

**目标**: 复刻NFLAT的完整架构

**工作量评估**:
- 修改数据加载（支持Lattice格式）：2天
- 实现InterFormerEncoder：1天
- 实现Four_Pos_Fusion_Embedding：1天
- 调试训练流程：2天
- 超参数调优：3天

**是否值得**: 取决于当前实验结果
- 如果改进后的Gated能达到97%+，暂时不需要
- 如果始终卡在96.8%左右，可以尝试完整移植

---

## 关键超参数对比

### NFLAT在不同数据集的最优配置

| 数据集 | n_heads | head_dims | num_layers | lr | batch_size | dropout | fc_dropout |
|--------|---------|-----------|------------|-----|-----------|---------|------------|
| Resume | 8 | 16 | 1 | 0.002 | 16 | 0.2 | 0 |
| Weibo | 12 | 16 | 1 | 0.003 | 16 | 0 | 0.2 |
| OntoNotes | 8 | 32 | 1 | 0.0008 | 16 | 0.2 | 0.2 |
| MSRA | 8 | 32 | 1 | 0.002 | 16 | 0.2 | 0 |

**给RedJujube的建议**:
- n_heads: 8
- head_dims: 32
- hidden_size: 256 (8×32)
- num_layers: 1
- lr: 0.001~0.002
- batch_size: 16~32
- dropout: 0.2~0.3
- fc_dropout: 0.1~0.2

---

## 总结与行动清单

### 核心收获

1. **架构设计**: 两阶段解耦（Self-Attention + Inter-Attention）
2. **内存优化**: 通过单向注意力节省50%内存
3. **位置编码**: 4种相对位置关系建模
4. **融合策略**: 'ff_two'性能最好

### 优先级排序

**✅ 立即可做（今晚）**:
1. 改进Gated融合（添加独立编码阶段）
2. 调整超参数（参考MSRA配置）

**🔸 短期尝试（本周）**:
1. 添加位置编码到ExpertDict
2. 实现简化版InterAttention

**🔹 中期探索（下周）**:
1. 层次化融合架构
2. 完整移植InterFormer（如果必要）

### 决策建议

**当前实验状态**:
- Gated: 96.46%
- SoftLex-v1: 训练中
- SoftLex-v2: 训练中

**建议策略**:
1. **等SoftLex-v2结果**: 如果≥96.5%，说明词典优化有效
2. **实现改进Gated**: 用v2词典 + 改进Gated，目标97%
3. **如果仍不达标**: 考虑层次化融合或InterFormer移植

---

## 参考资源

- **论文**: [NFLAT: Non-Flat-Lattice Transformer for Chinese NER](https://arxiv.org/abs/2205.05832)
- **代码**: `/home/shiwenlong/NERlabs/eznlp/examples/NFLAT4CNER-main/`
- **关键文件**:
  - `models/NFLAT.py`: 主模型
  - `modules/infomer.py`: InterFormer实现
  - `modules/rel_pos_embedding.py`: 位置编码
  - `main.py`: 训练脚本

---

**分析时间**: 2025-12-13  
**作者**: Qoder AI Assistant  
**项目**: RedJujube NER优化
