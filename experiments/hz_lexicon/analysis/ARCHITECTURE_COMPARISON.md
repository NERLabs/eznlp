# NFLAT架构对比图

## 架构演进对比

### FLAT架构（2020）- 全连接注意力

```
输入：字符序列 + Lattice词汇
         ↓
    [Flat Lattice]
    字:中 国 人 民
    词:  中国  人民
         ↓
  [Self-Attention] ← 全连接
    计算所有两两之间的注意力：
    - 字-字：中→国, 中→人, ...
    - 字-词：中→[中国], 国→[人民], ...
    - 词-词：[中国]→[人民]
         ↓
    复杂度: O((n+m)²)
    内存占用: 高
         ↓
      [CRF解码]
```

**问题**：
- 词-词交互是冗余的
- 内存占用大，无法用大词典
- 训练慢

---

### NFLAT架构（2022）- 两阶段解耦

```
输入：字符序列 + Lattice词汇
         ↓
    [Non-Flat结构]
    字符流: 中 国 人 民
    词汇流: 中国  人民
         ↓
    ┌─────────────────┐
    │ 阶段1: 字符自注意力 │ ← TransformerEncoder
    └─────────────────┘
         中 → 中,国,人,民  (Self-Attention)
         国 → 中,国,人,民
         人 → 中,国,人,民
         民 → 中,国,人,民
         ↓
    ┌─────────────────┐
    │ 阶段2: 字词交互注意力 │ ← InterFormer
    └─────────────────┘
         中 → [中国],[人民]  (Inter-Attention)
         国 → [中国],[人民]
         人 → [中国],[人民]
         民 → [中国],[人民]
         
         ⚠️ 注意：词汇不查询词汇！
         ↓
    复杂度: O(n²) + O(n×m)
    内存占用: 节省50%
         ↓
      [CRF解码]
```

**优势**：
- 解耦字符和词汇处理
- 砍掉冗余的词-词交互
- 内存节省，可用更大词典

---

### 你的项目当前架构（已实现）

```
输入：字符序列 + 专家词典 + 软词典
         ↓
    [BERT编码]
    字符: 中 国 人 民 → BERT特征(768维)
         ↓
    [ExpertDict匹配]
    B-ORG I-ORG O O → Expert特征(50维)
         ↓
    [SoftLexicon匹配]
    4通道BMES特征 → SoftLex特征(200维)
         ↓
    [特征融合] ← 你已实现4种融合方式！
    ├─ Concat:    直接拼接 → 96.87% 🏆
    ├─ Weighted:  加权融合 → 96.72%
    ├─ Gated:     门控融合 → 96.46%
    └─ Attention: 注意力融合 → 96.53% ✅
         ↓
    [BERT(768); Fused特征] → BiLSTM + CRF
```

**你的Attention实现**：
```python
# 特征级Attention融合
Q = Expert特征  (B, L, 50)
K = SoftLex特征 (B, L, 200)
V = SoftLex特征 (B, L, 200)
Attn_output = Attention(Q, K, V)  # Expert查询SoftLex
Fused = Concat([BERT, Attn_output])  # 与BERT拼接
```

**特点**：
- ✅ 词典特征之间有交互（Expert查询SoftLex）
- ✅ 比简单拼接更智能
- ❌ 但BERT和词典特征还是简单拼接
- ❌ 没有"字符查询词汇"的核心思想

**问题**：
- Attention融合(96.53%) < Concat(96.87%)
- 说明词典特征交互不如直接拼接有效
- 单独ExpertDict(97.00%)仍然最优

---

## 改进后的架构对比

### 重要说明：你的Attention融合 vs NFLAT Inter-Attention

**✅ 你已经实现了Attention融合**（96.53% F1），但这**不是**NFLAT的Inter-Attention！

**关键区别对比**：

| 特性 | 你的Attention融合 ✅ | NFLAT的Inter-Attention ❌ |
|------|---------------------|-------------------------|
| **实现状态** | **已完成** | **未实现** |
| **作用层次** | 特征级融合 | 序列级交互 |
| **Query来源** | Expert特征(50维) | **字符序列(768维)** |
| **Key/Value来源** | SoftLex特征(200维) | **词汇序列** |
| **交互对象** | 词典特征↔词典特征 | **字符↔词汇** |
| **架构位置** | 融合层（后处理） | 编码层（特征提取） |
| **核心思想** | 融合两个词典 | **字符查询词汇** |
| **测试F1** | 96.53% | 未测试 |

**你的Attention融合代码**（推测）：
```python
# 特征级Attention
expert_feat = expert_dict_emb(tokens)  # (B, L, 50)
softlex_feat = softlex_emb(tokens)     # (B, L, 200)

# Expert查询SoftLex
attn_output = MultiHeadAttention(
    query=expert_feat,      # 词典1
    key=softlex_feat,       # 词典2
    value=softlex_feat
)  # → (B, L, 50)

# 与BERT拼接
bert_feat = bert(tokens)  # (B, L, 768)
fused = concat([bert_feat, attn_output])  # (B, L, 818)
```

**NFLAT的Inter-Attention**（未实现）：
```python
# 序列级Inter-Attention
bert_feat = bert(tokens)              # (B, L_char, 768)
lexicon_feat = get_lexicon_feat()     # (B, L_word, 768)

# 字符查询词汇
enhanced_char = InterAttention(
    query=bert_feat,        # 字符序列
    key=lexicon_feat,       # 词汇序列
    value=lexicon_feat
)  # → (B, L_char, 768)

# 残差连接
output = enhanced_char + bert_feat  # (B, L_char, 768)
```

**核心区别**：
1. **你的Attention**：Expert特征和SoftLex特征之间交互
2. **NFLAT Inter-Attention**：BERT字符特征主动查询词汇特征

**为什么你的Attention效果不好？**
- ❌ Expert(50维) vs SoftLex(200维)维度不匹配
- ❌ SoftLex特征质量低（95.46% F1）
- ❌ 两个弱特征互相交互，无法产生增益
- ✅ 不如直接Concat（96.87%）

---

## 改进后的架构对比

### 方案1: 改进门控融合

```
输入：字符序列 + 专家词典 + 软词典
         ↓
    [BERT编码]
         ↓
    ┌──────────────┐
    │ 独立编码阶段  │ ← 新增！
    └──────────────┘
    Expert → [Encoder] → Expert'
    SoftLex → [Encoder] → SoftLex'
         ↓
    ┌──────────────┐
    │ 门控融合阶段  │ ← 改进！
    └──────────────┘
    Gate = σ(W[Expert'; SoftLex'])
    Fused = Gate⊙Expert' + (1-Gate)⊙SoftLex'
         ↓
    [BERT; Fused] → BiLSTM + CRF
```

**改进点**：
- 先独立编码，避免直接拼接
- 更深的门控网络（2层）
- LayerNorm稳定训练

**预期提升**：+0.2~0.3% F1

---

### 方案2: 层次化Inter-Attention

```
输入：字符序列 + 专家词典 + 软词典
         ↓
    [BERT编码]
    BERT_feat
         ↓
    ┌──────────────────┐
    │ 层次1: Expert增强  │ ← 借鉴NFLAT!
    └──────────────────┘
    Q = BERT_feat
    K,V = Expert_feat
    Enhanced1 = InterAttention(Q, K, V) + BERT_feat
         ↓
    ┌──────────────────┐
    │ 层次2: SoftLex增强 │ ← 借鉴NFLAT!
    └──────────────────┘
    Q = Enhanced1
    K,V = SoftLex_feat
    Enhanced2 = InterAttention(Q, K, V) + Enhanced1
         ↓
    [FFN增强]
    Final = FFN(Enhanced2) + Enhanced2
         ↓
    [CRF解码]
```

**核心思想**：
- BERT主动查询词典特征
- 层次化增强（先Expert后SoftLex）
- 残差连接保留原始信息

**预期提升**：+0.4~0.6% F1

---

## Inter-Attention详解

### 计算流程

```
输入：
    Query:  [B, L1, H]  ← 字符特征（BERT）
    Key:    [B, L2, H]  ← 词汇特征（ExpertDict）
    Value:  [B, L2, H]  ← 词汇特征（ExpertDict）

步骤1: 投影到多头
    Q = Linear(Query)  → [B, n_head, L1, d_head]
    K = Linear(Key)    → [B, n_head, L2, d_head]
    V = Linear(Value)  → [B, n_head, L2, d_head]

步骤2: 计算注意力得分
    Score = Q @ K^T / sqrt(d_head)  → [B, n_head, L1, L2]
    
    示例（L1=4字符，L2=2词汇）：
    Score[0,0,:,:] = 
        [[ 0.8, 0.2],  ← 字符0对词汇0和1的注意力
         [ 0.3, 0.7],  ← 字符1...
         [ 0.9, 0.1],
         [ 0.4, 0.6]]

步骤3: Softmax归一化（在词汇维度）
    Attn = Softmax(Score, dim=-1)  → [B, n_head, L1, L2]

步骤4: 加权求和
    Output = Attn @ V  → [B, n_head, L1, d_head]
    Output = Reshape(Output)  → [B, L1, H]

输出：
    增强后的字符特征 [B, L1, H]
```

### 与Self-Attention对比

| 特性 | Self-Attention | Inter-Attention |
|------|---------------|-----------------|
| Q来源 | 字符 | 字符 |
| K来源 | 字符 | **词汇** |
| V来源 | 字符 | **词汇** |
| 复杂度 | O(L²) | O(L1×L2) |
| 语义 | 字符内部交互 | **字符查询词汇** |
| 方向 | 双向 | **单向** |

---

## 位置编码对比

### BERT内置位置编码

```
Position Embedding:
    pos_emb[0] = [0.1, 0.3, ..., 0.8]   ← 位置0
    pos_emb[1] = [0.2, 0.4, ..., 0.7]   ← 位置1
    pos_emb[2] = [0.3, 0.5, ..., 0.6]
    ...

Token Embedding + Position Embedding = Input
```

**特点**：
- 绝对位置
- 固定（预训练好的）
- 字符级

---

### NFLAT的相对位置编码

```
字符：中(0,1) 国(1,2) 人(2,3) 民(3,4)
词汇：中国(0,2)       人民(2,4)

相对位置矩阵（字符0"中" vs 词汇0"中国"）：
    pos_ss = 0 - 0 = 0   (字符开始 - 词汇开始)
    pos_se = 0 - 2 = -2  (字符开始 - 词汇结束)
    pos_es = 1 - 0 = 1   (字符结束 - 词汇开始)
    pos_ee = 1 - 2 = -1  (字符结束 - 词汇结束)

位置编码融合：
    pe = Fusion([PE[pos_ss], PE[pos_se], PE[pos_es], PE[pos_ee]])
```

**特点**：
- 相对位置（更灵活）
- 4种关系（更细粒度）
- 字符-词汇级

---

## 参数量对比

### 各方案参数量估算

**基准配置**：
- BERT: 102M
- BiLSTM: 1M
- CRF: 0.2M

| 方案 | 额外参数 | 总参数 | 增加比例 |
|------|----------|--------|----------|
| Baseline | 0 | 103M | - |
| + ExpertDict | 0.02M | 103M | 0% |
| + SoftLex | 0.2M | 103M | 0% |
| Gated融合 | 10M | 113M | +10% |
| **Gated-Improved** | 13M | 116M | +13% |
| **Hierarchical-Inter** | 22M | 125M | +21% |

**分析**：
- 改进方案参数增加合理（<25%）
- 主要增加在融合层
- 内存占用可接受

---

## 训练时间对比

### 预估训练时间（30 epochs）

| 方案 | 每epoch | 总时间 | 相对速度 |
|------|---------|--------|----------|
| Baseline | 2min | 1h | 100% |
| + ExpertDict | 2min | 1h | 100% |
| + SoftLex | 2.4min | 1.2h | 120% |
| Gated融合 | 3min | 1.5h | 150% |
| Gated-Improved | 3.2min | 1.6h | 160% |
| Hierarchical-Inter | 4min | 2h | 200% |

**建议**：
- 先用小数据集快速验证
- 确认有效后再全量训练
- 使用混合精度训练加速

---

## 决策流程图

```
开始实验
    ↓
是否已有SoftLex-v2结果？
    ├─ 是 → 进入方案选择
    └─ 否 → 等待训练完成
         ↓
    方案选择
         ├─ 追求稳妥 → 方案1 (Gated-Improved)
         ├─ 追求性能 → 方案2 (Hierarchical-Inter)
         └─ 追求完美 → 方案1 + 方案3 (位置编码)
         ↓
    代码集成
         ├─ 复制类到extractor.py
         ├─ 修改train脚本
         └─ 创建运行脚本
         ↓
    启动训练
         ├─ 监控日志
         ├─ 观察收敛
         └─ 等待结果
         ↓
    结果评估
         ├─ Test F1 ≥ 97.0% → 成功！→ 消融实验
         ├─ 96.7~97.0% → 调参 → 重新训练
         └─ < 96.7% → 尝试下一个方案
         ↓
    最终报告
```

---

## 可视化对比总结

### 性能 vs 复杂度

```
性能提升 (F1 %)
    ↑
97.5|                    ⭐ Hierarchical-Inter (理论上限)
    |               
97.0|              ⭐ Gated-Improved + Position
    |         
96.5|    ⭐ Gated-Improved
    |    ⭐ ExpertDict(单独)
96.0|⭐ Gated(原始)
    |
95.5|
    +─────────────────────────────────────→
      低         中等              高         实现复杂度
```

### ROI对比

```
投入产出比
    ↑
高  |    ⭐ Gated-Improved
    |    (2h → +0.3%)
    |
中  |              ⭐ Hierarchical-Inter
    |              (1天 → +0.5%)
    |
低  |                          ⭐ 完整NFLAT
    |                          (7天 → +0.2%?)
    +─────────────────────────────────────→
      短期(今晚)    中期(本周)    长期(下周)
```

---

**建议**：
1. 今晚实施方案1（Gated-Improved）
2. 明天评估，如果≥97%就成功
3. 如果<97%，尝试方案2（Hierarchical-Inter）
4. 最后再考虑完整移植NFLAT

**期待好消息！🎉**
