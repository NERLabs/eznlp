# MSRA-ER 数据集 ExpertDict 实验报告

**实验时间**: 2025-12-07 ~ 2025-12-08  
**实验周期**: 12-3周  
**数据集**: MSRA-ER (通用命名实体识别)  
**实验目标**: 验证ExpertDict在通用NER数据集上的有效性  

---

## 📊 一、实验结果汇总

### 1.1 完整实验列表 (4个实验)

| 实验ID | 模型配置 | Epochs | 测试集P | 测试集R | 测试集F1 | 说明 |
|---------|---------|--------|---------|---------|-----------|------|
| 20251208-011438 | **MacBERT + ExpertDict** | 30 | 96.050% | 94.807% | **95.424%** 🏆 | 30轮训练 + 专家词典 |
| 20251207-233355 | MacBERT + ExpertDict | 10 | 95.802% | 94.888% | 95.343% | 10轮训练 + 专家词典 |
| 20251208-091001 | MacBERT Baseline | 30 | 95.200% | 94.985% | 95.092% | 30轮训练基线 |
| 20251208-002945 | MacBERT Baseline | 10 | 95.764% | 94.726% | 95.242% | 10轮训练基线 |

### 1.2 核心发现

✅ **ExpertDict在MSRA-ER上有效**:
- 30 epochs: **95.424%** vs 95.092% (+0.332%)
- 10 epochs: **95.343%** vs 95.242% (+0.101%)

✅ **训练轮数影响**:
- ExpertDict: 10轮→30轮 (+0.081%)
- Baseline: 10轮→30轮 (-0.150%)

⚠️ **与MSRA SOTA对比**:
- MSRA SOTA (Dice Loss): **96.72%**
- 我们的ExpertDict: **95.424%**
- 差距: **-1.296%**

---

## 📈 二、性能分析

### 2.1 ExpertDict vs Baseline

| 对比维度 | 10 Epochs | 30 Epochs | 改善幅度 |
|---------|-----------|-----------|---------|
| **F1提升** | +0.101% | +0.332% | 3.3x |
| **精确率提升** | +0.038% | +0.850% | 22.4x |
| **召回率提升** | +0.162% | -0.178% | - |

**关键观察**:
1. 30轮训练下ExpertDict优势更明显 (+0.332% vs +0.101%)
2. 精确率提升显著 (+0.850%)，说明ExpertDict有效减少误检
3. 召回率略有下降 (-0.178%)，可能需要调整词典覆盖率

### 2.2 与MSRA历史SOTA对比

| 排名 | 方法 | F1 | 发表年份 | 关键技术 |
|------|------|----|---------|---------| 
| 🥇 Rank 1 | Dice Loss | **96.72%** | 2020 | F1-oriented Loss |
| 🥈 Rank 2 | 未知 | 96.33% | - | - |
| 🥉 Rank 3 | Boundary Smoothing | 96.26% | 2020 | Label Smoothing |
| - | **我们的ExpertDict** | **95.424%** | 2025 | 专家词典特征 |
| - | 差距 | **-1.296%** | - | - |

**优化空间**:
- 🎯 **目标**: 通过SOTA技巧集成缩小差距到 **-0.2% ~ -0.7%**
- 📋 **策略**: Dice Loss + Boundary Smoothing + 对抗训练

---

## 🔬 三、数据集信息

### 3.1 MSRA-ER数据集概述

| 属性 | 值 |
|------|-----|
| **数据集名称** | MSRA Named Entity Recognition |
| **语言** | 中文 |
| **实体类型** | 3类 (PER, LOC, ORG) |
| **训练集规模** | 45,000+ 句子 |
| **测试集规模** | 4,365 句子 |
| **领域** | 新闻文本（通用） |

### 3.2 与RedJujube对比

| 对比维度 | MSRA-ER | RedJujube | 差异 |
|---------|---------|-----------|------|
| **领域** | 通用新闻 | 医疗 | 领域跨度大 |
| **实体类型** | 3类 | 14类 | 4.7倍 |
| **训练集规模** | 45,000+ | 5,372 | 8.4倍 |
| **ExpertDict提升** | +0.33% | +1.48% | 4.5倍 |
| **绝对F1** | 95.424% | 97.00% | +1.576% |

**关键洞察**:
1. **小样本场景下ExpertDict更有效**: RedJujube提升是MSRA的4.5倍
2. **医疗领域更依赖词典**: 14类细粒度实体需要专家知识
3. **通用数据集挑战更大**: MSRA-ER规模大但ExpertDict提升有限

---

## 🎯 四、改进方向

### 4.1 短期优化 (12-4周)

#### 方向1: 错误分析 🔥
- [ ] 分析95.424%的预测结果
- [ ] 对比Baseline差异
- [ ] 按实体类型统计 (PER/LOC/ORG)
- [ ] 典型Case Study

#### 方向2: 超参数优化
- [ ] `emb_dim`: 50 → 100/200
- [ ] `min_freq`: 2 → 3/5
- [ ] `agg_mode`: wtd_mean_pooling → max/attention

#### 方向3: SOTA技巧集成 ⭐
- [ ] **Dice Loss**: 直接优化F1
- [ ] **Boundary Smoothing**: 缓解边界标注噪声
- [ ] **对抗训练**: FGM/PGD增强鲁棒性

预期收益: **95.424% → 96.0~96.5% F1**

### 4.2 长期目标

#### 目标1: 缩小与SOTA差距
- 当前: **-1.296%**
- 保守目标: **-0.7%** (96.0% F1)
- 乐观目标: **-0.2%** (96.5% F1)

#### 目标2: 跨数据集验证
- ✅ MSRA-ER: 95.424% (已验证)
- ✅ RedJujube: 97.00% (已验证)
- 🎯 更多通用数据集验证

---

## 📝 五、实验配置

### 5.1 模型配置

```python
# MacBERT + ExpertDict
model = MacBertCRFNER(
    pretrained_model='hfl/chinese-macbert-base',
    expert_dict_config=ExpertDictConfig(
        emb_dim=50,
        num_channels=4,  # BMES
        min_freq=2,
        agg_mode='wtd_mean_pooling'
    ),
    lstm_hidden_size=256,
    lstm_layers=2,
    dropout=0.1
)
```

### 5.2 训练配置

| 参数 | 值 |
|------|-----|
| **优化器** | AdamW |
| **学习率** | 2e-5 (BERT) / 5e-4 (其他) |
| **批大小** | 16 |
| **最大轮数** | 10 / 30 |
| **早停** | Patience=5 |
| **设备** | GPU (CUDA) |

---

## 🔗 六、相关资源

### 文件路径
- **计划文档**: `../plans/12-3_soft_expert_joint.md`
- **RedJujube报告**: `./RedJujube_Complete_Experiments_20251213.md`
- **融合对比**: `./Fusion_Comparison_Report.md`
- **训练日志**: `../../cache/msra_er_experiments/`

### 下一步
→ 查看 [12-4周计划](../plans/12-4_improvements.md) 了解改进方向

---

## 📊 七、自定义脚本实验 (12-14补充)

### 实验5: SoftLexicon-v1 (CTB词典)

**完成时间**: 2024-12-14 05:23  
**脚本**: `_1CONFIG/msra/train_msra_ner_all_methods.py`

**配置**:
- BERT: hfl/chinese-macbert-base
- 编码器: BiLSTM (hidden=256, layers=1)
- 解码器: CRF
- 特征: SoftLexicon (CTB词典, 280K词)
- 参数量: 110,115,186

**训练参数**:
- Epochs: 30
- Batch Size: 16  
- LR: 0.002 / Finetune: 2e-5

**结果**:
- **测试集F1**: **94.62%**
- 测试Loss: 2.364

---

### 实验6: Soft+Expert Concat融合

**完成时间**: 2024-12-14 07:00  
**脚本**: `_1CONFIG/msra/train_msra_ner_all_methods.py`

**配置**:
- BERT: hfl/chinese-macbert-base
- 编码器: BiLSTM (hidden=256, layers=1)  
- 解码器: CRF
- 特征: SoftLexicon (CTB) + ExpertDict (自动提取)
- 融合方式: Concatenation
- 参数量: 104,510,836

**训练参数**:
- Epochs: 30
- Batch Size: 16
- LR: 0.002 / Finetune: 2e-5
- Seed: 42

**结果**:
- **测试集F1**: **94.75%**
- 测试Loss: 1.588
- vs SoftLex-v1: **+0.13%**

**关键发现**:
1. ✅ 融合有效: SoftLexicon + ExpertDict 带来提升
2. 📊 Loss更低: 融合模型测试Loss显著降低 (1.588 vs 2.364)
3. ⚠️ 参数更少: Concat融合参数量反而减少 (104.5M vs 110.1M)

---

**报告生成时间**: 2025-12-14  
**实验负责人**: 史文龙  
