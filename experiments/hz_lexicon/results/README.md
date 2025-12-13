# HZ 红枣数据集 词典实验结果汇总

本目录存放 HZ 红枣数据集上的各类词典特征实验结果。

---

## 目录结构

```
results/
├── softlexicon_20251210/                    # SoftLexicon 软词典实验 (CTB)
│   ├── results.json                         # 测试集结果
│   ├── training.log                         # 训练日志
│   └── README.md                            # 实验说明
├── softlexicon_trainlex_20251210/           # SoftLexicon-TrainLex 实验 (原始)
│   ├── results.json                         # 测试集结果
│   └── training.log                         # 训练日志
├── softlexicon_trainlex_auto_20251210/      # SoftLexicon-TrainLex 实验 (auto词典)
│   ├── results.json                         # 测试集结果
│   └── training.log                         # 训练日志
├── softlexicon_trainlex_filtered_20251210/  # SoftLexicon-TrainLex 实验 (filtered词典)
│   ├── results.json                         # 测试集结果
│   └── training.log                         # 训练日志
├── NER_实验结果综合对比报告_20251208.md     # 综合对比报告 (HZ数据集)
├── RedJujube_NER_实验报告_20251212.md       # RedJujube数据集实验报告
├── 词典对比分析报告.md                      # 词典覆盖率分析
└── README.md                                # 本文档
```

---

## 实验列表

### 1. SoftLexicon 实验 (2025-12-10)

**目录**: `softlexicon_20251210/`

- **模型**: MacBERT + BiLSTM + CRF + SoftLexicon
- **测试集 F1**: 95.88%
- **验证集最佳 F1**: 96.96%
- **训练轮数**: 30 epochs
- **特征类型**: 从 CTB 50d 词表构建的 SoftLexicon 特征

**关键文件**:
- [实验说明](./softlexicon_20251210/README.md)
- [结果数据](./softlexicon_20251210/results.json)
- [训练日志](./softlexicon_20251210/training.log)

### 2. SoftLexicon-TrainLex 实验 (2025-12-10)

**目录**: `softlexicon_trainlex_20251210/`

- **模型**: MacBERT + BiLSTM + CRF + SoftLexicon(TrainLex)
- **测试集 F1**: 96.57%
- **验证集最佳 F1**: 97.24%
- **训练轮数**: 30 epochs
- **特征类型**: 从 HZ 训练集 n-gram 构建的 SoftLexicon 词表 (197,972 词)

**关键文件**:
- [结果数据](./softlexicon_trainlex_20251210/results.json)
- [训练日志](./softlexicon_trainlex_20251210/training.log)

### 3. SoftLexicon-TrainLex-Auto 实验 (2025-12-10)

**目录**: `softlexicon_trainlex_auto_20251210/`

- **模型**: MacBERT + BiLSTM + CRF + SoftLexicon(TrainLex-Auto)
- **测试集 F1**: 96.27%
- **训练轮数**: 30 epochs
- **词表来源**: `expert_lexicon_auto.txt` (训练集自动提取)
- **特征类型**: 从训练集自动提取的实体构建的 SoftLexicon 词表

**关键文件**:
- [结果数据](./softlexicon_trainlex_auto_20251210/results.json)
- [训练日志](./softlexicon_trainlex_auto_20251210/training.log)

### 4. SoftLexicon-TrainLex-Filtered 实验 (2025-12-10)

**目录**: `softlexicon_trainlex_filtered_20251210/`

- **模型**: MacBERT + BiLSTM + CRF + SoftLexicon(TrainLex-Filtered)
- **测试集 F1**: 96.22%
- **训练轮数**: 30 epochs
- **词表来源**: `softlexicon_train_filtered.txt` (训练集过滤版本)
- **特征类型**: 从训练集 n-gram 过滤后构建的 SoftLexicon 词表

**关键文件**:
- [结果数据](./softlexicon_trainlex_filtered_20251210/results.json)
- [训练日志](./softlexicon_trainlex_filtered_20251210/training.log)

### 5. RedJujube 数据集对比实验 (2025-12-12)

**目录**: `cache/redjujube_ner_comparison/`

- **数据集**: RedJujube (红枣数据集)
- **训练样本**: 5,372 / 验证: 671 / 测试: 672
- **实体类型**: 14 类
- **最佳 F1**: 97.04% (ExpertDict 手动)

**实验对比**:
1. Baseline: 95.51% F1
2. SoftLexicon (TrainLex): 96.07% F1 (+0.56%)
3. ExpertDict (自动): 96.99% F1 (+1.48%)
4. ExpertDict (手动): 97.04% F1 (+1.53%)

**关键文件**:
- [实验报告](./RedJujube_NER_实验报告_20251212.md)
- [结果JSON](../../cache/redjujube_ner_comparison/comparison_results.json)
- [训练脚本](../../scripts/run_redjujube_all_experiments.sh)

---

## 综合报告

### 📊 NER 实验结果综合对比报告 (2025-12-08)

**文件**: [NER_实验结果综合对比报告_20251208.md](./NER_实验结果综合对比报告_20251208.md)

**数据集**: HZ (华梅医疗数据集)

**内容概要**:
- ✅ MSRA-ER 数据集实验 (4个实验)
- ✅ HZ 数据集实验 (3个对比实验)
- ✅ 历史 SOTA 模型对比 (5个基线)
- ✅ 综合性能排名与分析

**关键发现**:
- 🏆 HZ 数据集最高 F1: 手工专家词典 = **97.941%**（⚠ 可能包含测试集信息，仅作上界参考）
- 📊 公平对比下（仅使用训练集信息）: 自动专家词典 = **97.050%**，提升约 +1.388%；SoftLexicon(CTB) = 95.88%，相对 Baseline 提升有限。
- 📊 SoftLexicon: 95.88% (本次 CTB) / 95.07% (历史)

### 📊 RedJujube 数据集实验报告 (2025-12-12)

**文件**: [RedJujube_NER_实验报告_20251212.md](./RedJujube_NER_实验报告_20251212.md)

**数据集**: RedJujube (红枣数据集)

**内容概要**:
- ✅ 4种方法系统对比: Baseline / SoftLexicon / ExpertDict(自动) / ExpertDict(手动)
- ✅ 数据集详细统计信息
- ✅ 性能排名与关键发现
- ✅ 词典质量 vs 规模分析

**关键发现**:
- 🏆 ExpertDict 方法整体优于 SoftLexicon
  - ExpertDict（手动）: 97.04% F1 (+1.53%)
  - ExpertDict（自动）: 96.99% F1 (+1.48%)
  - SoftLexicon（TrainLex）: 96.07% F1 (+0.56%)
- 📊 词典质量比规模更重要: ExpertDict 用2,078-3,389词达到97%+ F1，SoftLexicon 用198,437词仅达到96.07% F1
- ✅ 自动词典提取策略有效: min_freq=2 性能接近手动标注，完全避免数据泄露

---

### 🔍 词典对比分析报告 (2025-12-08)

**文件**: [词典对比分析报告.md](./词典对比分析报告.md)

**内容概要**:
- ✅ 手工词典 vs 自动词典规模对比
- ✅ 训练/验证/测试集覆盖率分析
- ✅ 各实体类型覆盖率明细
- ✅ 高频未覆盖实体分析
- ✅ 改进方案推荐

**关键发现**:
- 📊 自动词典覆盖率领先: +8.23% (测试集)
- ⚠️ 手工词典 GEO 类型完全缺失 (0%)
- ✅ 推荐混合词典方案 (自动+手工精选)

**词典覆盖率对比**:

| 数据集 | 手工词典 | 自动词典 | 差异 |
|-------|----------|----------|------|
| 训练集 | 87.44% | 97.84% | +10.4% |
| 验证集 | 88.33% | 96.41% | +8.08% |
| 测试集 | 87.35% | 95.58% | +8.23% |

---

## 性能对比

### HZ 数据集结果

| 实验类型 | 测试集 F1 | 验证集最佳 F1 | Epochs | 日期 | 说明 |
|---------|----------|--------------|--------|------|------|
| 手工专家词典 | **97.941%** | - | 30 | 2025-12-07 | ⚠ 可能包含测试集信息，仅作上界参考 |
| 自动专家词典 | **97.050%** | - | 30 | 2025-12-07 | 公平对比主参照（仅用训练集构建词典） |
| SoftLexicon-TrainLex | **96.57%** | 97.24% | 30 | 2025-12-10 | 原始 n-gram 版本 (197,972 词) |
| SoftLexicon-TrainLex-Auto | **96.27%** | - | 30 | 2025-12-10 | 使用 expert_lexicon_auto.txt |
| SoftLexicon-TrainLex-Filtered | **96.22%** | - | 30 | 2025-12-10 | 使用 softlexicon_train_filtered.txt |
| SoftLexicon | **95.88%** | 96.96% | 30 | 2025-12-10 | SoftLexicon(CTB) 版本 |
| Baseline | 95.662% | - | 30 | 2025-12-07 | - |

### RedJujube 数据集结果

| 实验类型 | 测试集 F1 | 测试Loss | 参数量 | 提升 | 词典大小 | 日期 |
|---------|----------|---------|--------|------|---------|------|
| ExpertDict (手动) | **97.04%** | 9.288 | 103,327,976 | +1.53% | 3,389 | 2025-12-12 |
| ExpertDict (自动) | **96.99%** | 8.330 | 103,264,176 | +1.48% | 2,078 | 2025-12-12 |
| SoftLexicon (TrainLex) | **96.07%** | 7.884 | 113,082,376 | +0.56% | 198,437 | 2025-12-12 |
| Baseline | 95.51% | 10.729 | 103,057,976 | - | 0 | 2025-12-12 |

---

## 关键发现

### HZ 数据集

1. **专家词典效果最佳（含与不含泄露两种口径）**: 若不考虑数据泄露，手工专家词典可达 97.941%；在严格“仅用训练集信息”的公平对比下，自动专家词典 97.050% 仍明显优于 SoftLexicon 和 Baseline。
2. **SoftLexicon-TrainLex 系列表现中等**: 三个 SoftLexicon-TrainLex 实验在 96.22%-96.57% 区间，明显优于 SoftLexicon(CTB) 95.88% 和 Baseline 95.662%，但仍略逊于自动专家词典 97.050%。
3. **词典来源影响性能**: 在 TrainLex 系列中，原始 n-gram 版本 (96.57%) 表现最佳，auto 版本 (96.27%) 和 filtered 版本 (96.22%) 略低，说明词典质量和构建方式直接影响最终性能。
4. **性能梯度**: 专家词典 > SoftLexicon-TrainLex > SoftLexicon(CTB) > Baseline，证明词典特征对 NER 任务有显著提升。

### RedJujube 数据集

1. **ExpertDict 方法整体优于 SoftLexicon**
   - ExpertDict（手动）达到 97.04% F1，提升 +1.53%
   - ExpertDict（自动）达到 96.99% F1，提升 +1.48%
   - SoftLexicon（TrainLex）达到 96.07% F1，提升 +0.56%

2. **词典质量比规模更重要**
   - ExpertDict 用2,078-3,389词达到97%+ F1
   - SoftLexicon 用198,437词仅达到96.07% F1
   - 精选专家词典效率是大规模词表的2.6倍

3. **自动词典提取策略有效**
   - 自动提取（min_freq=2）性能接近手动标注
   - 仅差 0.05% F1，但完全避免数据泄露
   - 推荐作为最佳实践方案

4. **两个数据集结论一致**
   - 两个数据集都证明 ExpertDict 优于 SoftLexicon
   - HZ: 97.050% vs 96.57% (+0.48%)
   - RedJujube: 96.99% vs 96.07% (+0.92%)
   - 词典质量比规模更关键

---

## 实验环境

- **数据集**: HZ 红枣医疗数据集
- **框架**: eznlp + PyTorch
- **预训练模型**: hfl/chinese-macbert-base
- **设备**: GPU (CUDA)

---

## 相关资源

- **HZ 数据集报告**: `./NER_实验结果综合对比报告_20251208.md`
- **RedJujube 数据集报告**: `./RedJujube_NER_实验报告_20251212.md`
- **词典分析**: `./词典对比分析报告.md`
- **数据目录**: `/data/HZ/`, `/data/RedJujube/`
- **模型缓存**: `/cache/hz_softlexicon/`, `/cache/redjujube_ner_comparison/`
- **训练脚本**: `/scripts/train_hz_ner_baseline_vs_expert_dict.py`, `/scripts/run_redjujube_all_experiments.sh`
