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
├── NER_实验结果综合对比报告_20251208.md     # 综合对比报告
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

---

## 综合报告

### 📊 NER 实验结果综合对比报告 (2025-12-08)

**文件**: [NER_实验结果综合对比报告_20251208.md](./NER_实验结果综合对比报告_20251208.md)

**内容概要**:
- ✅ MSRA-ER 数据集实验 (4个实验)
- ✅ HZ 数据集实验 (3个对比实验)
- ✅ 历史 SOTA 模型对比 (5个基线)
- ✅ 综合性能排名与分析

**关键发现**:
- 🏆 HZ 数据集最高 F1: 手工专家词典 = **97.941%**（⚠ 可能包含测试集信息，仅作上界参考）
- 📊 公平对比下（仅使用训练集信息）: 自动专家词典 = **97.050%**，提升约 +1.388%；SoftLexicon(CTB) = 95.88%，相对 Baseline 提升有限。
- 📊 SoftLexicon: 95.88% (本次 CTB) / 95.07% (历史)

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

| 实验类型 | 测试集 F1 | 验证集最佳 F1 | Epochs | 日期 | 说明 |
|---------|----------|--------------|--------|------|------|
| 手工专家词典 | **97.941%** | - | 30 | 2025-12-07 | ⚠ 可能包含测试集信息，仅作上界参考 |
| 自动专家词典 | **97.050%** | - | 30 | 2025-12-07 | 公平对比主参照（仅用训练集构建词典） |
| SoftLexicon-TrainLex | **96.57%** | 97.24% | 30 | 2025-12-10 | 原始 n-gram 版本 (197,972 词) |
| SoftLexicon-TrainLex-Auto | **96.27%** | - | 30 | 2025-12-10 | 使用 expert_lexicon_auto.txt |
| SoftLexicon-TrainLex-Filtered | **96.22%** | - | 30 | 2025-12-10 | 使用 softlexicon_train_filtered.txt |
| SoftLexicon | **95.88%** | 96.96% | 30 | 2025-12-10 | SoftLexicon(CTB) 版本 |
| Baseline | 95.662% | - | 30 | 2025-12-07 | - |

---

## 关键发现

1. **专家词典效果最佳（含与不含泄露两种口径）**: 若不考虑数据泄露，手工专家词典可达 97.941%；在严格“仅用训练集信息”的公平对比下，自动专家词典 97.050% 仍明显优于 SoftLexicon 和 Baseline。
2. **SoftLexicon-TrainLex 系列表现中等**: 三个 SoftLexicon-TrainLex 实验在 96.22%-96.57% 区间，明显优于 SoftLexicon(CTB) 95.88% 和 Baseline 95.662%，但仍略逊于自动专家词典 97.050%。
3. **词典来源影响性能**: 在 TrainLex 系列中，原始 n-gram 版本 (96.57%) 表现最佳，auto 版本 (96.27%) 和 filtered 版本 (96.22%) 略低，说明词典质量和构建方式直接影响最终性能。
4. **性能梯度**: 专家词典 > SoftLexicon-TrainLex > SoftLexicon(CTB) > Baseline，证明词典特征对 NER 任务有显著提升。

---

## 实验环境

- **数据集**: HZ 红枣医疗数据集
- **框架**: eznlp + PyTorch
- **预训练模型**: hfl/chinese-macbert-base
- **设备**: GPU (CUDA)

---

## 相关资源

- **综合报告**: `./NER_实验结果综合对比报告_20251208.md`
- **词典分析**: `./词典对比分析报告.md`
- **数据目录**: `/data/HZ/`
- **模型缓存**: `/cache/hz_softlexicon/`
- **训练脚本**: `/scripts/train_hz_ner_baseline_vs_expert_dict.py`
