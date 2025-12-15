# HZ Lexicon 实验结果汇总

本目录存放 HZ/RedJujube 数据集上的词典特征实验结果，与 `plans/` 目录中的周计划对应。

---

## 📁 目录结构（按周计划组织）

```
results/
├── 12-1_baseline_expert_dict/           # 12-1周：Baseline + ExpertDict对比
│   ├── HZ_NER_实验报告_20251208.md
│   ├── RedJujube_NER_实验报告_20251212.md
│   └── 词典对比分析报告.md
├── 12-2_softlexicon/                    # 12-2周：SoftLexicon实验（进行中）🚀
│   ├── softlexicon_20251210/             # CTB词典实验
│   ├── softlexicon_trainlex_20251210/    # TrainLex原始实验
│   ├── softlexicon_trainlex_auto/        # TrainLex Auto实验
│   ├── softlexicon_trainlex_filtered/    # TrainLex Filtered实验
│   └── RedJujube_SoftLexicon_Complete_Report_20251213.md  # RedJujube完整实验报告
└── README.md                            # 本文档
```

---

## 📊 实验周期汇总

### 📌 12-1周：Baseline + ExpertDict 对比 (2025-12-02 ~ 12-08)

**计划文档**: [plans/12-1_baseline_expert_dict.md](../plans/12-1_baseline_expert_dict.md)

**实验报告**: 
- [HZ_NER_实验报告_20251208.md](./12-1_baseline_expert_dict/HZ_NER_实验报告_20251208.md)
- [RedJujube_NER_实验报告_20251212.md](./12-1_baseline_expert_dict/RedJujube_NER_实验报告_20251212.md)
- [词典对比分析报告.md](./12-1_baseline_expert_dict/词典对比分析报告.md)

**核心结论**:
- 🏆 HZ数据集: ExpertDict(自动) = **97.05% F1** (+1.39%)
- 🏆 RedJujube数据集: ExpertDict(自动) = **96.99% F1** (+1.48%)
- ✅ 证明ExpertDict方法有效性

**关键实验**:

| 数据集 | Baseline | ExpertDict(自动) | ExpertDict(手动) | 提升 |
|--------|----------|----------------|----------------|------|
| HZ | 95.66% | **97.05%** | 97.94% | +1.39% |
| RedJujube | 95.51% | **96.99%** | 97.04% | +1.48% |

---

### 📌 12-2周：SoftLexicon 实验 (2025-12-09 ~ 12-13) 🚀

**计划文档**: [plans/12-2_softlexicon.md](../plans/12-2_softlexicon.md)

**实验目录**: [12-2_softlexicon/](./12-2_softlexicon/)

**实验报告**:
- [RedJujube_SoftLexicon_Complete_Report_20251213.md](./12-2_softlexicon/RedJujube_SoftLexicon_Complete_Report_20251213.md) - RedJujube 14个完整实验

**核心结论**:
- ❌ SoftLexicon在RedJujube上**完全无效** (所有版本 < Baseline)
- ❌ 所有Soft+Expert融合方案**低于单独ExpertDict**
- ✅ ExpertDict稳定性极高（97.00% ± 0.01%）
- 💡 词典质量比规模更重要（ExpertDict 2k词 > SoftLex 200k词）

**关键实验**:

#### HZ数据集结果

| 实验 | 词典规模 | HZ F1 | 状态 |
|------|---------|-------|------|
| SoftLex-CTB | 280,930 | 95.88% | ✅ |
| SoftLex-TrainLex | 197,972 | **96.57%** | ✅ |
| SoftLex-Auto | 自动提取 | 96.27% | ✅ |
| SoftLex-Filtered | 过滤版 | 96.22% | ✅ |
| ExpertDict(自动) | 3,214 | **97.05%** | ✅ |
| Baseline | - | 95.66% | ✅ |

#### RedJujube数据集结果 (14个实验)

| 类别 | 实验 | 测试F1 | 说明 |
|------|------|--------|------|
| **ExpertDict** | Run1/Run2/Run3 | **97.00%** | 最优方案 🏆 |
| **融合方案** | Concat | 96.87% | 最佳融合 |
| | Weighted | 96.72% | 加权融合 |
| | Attention | 96.53% | 注意力融合 |
| | Gated | 96.46% | 门控融合 |
| **SoftLexicon** | v1 | 95.47% | 无效 ❌ |
| | v2 | 95.46% | 无效 ❌ |
| | Balanced | 94.63% | 反降 ❌ |
| **Baseline** | - | 95.51% | 基线 |

**关键发现**:
1. SoftLexicon在RedJujube上完全无效（所有版本都低于Baseline）
2. 所有融合方案都不如单独ExpertDict（Concat 96.87% vs Expert 97.00%）
3. ExpertDict稳定性极高（3次运行标准差 < 0.01%）
4. 词典质量比规模更重要（ExpertDict 2k词效率是SoftLex 20w词的2.6倍）

---

## 🔑 关键结论

### ✅ 已验证结论 (12-1周 + 12-2周)

1. **ExpertDict在两个数据集上都有效** 🏆
   - HZ (医疗): **97.05% F1** (+1.39%)
   - RedJujube (医疗): **97.00% F1** (+1.48%)
   - 跨数据集稳定性验证

2. **SoftLexicon效果不稳定** ⚠️
   - HZ: +0.95% (有效)
   - RedJujube: -0.05% (无效，所有版本都低于Baseline)
   - 受数据集特性影响大

3. **特征融合不如单独ExpertDict** 📉
   - Concat融合: 96.87% (-0.13%)
   - Weighted融合: 96.72% (-0.28%)
   - Attention融合: 96.53% (-0.47%)
   - Gated融合: 96.46% (-0.54%)
   - SoftLexicon是"拖累"而非"增益"

4. **ExpertDict稳定性极高** 💎
   - 3次运行标准差 < 0.01%
   - 97.00%, 97.01%, 97.01% (RedJujube)
   - 证明方法的可靠性和可复现性

5. **词典质量 > 词典规模** 🎯
   - ExpertDict: 2,078词 → 97.00% F1
   - SoftLex: 198,437词 → 96.07% F1 (HZ TrainLex)
   - 精选词典效率是大规模词表的2.6倍

6. **CTB vs 训练集词表** 📊
   - HZ: TrainLex (96.57%) > CTB (95.88%)
   - TrainLex避免数据泄露，性能更优
   - 推荐使用训练集构建词典

### 🎯 下周计划 (12-3周)

基于12-2周实验发现：

- ✅ 以 RedJujube 为主数据集
- ❌ 放弃 Soft+Expert 融合方案（已验证无效）
- 🎯 新方向：ExpertDict最优配置探索（词典规模、嵌入维度等）
- 📝 撰写阶段总结报告

---

## 📂 文件索引

### 12-1周报告
- [HZ_NER_实验报告_20251208.md](./12-1_baseline_expert_dict/HZ_NER_实验报告_20251208.md) - HZ数据集Baseline vs ExpertDict
- [RedJujube_NER_实验报告_20251212.md](./12-1_baseline_expert_dict/RedJujube_NER_实验报告_20251212.md) - RedJujube数据集完整对比
- [词典对比分析报告.md](./12-1_baseline_expert_dict/词典对比分析报告.md) - 手工vs自动词典覆盖率分析

### 12-2周报告
- [softlexicon_20251210/](./12-2_softlexicon/softlexicon_20251210/) - SoftLexicon (CTB)
- [softlexicon_trainlex_20251210/](./12-2_softlexicon/softlexicon_trainlex_20251210/) - SoftLexicon-TrainLex (原始)
- [softlexicon_trainlex_auto/](./12-2_softlexicon/softlexicon_trainlex_auto/) - SoftLexicon-TrainLex (Auto)
- [softlexicon_trainlex_filtered/](./12-2_softlexicon/softlexicon_trainlex_filtered/) - SoftLexicon-TrainLex (Filtered)
- [RedJujube_SoftLexicon_Complete_Report_20251213.md](./12-2_softlexicon/RedJujube_SoftLexicon_Complete_Report_20251213.md) - RedJujube 14个完整实验

---

## 🔗 相关资源

### 计划文档
- [12-1周计划](../plans/12-1_baseline_expert_dict.md) - Baseline + ExpertDict对比
- [12-2周计划](../plans/12-2_softlexicon.md) - SoftLexicon实验 (进行中) 🚀
- [12-3周计划](../plans/12-3_soft_expert_joint.md) - ExpertDict深度优化 (下周)

### 数据与模型
- **数据集**: `/data/HZ/`, `/data/RedJujube/`
- **模型缓存**: `/cache/redjujube_ner_comparison/`, `/cache/hz_softlexicon/`
- **训练脚本**: `/_1CONFIG/redjujube/train_redjujube_ner_comparison.py`

---

**文档维护**: 与 `plans/` 目录周计划保持同步  
**最后更新**: 2025-12-13 (当前为12-2周)  
**实验负责人**: 史文龙

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
