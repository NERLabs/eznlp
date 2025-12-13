# HZ 红枣数据集词典特征实验 - 计划汇总

本目录包含 HZ 数据集上词典特征相关实验的周计划和总体规划文档。

---

## 📁 计划文档列表

### 周计划

| 周次 | 文档 | 时间 | 状态 | 核心目标 |
|-----|------|------|------|------|
| 12-1周 | [12-1_baseline_expert_dict.md](./12-1_baseline_expert_dict.md) | 2025-12-07~08 | ✅ 已完成 | 基线与专家词典对比 |
| 12-2周 | [12-2_softlexicon.md](./12-2_softlexicon.md) | 2025-12-09~13 | ✅ 已完成 | SoftLexicon + 训练集词表 |
| 12-3周 | [12-3_soft_expert_joint.md](./12-3_soft_expert_joint.md) | 2025-12-14~20 | ⏳ 计划中 | Soft+Expert 联合 (RedJujube) |

### 总体规划

| 文档 | 说明 | 时间范围 |
|------|------|----------|
| [hz_lexicon_2weeks.md](./hz_lexicon_2weeks.md) | 两周实验总体计划 | 12-2 ~ 12-3周 |
| README.md | 本文档 - 计划汇总 | - |

---

## 📊 实验进度总览

### 已完成实验（12-1周）✅

**时间**: 2025-12-07 ~ 2025-12-08

**核心成果**:
1. ✅ 建立 HZ 和 MSRA 数据集基线
2. ✅ 对比手工 vs 自动专家词典
3. ✅ 完成词典覆盖率深度分析
4. ✅ 产出两份详细分析报告

**关键数据**:
- HZ Baseline: 95.618%
- HZ +ExpertDict(手工): **97.941%** (+2.323%)
- HZ +ExpertDict(自动): **97.050%** (+1.432%)
- MSRA +ExpertDict: **95.424%** (+0.332%)

**详细报告**:
- [12-1_baseline_expert_dict.md](./12-1_baseline_expert_dict.md)
- [NER实验结果综合对比报告](../results/NER_实验结果综合对比报告_20251208.md)
- [词典对比分析报告](../results/词典对比分析报告.md)

---

### 已完成实验（12-2周）✅

**时间**: 2025-12-09 ~ 2025-12-13

**核心成果**:
1. ✅ 实现 SoftLexicon 软词典方法
2. ✅ 对比 CTB 大词表 vs 训练集词表
3. ✅ RedJujube 数据集验证实验
4. ✅ 生成详细实验报告

**关键数据**:

*HZ 数据集*:
- SoftLexicon (CTB): **95.88%**
- SoftLexicon (TrainLex): **96.57%** (+0.95% vs Baseline)
- 自动 ExpertDict: **97.050%** (公平对比主参照)

*RedJujube 数据集*:
- SoftLexicon (TrainLex): **96.07%** (+0.56% vs Baseline)
- 自动 ExpertDict: **96.99%** (+1.48% vs Baseline)
- 手动 ExpertDict: **97.04%** (+1.53% vs Baseline)

**详细报告**:
- [12-2_softlexicon.md](./12-2_softlexicon.md)
- [RedJujube_NER_实验报告](../results/RedJujube_NER_实验报告_20251212.md)

---

### 计划中实验（12-3周）⏳

**预计时间**: 2025-12-14 ~ 2025-12-20

**核心任务**:
1. 实现 SoftLexicon + ExpertDict 联合模型
2. 在 **RedJujube 数据集**上验证（作为主数据集）
3. 词典组合策略对比实验
4. 消融实验分析各组件贡献
5. 性能调优与阶段总结

**预期目标**:
- Soft+Expert 联合模型在 RedJujube 上达到 **97.2%+** F1
- 证明两种特征的互补性
- 找到最优词典组合策略
- 完成阶段性总结报告

**数据集说明**:
- 从 12-3周开始，**RedJujube** 作为主数据集
- RedJujube 是 HZ 数据集的更新版本，数据质量更高
- HZ 实验结果保留作为历史对比参考

**详细计划**:
- [12-3_soft_expert_joint.md](./12-3_soft_expert_joint.md)
- [hz_lexicon_2weeks.md - 12-3周](./hz_lexicon_2weeks.md#第-2-周soft-vs-expert-vs-softexpert--策略完善hz)

---

## 🎯 实验目标与关键问题

### 总体目标

围绕 HZ 数据集，系统对比不同词典特征方法：
- **Baseline**: 无词典
- **ExpertDict**: 专家词典（手工/自动）
- **SoftLexicon**: 软词典（CTB/训练集词表）
- **Soft+Expert**: 联合方案

验证关键策略：
- 软词典候选词严格来源于训练集
- 混合词典方案的有效性

### 核心问题

1. **词典方法对比**
   - ExpertDict vs SoftLexicon，哪个更好？
   - 手工词典 vs 自动词典，质量差异在哪？

2. **数据泄露问题**
   - 使用外部大词表（CTB）是否导致数据泄露？
   - 训练集词表能否达到相近效果？

3. **最优方案**
   - 单一词典方法的天花板在哪？
   - Soft+Expert 联合是否能进一步提升？

---

## 📈 实验结果汇总

### 性能对比（测试集 F1）

#### HZ 数据集

| 方法 | HZ 数据集 | 词表来源 | 状态 |
|------|----------|---------|------|
| Baseline | 95.618% | - | ✅ |
| +ExpertDict(手工) | **97.941%** | 手工标注 | ✅ |
| +ExpertDict(自动) | **97.050%** | 训练集提取 | ✅ |
| SoftLexicon(CTB) | **95.88%** | CTB 50d | ✅ |
| SoftLexicon(TrainLex) | **96.57%** | 训练集n-gram | ✅ |
| Soft+Expert | ⏳ 计划中 | 混合 | ⏳ |

#### RedJujube 数据集

| 方法 | RedJujube 数据集 | 词表来源 | 状态 |
|------|----------------|---------|------|
| Baseline | 95.51% | - | ✅ |
| SoftLexicon(TrainLex) | **96.07%** | 训练集n-gram | ✅ |
| +ExpertDict(自动) | **96.99%** | 训练集实体(min_freq=2) | ✅ |
| +ExpertDict(手工) | **97.04%** | 训练集全量实体 | ✅ |

### 词典覆盖率对比（HZ 测试集）

| 词典类型 | 词表大小 | 覆盖率 | 特点 |
|---------|---------|-------|------|
| 手工专家词典 | 2,887词 | 87.35% | 专业术语全面，GEO缺失 |
| 自动专家词典 | 1,945词 | **95.58%** | 高频实体好，泛化能力强 |
| CTB 词表 | 280,930词 | - | 外部大词表，可能数据泄露 |
| 训练集词表 | 197,972词 | - | 仅用训练集，避免泄露 |

---

## 📝 重要发现

### 12-1周核心发现

1. **领域特性决定词典效果**
   - 医疗领域（HZ）：专家词典提升显著 (+2.32%)
   - 通用领域（MSRA）：专家词典提升有限 (+0.33%)

2. **词典质量 > 词典大小**
   - 手工2,371词性能优于自动3,214词
   - 但自动词典覆盖率更高（+8.23%）

3. **手工词典存在系统性缺陷**
   - GEO（地理）类型完全缺失（0%）
   - 建议混合方案：自动（基础）+ 手工（专业术语）

### 12-2周核心发现

1. **SoftLexicon 表现中等**
   - HZ: CTB词表 95.88%，仅比Baseline提升0.26%
   - HZ: TrainLex词表 96.57%，提升0.95%
   - 在医疗领域效果不如ExpertDict

2. **训练集词表策略**
   - TrainLex 优于 CTB词表 (+0.69%)
   - 避免数据泄露问题
   - 但仍逊于自动ExpertDict (-0.48%)

3. **RedJujube 数据集验证结论一致** (新增)
   - ExpertDict 方法优于 SoftLexicon
   - RedJujube: 96.99% vs 96.07% (+0.92%)
   - HZ: 97.050% vs 96.57% (+0.48%)
   - 词典质量比规模更关键：ExpertDict 用2k-3k词优于 SoftLexicon 的20w词

4. **自动词典提取策略高效** (新增)
   - RedJujube: 自动(96.99%) vs 手动(97.04%)，仅差0.05%
   - min_freq=2 策略非常有效
   - 完全避免数据泄露，推荐作为最佳实践

---

## 🔗 相关资源

### 数据与代码
- **数据目录**: `/data/HZ/`
- **训练脚本**: `/scripts/train_hz_ner_baseline_vs_expert_dict.py`
- **词典提取**: `/scripts/extract_lexicon_from_training.py`
- **软词典提取**: `/scripts/extract_softlexicon_from_training.py`
- **覆盖率分析**: `/scripts/test_expert_dict_coverage.py`

### 实验结果
- **结果目录**: `/experiments/hz_lexicon/results/`
- **HZ 模型缓存**: `/cache/hz_softlexicon/`, `/cache/hz_ner_comparison/`
- **RedJujube 模型缓存**: `/cache/redjujube_ner_comparison/`

### 分析报告
- [NER实验结果综合对比报告](../results/NER_实验结果综合对比报告_20251208.md) - HZ数据集
- [RedJujube NER实验报告](../results/RedJujube_NER_实验报告_20251212.md) - RedJujube数据集
- [词典对比分析报告](../results/词典对比分析报告.md)
- [results/README.md](../results/README.md)

---

## 📅 时间线

```
2025-12-07 ━━━━━━┓
                 ┃ 第0周：基线与专家词典
2025-12-08 ━━━━━━┛   ✅ 已完成

2025-12-09 ━━━━━━┓
                 ┃
2025-12-10 ━━━━━━┫ 第1周：SoftLexicon
                 ┃   ✅ 已完成
2025-12-13 ━━━━━━┛

2025-12-14 ━━━━━━┓
                 ┃ 第2周：Soft+Expert (RedJujube)
2025-12-20 ━━━━━━┛   ⏳ 计划中
```

---

**文档维护**: eznlp 项目组  
**最后更新**: 2025-12-13  
**版本**: v1.1
