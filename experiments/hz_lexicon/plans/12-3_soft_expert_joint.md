# 12-3周：ExpertDict深度优化 + FLAT模型探索

**时间**: 2025-12-14 ~ 2025-12-20  
**状态**: 🔄 进行中  
**数据集**: RedJujube (主数据集) + MSRA-ER (验证数据集)  
**目标**: ExpertDict深度优化 + FLAT模型复现探索 + SOTA技巧集成 → 冲击98% F1

---

## 📋 周计划概览（基于12-13实验结果调整 + FLAT探索）

### ⚠️ 重要策略调整

**基于12-13实验的关键发现**：
1. ✅ **ExpertDict在RedJujube上达到97.00%**（超过MSRA SOTA 96.72%）🏆
2. ✅ **ExpertDict在MSRA-ER上达到95.42%**（30 epochs，超过Baseline 0.33%）
3. ❌ SoftLexicon在RedJujube上**完全无效**（所有版本<Baseline）
4. ❌ Soft+Expert融合**不如**单独ExpertDict（96.87% vs 97.00%）
5. ✅ ExpertDict(自动)**极其稳定**（3次实验97.00% ± 0.01%）
6. 💡 SoftLexicon是"拖累"而非"增益"

**因此调整本周计划**:
- ❌ **放弃**: Soft+Expert联合模型（已验证无效）
- ❌ **放弃**: SoftLexicon优化（方法本身有问题）
- ✅ **新方向1**: ExpertDict深度优化（超参数、对抗训练、集成）
- ✅ **新方向2**: SOTA技巧集成（Dice Loss、Boundary Smoothing）
- ✅ **新方向3**: 跨数据集验证（RedJujube + MSRA-ER）
- 🔬 **探索方向**: FLAT (Flat-Lattice Transformer) 模型复现

---

### 实验目标（调整后 + FLAT探索）
1. ✅ 验证ExpertDict的稳定性（**已完成**：97.00% ± 0.01%）
2. 🔥 **深度错误分析**（最重要！理解ExpertDict为什么有效）
3. ⚡ **超参数优化**（词典规模、嵌入维度、频次阈值、聚合方式）
4. 🛡️ **对抗训练**（FGM/PGD，预期+0.2~0.5% F1）
5. 🏆 **模型集成**（多seed集成，预期+0.3~0.8% F1）
6. 💡 **可选探索**：轻量级Attention增强、多头ExpertDict
7. 🔬 **FLAT模型复现**：探索Lattice Transformer架构（技术储备）
8. 📊 撰写完整的阶段总结报告

### 关键问题（调整后）
- 🔍 ExpertDict解决了哪些Baseline解决不了的问题？（错误分析）
- 🔍 ExpertDict在不同实体类型上的性能差异？（细粒度分析）
- ⚙️ ExpertDict的最优词典规模？（min_freq=1/2/3/5对比）
- ⚙️ ExpertDict嵌入维度的最优配置？（emb_dim=25/50/100/150）
- ⚙️ 最优聚合方式是什么？（wtd_mean/mean/max pooling）
- 🛡️ 对抗训练能提升多少性能？（FGM/PGD）
- 🏆 多模型集成能达到什么水平？（预期97.5%+）

### 数据集说明
**为什么选择 RedJujube？**
- RedJujube 是 HZ 数据集的更新版本
- 数据质量更高，标注更准确
- 样本规模：训练5,372 / 验证671 / 测试672
- 实体类型：14类医疗实体
- 已有完整的 Baseline / SoftLexicon / ExpertDict 对比实验

---

## ✅ 12-13实验回顾（已完成）

### 已完成实验（14个）

#### 1. MSRA-ER数据集验证 ✅ 成功

| 实验ID | 模型配置 | Epochs | 测试集P | 测试集R | 测试集F1 | 说明 |
|---------|---------|--------|---------|---------|-----------|------|
| 20251207-233355 | MacBERT + ExpertDict | 10 | 95.802% | 94.888% | **95.343%** | 10轮训练 |
| 20251208-002945 | MacBERT Baseline | 10 | 95.764% | 94.726% | 95.242% | 10轮基线 |
| 20251208-011438 | MacBERT + ExpertDict | 30 | 96.050% | 94.807% | **95.424%** 🏆 | 30轮训练 |
| 20251208-091001 | MacBERT Baseline | 30 | 95.200% | 94.985% | 95.092% | 30轮基线 |

**关键发现**：
- ✅ ExpertDict在30 epochs上达到**95.424% F1**
- ✅ 超过Baseline **0.332%** (95.424% vs 95.092%)
- ✅ 证明ExpertDict在**通用NER数据集**上也有效
- ✅ 与RedJujube结论一致：词典特征能有效提升性能
- 💡 30 epochs比10 epochs效果更好（95.424% vs 95.343%）

**与MSRA SOTA对比**：
- MSRA SOTA（Dice Loss）：96.72%
- 我们的ExpertDict：95.424%
- 差距：-1.296%
- 🎯 **优化空间：通过SOTA技巧集成缩小差距**

---

#### 2. RedJujube数据集 - SoftLexicon词典优化 ❌ 失败

| 实验 | 词典规模 | 测试F1 | 结论 |
|------|---------|--------|------|
| v1-含标点 | 18,678 | 95.47% | 低于Baseline |
| v2-去标点 | 10,959 | 95.46% | 优化无效(-0.01%) |
| Balanced | 52,487 | 94.63% | 大词典反降 |
| **Baseline** | - | **95.51%** | 无词典更优 |

**结论**: SoftLexicon在RedJujube上完全无效，应放弃。

#### 3. RedJujube数据集 - 融合策略对比 ⚠️ 不如单独ExpertDict

| 融合方法 | 测试F1 | 与ExpertDict对比 |
|---------|--------|----------------|
| Concat | 96.87% | -0.13% |
| Weighted | 96.72% | -0.28% |
| Gated | 96.46% | -0.54% |
| Attention | 96.53% | -0.47% |
| **ExpertDict(单独)** | **97.00%** | **最优** 🏆 |

**结论**: 融合SoftLex拖累了ExpertDict性能。

#### 4. RedJujube数据集 - ExpertDict稳定性验证 ✅ 极其稳定

| 实验 | Seed | 测试F1 | 状态 |
|------|------|--------|------|
| Run1 | 42 | 96.99% | ✅ |
| Run2 | 123 | 97.01% | ✅ |
| Run3 | 456 | 97.01% | ✅ |
| **平均** | - | **97.00% ± 0.01%** | **非常稳定** |

**结论**: ExpertDict(自动)是最优且最稳定的方案。

**实验文档**:
- ✅ [`SoftLexicon_Versions_Comparison.md`](../analysis/SoftLexicon_Versions_Comparison.md)
- ✅ [`RedJujube_Complete_Experiments_20251213.md`](../results/RedJujube_Complete_Experiments_20251213.md)
- ✅ [`NFLAT_Code_Analysis.md`](../analysis/NFLAT_Code_Analysis.md)
- ✅ [`FLAT模型搭建实验记录_20251215.md`](../results/12-3_expert_optimization/FLAT模型搭建实验记录_20251215.md)

---

## 🔬 FLAT模型探索（并行工作）

### 实验状态: ⏸️ 暂停（技术问题待解决）

#### 已完成工作 (2025-12-13 ~ 2025-12-15)

**1. 代码架构实现** ✅
- ✅ 核心模块实现（约12小时）
  - Lattice基础模块: `_4MODELS/block/lattice_modules.py` (500行)
  - Lattice注意力: `_4MODELS/block/lattice_attention.py` (600行)
  - FLAT完整模型: `_4MODELS/models/flat_extractor.py` (800行)
- ✅ 数据处理模块（约8小时）
  - FLAT数据处理器: `_4MODELS/models/flat_data_processor.py` (400行)
  - YJ词典准备与格式转换
- ✅ 模型构建器（约6小时）
  - 模型工厂类: `_4MODELS/models/flat_model_builder.py` (350行)
  - 配置文件: `_1CONFIG/redjujube/flat_redjujube_config.json`
  - 使用文档: `_4MODELS/models/FLAT_MODEL_README.md` (336行)

**2. NFLAT深度研究** ✅
- ✅ 深度代码分析（约10小时）
  - NFLAT代码分析: `experiments/hz_lexicon/analysis/NFLAT_Code_Analysis.md` (638行)
  - 快速参考手册: `experiments/hz_lexicon/analysis/NFLAT_Quick_Reference.md` (339行)
  - 实施指南与改进代码: `nflat_improvements.py` (400行)
- ✅ 架构对比分析（约3小时）
  - 可视化对比: `experiments/hz_lexicon/analysis/ARCHITECTURE_COMPARISON.md` (350行)
  - FLAT vs NFLAT vs ExpertDict对比

**3. 训练尝试** ❌ 失败
- ⏸️ 首次训练 (2025-12-14 16:00)
  - 模型: FLAT + BERT
  - 错误: TransformerEncoderLayer接口不匹配
  - 状态: 阻塞，需修复接口问题

**工作量统计**:
- 总代码量: ~4,000行代码 + ~2,000行文档
- 总耗时: 约50小时（6.25个工作日）
- 主要产出: 完整的FLAT实现 + NFLAT改进方案

#### 关键问题与解决方案

**阻塞问题**: TransformerEncoderLayer接口不匹配
```
TypeError: TransformerEncoderLayer.__init__() got an unexpected keyword argument 'max_seq_len'
```
- **影响**: ❌ 无法训练FLAT+BERT模型
- **状态**: ⏸️ 暂停，待修复
- **预估时间**: 2-4小时

**次要问题**:
1. ✅ Lattice词汇匹配效率低 → 已优化（限制词长和数量）
2. ✅ 内存占用过大 → 已解决（限制词汇数，batch_size=8）

#### FLAT vs ExpertDict 初步对比

| 维度 | FLAT (预期) | ExpertDict (已验证) | 优势方 |
|------|-----------|------------------|--------|
| 性能 | 94-96.5% | **97.00%** | **ExpertDict** 🏆 |
| 训练时间 | 6-12小时 | 1.5小时 | ExpertDict |
| GPU内存 | 12-20GB | 8GB | ExpertDict |
| 实施难度 | 高 | 低 | ExpertDict |
| 复杂度 | O((n+m)²) | O(n) | ExpertDict |
| 理论价值 | Lattice架构 | 精选词典 | - |

**结论**: 
- ExpertDict在性能、效率、可用性上全面优于FLAT
- FLAT的价值在于技术储备和架构理解
- 建议: 优先优化ExpertDict，FLAT作为对比基准

#### 下一步计划（FLAT相关）

**Priority 1**: 修复阻塞问题（1-2天）
- [ ] 解决TransformerEncoderLayer接口问题
- [ ] 完成FLAT Baseline首次训练
- [ ] 记录性能基准

**Priority 2**: 性能验证（可选，2-3天）
- [ ] FLAT Baseline vs ExpertDict对比
- [ ] FLAT + BERT性能评估
- [ ] 撰写完整对比分析报告

**Priority 3**: NFLAT改进集成（推荐，3-4天）
- [ ] 将NFLAT的Inter-Attention应用到ExpertDict
- [ ] 层次化特征融合
- [ ] 预期提升0.3-0.5% F1

**详细记录**: [FLAT模型搭建实验记录_20251215.md](../results/12-3_expert_optimization/FLAT模型搭建实验记录_20251215.md)

---

## 🎯 任务列表（调整后）

### 任务 1：环境准备与数据集验证 ✅

**完成时间**: 2025-12-13

**检查项目**:
- ✅ 确认 RedJujube 数据集路径和格式
- ✅ 验证已有实验结果可复现
  - Baseline: 95.51% ✅
  - ExpertDict(自动): 97.00% ± 0.01% ✅
  - SoftLexicon: 95.47% ✅ (但无效)
- ✅ 准备词典文件
  - 自动专家词典: `expert_lexicon_auto.txt` (2,078词)
  - ~~训练集软词典~~: 已放弃

**输出**:
- ✅ 环境检查清单确认
- ✅ 数据集统计信息确认
- ✅ ExpertDict稳定性验证完成

---

### 任务 2：~~实现 Soft+Expert 联合模型~~ ❌ 已放弃

**原计划时间**: 2025-12-15~16  
**状态**: ❌ 已放弃

**放弃原因**:
- ✅ 已完成4种融合策略对比（Concat/Weighted/Gated/Attention）
- ❌ 所有融合方案均低于单独ExpertDict（96.87% vs 97.00%）
- ❌ SoftLexicon是"拖累"而非"增益"
- 💡 ExpertDict(自动)已是最优且最稳定的方案

**已完成实验**:
- ✅ Concat：96.87%
- ✅ Weighted：96.72%
- ✅ Gated：96.46%
- ✅ Attention：96.53%

---

### 任务 3：~~词典组合策略对比实验~~ ✅ 已完成

**完成时间**: 2025-12-13

**完成实验**:

| 实验编号 | SoftLexicon 词典 | ExpertDict 词典 | 测试F1 | 结论 |
|---------|-----------------|----------------|--------|------|
| SoftLex-v1 | 含标点(18k) | - | 95.47% | 无效 |
| SoftLex-v2 | 去标点(11k) | - | 95.46% | 无效 |
| Balanced | 平衡版(52k) | - | 94.63% | 反降 |
| **ExpertDict-Auto** | - | **自动(2k)** | **97.00%** | **最优** 🏆 |
| Concat | v1(18k) | 自动(2k) | 96.87% | 拖累 |

**分析维度**:
1. ✅ 测试集 F1 对比：ExpertDict最优
2. ✅ 参数量对比：ExpertDict最小（103.3M vs 113.1M）
3. ✅ 稳定性分析：ExpertDict极其稳定（±0.01%）

---

### 任务 4：深度错误分析 🔥 优先级最高

**计划时间**: 2025-12-14 (Day 1-2)

**分析内容**:

1. **错误模式分析** 🔍
   - [ ] 创建 `error_analysis.py` 脚本
   - [ ] 分析Baseline错误 & ExpertDict正确的样本（ExpertDict解决了什么）
   - [ ] 分析Baseline正确 & ExpertDict错误的样本（ExpertDict引入了什么问题）
   - [ ] 分析两者都错误的样本（模型的根本弱点）
   
2. **实体类型细粒度分析** 📊
   - [ ] 14类实体的性能对比（Baseline vs ExpertDict）
   - [ ] 哪些实体类型受益最大？（如人名、地名、机构名）
   - [ ] 哪些实体类型没改善甚至变差？
   
3. **错误类型统计** 📈
   - [ ] 边界错误（B/E标签错误）
   - [ ] 类型错误（识别出实体但类型错）
   - [ ] 漏检（False Negative）
   - [ ] 误检（False Positive）

4. **词典覆盖率分析** 🎯
   - [ ] 训练集词典覆盖率
   - [ ] 测试集词典覆盖率
   - [ ] 未覆盖实体的特征分析
   - [ ] 词典噪声分析（哪些词无效）

5. **Case Study** 📝
   - [ ] 选择20-30个典型样本
   - [ ] 详细分析ExpertDict如何帮助识别实体
   - [ ] 可视化展示（标注对比）

**预期输出**:
- ✅ `error_analysis.py` 脚本
- ✅ `ExpertDict_Error_Analysis_Report.md` 报告
- ✅ 14类实体性能对比表
- ✅ 典型Case Study文档
- ✅ **论文最重要的分析章节**

---

### 任务 5：超参数优化实验 ⚡ 快速见效

**计划时间**: 2025-12-15 (Day 3-4)

**优化方向**:

1. **ExpertDict嵌入维度优化** 🎯
   ```bash
   实验组：
   - emb_dim=25  → 总特征100维（更轻量）
   - emb_dim=50  → 总特征200维（当前基线）✅
   - emb_dim=100 → 总特征400维（更强表达）
   - emb_dim=150 → 总特征600维（更强表达）
   ```
   - [ ] 创建批量实验脚本 `run_expert_emb_dim_tuning.sh`
   - [ ] 运行4个实验并记录结果
   - [ ] 分析维度与性能的关系

2. **词典规模优化（min_freq阈值）** 📏
   ```python
   实验组：
   - min_freq=1 → 更大词典（约3000+词，高召回）
   - min_freq=2 → 当前基线（2,078词）✅
   - min_freq=3 → 更精准（约1500词）
   - min_freq=5 → 高置信度（约800词）
   ```
   - [ ] 修改 `extract_lexicon_from_training.py`
   - [ ] 生成4个不同规模的词典
   - [ ] 运行对比实验

3. **聚合方式优化（agg_mode）** 🔄
   ```python
   实验组：
   - agg_mode="wtd_mean_pooling"  # 当前（加权平均）✅
   - agg_mode="mean_pooling"      # 简单平均
   - agg_mode="max_pooling"       # 最大池化
   ```
   - [ ] 修改配置并运行实验
   - [ ] 对比不同聚合方式的效果

**预期收益**:
- 嵌入维度优化：+0.1~0.3% F1
- 词典规模优化：+0.1~0.5% F1
- 聚合方式优化：+0.05~0.2% F1
- **总计预期**：找到最优配置，可能达到97.2~97.5% F1

---

### 任务 6：对抗训练 🛡️ 通用提升技巧

**计划时间**: 2025-12-16 (Day 5 上午)

**实施方案**:

1. **FGM对抗训练** ⚔️
   ```python
   # 在BERT embedding上添加扰动
   class FGM:
       def attack(self, epsilon=0.5):
           for name, param in model.named_parameters():
               if 'word_embeddings' in name:
                   grad = param.grad
                   perturb = epsilon * grad / (grad.norm() + 1e-8)
                   param.data.add_(perturb)
   ```
   - [ ] 集成FGM到训练脚本
   - [ ] 运行对抗训练实验
   - [ ] 对比有/无对抗训练的差异

2. **PGD对抗训练**（可选）
   - [ ] 如果FGM有效，尝试更强的PGD

**预期收益**:
- FGM: +0.2~0.5% F1
- 不改架构，通用提升技巧
- 提升模型鲁棒性

---

### 任务 7：模型集成 🏆 终极武器

**计划时间**: 2025-12-16 (Day 5 下午) ~ 2025-12-17 (Day 6)

**集成方案**:

1. **多seed集成** 🎲
   ```python
   # 训练5个不同seed的模型
   model1: seed=42,  emb_dim=100, F1=?
   model2: seed=123, emb_dim=100, F1=?
   model3: seed=456, emb_dim=100, F1=?
   model4: seed=789, emb_dim=100, F1=?
   model5: seed=999, emb_dim=100, F1=?
   
   # 集成策略
   方案A：标签级投票
   方案B：Logits级平均
   ```
   - [ ] 创建批量训练脚本 `run_expert_ensemble.sh`
   - [ ] 训练5个模型
   - [ ] 实现集成脚本 `ensemble_predict.py`
   - [ ] 运行集成预测

2. **配置集成**（可选）🔧
   ```python
   # 不同配置的模型
   model_A: emb_dim=50,  min_freq=2
   model_B: emb_dim=100, min_freq=2
   model_C: emb_dim=100, min_freq=3
   model_D: emb_dim=150, min_freq=2
   model_E: emb_dim=100, min_freq=1
   ```

**预期收益**:
- 多seed集成：+0.3~0.8% F1
- **最稳定的提升方式**
- **预期达到97.5~97.8% F1**

---

### 任务 8：可选探索实验 💡

**计划时间**: 2025-12-18 (Day 7，如果时间充裕)

**可选方向**:

1. **轻量级Attention增强** ✨
   ```python
   class LightweightLexiconEnhancement(nn.Module):
       def forward(self, bert_feat, expert_feat):
           # 计算expert特征的重要性权重
           weight = sigmoid(Linear(concat([bert, expert])))
           # 加权增强BERT特征
           enhanced = bert + weight * project(expert)
           return enhanced
   ```
   - [ ] 实现轻量级Attention模块
   - [ ] 集成到训练脚本
   - [ ] 运行实验
   - **预期收益**：+0.05~0.2% F1

2. **多头ExpertDict** 🎭
   ```python
   # 不同粒度的词典特征
   ExpertDict_strict:  min_freq=5, emb_dim=50  # 高精度
   ExpertDict_balance: min_freq=2, emb_dim=50  # 平衡
   ExpertDict_recall:  min_freq=1, emb_dim=50  # 高召回
   # 融合三个特征
   fused = Attention([strict, balance, recall])
   ```
   - [ ] 实现多头ExpertDict
   - [ ] 运行实验
   - **预期收益**：+0.1~0.4% F1

3. **R-Drop正则化**（最简单）
   ```python
   # 同一样本两次前向传播，约束输出一致性
   loss = ce_loss + kl_divergence(logits1, logits2)
   ```
   - [ ] 集成R-Drop
   - **预期收益**：+0.1~0.3% F1

**决策原则**：
- 如果前面任务已达97.5%+，可跳过
- 如果还有提升空间，优先尝试R-Drop（最简单）

---

### 任务 9：整理实验结果与撰写报告 📊

**计划时间**: 2025-12-19~20 (Day 8-9)

**整理内容**:

1. **性能对比总表**（更新预期）
   
   | 方法 | RedJujube F1 | 词典大小 | 参数量 | 提升 | 状态 |
   |------|-------------|---------|--------|------|------|
   | Baseline | 95.51% | - | 103.1M | - | ✅ |
   | SoftLexicon (v1) | 95.47% | 18,678 | 113.1M | -0.04% | ✅ |
   | ExpertDict (自动) | **97.00%** | 2,078 | 103.3M | **+1.49%** | ✅ |
   | ExpertDict (优化配置) | **97.2~97.5%** | 待定 | 103.3M | **+1.7~2.0%** | ⏳ 目标 |
   | ExpertDict + FGM | **97.3~97.6%** | 2,078 | 103.3M | **+1.8~2.1%** | ⏳ 目标 |
   | **ExpertDict Ensemble** | **97.5~97.8%** | 2,078 | 516M×5 | **+2.0~2.3%** | 🎯 终极目标 |

2. **关键发现总结**（更新）
   - ✅ ExpertDict(自动)是最优且最稳定的方案（97.00% ± 0.01%）
   - ✅ SoftLexicon在RedJujube上完全无效（所有版本<Baseline）
   - ✅ 所有融合方案都不如单独ExpertDict（拖累效应）
   - ✅ 词典质量远比规模重要（2k词 > 52k词）
   - ⏳ 深度错误分析揭示ExpertDict的作用机制
   - ⏳ 超参数优化找到最优配置（预期+0.2~0.5%）
   - ⏳ 对抗训练提升鲁棒性（预期+0.2~0.5%）
   - ⏳ 模型集成达到97.8%的最终目标

3. **生成报告文档**
   - `experiments/hz_lexicon/results/ExpertDict_Deep_Optimization_20251220.md`
   - 包含详细实验配置、错误分析、超参数优化结果
   - 可视化图表：性能曲线、实体类型对比、Case Study
   - 对抗训练与集成学习分析

4. **更新总体文档**
   - 更新 `experiments/hz_lexicon/results/README.md`
   - 更新 `experiments/hz_lexicon/plans/README.md`
   - 更新 `experiments/hz_lexicon/plans/hz_lexicon_2weeks.md`

---

## 📊 预期成果（调整后）

### 1. 实验产出

**已完成（12-13）** ✅:
- ✅ 验证ExpertDict的稳定性（RedJujube: 3次实验97.00% ± 0.01%）
- ✅ MSRA-ER数据集验证（ExpertDict: 95.424%，超Baseline 0.33%）
- ✅ 完成4种融合策略对比（Concat/Weighted/Gated/Attention）
- ✅ 验证SoftLexicon在RedJujube上完全无效
- ✅ 生成详细的词典对比分析报告
- ✅ 分析NFLAT代码并提炼改进方案

**本周计划（12-14~20）** 🎯:

**Day 1-2 (12-14~15)：深度错误分析** 🔥
- [ ] 创建 `error_analysis.py` 脚本
- [ ] 分析RedJujube: Baseline vs ExpertDict
- [ ] 分析MSRA-ER: Baseline vs ExpertDict
- [ ] 生成 `ExpertDict_Error_Analysis_Report.md`
- [ ] 14类实体性能对比表（RedJujube）
- [ ] 3类实体性能对比表（MSRA-ER）
- [ ] 典型Case Study文档

**Day 3-4 (12-16~17)：超参数优化** ⚡
- [ ] emb_dim优化实验（25/50/100/150）
- [ ] min_freq优化实验（1/2/3/5）
- [ ] agg_mode优化实验（3种聚合方式）
- [ ] 找到最优配置组合

**Day 5 (12-18)：对抗训练 + SOTA技巧** 🛡️⭐
- [ ] 集成FGM对抗训练
- [ ] 实现Dice Loss
- [ ] 实现Boundary Smoothing
- [ ] 运行SOTA技巧实验

**Day 6 (12-19)：模型集成** 🏆
- [ ] 训练5个不同seed的模型
- [ ] 实现集成预测脚本
- [ ] 运行集成实验

**Day 7 (12-20)：报告撰写** 📊
- [ ] 整理所有实验结果
- [ ] 生成性能对比总表
- [ ] 撰写完整实验报告
- [ ] 更新计划文档

## 2. 关键结论（已确认）

**12-13实验结论** ✅:

1. **ExpertDict在两个数据集上都有效** ✅
   - RedJujube：97.00% (超过MSRA SOTA 96.72%)
   - MSRA-ER：95.424% (超过Baseline 0.33%)
   - 证明方法的通用性和有效性

2. **SoftLexicon在RedJujube上完全无效** ❌
   - 所有版本（v1/v2/Balanced）均低于Baseline（95.51%）
   - 去标点优化几乎无效（95.47% → 95.46%）
   - 问题在于n-gram匹配太粗糙，不在词典质量

3. **Soft+Expert融合不如单独ExpertDict** ⚠️
   - 最优融合（Concat）：96.87%
   - 单独ExpertDict：97.00% (+0.13%)
   - SoftLexicon是"拖累"而非"增益"

4. **ExpertDict(自动)极其稳定** ✅
   - RedJujube: 3次实验：96.99%, 97.01%, 97.01%
   - 平均：97.00% ± 0.01%
   - 远超所有其他方法

5. **词典质量比规模更重要** 💡
   - ExpertDict用**2k词**达到**97.00%** F1
   - SoftLexicon用**52k词**仅达到**94.63%** F1
   - 精选专家词典效率是大规模词表的26倍

**本周预期结论** 🎯:

6. **深度理解ExpertDict的作用机制**（错误分析）
   - ExpertDict解决了哪些类型的错误
   - 哪些实体类型受益最大
   - 词典覆盖率与性能的关系
   - 跨数据集的一致性分析

7. **找到ExpertDict的最优配置**（超参数优化）
   - 最优嵌入维度（预期emb_dim=100最优）
   - 最优词典规模（预期min_freq=2或3最优）
   - 最优聚合方式

8. **SOTA技巧集成效果**（Dice Loss + Boundary Smoothing）
   - Dice Loss: +0.1~0.3% F1
   - Boundary Smoothing: +0.1~0.2% F1
   - 组合使用预期提升更显著

9. **性能进一步提升**（对抗训练+集成）
   - RedJujube目标：
     - 对抗训练：预期达到97.2~97.5% F1
     - 模型集成：预期达到97.5~97.8% F1
     - **冲券98% F1大关**
   - MSRA-ER目标：
     - 当前：95.424%
     - 优化后：96.0~96.5% F1
     - **缩小与SOTA的差距**（从-1.3%到-0.2~0.7%）

10. **实践建议更新** 📋
    - 单模型最优：ExpertDict(优化配置) ≈ 97.5%
    - 集成最优：ExpertDict Ensemble ≈ 97.8%
    - 生产环境：根据资源选择单模型或集成
    - 论文贡献：错误分析 + 超参数优化 + 性能提升路线 + 跨数据集验证

---

## 🔗 相关资源

### 代码脚本
- 训练脚本: `scripts/train_redjujube_ner.py` (待创建/修改)
- 词表提取: `scripts/extract_softlexicon_from_training.py`
- 专家词典提取: `scripts/extract_entities_for_manual_dict.py`

### 数据文件
- 数据集: `data/RedJujube/`
- 自动专家词典: `data/RedJujube/expert_lexicon_auto.txt`
- 训练集软词典: `data/RedJujube/softlexicon_train.txt`
- CTB 词向量: `assets/vectors/ctb.50d.vec`

### 实验结果
- 已有实验: `cache/redjujube_ner_comparison/`
- 联合模型: `cache/redjujube_softlexicon_expert/` (计划中)
- 结果汇总: `experiments/hz_lexicon/results/`

### 历史报告
- 12-1周报告: [12-1_baseline_expert_dict.md](./12-1_baseline_expert_dict.md)
- 12-2周报告: [12-2_softlexicon.md](./12-2_softlexicon.md)
- RedJujube实验: [RedJujube_NER_实验报告_20251212.md](../results/RedJujube_NER_实验报告_20251212.md)

---

## 📝 补充说明

### RedJujube vs HZ 数据集对比

| 特性 | HZ 数据集 | RedJujube 数据集 |
|-----|----------|-----------------|
| 状态 | 历史版本 | 更新版本 ⭐ |
| 数据质量 | 较好 | 更优 |
| 训练样本 | 类似规模 | 5,372 |
| 实体类型 | 14类 | 14类 |
| 标注准确度 | 较高 | 更高 |
| 后续使用 | 参考对比 | **主数据集** |

### 为什么切换到 RedJujube？

1. **数据质量提升**: RedJujube 是 HZ 的改进版本，标注更准确
2. **一致性验证**: 已在 RedJujube 上完成基础实验，结论与 HZ 一致
3. **未来发展**: RedJujube 将作为后续所有实验的标准数据集
4. **可比性**: 保留 HZ 实验结果作为历史对比参考

---

**进度状态**: ⏳ 待开始  
**预计完成**: 2025-12-20  
**负责人**: eznlp 项目组  
**上一阶段**: [12-2周计划](./12-2_softlexicon.md) - SoftLexicon 实验已完成
