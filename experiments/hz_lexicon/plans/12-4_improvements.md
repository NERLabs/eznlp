# 12-4周：传统NER方法改进实验

**时间**: 2025-12-13 ~ 2025-12-20  
**状态**: 🚀 进行中  
**目标**: 不使用大模型微调，仅通过传统方法优化提升NER性能

---

## 📋 改进方案总览

### 当前性能基线

| 方法 | 测试集F1 | 词典规模 | 状态 |
|------|---------|---------|------|
| Baseline | 95.51% | - | ✅ |
| SoftLexicon(TrainLex) | 96.07% | 198,437词 | ✅ |
| ExpertDict(自动) | **96.99%** | 2,078词 | ✅ 当前最优 |
| Soft+Expert(Concat) | 96.76% | 200k+词 | ✅ |
| Soft+Expert(Weighted) | 96.72% | 200k+词 | ✅ |
| Soft+Expert(Attention) | 96.53% | 200k+词 | ✅ |
| Soft+Expert(Gated) | ? | 200k+词 | ⏳ 训练中 |

### 改进目标

🎯 **终极目标**: 突破 **97.2% F1**（不使用大模型微调）

---

## 🚀 三大改进方向

### 方向1️⃣：优化软词典质量 ⭐⭐⭐⭐⭐

**问题诊断**：
- 当前SoftLexicon：198,437词（太大，质量参差不齐）
- 包含大量低频、低质量n-gram
- 引入噪声，干扰模型学习

**改进策略**：
```python
# 高质量过滤
- 提高频次阈值：min_freq 2 → 10
- 长度限制：只保留 2-4字词
- 实体优先：权重 3.0x
- PMI互信息过滤：阈值 2.0
- 停用词过滤
```

**预期效果**：
- 词典规模：198k → **18.6k**（缩小90%+）
- 性能提升：96.07% → **96.3~96.5%**
- 训练加速：10-20%

**执行命令**：
```bash
# 方案1a: 测试过滤版SoftLexicon
task train:redjujube:softlexicon-filtered

# 方案1b: 改进融合（ExpertDict + SoftLex过滤版）
task train:redjujube:fusion-improved
```

**预期提升**: +0.3~0.5% F1

---

### 方向2️⃣：超参数调优 ⭐⭐⭐⭐

**当前配置**（ExpertDict自动）：
```python
hid_dim: 256
num_layers: 1
dropout: 0.5
batch_size: 16
lr: 2e-3
```

**调优实验**：

| 实验 | 改动参数 | 预期效果 |
|------|---------|---------|
| Exp1 | hid_dim=384 | 更强表达能力 |
| Exp2 | num_layers=2 | 更深层次特征 |
| Exp3 | dropout=0.3 | 降低欠拟合 |
| Exp4 | batch_size=32 | 更稳定梯度 |

**执行命令**：
```bash
task train:redjujube:hyperparameter-tuning
```

**预期提升**: +0.1~0.3% F1

---

### 方向3️⃣：数据增强（简单版） ⭐⭐⭐

**策略**：
1. **同义词替换**
   - 红枣 ↔ 大枣
   - 种植 ↔ 栽培
   
2. **实体遮蔽（Entity Masking）**
   - 随机遮蔽10%实体，让模型从上下文学习
   
3. **上下文扩充**
   - 从训练集生成相似句子变体

**执行方式**：
```bash
# 需要开发数据增强脚本（约1-2天）
python _3DATA_PROCESS/data_augmentation.py \
  --train_path _2DATA/RedJujube/redjujube_train.bmes \
  --output_path _2DATA/RedJujube/redjujube_train_aug.bmes \
  --augmentation_ratio 0.3
```

**预期提升**: +0.2~0.5% F1

---

## 📅 执行计划

### 第1周（12-13 ~ 12-16）

**Day 1-2: 软词典优化** ✅
- [x] 创建过滤版提取脚本
- [x] 生成18.6k高质量词典
- [ ] 训练SoftLex(过滤版)模型
- [ ] 训练改进融合模型

**Day 3-4: 超参数调优**
- [ ] 运行4个超参数实验
- [ ] 分析结果，找到最优配置
- [ ] 可选：组合最优参数再训练

### 第2周（12-17 ~ 12-20）

**Day 5-6: 数据增强（可选）**
- [ ] 开发数据增强脚本
- [ ] 生成增强数据
- [ ] 训练验证效果

**Day 7: 结果汇总**
- [ ] 收集所有实验结果
- [ ] 生成完整对比报告
- [ ] 更新文档体系

---

## 🎯 预期最终性能

### 保守预期

| 改进方向 | 提升幅度 |
|---------|---------|
| SoftLex过滤版 | +0.3% |
| 超参数调优 | +0.2% |
| **总计** | **+0.5%** |

**最终性能**: 96.99% + 0.5% = **97.49% F1** ✅

### 乐观预期

| 改进方向 | 提升幅度 |
|---------|---------|
| SoftLex过滤版 | +0.5% |
| 超参数调优 | +0.3% |
| 数据增强 | +0.3% |
| **总计** | **+1.1%** |

**最终性能**: 96.99% + 1.1% = **98.09% F1** 🚀

---

## 📊 实验追踪

### ✅ 已完成

1. ✅ 创建高质量词典过滤脚本
2. ✅ 生成softlexicon_filtered.txt（18.6k词）
3. ✅ 创建训练脚本（3个方向）
4. ✅ 更新Taskfile任务

### ⏳ 进行中

1. ⏳ Gated融合模型训练（后台运行）

### 📋 待执行

1. [ ] SoftLex过滤版训练
2. [ ] 改进融合模型训练
3. [ ] 超参数调优实验
4. [ ] 结果收集与分析

---

## 🔗 相关文件

### 脚本文件
- `_3DATA_PROCESS/extract_softlexicon_filtered.py` - 过滤版词典提取
- `_1CONFIG/redjujube/run_softlexicon_filtered.sh` - SoftLex过滤版训练
- `_1CONFIG/redjujube/run_fusion_improved.sh` - 改进融合训练
- `_1CONFIG/redjujube/run_hyperparameter_tuning.sh` - 超参数调优

### 数据文件
- `_2DATA/RedJujube/softlexicon_filtered.txt` - 18.6k高质量词典
- `_2DATA/RedJujube/softlexicon_train.txt` - 198k原版词典
- `_2DATA/RedJujube/expert_lexicon_auto.txt` - 2k自动专家词典

### 结果目录
- `cache/redjujube_softlexicon_filtered/` - SoftLex过滤版结果
- `cache/redjujube_fusion_improved/` - 改进融合结果
- `cache/redjujube_expert_tuning/` - 超参数调优结果

---

## 💡 技术洞察

### 为什么软词典需要过滤？

1. **质量 > 数量**
   - ExpertDict: 2k词 → 96.99%
   - SoftLexicon: 200k词 → 96.07%
   - 说明词典质量远比规模重要

2. **噪声问题**
   - 低频n-gram引入噪声
   - 干扰模型学习正确特征
   - 增加过拟合风险

3. **计算效率**
   - 词典越大，特征维度越高
   - 训练速度变慢
   - 推理延迟增加

### 传统方法优化空间

在不使用大模型的前提下，传统NER方法仍有优化空间：

✅ **词典质量优化** - 本次重点  
✅ **超参数调优** - 本次重点  
✅ **数据增强** - 可选探索  
⚠️ **集成学习** - 训练成本高  
⚠️ **主动学习** - 需要标注资源  

---

## 📝 参考资料

### 相关论文
- SoftLexicon: Ma et al. (2020) "Simplify the Usage of Lexicon in Chinese NER"
- FLAT: Li et al. (2020) "Chinese NER using Flat-Lattice Transformer"
- Lattice-LSTM: Zhang & Yang (2018) "Chinese NER Using Lattice LSTM"

### 历史实验
- [12-1_baseline_expert_dict.md](./12-1_baseline_expert_dict.md) - ExpertDict基线
- [12-2_softlexicon.md](./12-2_softlexicon.md) - SoftLexicon实验
- [12-3_soft_expert_joint.md](./12-3_soft_expert_joint.md) - 融合实验

---

**更新时间**: 2025-12-13 20:10  
**负责人**: AI助手 + 用户  
**状态**: 🚀 进行中
