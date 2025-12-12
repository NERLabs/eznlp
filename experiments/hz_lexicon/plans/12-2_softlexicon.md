# 12-2周：SoftLexicon 软词典实验

**时间**: 2025-12-09 ~ 2025-12-13  
**状态**: 🔄 进行中  
**目标**: 验证 SoftLexicon 方法，对比 CTB 词表 vs 训练集词表

---

## 📋 周计划概览

### 实验目标
1. 实现 SoftLexicon 软词典方法
2. 对比两种词表来源：CTB 大词表 vs 训练集词表
3. 验证"训练集候选词"策略，避免数据泄露
4. 对比 SoftLexicon vs ExpertDict 的性能差异

### 关键问题
- SoftLexicon 在医疗领域（HZ）的效果如何？
- CTB 大词表是否导致数据泄露？
- 仅用训练集构建的词表能否达到相近效果？
- SoftLexicon 与 ExpertDict 相比，孰优孰劣？

---

## 🎯 任务列表

### 任务 1：环境确认 & 词向量准备 ✅

**执行时间**: 2025-12-09

**检查项目**:
- ✅ `assets/vectors/ctb.50d.vec` 存在且可正常加载
- ✅ `load_vectors("chinese", 50)` 功能正常
- ✅ 训练脚本可正常运行

**结果**: 环境就绪，词向量加载正常

---

### 任务 2：SoftLexicon (CTB 词表) 基线实验 ✅

**执行时间**: 2025-12-10

**实验配置**:
- 模型: MacBERT + BiLSTM + CRF + SoftLexicon
- 训练轮数: 30 epochs
- 批次大小: 16
- 学习率: 2e-3 (主网络) / 2e-5 (BERT)
- 词表来源: CTB 50d 词向量 (280,930 个词)

**执行命令**:
```bash
python scripts/train_hz_ner_baseline_vs_expert_dict.py \
  --data_dir data/HZ \
  --save_dir cache/hz_softlexicon \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 256 \
  --num_layers 1 \
  --dropout 0.5 \
  --expert_dict_dim 50 \
  --num_epochs 30 \
  --batch_size 16 \
  --lr 2e-3 \
  --finetune_lr 2e-5 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --disp_every_steps 50 \
  --eval_every_steps 200 \
  --seed 42 \
  --run_softlexicon
```

**实验结果**:

| 指标 | 数值 | 说明 |
|-----|------|------|
| 测试集 F1 | **95.88%** | 最终性能 |
| 验证集最佳 F1 | **96.96%** | 第22轮 |
| 训练集 F1 | 99.94% | 第30轮 |
| 词表大小 | 280,930词 | CTB 50d |

**实验目录**: `cache/hz_softlexicon/softlexicon_20251210-191021/`

**关键发现**:
- ✅ SoftLexicon 优于 Baseline (+0.262%)
- ⚠️ 提升幅度远小于 ExpertDict (+2.323%)
- ⚠️ 在医疗领域效果有限
- ⚠️ CTB 大词表可能存在数据泄露风险

---

### 任务 3：训练集词表提取脚本开发 ✅

**执行时间**: 2025-12-10

**开发内容**:
创建 `scripts/extract_softlexicon_from_training.py` 脚本

**功能特性**:
1. 从 BMES 标注提取实体作为候选词
2. 提取 n-gram (1~5字) 作为候选词
3. 频次过滤（min_freq=2）
4. 按频次排序输出

**执行命令**:
```bash
python scripts/extract_softlexicon_from_training.py \
  --train_path data/HZ/hz_train.bmes \
  --output_path data/HZ/softlexicon_train.txt \
  --min_freq 2 \
  --ngram_max_len 5
```

**提取结果**:

| 指标 | 数值 |
|-----|------|
| 词表大小 | 197,972 个词 |
| 唯一实体数 | 3,214 个 |
| 总频次 | 2,550,182 |
| 平均频次 | 12.88 |

**词长分布**:
- 1字词: 2,496 个
- 2字词: 30,046 个
- 3字词: 50,910 个
- 4字词: 57,033 个
- 5字词: 57,406 个

**输出文件**: `data/HZ/softlexicon_train.txt`

---

### 任务 4：SoftLexicon (训练集词表) 实验 ✅

**执行时间**: 2025-12-10 (已完成)

**实验配置**:
- 模型: MacBERT + BiLSTM + CRF + SoftLexicon
- 训练轮数: 30 epochs
- 词表来源: 训练集提取 (197,972 个词)
- 向量初始化: CTB 50d (有的用预训练，无的随机初始化)

**执行命令**:
```bash
python scripts/train_hz_ner_baseline_vs_expert_dict.py \
  --data_dir data/HZ \
  --save_dir cache/hz_softlexicon \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 256 \
  --num_layers 1 \
  --dropout 0.5 \
  --num_epochs 30 \
  --batch_size 16 \
  --lr 2e-3 \
  --finetune_lr 2e-5 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --disp_every_steps 50 \
  --eval_every_steps 200 \
  --seed 42 \
  --run_softlexicon_trainlex
```

**实验结果** (2025-12-10 20:36):
- 测试集 F1: **96.57%** (Metric 0 = 0.9657)
- 实验目录: `cache/hz_softlexicon/softlexicon_trainlex_20251210-195654/`

**待分析**:
- [x] 最终测试集 F1：**96.57%**
- [x] 与 CTB 词表的性能对比：TrainLex 96.57% vs CTB 95.88%（+0.69%）
- [ ] 词表覆盖率分析
- [ ] 训练曲线对比

---

### 任务 5：整理本周实验结果 ⏳

**计划时间**: 2025-12-13

**整理内容**:

1. **性能对比表**
   | 方法 | 测试集 F1 | 词表大小 | 词表来源 | 提升 |
   |------|----------|---------|---------|------|
   | Baseline | 95.618% | - | - | - |
   | SoftLexicon (CTB) | 95.88% | 280,930 | CTB 50d | +0.262% |
   | SoftLexicon (TrainLex) | 96.57% | 197,972 | 训练集 | +0.952% |
   | ExpertDict (手工) | 97.941% | 2,371 | 手工标注 | +2.323% |

2. **关键发现总结**
   - **SoftLexicon vs Baseline**: SoftLexicon(CTB 95.88%) 和 SoftLexicon(TrainLex 96.57%) 均优于 Baseline(95.618%)，其中 TrainLex 相比 Baseline 提升约 **+0.95%**。
   - **SoftLexicon vs ExpertDict（公平对比）**: 以自动 ExpertDict(97.050%) 为主，对比 SoftLexicon(TrainLex 96.57%) 时差距约 **-0.48%**；手工 ExpertDict(97.941%) 可能包含测试集信息，仅作上界参考。
   - **CTB 词表 vs 训练集词表**: TrainLex 在测试集上比 CTB 高约 **+0.69%** (96.57% vs 95.88%)，且词表规模更小、更聚焦于训练数据。
   - **数据泄露问题分析**: CTB 词表可能包含测试集词汇，存在潜在数据泄露风险；TrainLex 词表严格来源训练集，更符合数据规范，并且性能略优于 CTB。

3. **更新实验记录**
   - 更新 `experiments/hz_lexicon/results/README.md`
   - 保存实验数据到 results 目录

---

## 📊 周总结

### 核心成果 (截至 2025-12-10)

1. **成功实现 SoftLexicon 方法**
   - ✅ CTB 词表版本完成（F1: 95.88%）
   - ✅ 训练集词表版本完成（F1: 96.57%）

2. **开发了词表提取工具**
   - ✅ `extract_softlexicon_from_training.py`
   - ✅ 支持实体提取 + n-gram 生成
   - ✅ 频次过滤与统计分析

3. **建立了对比基线**
   - Baseline: 95.618%
   - SoftLexicon (CTB): 95.88%
   - SoftLexicon (TrainLex): 96.57%
   - ExpertDict (自动): 97.050%（训练集自动抽取，作为公平对比主参照）
   - ExpertDict (手工): 97.941%（可能包含测试集信息，仅作上界参考）

### 初步结论

1. **SoftLexicon 在医疗领域的效果**
   - SoftLexicon(CTB) 提升幅度: +0.262% (vs Baseline)，TrainLex 版本约 +0.95%。
   - 在公平对比下，SoftLexicon(TrainLex 96.57%) 仍低于自动 ExpertDict(97.050%)。
   - 手工 ExpertDict(+2.323%) 因可能包含测试集信息，仅作参考上界。

2. **CTB 大词表的利弊**
   - 优势: 词表全面，覆盖广
   - 劣势: 可能存在数据泄露风险
   - 词表大小: 280,930词（非常大）

3. **训练集词表策略的价值**
   - 目的: 避免数据泄露
   - 词表规模: 197,972词（更合理）
   - 性能: 测试集 F1 = 96.57%，略高于 CTB (95.88%)

### 待解决问题

1. **训练集词表实验结果**
   - ✅ 训练完成，测试集 F1 = 96.57%
   - ✅ 与 CTB 词表的性能对比完成（TrainLex +0.69%）
   - ⏳ 覆盖率对比

2. **方法选择建议**
   - 在医疗领域，ExpertDict 明显优于 SoftLexicon
   - 下周探索 Soft+Expert 联合方案

### 下周计划

- [ ] 完成 SoftLexicon-TrainLex 实验
- [ ] 详细对比 Soft vs Expert
- [ ] 实现 Soft+Expert 联合模型
- [ ] 探索混合词典优化方案

---

## 📁 相关资源

### 代码脚本
- 训练脚本: `scripts/train_hz_ner_baseline_vs_expert_dict.py`
- 词表提取: `scripts/extract_softlexicon_from_training.py`
- 向量工具: `eznlp/vectors.py`

### 数据文件
- 训练数据: `data/HZ/hz_train.bmes`
- CTB 词向量: `assets/vectors/ctb.50d.vec`
- 训练集词表: `data/HZ/softlexicon_train.txt`

### 实验结果
- SoftLexicon (CTB): `cache/hz_softlexicon/softlexicon_20251210-191021/`
- SoftLexicon (TrainLex): `cache/hz_softlexicon/softlexicon_trainlex_20251210-195715/`
- 结果记录: `experiments/hz_lexicon/results/`

### 分析报告
- 本周计划: [12-2_softlexicon.md](./12-2_softlexicon.md)
- 总体计划: [hz_lexicon_2weeks.md](./hz_lexicon_2weeks.md)
- 上周总结: [12-1_baseline_expert_dict.md](./12-1_baseline_expert_dict.md)

---

**进度状态**: 🔄 60% 完成  
**预计完成**: 2025-12-13  
**负责人**: eznlp 项目组