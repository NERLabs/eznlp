# 12-3周：Soft+Expert 联合模型与深度优化

**时间**: 2025-12-14 ~ 2025-12-20  
**状态**: ⏳ 计划中  
**数据集**: RedJujube (HZ数据集更新版本，作为主数据集)  
**目标**: 实现 SoftLexicon + ExpertDict 联合模型，探索混合词典方案，性能调优

---

## 📋 周计划概览

### 实验目标
1. 实现 SoftLexicon + ExpertDict 联合特征模型
2. 在 RedJujube 数据集上验证联合模型效果
3. 探索不同词典组合策略的性能差异
4. 进行消融实验，分析各组件贡献
5. 撰写完整的阶段总结报告

### 关键问题
- Soft+Expert 联合是否能进一步提升性能？
- 两种特征如何有效融合？互补性体现在哪里？
- 最优的词典组合策略是什么？
- RedJujube 数据集作为主数据集的优势在哪？

### 数据集说明
**为什么选择 RedJujube？**
- RedJujube 是 HZ 数据集的更新版本
- 数据质量更高，标注更准确
- 样本规模：训练5,372 / 验证671 / 测试672
- 实体类型：14类医疗实体
- 已有完整的 Baseline / SoftLexicon / ExpertDict 对比实验

---

## 🎯 任务列表

### 任务 1：环境准备与数据集验证 ⏳

**计划时间**: 2025-12-14

**检查项目**:
- [ ] 确认 RedJujube 数据集路径和格式
- [ ] 验证已有实验结果可复现
  - Baseline: 95.51%
  - ExpertDict(自动): 96.99%
  - SoftLexicon(TrainLex): 96.07%
- [ ] 准备词典文件
  - 自动专家词典: `expert_lexicon_auto.txt` (2,078词)
  - 训练集软词典: `softlexicon_train.txt` (198,437词)

**预期输出**:
- 环境检查清单
- 数据集统计信息确认

---

### 任务 2：实现 Soft+Expert 联合模型 ⏳

**计划时间**: 2025-12-15~16

**开发内容**:

1. **修改训练脚本** (`scripts/train_redjujube_ner.py` 或扩展现有脚本)
   - 新增 `--run_softlexicon_expert` 参数
   - 同时构建两种特征：
     ```python
     # SoftLexicon 特征
     entry["tokens"].build_softwords(soft_tokenizer.tokenize)
     entry["tokens"].build_softlexicons(soft_tokenizer.tokenize)
     
     # ExpertDict 特征
     entry["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
     ```

2. **配置模型架构**
   - 创建 `build_softlexicon_expert_config()` 函数
   - 在 `nested_oshots` 中同时添加两种嵌入配置
   - 确保特征维度合理配置

3. **训练配置**
   - 模型: MacBERT + BiLSTM + CRF + SoftLexicon + ExpertDict
   - 训练轮数: 30 epochs
   - 批次大小: 16
   - 学习率: 2e-3 (主网络) / 2e-5 (BERT)
   - 保存目录: `cache/redjujube_softlexicon_expert/`

**执行命令** (预期):
```bash
python scripts/train_redjujube_ner.py \
  --data_dir data/RedJujube \
  --save_dir cache/redjujube_softlexicon_expert \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 256 \
  --num_layers 1 \
  --dropout 0.5 \
  --num_epochs 30 \
  --batch_size 16 \
  --lr 2e-3 \
  --finetune_lr 2e-5 \
  --seed 42 \
  --run_softlexicon_expert \
  --softlex_path data/RedJujube/softlexicon_train.txt \
  --expert_dict_path data/RedJujube/expert_lexicon_auto.txt
```

**预期目标**:
- 测试集 F1 > 97.04% (超越单独 ExpertDict)
- 验证联合模型的互补性

---

### 任务 3：词典组合策略对比实验 ⏳

**计划时间**: 2025-12-16~17

**实验方案**:

对比不同的词典组合策略：

| 实验编号 | SoftLexicon 词典 | ExpertDict 词典 | 说明 |
|---------|-----------------|----------------|------|
| Exp-1 | TrainLex (full) | Auto (min_freq=2) | 完整组合 |
| Exp-2 | TrainLex (filtered) | Auto (min_freq=2) | 过滤低频词 |
| Exp-3 | TrainLex (full) | Manual (full) | 使用手工词典 |
| Exp-4 | - | Auto (min_freq=1) | 仅Expert，更大词表 |

**分析维度**:
1. 测试集 F1 对比
2. 训练时间对比
3. 参数量对比
4. 收敛速度分析

---

### 任务 4：消融实验 - 分析各组件贡献 ⏳

**计划时间**: 2025-12-17~18

**消融实验设计**:

| 模型配置 | Baseline | +SoftLex | +Expert | +Both | 说明 |
|---------|----------|----------|---------|-------|------|
| MacBERT | ✅ | ✅ | ✅ | ✅ | 基础编码器 |
| BiLSTM | ✅ | ✅ | ✅ | ✅ | 序列编码 |
| CRF | ✅ | ✅ | ✅ | ✅ | 序列标注 |
| SoftLexicon | ❌ | ✅ | ❌ | ✅ | 软词典特征 |
| ExpertDict | ❌ | ❌ | ✅ | ✅ | 专家词典特征 |
| **预期F1** | 95.51% | 96.07% | 96.99% | **>97%** | 目标 |

**分析内容**:
1. 各组件对性能的贡献度
2. SoftLex 和 Expert 的互补性分析
3. 错误类型分布对比
4. 不同实体类型的性能差异

**Case Study**:
- 选择 50-100 个测试样本
- 对比不同模型的预测结果
- 分析 SoftLex 和 Expert 分别擅长捕捉的实体类型

---

### 任务 5：性能调优实验 ⏳

**计划时间**: 2025-12-18~19

**调优方向**:

1. **超参数调优**
   - 学习率: [1e-3, 2e-3, 5e-3]
   - Dropout: [0.3, 0.5, 0.7]
   - BiLSTM 隐藏层维度: [128, 256, 512]

2. **词典参数调优**
   - SoftLexicon 嵌入维度: [25, 50, 100]
   - ExpertDict 嵌入维度: [25, 50, 100]
   - SoftLexicon 最小频次: [1, 2, 5]

3. **训练策略优化**
   - Warm-up 步数调整
   - 学习率调度策略
   - 早停策略优化

**目标**:
- 在 RedJujube 测试集上达到 **97.2%+** F1
- 超越所有单独方法

---

### 任务 6：整理实验结果与撰写报告 ⏳

**计划时间**: 2025-12-19~20

**整理内容**:

1. **性能对比总表**
   
   | 方法 | RedJujube F1 | 词典大小 | 参数量 | 提升 |
   |------|-------------|---------|--------|------|
   | Baseline | 95.51% | - | 103.1M | - |
   | SoftLexicon (TrainLex) | 96.07% | 198,437 | 113.1M | +0.56% |
   | ExpertDict (自动) | 96.99% | 2,078 | 103.3M | +1.48% |
   | ExpertDict (手动) | 97.04% | 3,389 | 103.3M | +1.53% |
   | **Soft+Expert (联合)** | **>97.2%** | 200k+ | ~113M | **>1.7%** |

2. **关键发现总结**
   - Soft+Expert 联合是否优于单独方法
   - 两种特征的互补性体现
   - 最优词典组合策略
   - RedJujube 数据集的特点分析

3. **生成报告文档**
   - `experiments/hz_lexicon/results/Soft_Expert_Joint_实验报告_20251220.md`
   - 包含详细实验配置、结果分析、可视化图表
   - 错误分析与Case Study

4. **更新总体文档**
   - 更新 `experiments/hz_lexicon/results/README.md`
   - 更新 `experiments/hz_lexicon/plans/README.md`
   - 更新 `experiments/hz_lexicon/plans/hz_lexicon_2weeks.md`

---

## 📊 预期成果

### 1. 实验产出

- ✅ 完成 Soft+Expert 联合模型实现
- ✅ 完成 4+ 组词典组合对比实验
- ✅ 完成详细消融实验
- ✅ 完成性能调优实验
- ✅ 生成完整实验报告

### 2. 关键结论（预期）

1. **联合模型性能**
   - Soft+Expert 联合模型在 RedJujube 上达到 **97.2%+** F1
   - 相比单独方法有显著提升
   - 证明两种特征具有互补性

2. **词典组合策略**
   - 最优组合: TrainLex (SoftLex) + Auto (ExpertDict)
   - 词典质量比规模更重要
   - 自动词典提取策略有效且安全

3. **RedJujube 数据集优势**
   - 数据质量高，标注准确
   - 实体类型丰富，覆盖全面
   - 适合作为医疗NER研究的标准数据集

4. **实践建议**
   - 生产环境: ExpertDict (自动) 作为首选
   - 研究实验: Soft+Expert 联合模型追求极致性能
   - 快速部署: Baseline 已足够优秀 (95.51%)

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
