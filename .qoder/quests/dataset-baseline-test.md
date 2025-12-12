# 红枣数据集 NER Baseline 测试设计

## 需求概述

针对新的红枣数据集，进行命名实体识别（NER）任务的基线测试与对比实验，涵盖 Baseline、SoftLexicon、自动词典和手动词典四种配置，最终输出综合对比结果。

## 任务目标

执行以下四组对比实验：
1. Baseline 模型（无词典增强）
2. SoftLexicon 模型（使用软词典嵌入）
3. 自动词典模型（训练集自动提取的专家词典）
4. 手动词典模型（人工标注的专家词典）

完成后生成统一的实验结果对比报告。

## 数据准备规划

### 数据目录结构

创建新的数据集目录 `data/RedJujube`，按以下结构组织：

```
data/RedJujube/
├── redjujube_train.bmes      # 训练集（BMES 标注格式）
├── redjujube_dev.bmes         # 验证集
├── redjujube_test.bmes        # 测试集
├── expert_lexicon.txt         # 手动标注的专家词典
├── expert_lexicon_auto.txt    # 自动提取的专家词典（稍后生成）
└── softlexicon_train.txt      # 训练集软词典（稍后生成）
```

### 数据准备步骤

#### 步骤1：创建数据目录

在 `data/` 下创建 `RedJujube` 目录，用于存放所有相关数据文件。

#### 步骤2：准备标注数据

将红枣数据集按 BMES 标注格式转换后，分别保存为：
- `redjujube_train.bmes`
- `redjujube_dev.bmes`
- `redjujube_test.bmes`

每行格式为：`字符 标签`，句子之间空行分隔。

标注方案：
- B：实体开始
- M：实体中间
- E：实体结束
- S：单字实体

#### 步骤3：导入手动专家词典

将人工标注的专家词典保存为 `expert_lexicon.txt`，每行一个词条。

格式要求：
- 每行一个词语
- UTF-8 编码
- 去除空行和重复项

## 词典生成规划

### 自动词典提取

使用现有脚本 `scripts/extract_lexicon_from_training.py` 从训练集自动提取专家词典。

提取策略：
- 从 BMES 标注中提取所有实体
- 频次过滤（建议 min_freq=2）
- 按频次降序排序

输出文件：`data/RedJujube/expert_lexicon_auto.txt`

### 软词典生成

使用现有脚本 `scripts/extract_softlexicon_from_training.py` 从训练集生成软词典。

生成策略：
- 从 BMES 标注提取实体作为候选词
- 提取 n-gram（1-5字）作为候选词
- 频次过滤（建议 min_freq=2）
- 按频次降序排序

输出文件：`data/RedJujube/softlexicon_train.txt`

## 实验配置规划

### 实验1：Baseline 模型

模型架构：MacBERT + BiLSTM + CRF

配置参数：
- BERT 模型：hfl/chinese-macbert-base
- LSTM 隐藏层维度：256
- LSTM 层数：1
- Dropout：0.5
- 训练轮数：30
- 批次大小：16
- 主网络学习率：2e-3
- BERT 微调学习率：2e-5
- 权重衰减：1e-4
- 梯度裁剪：5.0

特性：不使用任何词典特征，作为基准对比。

### 实验2：SoftLexicon 模型

模型架构：MacBERT + BiLSTM + CRF + SoftLexicon

配置参数：
- 基础参数同 Baseline
- 软词典嵌入维度：50
- 聚合模式：wtd_mean_pooling（加权平均池化）
- 词表来源：训练集提取的软词典
- 向量初始化：CTB 50d 词向量（存在于词表中的使用预训练向量，否则随机初始化）

特性：使用软词典嵌入提供词汇级别的特征增强。

### 实验3：自动词典模型

模型架构：MacBERT + BiLSTM + CRF + ExpertDict

配置参数：
- 基础参数同 Baseline
- 专家词典嵌入维度：50
- 聚合模式：wtd_mean_pooling
- 词典来源：从训练集自动提取的实体词典

特性：使用自动提取的专家词典特征，避免数据泄露。

### 实验4：手动词典模型

模型架构：MacBERT + BiLSTM + CRF + ExpertDict

配置参数：
- 基础参数同 Baseline
- 专家词典嵌入维度：50
- 聚合模式：wtd_mean_pooling
- 词典来源：人工标注的专家词典

特性：使用人工构建的专家词典，验证领域知识的性能上界。

## 实验执行流程

### 第一阶段：数据与词典准备

#### 任务1：创建数据目录
在 `data/` 下创建 `RedJujube/` 目录。

#### 任务2：导入标注数据
将 BMES 格式的训练、验证、测试集放入数据目录。

#### 任务3：导入手动词典
将人工标注的专家词典保存到 `data/RedJujube/expert_lexicon.txt`。

#### 任务4：生成自动词典
执行词典提取命令，从训练集自动生成专家词典。

#### 任务5：生成软词典
执行软词典提取命令，从训练集生成 n-gram 候选词表。

### 第二阶段：模型训练与评估

#### 任务6：Baseline 实验
训练不含词典特征的基准模型，记录验证集和测试集性能。

#### 任务7：SoftLexicon 实验
训练使用软词典嵌入的模型，对比 Baseline 性能提升。

#### 任务8：自动词典实验
训练使用自动提取专家词典的模型，验证自动化方案的效果。

#### 任务9：手动词典实验
训练使用人工标注专家词典的模型，验证领域知识的价值。

### 第三阶段：结果汇总与分析

#### 任务10：收集实验结果
从各实验目录提取关键指标：测试集 F1、Loss、参数量、训练时间等。

#### 任务11：生成对比报告
整理所有实验结果，生成统一的对比分析表格。

#### 任务12：结论总结
分析各方法的性能差异，给出最佳方案建议。

## 预期输出

### 实验结果目录

所有实验结果统一保存在 `cache/redjujube_ner_comparison/` 下：

```
cache/redjujube_ner_comparison/
├── baseline_<timestamp>/
│   ├── best_model.pt
│   ├── results.json
│   └── training.log
├── softlexicon_trainlex_<timestamp>/
│   ├── best_model.pt
│   ├── results.json
│   └── training.log
├── expert_dict_auto_<timestamp>/
│   ├── best_model.pt
│   ├── results.json
│   └── training.log
├── expert_dict_manual_<timestamp>/
│   ├── best_model.pt
│   ├── results.json
│   └── training.log
└── comparison_<timestamp>.json
```

### 实验对比表

所有实验完成后，生成如下格式的对比表：

| 方法 | 测试集 F1 | 测试集 Loss | 词表大小 | 词表来源 | F1 提升 | 参数量 |
|------|----------|------------|---------|---------|---------|--------|
| Baseline | X.XX% | X.XXXX | - | - | - | XXX,XXX |
| SoftLexicon (TrainLex) | X.XX% | X.XXXX | ~200K | 训练集 | +X.XX% | XXX,XXX |
| ExpertDict (自动) | X.XX% | X.XXXX | ~3K | 训练集 | +X.XX% | XXX,XXX |
| ExpertDict (手动) | X.XX% | X.XXXX | ~2K | 人工 | +X.XX% | XXX,XXX |

### 分析结论

对比分析应涵盖：
1. 各方法相对 Baseline 的性能提升
2. SoftLexicon 与 ExpertDict 的效果对比
3. 自动词典与手动词典的性能差距
4. 词典规模与性能增益的关系
5. 针对红枣数据集的最佳方案推荐

## 技术依赖

### 核心脚本

- 训练脚本：`scripts/train_hz_ner_baseline_vs_expert_dict.py`
- 自动词典提取：`scripts/extract_lexicon_from_training.py`
- 软词典提取：`scripts/extract_softlexicon_from_training.py`

### 配置模块

- 模型配置：`eznlp.model.ExtractorConfig`
- BERT 配置：`eznlp.model.BertLikeConfig`
- 专家词典配置：`eznlp.model.ExpertDictConfig`
- 软词典配置：`eznlp.model.SoftLexiconConfig`
- 解码器配置：`eznlp.model.decoder.SequenceTaggingDecoderConfig`

### 数据处理

- IO 处理：`eznlp.io.ConllIO`
- 词典分词器：`eznlp.token.LexiconTokenizer`
- 数据集构建：`eznlp.dataset.Dataset`

### 训练工具

- 训练器：`eznlp.training.Trainer`
- 向量加载：`utils.load_vectors`

## 关键注意事项

### 数据泄露防护

1. 自动词典必须仅从训练集提取
2. 软词典候选词表必须仅从训练集生成
3. 手动词典应避免包含测试集特有的词汇

### 实验公平性

1. 所有实验使用相同的随机种子（seed=42）
2. 所有实验使用相同的训练参数配置
3. 所有实验使用相同的评估指标

### 性能预期

基于 HZ 数据集的历史经验：
- Baseline F1 预期：~95.6%
- SoftLexicon 提升：+0.3% ~ +1.0%
- ExpertDict (自动) 提升：+1.0% ~ +1.5%
- ExpertDict (手动) 提升：+2.0% ~ +2.5%

实际效果可能因数据集特性而异，需以实验结果为准。

## 任务执行清单

### 准备阶段
- [ ] 创建 `data/RedJujube/` 目录
- [ ] 导入训练集、验证集、测试集（BMES 格式）
- [ ] 导入手动专家词典 `expert_lexicon.txt`
- [ ] 生成自动专家词典 `expert_lexicon_auto.txt`
- [ ] 生成软词典 `softlexicon_train.txt`

### 实验阶段
- [ ] 执行 Baseline 实验
- [ ] 执行 SoftLexicon 实验
- [ ] 执行自动词典实验
- [ ] 执行手动词典实验

### 分析阶段
- [ ] 收集所有实验结果
- [ ] 生成对比分析表
- [ ] 编写结论与建议

## 成功验收标准

1. 所有四组实验成功完成训练并保存结果
2. 测试集评估指标完整记录
3. 生成统一的对比分析报告
4. 实验结果文件完整保存在指定目录
5. 对比表格清晰展示各方法的性能差异
