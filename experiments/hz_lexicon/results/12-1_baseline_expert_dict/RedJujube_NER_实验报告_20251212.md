# RedJujube NER Comparison

**实验日期**: 2025-12-12

---

## 📊 数据集信息

- **数据集名称**: RedJujube
- **训练集样本数**: 5,372
- **验证集样本数**: 671
- **测试集样本数**: 672
- **实体总数**: 56,644
- **实体类型数**: 14
- **平均句长**: 116.7

---

## 🎯 实验结果对比

| 方法 | 测试集 F1 | 测试Loss | 参数量 | 提升幅度 | 词典大小 | 词典来源 |
|------|----------|---------|--------|---------|---------|---------|
| Baseline | 95.51% | 10.729 | 103,057,976 | 0.00% | 0 | - |
| SoftLexicon (TrainLex) | 96.07% | 7.884 | 113,082,376 | +0.56% | 198,437 | 训练集 n-gram |
| ExpertDict (自动) | 96.99% | 8.330 | 103,264,176 | +1.48% | 2,078 | 训练集实体 (min_freq=2) |
| ExpertDict (手动) | 97.04% | 9.288 | 103,327,976 | +1.53% | 3,389 | 训练集全量实体 |

---

## 🏆 性能排名

| 排名 | 方法 | F1 Score | 相对提升 |
|-----|------|----------|---------|
| 1 | ExpertDict (手动) | 97.04% | +1.53% |
| 2 | ExpertDict (自动) | 96.99% | +1.48% |
| 3 | SoftLexicon (TrainLex) | 96.07% | +0.56% |
| 4 | Baseline | 95.51% | 0.00% |

---

## 💡 关键发现

1. ExpertDict (手动) 获得最佳性能: 97.04% F1
2. ExpertDict (自动) 性能接近手动版本，仅差 0.05%
3. ExpertDict 方法优于 SoftLexicon，提升幅度更大 (+1.48% vs +0.56%)
4. 词典质量比规模更重要：ExpertDict 用 1% 词表达到 SoftLexicon 2.6倍提升
5. 自动词典提取策略（min_freq=2）非常有效，避免数据泄露

---

## 📋 推荐方案

- **最佳性能**: ExpertDict (手动) - 97.04% F1
- **最佳实践**: ExpertDict (自动) - 96.99% F1, 无数据泄露
- **快速部署**: Baseline - 95.51% F1, 无需词典
- **不推荐**: SoftLexicon - 性价比低，参数多

---

## ⚙️ 模型配置

- **基础模型**: hfl/chinese-macbert-base
- **模型架构**: MacBERT + BiLSTM + CRF
- **训练轮数**: 30
- **批次大小**: 16
- **学习率**: 0.002
- **微调学习率**: 2e-05
- **随机种子**: 42

---

## 📁 实验文件

- **实验缓存目录**: `cache/redjujube_ner_comparison/`
- **结果JSON**: `cache/redjujube_ner_comparison/comparison_results.json`
- **训练脚本**: `scripts/run_redjujube_all_experiments.sh`

### 各实验模型目录

```
cache/redjujube_ner_comparison/
├── baseline_20251212-200053/
│   └── results.json
├── expert_dict_auto_20251212-202537/
│   └── results.json
├── expert_dict_manual_20251212-202537/
│   └── results.json
├── softlexicon_trainlex_20251212-202537/
│   └── results.json
└── comparison_results.json
```

---

## 📝 结论

本次 RedJujube 数据集 NER 实验系统对比了多种词典特征方法，主要结论如下：

1. **ExpertDict 方法整体优于 SoftLexicon**
   - ExpertDict（手动）达到 97.04% F1，提升 +1.53%
   - ExpertDict（自动）达到 96.99% F1，提升 +1.48%
   - SoftLexicon（TrainLex）达到 96.07% F1，提升 +0.56%

2. **词典质量比规模更重要**
   - ExpertDict 用 2,078-3,389 词达到 97%+ F1
   - SoftLexicon 用 198,437 词仅达到 96.07% F1
   - 精选专家词典效率是大规模词表的 2.6 倍

3. **自动词典提取策略有效**
   - 自动提取（min_freq=2）性能接近手动标注
   - 仅差 0.05% F1，但完全避免数据泄露
   - 推荐作为最佳实践方案

4. **实践建议**
   - 生产环境推荐：ExpertDict（自动），平衡性能与安全
   - 研究实验推荐：ExpertDict（手动），追求最高性能
   - 快速部署推荐：Baseline，95.51% F1 已足够优秀

---

**报告生成时间**: 2025-12-13 15:22:35
