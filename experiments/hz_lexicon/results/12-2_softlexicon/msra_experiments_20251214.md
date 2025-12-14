# MSRA数据集实验结果报告

**实验日期**: 2024-12-14  
**实验批次**: msra_complete_experiments_20251214  
**实验脚本**: `experiments/hz_lexicon/plans/run_msra_complete_experiments.sh`

---

## 📊 实验概况

### 实验进度
- **计划实验数**: 14个
- **已完成实验**: 2个
- **运行中实验**: 1个 (fusion_concat_run2, Epoch 23/30)
- **失败/未完成**: 11个

### 当前状态
截至 2024-12-14 10:55，有1个实验进程仍在运行：
```
PID: 1332094
实验: fusion_concat_run2 (Soft+Expert Concat融合, seed=123)
进度: Epoch 23/30, Dev F1: 92.00%
```

---

## ✅ 已完成实验结果

### 1. SoftLexicon-v1 (CTB原版词典)
- **完成时间**: 2024-12-14 05:23
- **模型配置**: MacBERT + BiLSTM + CRF + SoftLexicon
- **词典来源**: CTB词典(原版)
- **训练轮数**: 30 epochs
- **批次大小**: 16
- **测试集F1**: **94.62%**
- **测试Loss**: 2.364

**关键参数**:
- 总参数量: 110,115,186
- 学习率: 0.002 (BiLSTM) / 2e-05 (BERT微调)
- Expert Dict维度: 50

---

### 2. Soft+Expert Concat融合
- **完成时间**: 2024-12-14 07:00
- **模型配置**: MacBERT + BiLSTM + CRF + SoftLexicon + ExpertDict
- **融合方式**: Concatenation
- **词典来源**: 
  - SoftLexicon: CTB词典
  - ExpertDict: MSRA训练集自动提取 (`expert_lexicon_auto.txt`)
- **训练轮数**: 30 epochs
- **批次大小**: 16
- **测试集F1**: **94.75%**
- **测试Loss**: 1.588

**关键参数**:
- 总参数量: 104,510,836
- 学习率: 0.002 (BiLSTM) / 2e-05 (BERT微调)
- Expert Dict维度: 50
- Seed: 42 (默认)

**性能提升**:
- vs SoftLexicon-v1: +0.13% F1

---

## 🔄 运行中实验

### 3. Soft+Expert Concat融合 Run2 (稳定性验证)
- **开始时间**: 2024-12-14 07:00
- **当前进度**: Epoch 23/30
- **当前Dev F1**: 91.52% - 92.16% (波动中)
- **Seed**: 123
- **预计完成**: 约需2-3小时

---

## ❌ 实验失败记录

### softlex_v1 早期失败 (7次)
**时间段**: 00:00 - 00:18

多次启动失败,错误模式:
- softlexicon_20251214-000041: 训练刚启动(epoch 1, step 50)即中断
- softlexicon_20251214-000302: 类似问题
- softlexicon_20251214-000746 ~ 001515: 连续失败

**可能原因**:
1. 资源冲突(GPU占用)
2. 配置问题
3. 数据加载错误

**最终成功**: softlexicon_20251214-001836 (00:18启动, 05:23完成)

---

## 📈 实验对比分析

| 实验名称 | 模型类型 | 测试F1 | 相对Baseline | 参数量 |
|---------|---------|--------|-------------|--------|
| SoftLexicon-v1 | SoftLexicon | 94.62% | - | 110.1M |
| Soft+Expert Concat | Soft+Expert融合 | 94.75% | +0.13% | 104.5M |

**关键发现**:
1. ✅ **融合有效**: ExpertDict与SoftLexicon融合带来小幅提升(+0.13%)
2. ⚠️ **参数更少**: Concat融合模型参数量反而减少(104.5M vs 110.1M)
3. 📊 **Loss更低**: Concat融合的测试Loss显著更低(1.588 vs 2.364)

---

## 🚧 待完成实验 (11个)

根据原计划 `run_msra_complete_experiments.sh`:

### 未启动实验:
1. **Baseline** - MacBERT + BiLSTM + CRF
2. **ExpertDict** - 专家词典(自动提取)
3. **SoftLexicon-v2** - 去标点版
4. **SoftLexicon-Balanced** - 均衡版
5. **Weighted融合** - 加权融合策略
6. **Attention融合** - 注意力融合
7. **Gated融合** - 门控融合
8. **ExpertDict Run2-5** - 稳定性验证(4次运行, seeds: 123/456/789/999)

---

## 💡 建议与后续计划

### 短期建议:
1. **等待当前实验完成**: fusion_concat_run2预计2-3小时完成
2. **分析失败原因**: 检查早期7次失败的根本原因
3. **继续剩余实验**: 按照脚本顺序继续执行

### 实验策略优化:
1. **错误重试机制**: 增加自动重试逻辑
2. **资源监控**: 训练前检查GPU可用性
3. **断点续训**: 对长时间实验启用checkpoint机制

### 数据分析:
1. **稳定性验证**: 等待多次运行结果评估方差
2. **融合策略对比**: 完成所有融合方法后横向对比
3. **词典质量分析**: 自动提取词典 vs 手动构建的效果差异

---

## 📝 技术细节

### 自动提取词典
- **路径**: `_2DATA/MSRA/expert_lexicon_auto.txt`
- **提取方法**: 从MSRA训练集自动提取实体
- **优势**: 避免跨数据集词典污染(不使用RedJujube手动词典)

### 模型配置一致性
所有实验统一配置:
- BERT模型: `hfl/chinese-macbert-base`
- BERT dropout: 0.2
- BiLSTM hidden: 256
- BiLSTM layers: 1
- BiLSTM dropout: 0.5
- 优化器: Adam
- 梯度裁剪: 5.0

---

## 📂 结果文件位置

```
cache/msra_complete_experiments_20251214/
├── softlex_v1/
│   └── softlexicon_20251214-001836/
│       ├── results.json         # ✅ 完成
│       ├── training.log
│       └── best_model.pt (405M)
├── fusion_concat/
│   └── softlexicon_expert_concat_20251214-005134/
│       ├── results.json         # ✅ 完成
│       ├── training.log
│       └── best_model.pt (399M)
└── fusion_concat_run2/
    └── softlexicon_expert_concat_20251214-070046/
        ├── training.log         # 🔄 运行中
        └── best_model.pt (399M)
```

---

**报告生成时间**: 2024-12-14 10:55  
**下次更新**: 实验完成后
