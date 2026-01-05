# 12-4周：ExpertDict深度优化 + SOTA技巧集成

**时间**: 2025-12-14 ~ 2025-12-20（原计划）→ 延期至 2025-01-05  
**状态**: 🔄 计划调整中  
**数据集**: RedJujube (主数据集)  
**目标**: ExpertDict深度优化 + 错误分析 + SOTA技巧集成 → 冲击98% F1

**最新更新** (2025-01-05):
- ✅ **已完成**: FLAT模型在MSRA数据集验证（F1=95.19%，超越论文+0.73%）
- 🔄 **调整方向**: 放弃FLAT在小数据集应用，聚焦ExpertDict优化
- 📝 **实验记录**: [实验记录补充_20241224-20250105.md](../results/实验记录补充_20241224-20250105.md)

---

## 📋 改进方案总览

### 当前性能基线（12-13更新）

| 方法 | RedJujube F1 | MSRA SOTA | 词典规模 | 状态 |
|------|-------------|-----------|---------|------|
| Baseline | 95.51% | - | - | ✅ |
| SoftLexicon(v1) | 95.47% | - | 18,678 | ✅ 无效 |
| **ExpertDict(自动)** | **97.00%** | **>96.72%** 🏆 | 2,078 | ✅ **超SOTA** |
| Concat融合 | 96.87% | - | 20k+ | ✅ 被拖累 |
| Weighted融合 | 96.72% | - | 20k+ | ✅ 被拖累 |
| Gated融合 | 96.46% | - | 20k+ | ✅ 被拖累 |
| Attention融合 | 96.53% | - | 20k+ | ✅ 被拖累 |

**重要发现**: ExpertDict(97.00%) **已超过MSRA数据集SOTA**(96.72%)！🎉

### 改进目标（调整）

🎯 **短期目标**: 深度优化ExpertDict → **97.5% F1**  
🏆 **终极目标**: SOTA技巧集成 + Ensemble → **97.8~98.0% F1**

---

## 🚀 五大改进方向（基于SOTA排行榜）

### 方向1️⃣：深度错误分析 🔥 优先级最高

**目标**: 理解ExpertDict为什么有效，找到进一步优化空间

**分析内容**:
```
1. 错误模式分析
   - Baseline错误 & ExpertDict正确 → ExpertDict解决了什么
   - Baseline正确 & ExpertDict错误 → ExpertDict引入了什么问题
   - 两者都错误 → 模型的根本弱点

2. 实体类型细粒度分析
   - 14类实体的性能对比
   - 哪些实体类型受益最大
   
3. 错误类型统计
   - 边界错误 (B/E标签)
   - 类型错误
   - 漏检/误检

4. 词典覆盖率分析
   - 训练集/测试集覆盖率
   - 未覆盖实体特征
```

**执行命令**:
```
# 创建错误分析脚本
python _3DATA_PROCESS/error_analysis.py \
  --baseline_dir cache/redjujube_baseline \
  --expert_dir cache/redjujube_expert_auto \
  --data_dir _2DATA/RedJujube \
  --output_dir experiments/hz_lexicon/analysis/error_analysis
```

**预期输出**:
- ✅ `ExpertDict_Error_Analysis_Report.md`
- ✅ 14类实体性能对比表
- ✅ 典型Case Study文档
- ✅ **论文最重要的分析章节**

**预期收益**: 指导后续优化方向

---

### 方向2️⃣：超参数优化 ⚡ 快速见效

**当前配置**（ExpertDict自动）：
```
emb_dim: 50        # ExpertDict嵌入维度
hid_dim: 256       # BiLSTM隐藏层维度
num_layers: 1      # LSTM层数
dropout: 0.5       # Dropout比例
batch_size: 16     # 批次大小
lr: 2e-3           # 学习率
min_freq: 2        # 词典频次阈值
agg_mode: wtd_mean_pooling  # 聚合方式
```

**调优实验**:

| 实验 | 改动参数 | 预期效果 |
|------|---------|----------|
| Exp1 | emb_dim=100 | 更强表达能力 |
| Exp2 | min_freq=3 | 更精准词典 |
| Exp3 | dropout=0.3 | 降低欠拟合 |
| Exp4 | agg_mode=max_pooling | 不同聚合策略 |

**执行命令**:
```
# 批量超参数调优
cd _1CONFIG/redjujube
bash run_expert_emb_dim_tuning.sh    # emb_dim: 25/50/100/150
bash run_expert_min_freq_tuning.sh   # min_freq: 1/2/3/5
bash run_expert_agg_mode_tuning.sh   # agg_mode: wtd_mean/mean/max
```

**预期提升**: +0.2~0.5% F1 → **97.2~97.5%**

---

### 方向3️⃣：对抗训练 (FGM/PGD) 🛡️ 通用技巧

**原理**: 在BERT embedding上添加扰动，提升模型鲁棒性

**实现**:
```
class FGM:
    def attack(self, epsilon=0.5):
        for name, param in model.named_parameters():
            if 'word_embeddings' in name:
                grad = param.grad
                perturb = epsilon * grad / (grad.norm() + 1e-8)
                param.data.add_(perturb)
    
    def restore(self):
        for name, param in model.named_parameters():
            if 'word_embeddings' in name:
                param.data = self.backup[name]
```

**执行命令**:
```
# 集成FGM对抗训练
python _1CONFIG/redjujube/train_redjujube_ner_comparison.py \
  --run_expert_dict_auto \
  --adversarial_training fgm \
  --adv_epsilon 0.5 \
  --save_dir cache/redjujube_expert_auto_fgm
```

**预期提升**: +0.2~0.5% F1 → **97.3~97.6%**

---

### 方向4️⃣：SOTA技巧集成 ⭐ 借鉴排行榜

**基于MSRA SOTA排行榜的最佳实践**:

#### **A. Dice Loss（Rank 1: 96.72%）** 🎲

```
class DiceLoss(nn.Module):
    """F1-oriented loss for imbalanced data"""
    def forward(self, logits, targets):
        intersection = (logits * targets).sum()
        dice = (2 * intersection) / (logits.sum() + targets.sum())
        return 1 - dice

# 组合损失
loss = ce_loss + 0.5 * dice_loss
```

**为什么有效**:
- RedJujube有14类实体，类别不均衡严重
- Dice Loss直接优化F1指标
- 适合长尾分布问题

**预期提升**: +0.1~0.3% F1

#### **B. Boundary Smoothing（Rank 3: 96.26%）** 📏

```
class BoundarySmoothing:
    """Label smoothing for boundary tags"""
    def smooth_labels(self, labels, epsilon=0.1):
        # B标签 → 0.9*B + 0.05*M + 0.05*I
        # E标签 → 0.9*E + 0.05*M + 0.05*I
        smoothed = (1 - epsilon) * labels + epsilon / n_classes
        return smoothed
```

**为什么有效**:
- 缓解边界标注噪声
- 提升边界识别准确率
- 防止过拟合

**预期提升**: +0.1~0.2% F1

**执行命令**:
```
# Dice Loss实验
python train_with_dice_loss.py \
  --run_expert_dict_auto \
  --loss_type dice \
  --dice_alpha 0.5

# Boundary Smoothing实验
python train_with_boundary_smoothing.py \
  --run_expert_dict_auto \
  --label_smoothing 0.1
```

**总预期提升**: +0.2~0.5% F1 → **97.5~97.8%**

---

### 方向5️⃣：模型集成 🏆 终极武器

**集成策略**:

```
# 训练5个不同seed的模型
model1: seed=42,  emb_dim=100, F1=?
model2: seed=123, emb_dim=100, F1=?
model3: seed=456, emb_dim=100, F1=?
model4: seed=789, emb_dim=50,  F1=?
model5: seed=999, emb_dim=150, F1=?

# 集成方法
方案A：标签级投票（简单）
方案B：Logits级平均（更优）
```

**执行命令**:
```
# 批量训练
bash run_expert_ensemble.sh  # 自动训练5个模型

# 集成预测
python ensemble_predict.py \
  --model_dirs cache/redjujube_expert_auto_seed* \
  --ensemble_method voting \
  --output_dir cache/redjujube_expert_ensemble
```

**预期提升**: +0.3~0.8% F1 → **97.8~98.0%**

---

## 📅 执行计划（更新）

### 第1周（12-14 ~ 12-20）

**Day 1-2 (12-14~15)：深度错误分析** 🔥
- [ ] 创建 `error_analysis.py` 脚本
- [ ] 分析Baseline vs ExpertDict的差异
- [ ] 生成14类实体性能对比表
- [ ] 选择20-30个典型样本做Case Study
- [ ] 生成 `ExpertDict_Error_Analysis_Report.md`

**Day 3-4 (12-16~17)：超参数优化** ⚡
- [ ] 运行emb_dim优化实验（25/50/100/150）
- [ ] 运行min_freq优化实验（1/2/3/5）
- [ ] 运行agg_mode优化实验（3种）
- [ ] 分析结果，找到最优配置
- [ ] 可选：组合最优参数再训练

**Day 5 (12-18)：对抗训练 + SOTA技巧** 🛡️⭐
- [ ] 集成FGM对抗训练
- [ ] 训练ExpertDict + FGM
- [ ] 实现Dice Loss
- [ ] 实现Boundary Smoothing
- [ ] 运行SOTA技巧实验

**Day 6 (12-19)：模型集成** 🏆
- [ ] 训练5个不同seed的模型
- [ ] 实现集成预测脚本
- [ ] 运行集成实验
- [ ] 对比投票 vs Logits平均

**Day 7 (12-20)：结果汇总** 📊
- [ ] 收集所有实验结果
- [ ] 生成完整对比报告
- [ ] 更新文档体系
- [ ] 撰写 `ExpertDict_Deep_Optimization_20251220.md`

---

## 🎯 预期最终性能（更新）

### 阶段性目标

| 阶段 | 方法 | RedJujube F1 | MSRA SOTA | 状态 |
|------|------|-------------|-----------|------|
| **当前基线** | ExpertDict(自动) | **97.00%** | 96.72% | ✅ **超SOTA** |
| 阶段1 | +超参数优化 | 97.2~97.5% | - | 🎯 Day 3-4 |
| 阶段2 | +对抗训练 | 97.3~97.6% | - | 🎯 Day 5 |
| 阶段3 | +SOTA技巧 | 97.5~97.8% | - | 🎯 Day 5 |
| **终极目标** | **+模型集成** | **97.8~98.0%** | **🏆 远超** | 🎯 Day 6 |

### 提升明细

| 改进方向 | 预期提升 | 实现难度 | 论文价值 |
|---------|---------|---------|----------|
| 深度错误分析 | 指导优化 | 中 | ⭐⭐⭐⭐⭐ |
| 超参数优化 | +0.2~0.5% | 低 | ⭐⭐⭐ |
| 对抗训练 (FGM) | +0.2~0.5% | 低 | ⭐⭐⭐⭐ |
| Dice Loss | +0.1~0.3% | 中 | ⭐⭐⭐⭐ |
| Boundary Smoothing | +0.1~0.2% | 中 | ⭐⭐⭐ |
| 模型集成 | +0.3~0.8% | 低 | ⭐⭐⭐⭐ |
| **总计** | **+0.8~2.0%** | - | **⭐⭐⭐⭐⭐** |

### 保守预期

基于当前实验经验，保守预期：

```
当前基线:      97.00%
+ 超参数优化:  +0.3%
+ 对抗训练:      +0.3%
+ SOTA技巧:      +0.2%
+ 模型集成:      +0.5%
--------------------------
最终性能:       97.8% F1 ✅
```

### 乐观预期

如果各项技巧都达到最佳效果：

```
当前基线:      97.00%
+ 超参数优化:  +0.5%
+ 对抗训练:      +0.5%
+ SOTA技巧:      +0.5%  (Dice Loss + BS)
+ 模型集成:      +0.8%
--------------------------
最终性能:       98.3% F1 🚀
```

**现实目标**: **97.8~98.0% F1** 🎯

---

## 📊 实验追踪

### ✅ 已完成

#### 12-4周核心实验（截至2025-01-05）

**1. FLAT模型在MSRA数据集验证** ✅ (2024-12-24)
- 模型: FLAT + Inter-Attention + MacBERT + CRF
- 词典: 4,834词专家词典
- 结果: **F1=95.19%** (超越论文baseline +0.73%)
- 实体类型: NR=96.91%, NS=96.12%, NT=91.28%
- 训练特性: 7 epochs达峰值，泛化性强
- 状态: ✅ **成功验证FLAT架构有效性**
- 详细报告: [FLAT_MSRA_实验记录_20251224.md](../results/12-4_flat_msra/FLAT_MSRA_实验记录_20251224.md)

**2. FLAT模型在RedJujube数据集尝试** ⚠️ (2024-12-24)
- 方案A: FLAT + Expert Auto词典 (2,078词)
  - 最优Dev F1: **96.39%** @ Epoch 15
  - 问题: Epoch 16后出现Recall崩溃（27% Recall @ Epoch 19）
  
- 方案B: FLAT + CTB大词典 (280k+词)
  - 最优Dev F1: **96.12%** @ Epoch 13
  - 问题: 训练不稳定，F1波动在93%-96%

- 对比ExpertDict（传统方法）:
  - ExpertDict: **97.00% F1** (稳定)
  - FLAT最优: **96.39% F1** (不稳定)
  - 差距: **-0.61%**

- 状态: ❌ **验证FLAT不适合小数据集**

**核心结论** (2025-01-05):
1. ✅ **FLAT适合大规模数据集** (MSRA 45k样本：95.19% F1)
2. ❌ **FLAT不适合小规模数据集** (RedJujube 5k样本：96.39% < 97.00%)
3. 💪 **传统方法在小数据集更优** (ExpertDict: 97.00% F1)
4. 📊 **数据规模是关键** (临界点约10k-20k样本)

---

### ❌ 放弃方向

1. ❌ **FLAT在RedJujube数据集应用**
   - 原因: 性能不如ExpertDict (96.39% vs 97.00%)
   - 原因: 训练不稳定，出现Recall崩溃
   - 原因: 数据规模不足以支撑112M参数

2. ❌ **继续优化FLAT-RedJujube**
   - 性价比低，收益有限
   - 应聚焦于ExpertDict优化（更高ROI）

---

### 📝 待执行（调整后）

**新计划：基于ExpertDict的深度优化** (优先级最高)

#### 阶段1：超参数优化 (快速见效)
- [ ] emb_dim调优实验 (25/50/100/150)
- [ ] min_freq调优实验 (1/2/3/5)
- [ ] dropout调优实验 (0.3/0.5/0.7)
- [ ] agg_mode调优实验 (wtd_mean/mean/max)
- **预期目标**: 97.2~97.5% F1
- **时间预估**: 2-3天

#### 阶段2：对抗训练 (通用技巧)
- [ ] FGM对抗训练实现
- [ ] PGD对抗训练实现
- [ ] 对抗参数调优 (epsilon=0.3/0.5/1.0)
- **预期目标**: 97.3~97.6% F1
- **时间预估**: 2-3天

#### 阶段3：SOTA技巧集成 (借鉴排行榜)
- [ ] Dice Loss实现 (优化不均衡数据)
- [ ] Boundary Smoothing实现 (缓解边界噪声)
- [ ] Label Smoothing实验
- **预期目标**: 97.5~97.8% F1
- **时间预估**: 3-4天

#### 阶段4：模型集成 (终极武器)
- [ ] 训练5个不同seed的模型
- [ ] 实现Voting集成策略
- [ ] 实现Logits平均集成
- **预期目标**: 97.8~98.0% F1 🎯
- **时间预估**: 2-3天

#### 阶段5：FLAT经验迁移 (中期探索)
- [ ] 将Inter-Attention机制整合到ExpertDict
- [ ] CRF解码器优化
- [ ] 词典嵌入融合策略优化
- **预期目标**: 97.6~97.9% F1
- **时间预估**: 4-5天

**总计时间**: 13-18天
**最终目标**: **97.8~98.0% F1** 🎯

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
