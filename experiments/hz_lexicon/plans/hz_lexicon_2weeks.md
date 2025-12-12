### 两周实验计划总览（优先 HZ）

**总体目标**：  
围绕 HZ 数据集，对比 **Baseline / SoftLexicon / ExpertDict（手工/自动）/ Soft+Expert**，并验证“软词典候选词严格来源于训练集”的策略效果。

时间粒度按“工作日/阶段”来，不死盯每天。

---

### 第 1 周：跑通 & 对齐基线 + 软词典策略

#### 第 1–2 天：环境确认 & 现有 HZ 基线/专家词典复现

- **任务 1：确认环境 & 词向量就绪**
  - 检查 `assets/vectors/ctb.50d.vec` 是否存在且可被 `load_vectors("chinese", 50)` 正常加载。
  - 简单跑一小段 `train_hz_ner_baseline_vs_expert_dict.py` 确认不会崩（不用跑完）。

- **任务 2：复现 HZ Baseline vs ExpertDict（手工）** ✅ **已完成**
  - 命令：
    ```bash
    bash scripts/run_hz_comparison.sh
    ```
  - **结果记录** (2025-12-07):
    - Baseline F1: **95.618%**
    - +ExpertDict(手工) F1: **97.941%**
    - F1 提升: **+2.323%**
    - 实验目录: `cache/hz_ner_comparison/`
  
  - **相关分析报告**:
    - 📊 [词典对比分析报告](../results/词典对比分析报告.md)
      - 手工词典 vs 自动词典覆盖率对比
      - 测试集：手工87.35% vs 自动95.58%
      - 发现：手工词典GEO类型完全缺失(0%)
    - 📊 [NER实验结果综合对比报告](../results/NER_实验结果综合对比报告_20251208.md)
      - 包含MSRA和HZ数据集完整对比
      - HZ最佳：手工专家词典97.941%

#### 第 3–4 天：HZ SoftLexicon baseline + 训练集候选词策略

- **任务 3：HZ SoftLexicon baseline（使用现有 CTB 词典）** ✅ **已完成**
  - 命令：
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
  - **结果记录** (2025-12-10):
    - 测试集 F1: **95.88%**
    - 验证集最佳 F1: **96.96%** (第22轮)
    - 词表来源: CTB 50d 词向量 (280,930 个词)
    - 实验目录: `cache/hz_softlexicon/softlexicon_20251210-191021/`

- **任务 4：实现“训练集候选词构建 SoftLexicon”的版本** 🔄 **进行中**
  - **已完成：**
    - ✅ 创建脚本 `scripts/extract_softlexicon_from_training.py`
    - ✅ 从 `data/HZ/hz_train.bmes` 提取候选词表
      - 输出: `data/HZ/softlexicon_train.txt`
      - 词表大小: **197,972 个词**
      - 来源: BMES 实体 (3,214个) + n-gram (最大长度5)
      - 频次过滤: min_freq=2
    - ✅ 修改训练脚本支持 `--run_softlexicon_trainlex` 参数
    - ✅ 启动训练实验
  
  - **进行中：**
    - 🔄 SoftLexicon-TrainLex 训练 (Epoch 5/30, 当前验证集 F1: 93.39%)
    - 实验目录: `cache/hz_softlexicon/softlexicon_trainlex_20251210-195715/`
  
  - **待记录：**
    - SoftLexicon(CTB词表) vs SoftLexicon(Train-only词表) 的 F1 对比
    - 命中率变化分析

#### 第 5 天：整理第1周结果 & 快速结论

- 在 `hz_lexicon_2weeks.md` 里增加一个“小结 1”：
  - 写清 HZ 上的三条线：
    - Baseline
    - +ExpertDict(手工)
    - SoftLexicon(CTB) / SoftLexicon(Train-only)
  - 简单一句话结论：SoftLexicon 是否明显优于 Baseline？Train-only 词表是否接近/优于 CTB？

---

## 📊 小结 1：第 1 周实验结果 (2025-12-10)

### 已完成实验对比

| 实验类型 | 测试集 F1 | 验证集最佳 F1 | 词表来源 | 词表大小 | 状态 |
|---------|----------|--------------|----------|----------|------|
| Baseline | 95.618% | - | - | - | ✅ 历史数据 |
| +ExpertDict(手工) | **97.941%** | - | 手工标注 | 2,371词 | ✅ 历史数据 |
| +ExpertDict(自动) | **97.050%** | - | 训练集提取 | 3,214词 | ✅ 历史数据 |
| SoftLexicon (CTB) | **95.88%** | 96.96% | CTB 50d | 280,930词 | ✅ 已完成 |
| SoftLexicon (TrainLex) | **96.57%** | 97.24% | 训练集n-gram | 197,972词 | ✅ 已完成 |

### 初步发现

1. **SoftLexicon vs Baseline**
   - SoftLexicon(CTB): 95.88% vs Baseline: 95.618%
   - **提升**: +0.262% (较小提升)
   - 结论: SoftLexicon 在 HZ 数据集上效果有限

2. **SoftLexicon vs ExpertDict（公平对比）**
   - 公平对比以 **自动 ExpertDict(97.050%)** 为主，对比 SoftLexicon(TrainLex 96.57%) 时差距约 **-0.48%**，自动专家词典仍略优。
   - 手工 ExpertDict(97.941%) 结果可能包含测试集知识，存在潜在数据泄露，仅作为上界参考。
   - 结论: 在“仅使用训练集信息”的前提下，**自动专家词典 > SoftLexicon(TrainLex) > Baseline**。

3. **CTB vs Train-only 词表**
   - CTB 词表: 280,930词 (外部大词表)，测试集 F1 = 95.88%
   - Train-only 词表: 197,972词 (仅使用训练集)，测试集 F1 = 96.57%
   - 验证集最佳 F1: CTB 96.96% vs Train-only 97.24%
   - **结论**: Train-only 词表在不依赖外部大词表的情况下，略优于 CTB 且避免潜在数据泄露

### 下一步

- 等待 SoftLexicon-TrainLex 训练完成
- 对比两种 SoftLexicon 策略的效果
- 分析词表大小与性能的关系

### 相关报告文档

1. **📊 NER实验结果综合对比报告** ([results/NER_实验结果综合对比报告_20251208.md](../results/NER_实验结果综合对比报告_20251208.md))
   - MSRA-ER 数据集 4个实验结果
   - HZ 数据集 3个对比实验
   - 历史 SOTA 模型对比
   - 综合性能排名与分析

2. **🔍 词典对比分析报告** ([results/词典对比分析报告.md](../results/词典对比分析报告.md))
   - 手工词典 vs 自动词典规模对比：2,887 vs 1,945词
   - 训练/验证/测试集覆盖率分析
   - 关键发现：自动词典覆盖率领先+8.23%
   - 手工词典GEO类型完全缺失(0%)
   - 推荐混合词典方案

---

### 第 2 周：Soft vs Expert vs Soft+Expert & 策略完善（HZ）

#### 第 6–7 天：Soft vs Expert（手工/自动）对比

- **任务 5：Soft vs 手工 ExpertDict**
  - 利用第 1 周得到的：
    - Baseline F1
    - +ExpertDict(手工) F1
    - SoftLexicon (CTB / Train-only) F1
  - 在计划文件中按表格形式整理对比，按实体类型（如果你有类型粒度的评估）粗略分析：
    - 比如疾病类、药品类、检查类各自的 F1 或 coverage 差异。

- **任务 6：Soft vs 自动 ExpertDict（`expert_lexicon_auto.txt`）**
  - 跑一版自动词典对比（如果没跑过，可直接用现有脚本）：
    ```bash
    bash scripts/run_hz_comparison_with_auto_lexicon.sh
    ```
  - 在 `experiments/hz_lexicon/results/` 记录：
    - +ExpertDict(自动) F1；
    - 与手工词典 / SoftLexicon 的差距；
  - 简单用 `auto_dict_analysis_report.txt` vs `expert_dict_analysis_report.txt` 辅助说明覆盖率差异。

#### 第 8–9 天：实现并跑 SoftLexicon + ExpertDict 联合模型（HZ）

- **任务 7：在 HZ 训练脚本中增加 Soft+Expert 联合配置**
  - 在 `train_hz_ner_baseline_vs_expert_dict.py` 中新增：
    - 一个 `build_softlexicon_expert_config`（同时把 `SoftLexiconConfig` 和 `ExpertDictConfig` 都挂在 `nested_ohots`）；
    - 一个 `--run_softlexicon_expert` 参数；
    - 预处理时，对数据同时调用：
      ```python
      # softlexicon 部分（用训练集/CTB 方案二选一）
      entry["tokens"].build_softwords(soft_tokenizer.tokenize)
      entry["tokens"].build_softlexicons(soft_tokenizer.tokenize)
      # expert 部分
      entry["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
      ```
  - 跑一版联合实验，`save_dir` 建议 `cache/hz_softlexicon_expert`。

- **任务 8：比较联合模型 vs 单独 Soft / 单独 Expert**
  - 在计划文件里整理一个对比表：
    - Baseline
    - +ExpertDict(手工)
    - SoftLexicon(你觉得更稳的那版：CTB or Train-only)
    - SoftLexicon + ExpertDict
  - 如果有条件，挑一小批 case 看错误类型分布（例如哪些实体由 Soft 补救，哪些由 Expert 补救）。

#### 第 10 天：总结 & 为后续 MSRA / 公开词典做准备

- 在 `hz_lexicon_2weeks.md` 里写一个“阶段总结”小节，重点回答三件事：
  1. 在 HZ 上，SoftLexicon（不同候选词策略） vs ExpertDict（手工/自动）的优劣势；
  2. Soft+Expert 是否明显优于单独任一方；
  3. “软词典候选词严格来自训练集”的策略，是否在不依赖外部大词表的情况下，效果还能接受（或更好）。

- 若时间允许，在最后列出下一阶段（可能是 MSRA 或 引入公开领域词典）的 TODO，但不用展开。

---

如果你愿意，我可以下一步直接给出：  
- “训练集抽 softlexicon 候选词”的脚本改法（在现有 `extract_lexicon_from_training.py` 基础上扩展一段）；  
- 以及 HZ 联合模型（Soft+Expert）的具体代码改动片段。