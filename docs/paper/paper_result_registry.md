# 红枣栽培 NER 投稿稿结果注册表

本文档用于执行 `plans/农业机械学报投稿论文_goal计划.md` 的 Phase 1：证据归档与数值统一。所有进入投稿稿的结果必须能追溯到本表中的本地文件或现有论文表格。

## 1. 采用原则

1. 摘要、结论和主对比表采用 **seed=42 单次同口径结果**，以便纳入 SoftLexicon、FLAT 和 FLAT+BERT 等扩展基线；涉及稳定性或泛化性的其他表格可继续沿用各自登记的均值 ± 标准差口径。
2. 单次实验结果可用于分类分析、案例分析和误差分析，但不能与均值结果混写。
3. 现有主稿中的 `EDBP/EDBS` 统一为 **EDBP**，中文统一为“专家词典与边界预测模型”。
4. “边界预测”是论文问题表述；代码实现对应 `BoundarySelectionDecoder`，正文表述为“边界预测模块采用边界选择解码形式”。
5. 当前投稿稿先采用现有主稿中已经整理过的 RJND 结果，不将 2026-03 数据重构后的新数据结果直接替换主结果，除非后续完成全表重算。

## 2. RJND 主结果

### 2.1 历史三种子主对比结果，保留为稳定性证据

| 模型 | P/% | R/% | F1/% | 证据来源 | 采用状态 | 说明 |
|---|---:|---:|---:|---|---|---|
| BiLSTM-CRF | - | - | 78.69 | `docs/paper/基于词典和边界预测的红枣栽培命名实体识别.md` 表 6 | 历史保留 | 3 seeds 无预训练弱基线 |
| BERT-wwm-ext+BiLSTM+CRF | - | - | 85.48±0.25 | 同上，表 6 | 历史保留 | 3 seeds 强基线 |
| MacBERT-base+BiLSTM+CRF | - | - | 85.57±0.29 | 同上，表 6 | 历史保留 | 3 seeds 强基线 |
| EDBP/EDBS 完整模型 | - | - | 88.28±0.22 | `experiments/EXP-010-optimization/results_newdata/Q_bs_focal/` | 历史保留 | 3 seeds 均值，不作为当前扩展表 3 展示口径 |

历史三种子计算：

- EDBP 相对 BiLSTM-CRF：88.28 - 78.69 = **+9.59** 个百分点。
- EDBP 相对 BERT-wwm-ext+BiLSTM+CRF：88.28 - 85.48 = **+2.80** 个百分点。
- EDBP 相对 MacBERT-base+BiLSTM+CRF：88.28 - 85.57 = **+2.71** 个百分点。

### 2.1.1 主对比三种子原始结果与显著性检验

2026-05-23 重新整理 `results_newdata` 后，定位到三种子原始结果。该组结果用于保留稳定性和显著性证据；当前投稿稿表 3 因补入 SoftLexicon、FLAT 和 FLAT+BERT，改用 2.1.2 中 seed=42 单次同口径结果。

| 模型 | seed 42 | seed 43 | seed 44 | 均值±标准差/% | 结果路径 |
|---|---:|---:|---:|---:|---|
| BiLSTM-CRF | 78.30 | 78.84 | 78.92 | 78.69±0.34 | `experiments/EXP-010-optimization/results_newdata/G_bilstm_baseline/` |
| BERT-wwm-ext+BiLSTM+CRF | 85.21 | 85.54 | 85.69 | 85.48±0.25 | `experiments/EXP-010-optimization/results_newdata/CRF_nodict_bertwwm/` |
| MacBERT-base+BiLSTM+CRF | 85.36 | 85.90 | 85.45 | 85.57±0.29 | `experiments/EXP-010-optimization/results_newdata/CRF_nodict/` |
| EDBP | 88.16 | 88.54 | 88.15 | 88.28±0.22 | `experiments/EXP-010-optimization/results_newdata/Q_bs_focal/` |

配对 t 检验结果：

- EDBP vs. BiLSTM-CRF：均值差 9.60 个百分点，p=0.0004。
- EDBP vs. BERT-wwm-ext+BiLSTM+CRF：均值差 2.80 个百分点，p=0.0036。
- EDBP vs. MacBERT-base+BiLSTM+CRF：均值差 2.71 个百分点，p=0.0003。

说明：三种子结果不再作为当前投稿稿表 3 的展示口径，正文表 3 不再写 `±` 或配对 t 检验。

### 2.1.2 RJND seed=42 扩展主对比结果

为回应主模型对比模型数量偏少的问题，投稿稿表 3 改为采用 seed=42 单次同口径结果，并补入 SoftLexicon、FLAT 和 FLAT+BERT 三个词典或 lattice 增强基线。LatticeLSTM 和 NFLAT 暂未找到 RJND 同口径结果，不进入正文主表。

| 模型 | seed | 测试 F1/% | BERT/MacBERT | 词典 | 结果路径 | 采用状态 |
|---|---:|---:|---|---|---|---|
| BiLSTM-CRF | 42 | 78.30 | 否 | 否 | `experiments/EXP-010-optimization/results_newdata/G_bilstm_baseline/.../results.json` | 表 3 采用 |
| BERT-wwm-ext+BiLSTM+CRF | 42 | 85.21 | BERT-wwm-ext | 否 | `experiments/EXP-010-optimization/results_newdata/CRF_nodict_bertwwm/.../results.json` | 表 3 采用 |
| MacBERT-base+BiLSTM+CRF | 42 | 85.36 | MacBERT-base | 否 | `experiments/EXP-010-optimization/results_newdata/CRF_nodict/.../results.json` | 表 3 采用 |
| Boundary Smoothing | 42 | 86.48 | MacBERT-base | 否 | `experiments/EXP-010-optimization/results_needed_20260524/BS_nodict_seed42_current/bert_bs_pure_20260524-164044/results.json` | 表 3 补强候选；当前路径补跑 |
| SoftLexicon-TrainLex | 42 | 85.55 | MacBERT-base | 当前 RedJujube 训练集词表 + 中文 50d 词向量 | `experiments/EXP-010-optimization/results_needed_20260524/SoftLexicon_trainlex_seed42_current/softlexicon_trainlex_20260524-171342/results.json` | 表 3 候选；当前路径补跑 |
| AdaSeq BERT-CRF | 42 | 85.16 | MacBERT-base | 否 | `experiments/EXP-010-optimization/results_needed_20260524/AdaSeq_bert_crf_seed42_current/metrics_summary.json` | 表 3 候选；BMES 转 BIO 后补跑 |
| FLAT | 42 | 79.78 | 否 | CTB lattice 词表 | `experiments/baselines/flat_no_bert_v2/.../results.json` | 表 3 采用 |
| FLAT+BERT | 42 | 79.40 | MacBERT-base | CTB lattice 词表 | `experiments/baselines/flat_bert_fixed/.../results.json` | 表 3 采用 |
| EDBP | 42 | 88.16 | MacBERT-base | 训练集自动词典，min_freq=2，1842 词 | `experiments/EXP-010-optimization/results_newdata/Q_bs_focal/.../results.json` | 表 3 采用 |
| LatticeLSTM | 42 | 待补 | 否 | 目标为 gazetteer | 未找到 RJND 结果 | 不进入表 3 |
| NFLAT | 42 | 待补 | 否 | YangJie/CTB lattice | 未找到 RJND 结果 | 不进入表 3 |

主结论计算：

- EDBP 相对 BiLSTM-CRF：88.16 - 78.30 = **+9.86** 个百分点。
- EDBP 相对 BERT-wwm-ext+BiLSTM+CRF：88.16 - 85.21 = **+2.95** 个百分点。
- EDBP 相对 MacBERT-base+BiLSTM+CRF：88.16 - 85.36 = **+2.80** 个百分点。
- EDBP 相对 Boundary Smoothing：88.16 - 86.48 = **+1.68** 个百分点。
- EDBP 相对 SoftLexicon-TrainLex：88.16 - 85.55 = **+2.61** 个百分点。
- EDBP 相对 AdaSeq BERT-CRF：88.16 - 85.16 = **+3.00** 个百分点。
- EDBP 相对 FLAT：88.16 - 79.78 = **+8.38** 个百分点。
- EDBP 相对 FLAT+BERT：88.16 - 79.40 = **+8.76** 个百分点。

处理决策：

- 投稿稿表 3 使用本节 seed=42 扩展主对比结果，不再在该表中使用 `±` 或配对 t 检验。
- `Boundary Smoothing` 为 2026-05-24 当前路径补跑结果，可供论文端决定是否补入表 3；若补入，摘要和结论需同步加入相对提升 **+1.68** 个百分点。
- 三种子结果保留为历史稳定性证据，但不作为扩展表 3 的展示口径。

### 2.1.2 RJND seed=42 经典词典与 lattice 基线补充检索

2026-05-23 按 `RJND/RedJujube + seed=42 + test F1` 口径检索本地结果。下表只登记 RedJujube 主数据集结果，不混入 MSRA、Resume、Boson、CLUENER 或旧 cache 结果。

| 模型名 | 数据集路径/名称 | seed | 测试集 F1/% | 结果文件路径 | 是否使用 BERT/MacBERT | 是否使用外部词典/训练集词典 | 状态 |
|---|---|---:|---:|---|---|---|---|
| BiLSTM-CRF | `_2DATA/RedJujube` | 42 | 78.30 | `experiments/EXP-010-optimization/results_newdata/G_bilstm_baseline/bilstm_crf_20260319-211534/results.json` | 否 | 否 | 表 3 seed=42 单次值 |
| BERT-wwm-ext+BiLSTM+CRF | `_2DATA/RedJujube` | 42 | 85.21 | `experiments/EXP-010-optimization/results_newdata/CRF_nodict_bertwwm/bert_bilstm_crf_20260331-132603/results.json` | 是，`hfl/chinese-bert-wwm-ext` | 否 | 表 3 seed=42 单次值 |
| MacBERT-base+BiLSTM+CRF | `_2DATA/RedJujube` | 42 | 85.36 | `experiments/EXP-010-optimization/results_newdata/CRF_nodict/bert_bilstm_crf_20260319-211534/results.json` | 是，`hfl/chinese-macbert-base` | 否 | 表 3 seed=42 单次值 |
| Boundary Smoothing | `datasets/raw/RedJujube` | 42 | 86.48 | `experiments/EXP-010-optimization/results_needed_20260524/BS_nodict_seed42_current/bert_bs_pure_20260524-164044/results.json` | 是，`hfl/chinese-macbert-base` | 否 | 2026-05-24 当前路径补跑；P/R/F1=87.36/85.61/86.48，可作表 3 补强候选 |
| EDBP | `_2DATA/RedJujube` | 42 | 88.16 | `experiments/EXP-010-optimization/results_newdata/Q_bs_focal/expert_boundary_20260319-182029/results.json` | 是，`hfl/chinese-macbert-base` | 是，训练集自动专家词典；`min_freq=2`，1842 个词，保存为同目录 `auto_lexicon.txt` | 表 3 seed=42 单次值 |
| SoftLexicon-TrainLex | `datasets/raw/RedJujube` | 42 | 85.55 | `experiments/EXP-010-optimization/results_needed_20260524/SoftLexicon_trainlex_seed42_current/softlexicon_trainlex_20260524-171342/results.json` | 是，`hfl/chinese-macbert-base` | 是，匹配词表 `datasets/raw/RedJujube/softlexicon_train.txt`，嵌入初始化来自中文 50d 词向量 | 当前路径补跑；建议替代旧 HZ 词典 SoftLexicon |
| SoftLexicon-External | `datasets/raw/RedJujube` | 42 | 84.98 | `experiments/EXP-010-optimization/results_needed_20260524/SoftLexicon_external_seed42_current/softlexicon_20260524-173106/results.json` | 是，`hfl/chinese-macbert-base` | 是，匹配词表来自 `assets/vectors/ctb.50d.vec` | 当前路径补跑；外部词典对照 |
| AdaSeq BERT-CRF | `datasets/raw/RedJujube` -> BIO | 42 | 85.16 | `experiments/EXP-010-optimization/results_needed_20260524/AdaSeq_bert_crf_seed42_current/metrics_summary.json` | 是，`hfl/chinese-macbert-base` | 否 | 当前路径补跑；BMES 转 BIO 后完成 |
| FLAT | `_2DATA/RedJujube` | 42 | 79.78 | `experiments/baselines/flat_no_bert_v2/flat_20260506-101813/results.json` | 否 | 是，外部 lattice 词表/向量 `assets/vectors/ctb.50d.vec`，698668 词 | 可作为无 BERT FLAT 对照 |
| BERT-FLAT / FLAT+BERT | `_2DATA/RedJujube` | 42 | 79.40 | `experiments/baselines/flat_bert_fixed/flat_20260423-211828/results.json` | 是，MacBERT-base 本地快照 | 是，外部 lattice 词表/向量 `assets/vectors/ctb.50d.vec`，698668 词 | 可作为 BERT-FLAT 对照 |
| LatticeLSTM | `_2DATA/RedJujube` | 42 | 待补 | 未找到 RJND 结果文件 | 否 | 目标模型通常使用外部 gazetteer；本地代码默认 `data/ctb.50d.vec` | 仅找到 `projects/LatticeLSTM-master` 代码和 Resume/demo 数据，无 RedJujube 结果 |
| NFLAT | `_2DATA/RedJujube` | 42 | 待补 | 未找到 RJND 结果文件 | 否 | 目标代码默认使用 YangJie/CTB lattice 词表与词向量 | 仅找到 `projects/NFLAT4CNER-main/nflat_msra_result.txt`，属于 MSRA，已排除 |

补实验状态：

- `Boundary Smoothing` 已按当前 `datasets/raw/RedJujube`、seed=42、`hfl/chinese-macbert-base`、无专家词典补跑完成。`results.json` 保存 `test_f1=0.8647850950009224`；`research/evaluation/test_redjujube_baseline.py --model_type baseline` 复评得到 Micro P/R/F1=0.8736/0.8561/0.8648。
- `BERT-MRC` / `BERT-MRC+DSC` 暂无最终可登记结果，但旧失败已解除：
  `span_loss_candidates=pred_and_gold` 已改为 `gold_pred` 默认值，随机矩阵 `.cuda()`
  已改为按目标张量设备迁移。`dice_ohem=0.3` 在当前 PyTorch/CUDA 下触发
  INT_MAX 或 OOM；`dice_ohem=0`、`train_batch_size=4` 的 20-step GPU smoke
  已通过，完整 10 epoch 长实验正在 tmux `rjnd-mrc-dsc-20260524` 中运行。
- `RA_NER / AdaSeq BERT-CRF` 旧失败已解除：AdaSeq conll builder 不能直接吃 BMES，
  转为 BIO 后 seed=42 test P/R/F1=84.42/85.90/85.16。
- LatticeLSTM 当前不能直接补跑：源码位于
  `references/external_projects/LatticeLSTM-master`，为 Python 2/PyTorch 0.3
  风格；服务器 `/usr/bin/python2` 无 `torch`。需要先恢复旧环境或将代码迁移到
  Python 3/PyTorch 当前接口，并将 seed 改为 42。
- NFLAT 已完成前置适配但尚未完整训练：源码位于
  `references/external_projects/NFLAT4CNER-main`，已补 `redjujube` 数据配置、
  `seed` 参数、`n_epochs` 参数和当前向量路径；`flat37` 已安装 `prettytable`
  并可导入 FastNLP。完整训练需等待 GPU 从 MRC/DSC 长实验释放。
- `cache/redjujube_softlexicon_*` 与 `cache/flat_complete*` 中存在更高 F1 的旧结果，但不属于本次统一的 `results_newdata`/表 3 口径，暂不登记为投稿主证据。

### 2.2 单次分类分析结果，不进入摘要主结果

| 指标 | 数值 | 证据来源 | 采用状态 | 说明 |
|---|---:|---|---|---|
| Precision | 89.51 | `docs/paper/基于词典和边界预测的红枣栽培命名实体识别.md` 表 3/表 5 | 限分类分析采用 | 单次或特定实验配置结果 |
| Recall | 87.58 | 同上 | 限分类分析采用 | 单次或特定实验配置结果 |
| F1 | 88.54 | 同上 | 限分类分析采用 | 不作为摘要主结果，避免与 88.16 冲突 |

处理决策：

- 摘要写：EDBP 在 RJND 上 F1 为 **88.16%**。
- 分类分析写：在代表性运行中，整体 P/R/F1 为 **89.51%/87.58%/88.54%**。
- 不再写“摘要 P/R/F1 分别达到 89.51、87.58、88.54”作为全局主结论。

## 3. 消融实验结果

| 方法 | 词典 | 边界预测 | Focal Loss | F1/% | 证据来源 | 采用状态 |
|---|---|---|---|---:|---|---|
| BERT+LSTM+CRF | 否 | 否 | - | 85.57 | `docs/paper/基于词典和边界预测的红枣栽培命名实体识别.md` 表 2 | 采用 |
| +专家词典 | 是 | 否 | - | 86.71 | 同上 | 采用 |
| +边界预测 | 否 | 是 | 否 | 86.68 | 同上 | 采用 |
| +专家词典+边界预测 | 是 | 是 | 否 | 87.66 | 同上 | 采用 |
| +边界预测+Focal | 否 | 是 | 是 | 86.58 | 同上 | 采用 |
| +专家词典+边界预测+Focal | 是 | 是 | 是 | 88.28 | 同上 | 采用 |

关键计算：

- 专家词典单独增益：86.71 - 85.57 = **+1.14**。
- 边界预测单独增益：86.68 - 85.57 = **+1.11**。
- 词典 + 边界预测协同增益：87.66 - 85.57 = **+2.09**。
- Focal Loss 在词典 + 边界预测条件下增益：88.28 - 87.66 = **+0.62**。

## 4. 词典构建策略结果

词典构建策略采用 `experiments/EXP-011-lexicon_strategy/analysis/candidate_proxy_table.csv` 中 RedJujube 行。该表为训练集词典匹配代理指标，用于解释主模型采用最小词频阈值为 2 的专家词典；不作为测试集 NER 主性能。主模型结果 JSON 的 `args.min_freq=2`，对应 `results_newdata/Q_bs_focal`。

| 最小词频阈值 | 词典规模 | 短实体覆盖率/% | 长实体覆盖率/% | 短实体匹配 F1/% | 长实体匹配 F1/% | 平衡 F1/% | 采用状态 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5 317 | 100.00 | 77.75 | 61.69 | 81.67 | 62.76 | 对比 |
| 2 | 1 842 | 85.45 | 31.04 | 58.51 | 45.11 | 57.79 | 主模型采用 |
| 3 | 1 087 | 78.61 | 16.94 | 56.73 | 28.14 | 55.19 | 对比 |

2026-05-24 按当前 `datasets/raw/RedJujube`、seed=42、同一 EDBP 训练脚本补跑
`min_freq=1`、`min_freq=3` 和 `min_freq=4`，将词典阈值对比从代理指标补充为真实测试集 NER 指标。
`min_freq=2` 主模型结果仍采用 2.1.2 登记的表 3 口径。

| 最小词频阈值 | seed | 词典规模 | P/% | R/% | F1/% | 结果路径 | 采用状态 |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 42 | 5 317 | 86.94 | 81.92 | 84.36 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current/expert_boundary_20260524-124317/results.json` | 词典阈值真实 NER 对照 |
| 2 | 42 | 1 842 | - | - | 88.16 | `experiments/EXP-010-optimization/results_newdata/Q_bs_focal/expert_boundary_20260319-182029/results.json` | 表 3 主模型采用 |
| 3 | 42 | 1 087 | 88.56 | 86.49 | 87.51 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/expert_boundary_20260524-125722/results.json` | 词典阈值真实 NER 对照 |
| 4 | 42 | 786 | 87.79 | 86.12 | 86.95 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq4_seed42_current/expert_boundary_20260524-132104/results.json` | 词典阈值真实 NER 对照 |

复评说明：补跑结果的 `results.json` 只保存 loss/F1；P/R 由
`research/evaluation/test_redjujube_baseline.py` 加载同一最佳模型和 `auto_lexicon.txt`
复评得到。执行命令和 tmux 查询方式见
`docs/paper/plans/2026-05-24-needed-experiments-execution.md`。

处理决策：

- 投稿稿新增词典构建策略对比表，说明低频复合术语对词典覆盖的影响，并明确主模型采用 min_freq=2。
- 代理指标表只用于词典阈值选择解释；真实 NER 指标表可用于说明 `min_freq=2`
  相比 `min_freq=1/3/4` 的测试集 F1 更优。

## 5. 解码器与损失函数对比

| 解码器/损失 | P/% | R/% | F1/% | 证据来源 | 采用状态 |
|---|---:|---:|---:|---|---|
| CRF | 86.32 | 87.80 | 87.05 | `docs/paper/基于词典和边界预测的红枣栽培命名实体识别.md` 表 3 | 采用 |
| 边界预测 | 90.12 | 85.61 | 87.81 | 同上 | 采用 |
| 边界预测 + Focal Loss | 89.51 | 87.58 | 88.54 | 同上 | 限该表采用 |

解释口径：

- 该表用于说明 Focal Loss 对召回率的提升，不能与三种子均值主表混为同一统计口径。
- 投稿稿中需要在表注中注明“该表为代表性运行结果”。

## 6. 公开数据集泛化结果

现有主稿整理结果如下：

| 数据集 | EDBP F1/% | 证据来源 | 采用状态 |
|---|---:|---|---|
| MSRA | 95.19±0.22 | `docs/paper/基于词典和边界预测的红枣栽培命名实体识别.md` 表 8 | 采用 |
| WeiboNER | 72.27±1.03 | 同上 | 采用 |
| ResumeNER | 96.13±0.29 | 同上 | 采用 |
| Boson | 85.60±0.12 | 同上 | 采用 |
| CLUENER | 80.06±0.38 | 同上 | 采用 |

本地结果校验：

- MSRA seed 42/43/44：`experiments/EXP-010-optimization/results_public/msra_bs_dict_focal/`
- WeiboNER seed 42/43/44：`experiments/EXP-010-optimization/results_public/weibo_bs_dict_focal/`
- ResumeNER seed 42/43/44：`experiments/EXP-010-optimization/results_public/resume_bs_dict_focal/`
- Boson seed 42/43/44：`experiments/EXP-010-optimization/results_public/boson_bs_dict_focal/`
- CLUENER seed 42/43/44：`experiments/EXP-010-optimization/results_public/clue_bs_dict_focal/`

注意：

- 部分目录存在同 seed 重跑结果，投稿稿采用已在主稿中汇总的均值表。
- 后续若要严格重算均值，需要先确定每个 seed 采用哪一次运行，避免重复运行污染均值。

## 7. 2026-03 新数据结果

| 实验组 | 代表文件 | 结果 | 采用状态 | 原因 |
|---|---|---|---|---|
| H_bs_baseline | `experiments/EXP-010-optimization/results_newdata/H_bs_baseline/` | test F1 约 87.55-87.81 | 暂不进入主稿 | 数据重构后样本粒度变化，不能与旧稿 RJND 主表混用 |
| Q_bs_focal | `experiments/EXP-010-optimization/results_newdata/Q_bs_focal/` | test F1 约 88.15-88.54 | 暂不进入主稿 | 与旧稿主表统计口径不同 |
| Q_bs_focal_attnv1 | `experiments/EXP-010-optimization/results_newdata/Q_bs_focal_attnv1_s*/` | test F1 约 88.10-88.73 | 可作为后续补充实验候选 | 若采用需重写方法为 CA-BMES，并补消融 |

相关风险报告：

- `docs/RedJujube_DatasetAnalysis_Report_2026-03-18.md`
- `experiments/EXP-010-optimization/REDJUJUBE_DEV_TEST_ANALYSIS.md`
- `experiments/EXP-010-optimization/QUICK_SUMMARY_ZH.txt`
- `experiments/EXP-010-optimization/SUMMARY_TABLES.txt`

处理决策：

- 当前投稿稿不混入新旧数据结果。
- 若后续决定使用新数据版本，需要重新生成完整论文结果表，包括数据集统计、主对比、消融、公开数据集泛化与显著性分析。

## 8. 投稿稿统一写法

推荐摘要结果句：

> 结果表明，EDBP 在 RJND 数据集上的 F1 值达到 88.16%，较 BiLSTM-CRF、BERT-wwm-ext+BiLSTM+CRF、MacBERT-base+BiLSTM+CRF、SoftLexicon-TrainLex、AdaSeq BERT-CRF、FLAT 和 FLAT+BERT 分别提升 9.86、2.95、2.80、2.61、3.00、8.38 和 8.76 个百分点。

推荐结果分析补充句：

> 在代表性运行的分类统计中，EDBP 的精确率、召回率和 F1 值分别为 89.51%、87.58% 和 88.54%，其中病虫害、虫害、部位等类别取得较高识别效果，低频肥料和分类实体仍是主要误差来源。
