# 论文补充实验结果查询清单

本文档用于服务器侧查询或补跑红枣栽培 NER 论文仍缺的实验结果。除特别说明外，所有结果均要求使用 **RJND/RedJujube 当前投稿稿数据划分**、**seed=42**、**同一评估脚本的测试集 F1/%**，以便和正文表 3 的主对比口径一致。

## 1. 必须优先查询

| 优先级 | 模型 | seed | 数据集 | 需要结果 | 目标用途 | 备注 |
|---:|---|---:|---|---|---|---|
| 1 | LatticeLSTM | 42 | RJND/RedJujube | 测试集 P/R/F1、结果路径、配置文件路径 | 判断是否纳入正文表 3 | 当前 registry 只登记为“待补”，未找到 RJND 同口径结果 |
| 2 | NFLAT | 42 | RJND/RedJujube | 测试集 P/R/F1、结果路径、配置文件路径 | 判断是否纳入正文表 3 | 当前 registry 只找到 MSRA 相关结果，未找到 RJND 同口径结果 |
| 3 | EDBP min_freq=1 | 42 | RJND/RedJujube | 测试集 P/R/F1、词典规模、结果路径 | 支撑表 5 “不同词频”从代理指标升级为真实 NER 指标 | 2026-05-24 已补跑，见 1.1 |
| 4 | EDBP min_freq=3 | 42 | RJND/RedJujube | 测试集 P/R/F1、词典规模、结果路径 | 支撑表 5 “不同词频”从代理指标升级为真实 NER 指标 | 2026-05-24 已补跑，见 1.1 |

### 1.1 已补跑结果（2026-05-24）

下表使用 `datasets/raw/RedJujube`、seed=42、`hfl/chinese-macbert-base`、
`sb_epsilon=0.1`、`sb_size=2`、`fl_gamma=2.0`、`no_fgm`、`no_ema`。
F1 来自训练脚本保存的最佳模型测试结果；Precision 和 Recall 由
`research/evaluation/test_redjujube_baseline.py` 加载同一最佳模型复评得到。

| 模型 | seed | 词典规模 | P/% | R/% | F1/% | result_path | config_path | 状态 |
|---|---:|---:|---:|---:|---:|---|---|---|
| EDBP min_freq=1 | 42 | 5 317 | 86.94 | 81.92 | 84.36 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current/expert_boundary_20260524-124317/results.json` | `docs/paper/plans/2026-05-24-needed-experiments-execution.md` | 已完成 |
| EDBP min_freq=3 | 42 | 1 087 | 88.56 | 86.49 | 87.51 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/expert_boundary_20260524-125722/results.json` | `docs/paper/plans/2026-05-24-needed-experiments-execution.md` | 已完成 |
| EDBP min_freq=4 | 42 | 786 | 87.79 | 86.12 | 86.95 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq4_seed42_current/expert_boundary_20260524-132104/results.json` | `docs/paper/plans/2026-05-24-needed-experiments-execution.md` | 已完成 |

执行约定：后续长实验统一使用 tmux 后台会话，查询使用 `tmux ls`、
`tmux capture-pane -pt <session> -S -80` 和对应 `training.log`。

## 2. 建议补充查询

| 优先级 | 模型 | seed | 数据集 | 需要结果 | 目标用途 | 备注 |
|---:|---|---:|---|---|---|---|
| 5 | SoftLexicon / BERT-SoftLexicon | 43、44 | RJND/RedJujube | 测试集 P/R/F1、结果路径 | 如果后续需要三种子均值主表，可补齐 SoftLexicon 的 43/44 | seed=42 已登记为 84.75 |
| 6 | FLAT | 43、44 | RJND/RedJujube | 测试集 P/R/F1、结果路径 | 如果后续需要三种子均值主表，可补齐 FLAT 的 43/44 | seed=42 已登记为 79.78 |
| 7 | FLAT+BERT | 43、44 | RJND/RedJujube | 测试集 P/R/F1、结果路径 | 如果后续需要三种子均值主表，可补齐 FLAT+BERT 的 43/44 | seed=42 已登记为 79.40 |
| 8 | BiLSTM-CRF、BERT-wwm-ext+BiLSTM+CRF、MacBERT-base+BiLSTM+CRF、EDBP | 42 | RJND/RedJujube | 复核测试集 P/R/F1 和路径 | 复核正文表 3 当前主口径 | 目前 F1 已登记：78.30、85.21、85.36、88.16 |

## 3. 查询时必须记录的字段

每条结果请至少记录以下字段，便于直接同步到 `docs/paper/paper_result_registry.md`：

| 字段 | 说明 |
|---|---|
| model_name | 论文表中模型名，例如 `LatticeLSTM`、`NFLAT`、`EDBP min_freq=1` |
| seed | 随机种子，正文主表优先使用 42 |
| dataset_split | 数据划分名称或路径，必须确认是 RJND/RedJujube 当前投稿稿划分 |
| precision | 测试集 Precision/%；若结果文件没有可留空 |
| recall | 测试集 Recall/%；若结果文件没有可留空 |
| f1 | 测试集 F1/% |
| result_path | `results.json` 或日志文件路径 |
| config_path | 训练配置、命令或脚本路径 |
| lexicon | 是否使用词典、词典来源、词典规模 |
| bert_backbone | 是否使用 BERT/MacBERT，若使用需写模型名 |
| note | 环境限制、未完成原因或与正文口径差异 |

## 4. 当前不要混入正文主表的结果

| 类型 | 原因 |
|---|---|
| 旧 HZ 数据集结果 | 与当前 RJND 投稿稿数据划分不同，不能直接和 88.16 比较 |
| MSRA/Resume/Boson/CLUENER/Weibo 公开数据集结果 | 只用于泛化实验，不用于 RJND 表 3 主模型对比 |
| seed 不是 42 的单次结果 | 当前正文表 3 使用 seed=42 单组同口径结果；其他 seed 可作为补充稳定性证据 |
| 不同评估脚本或不同标签集合结果 | 会造成 F1 不可比，必须单独标注 |

## 5. 已登记的当前正文主表结果

| 模型 | seed | F1/% | 当前状态 |
|---|---:|---:|---|
| BiLSTM-CRF | 42 | 78.30 | 已采用 |
| BERT-wwm-ext+BiLSTM+CRF | 42 | 85.21 | 已采用 |
| MacBERT-base+BiLSTM+CRF | 42 | 85.36 | 已采用 |
| SoftLexicon | 42 | 84.75 | 已采用 |
| FLAT | 42 | 79.78 | 已采用 |
| FLAT+BERT | 42 | 79.40 | 已采用 |
| EDBP | 42 | 88.16 | 已采用；专家词典 `min_freq=2`，词典规模 1 842 |
