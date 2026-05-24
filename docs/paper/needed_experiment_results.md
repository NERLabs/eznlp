# 论文补充实验结果查询清单

本文档用于服务器侧查询或补跑红枣栽培 NER 论文仍缺的实验结果。除特别说明外，所有结果均要求使用 **RJND/RedJujube 当前投稿稿数据划分**、**seed=42**、**同一评估脚本的测试集 F1/%**，以便和正文表 3 的主对比口径一致。

服务器端优先读取 `docs/paper/current_rjnd_experiment_requirements.md`。该文件是当前最新版补实验任务包，明确旧版红枣结果不能直接进入当前论文主表。

## 1. 必须优先查询

| 优先级 | 模型 | seed | 数据集 | 需要结果 | 目标用途 | 备注 |
|---:|---|---:|---|---|---|---|
| 1 | Boundary Smoothing | 42 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径、配置文件路径、评估脚本、backbone | 判断是否纳入正文表 3 | 与本文边界预测主题最相关；旧版红枣结果不能直接采用 |
| 2 | BERT+SoftLexicon / SoftLexicon+BERT | 42 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径、配置文件路径、词典来源、词典规模、backbone | 词典增强强基线 | 必须说明词典是否来自训练集、外部词典或旧 HZ 词典 |
| 3 | BERT-MRC | 42 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径、配置文件路径、MRC 转换脚本、query 模板 | 经典 MRC-NER 强基线 | 需使用当前 RJND 重新生成 MRC 格式数据 |
| 4 | BERT-MRC+DSC | 42 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径、配置文件路径、Dice/DSC 参数、query 模板 | 类别不平衡损失强基线 | 可与本文 Focal Loss 讨论关联 |
| 5 | RA_NER / AdaSeq BERT-CRF | 42 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径、配置文件路径、dataset adapter、backbone | 强 BERT-CRF 或检索增强基线 | 若使用检索增强，必须记录外部数据或检索库来源 |

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
| 6 | W2NER | 42 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径、配置文件路径 | 现代统一 NER 建模补充对比 | 工作量较高，非当前最急 |
| 7 | DiffusionNER | 42 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径、配置文件路径 | 新模型补充对比 | 与本文机制距离较远 |
| 8 | PIQN | 42 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径、配置文件路径 | query/span 类补充对比 | 非当前最急 |
| 9 | LatticeLSTM | 42 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径、配置文件路径、词典来源 | 经典 lattice 对比 | 本地旧环境阻塞，若服务器已有环境可跑 |
| 10 | NFLAT | 42 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径、配置文件路径、词典来源 | 新 lattice 对比 | 本地旧环境阻塞，若服务器已有环境可跑 |
| 11 | SoftLexicon / BERT-SoftLexicon | 43、44 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径 | 如果后续需要三种子均值主表，可补齐 SoftLexicon 的 43/44 | seed=42 已登记为 84.75 |
| 12 | FLAT | 43、44 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径 | 如果后续需要三种子均值主表，可补齐 FLAT 的 43/44 | seed=42 已登记为 79.78 |
| 13 | FLAT+BERT | 43、44 | RJND/RedJujube 当前投稿稿划分 | 测试集 P/R/F1、结果路径 | 如果后续需要三种子均值主表，可补齐 FLAT+BERT 的 43/44 | seed=42 已登记为 79.40 |

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
| 旧版红枣实验结果 | 与当前论文数据不是同一版，不能直接和 88.16 比较 |
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
