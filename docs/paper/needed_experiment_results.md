# 论文补充实验结果查询清单

本文档用于服务器侧查询或补跑红枣栽培 NER 论文仍缺的实验结果。除特别说明外，所有结果均要求使用 **RJND/RedJujube 当前投稿稿数据划分**、**seed=42**、**同一评估脚本的测试集 F1/%**，以便和正文表 3 的主对比口径一致。

## 1. 必须优先查询

| 优先级 | 模型 | seed | 数据集 | 需要结果 | 目标用途 | 备注 |
|---:|---|---:|---|---|---|---|
| 1 | LatticeLSTM | 42 | RJND/RedJujube | 测试集 P/R/F1、结果路径、配置文件路径 | 判断是否纳入正文表 3 | 当前 registry 只登记为“待补”，未找到 RJND 同口径结果 |
| 2 | NFLAT | 42 | RJND/RedJujube | 测试集 P/R/F1、结果路径、配置文件路径 | 判断是否纳入正文表 3 | 当前 registry 只找到 MSRA 相关结果，未找到 RJND 同口径结果 |
| 3 | EDBP min_freq=1 | 42 | RJND/RedJujube | 测试集 P/R/F1、词典规模、结果路径 | 支撑表 5 “不同词频”从代理指标升级为真实 NER 指标 | 当前只有词典覆盖/匹配代理指标，缺测试 F1 |
| 4 | EDBP min_freq=3 | 42 | RJND/RedJujube | 测试集 P/R/F1、词典规模、结果路径 | 支撑表 5 “不同词频”从代理指标升级为真实 NER 指标 | 当前只有词典覆盖/匹配代理指标，缺测试 F1 |

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
