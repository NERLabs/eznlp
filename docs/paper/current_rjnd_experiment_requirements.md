# 当前 RJND 论文补实验需求清单

本文档供服务器实验端读取，用于设计、排队和运行红枣栽培 NER 论文补充实验。

## 1. 统一实验口径

所有计划进入当前投稿稿正文主对比表的结果，必须满足以下条件：

| 项目 | 要求 |
|---|---|
| 数据集 | 当前论文使用的 RJND/RedJujube 数据划分 |
| 随机种子 | `seed=42` |
| 评估集 | test split |
| 评估方式 | 实体级严格 P/R/F1，单位为 `%` |
| 主指标 | test F1/% |
| 结果记录 | 必须保存 result/log/config 路径 |
| 可比性 | 不混入旧版红枣数据、HZ 数据、公开数据集或不同标签集合结果 |

旧版实验表中的红枣结果，如 `Boundary Smoothing=96.30`、`BERT+SoftLexicon=95.07`、
`BERT-MRC=91.84` 等，只能作为模型筛选参考，不能直接作为当前论文证据。

## 2. 必跑优先级

| 优先级 | 模型 | seed | 数据集 | 需要返回 | 目标用途 | 备注 |
|---:|---|---:|---|---|---|---|
| 1 | Boundary Smoothing | 42 | 当前 RJND/RedJujube | test P/R/F1、result_path、config_path、eval_script、backbone | 判断是否替换或补强正文表 3 | 与本文边界预测主题最相关，最高优先级 |
| 2 | BERT+SoftLexicon / SoftLexicon+BERT | 42 | 当前 RJND/RedJujube | test P/R/F1、result_path、config_path、词典来源、词典规模、backbone | 词典增强强基线 | 必须说明词典来自训练集、外部词典还是旧 HZ 词典 |
| 3 | BERT-MRC | 42 | 当前 RJND/RedJujube | test P/R/F1、result_path、config_path、MRC 转换脚本、query 模板 | 经典 MRC-NER 强基线 | 使用当前 RJND 重新生成 MRC 格式数据 |
| 4 | BERT-MRC+DSC | 42 | 当前 RJND/RedJujube | test P/R/F1、result_path、config_path、Dice/DSC 参数、query 模板 | 类别不平衡损失强基线 | 可与本文 Focal Loss 讨论关联 |
| 5 | RA_NER / AdaSeq BERT-CRF | 42 | 当前 RJND/RedJujube | test P/R/F1、result_path、config_path、dataset adapter、backbone | 强 BERT-CRF 或检索增强基线 | 若使用检索增强，必须记录外部数据或检索库来源 |

## 3. 可选补跑

| 优先级 | 模型 | seed | 数据集 | 需要返回 | 目标用途 | 备注 |
|---:|---|---:|---|---|---|---|
| 6 | W2NER | 42 | 当前 RJND/RedJujube | test P/R/F1、result_path、config_path | 现代统一 NER 建模补充对比 | 工作量较高，非当前最急 |
| 7 | DiffusionNER | 42 | 当前 RJND/RedJujube | test P/R/F1、result_path、config_path | 新模型补充对比 | 与本文机制距离较远 |
| 8 | PIQN | 42 | 当前 RJND/RedJujube | test P/R/F1、result_path、config_path | query/span 类补充对比 | 非当前最急 |
| 9 | LatticeLSTM | 42 | 当前 RJND/RedJujube | test P/R/F1、result_path、config_path、词典来源 | 经典 lattice 对比 | 本地旧环境阻塞，若服务器已有环境可跑 |
| 10 | NFLAT | 42 | 当前 RJND/RedJujube | test P/R/F1、result_path、config_path、词典来源 | 新 lattice 对比 | 本地旧环境阻塞，若服务器已有环境可跑 |

## 4. 已有当前口径结果

下列结果已经登记，可作为当前正文主表基础；服务器端无需重复跑，除非要统一重算全部模型。

| 模型 | seed | test F1/% | 状态 |
|---|---:|---:|---|
| BiLSTM-CRF | 42 | 78.30 | 已登记 |
| BERT-wwm-ext+BiLSTM+CRF | 42 | 85.21 | 已登记 |
| MacBERT-base+BiLSTM+CRF | 42 | 85.36 | 已登记 |
| SoftLexicon | 42 | 84.75 | 已登记 |
| FLAT | 42 | 79.78 | 已登记 |
| FLAT+BERT | 42 | 79.40 | 已登记 |
| EDBP | 42 | 88.16 | 已登记，`min_freq=2`，词典规模 1 842 |

## 5. 结果返回格式

每个实验完成后，请按以下字段登记，便于直接同步到
`docs/paper/paper_result_registry.md` 和论文表 3。

| 字段 | 示例或说明 |
|---|---|
| model_name | `Boundary Smoothing` |
| seed | `42` |
| dataset_split | 当前 RJND/RedJujube 数据路径或划分名 |
| precision | test Precision/% |
| recall | test Recall/% |
| f1 | test F1/% |
| result_path | `results.json`、`metrics.json` 或日志路径 |
| config_path | yaml、shell、命令记录或参数文件路径 |
| eval_script | 使用的评估脚本路径 |
| bert_backbone | 如 `hfl/chinese-macbert-base`、`hfl/chinese-bert-wwm-ext` |
| lexicon | 是否使用词典；词典来源；词典规模 |
| data_process | 是否转换数据格式，如 MRC 格式、span 格式、word-word relation 格式 |
| note | 口径差异、环境限制、异常现象 |

## 6. 不要进入当前正文主表的结果

| 类型 | 原因 |
|---|---|
| 旧版红枣实验结果 | 与当前论文数据不是同一版 |
| HZ 数据集结果 | 与当前 RJND 划分不同 |
| MSRA、Resume、Boson、CLUENER、WeiboNER 结果 | 只能用于公开数据集泛化实验 |
| seed 不是 42 的单次结果 | 当前正文表 3 采用 seed=42 单组同口径 |
| 不同标签集合或不同评估脚本结果 | F1 不可直接比较 |
| 只报告 dev F1 的结果 | 正文主表需要 test F1 |

