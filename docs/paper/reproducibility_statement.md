# 红枣 NER 投稿稿复现实验说明

本文档用于补充 `农业机械学报_红枣NER_投稿稿.md` 的实验可复现信息。投稿正文不直接暴露本地工程路径；本文件用于内部交接、审稿补充材料准备和后续重跑实验。

## 1. 数据

| 数据/资源 | 路径 | 用途 |
|---|---|---|
| RJND 训练集 | `datasets/raw/RedJujube/redjujube_train.bmes` | 主实验训练数据 |
| RJND 验证集 | `datasets/raw/RedJujube/redjujube_dev.bmes` | 模型选择和调参 |
| RJND 测试集 | `datasets/raw/RedJujube/redjujube_test.bmes` | 主结果测试 |
| 专家词典 min_freq=2 | 训练集自动抽取词典，词典规模 1 842 | EDBP 专家词典特征 |
| MSRA | `datasets/raw/MSRA/` | 公开数据集泛化实验 |
| WeiboNER | `datasets/raw/WeiboNER/` | 公开数据集泛化实验 |
| ResumeNER | `datasets/raw/ResumeNER/` | 公开数据集泛化实验 |
| Boson | `datasets/raw/boson/` | 公开数据集泛化实验 |
| CLUENER | `datasets/raw/clue/` | 公开数据集泛化实验 |

## 2. 代码入口

| 任务 | 入口 |
|---|---|
| Taskfile 任务定义 | `Taskfile.yml` |
| 红枣基础训练脚本 | `research/training/train_redjujube_ner.py` |
| 红枣边界预测训练脚本 | `research/training/train_redjujube_expert_boundary.py` |
| 通用专家词典边界预测训练脚本 | `research/training/train_general_expert_boundary.py` |
| 红枣实验脚本目录 | `research/configs/redjujube/` |
| eznlp 编码器实现 | `eznlp/model/encoder.py` |
| 专家词典嵌入实现 | `eznlp/model/nested_embedder.py` |
| 边界选择/边界预测解码器 | `eznlp/model/decoder/boundary_selection.py` |
| 词典抽取工具 | `research/data_processing/extract_lexicon_from_training.py` |

## 3. 推荐复现实验命令

`Taskfile.yml` 中部分早期任务仍指向旧实验脚本，复现实验前应优先核对 `research/training/` 和 `research/configs/redjujube/` 中当前实际脚本。基础基线：

```bash
task train:redjujube:baseline
```

专家词典模型：

```bash
task train:redjujube:expert-auto
```

公开数据集与边界预测相关实验主要在 `research/configs/redjujube/` 下通过批处理脚本运行，例如：

```bash
bash research/configs/redjujube/run_public_all_sequential.sh
bash research/configs/redjujube/run_bs_optimization_experiments.sh
```

## 4. 投稿稿采用结果来源

| 结果项 | 采用值 | 证据来源 |
|---|---:|---|
| RJND 主结果 | 88.28%±0.22% | `docs/paper/基于词典和边界预测的红枣栽培命名实体识别.md` 表 6；`paper_result_registry.md` |
| 代表性运行 P/R/F1 | 89.51/87.58/88.54 | `docs/paper/基于词典和边界预测的红枣栽培命名实体识别.md` 表 3/表 5 |
| 词典策略 | min_freq=2，词典规模 1 842 | `experiments/EXP-011-lexicon_strategy/analysis/candidate_proxy_table.csv` 与 `results_newdata/Q_bs_focal/.../results.json` |
| MSRA | 95.19%±0.22% | `paper_result_registry.md`；`experiments/EXP-010-optimization/results_public/msra_bs_dict_focal/` |
| WeiboNER | 72.27%±1.03% | `paper_result_registry.md`；`experiments/EXP-010-optimization/results_public/weibo_bs_dict_focal/` |
| ResumeNER | 96.13%±0.29% | `paper_result_registry.md`；`experiments/EXP-010-optimization/results_public/resume_bs_dict_focal/` |
| Boson | 85.60%±0.12% | `paper_result_registry.md`；`experiments/EXP-010-optimization/results_public/boson_bs_dict_focal/` |
| CLUENER | 80.06%±0.38% | `paper_result_registry.md`；`experiments/EXP-010-optimization/results_public/clue_bs_dict_focal/` |

## 5. 结果采用原则

1. 投稿稿主结果采用旧稿已汇总的 RJND 三随机种子均值 `88.28%±0.22%`。
2. `89.51/87.58/88.54` 仅用于代表性运行、类别分析和解码器/损失函数行为解释。
3. `experiments/EXP-010-optimization/results_newdata/` 中 2026-03 新数据结果暂不进入主稿，避免新旧数据划分和统计口径混用。
4. 词典策略表使用 EXP-011 的训练集词典匹配代理指标，只解释阈值选择，不作为测试集 NER 性能。

## 6. 重跑注意事项

1. 需安装 PyTorch、Transformers、eznlp 项目依赖和中文预训练模型 `hfl/chinese-macbert-base`。
2. 慢实验建议固定随机种子 42、43、44 后分别运行，并记录每个 seed 的 `results.json`。
3. 若重算主结果，应同时重算主对比、消融、类别分析、公开数据集泛化和图表；不能只替换摘要或单个表格。
4. 若采用 2026-03 新数据结果，必须重新生成 `paper_result_registry.md` 并重新通过全量投稿检查。
