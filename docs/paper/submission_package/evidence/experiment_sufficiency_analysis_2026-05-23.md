# 实验充分性补强分析

生成脚本：`docs/paper/tools/analyze_experiment_sufficiency.py`

## 1 主要结论

- 现有实验已经覆盖主模型、无词典/无边界/Focal 消融、公开数据集泛化、词典阈值和新数据分布鲁棒性分析；短期不建议为了追求更高单点 F1 直接替换投稿主结果。
- 当前最值得补进投稿材料的是“实验工作量说明”和“鲁棒性审计”，而不是把新数据版本结果与旧 RJND 主表混用。
- 已在 `results_newdata` 中定位到投稿稿表 3 对应的三种子原始结果：`CRF_nodict_bertwwm`、`CRF_nodict` 和 `Q_bs_focal`，可支撑配对 t 检验。
- 新数据版本中存在若干单种子高分配置，但只有三种子结果才适合进入主表；短期不建议用单种子通道注意力结果替换当前主结果。

## 2 新数据版本结果概览

| 实验组 | n | Dev F1/% | Test F1/% | Test-Dev/pp |
|---|---:|---:|---:|---:|
| `experiments/EXP-010-optimization/results_newdata/Q_bs_focal_attnv1_s42` | 1 | 85.80±0.00 | 88.73±0.00 | 2.93 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_ce_attnv1_s44` | 1 | 86.66±0.00 | 88.36±0.00 | 1.71 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_focal_attnv2_bmesaux_s42` | 1 | 86.07±0.00 | 88.33±0.00 | 2.27 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_focal_attnv1_s44` | 1 | 85.96±0.00 | 88.30±0.00 | 2.34 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_focal` | 3 | 86.13±0.19 | 88.28±0.22 | 2.15 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_focal_attnv2_s44` | 1 | 85.83±0.00 | 88.28±0.00 | 2.46 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_focal_attnv1_s43_diag` | 1 | 86.11±0.00 | 88.28±0.00 | 2.17 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_focal_attnv2_s42` | 1 | 85.72±0.00 | 88.27±0.00 | 2.55 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_focal_attnv2_s43` | 1 | 85.70±0.00 | 88.21±0.00 | 2.50 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_focal_attnv1_s43` | 1 | 85.61±0.00 | 88.11±0.00 | 2.50 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_ce_attnv1_s42` | 1 | 86.27±0.00 | 87.90±0.00 | 1.63 |
| `experiments/EXP-010-optimization/results_newdata/R_bs_typeaware_dict` | 1 | 86.32±0.00 | 87.79±0.00 | 1.47 |

## 3 泛化稳定性排序

| 实验组 | n | Test F1/% | |Test-Dev|/pp | 判断 |
|---|---:|---:|---:|---|
| `experiments/EXP-010-optimization/results_newdata/G_bilstm_baseline` | 3 | 78.69±0.34 | 0.61 | 稳定 |
| `experiments/EXP-010-optimization/results_newdata/CRF_nodict_bertwwm` | 3 | 85.48±0.25 | 0.86 | 稳定 |
| `experiments/EXP-010-optimization/results_newdata/BS_nodict` | 3 | 86.68±0.10 | 1.22 | 需谨慎 |
| `experiments/EXP-010-optimization/results_newdata/S_bs_srg` | 3 | 87.68±0.28 | 1.24 | 需谨慎 |
| `experiments/EXP-010-optimization/results_newdata/U_bs_adaptive_sbsize` | 3 | 87.58±0.26 | 1.26 | 需谨慎 |
| `experiments/EXP-010-optimization/results_newdata/CRF_nodict` | 3 | 85.57±0.29 | 1.29 | 需谨慎 |
| `experiments/EXP-010-optimization/results_newdata/A_baseline` | 3 | 86.71±0.35 | 1.41 | 需谨慎 |
| `experiments/EXP-010-optimization/results_newdata/W_enhanced_size_emb` | 3 | 87.64±0.25 | 1.43 | 需谨慎 |
| `experiments/EXP-010-optimization/results_newdata/H_bs_baseline` | 3 | 87.67±0.13 | 1.48 | 需谨慎 |
| `experiments/EXP-010-optimization/results_newdata/BS_focal_nodict` | 3 | 86.58±0.16 | 1.51 | 需谨慎 |
| `experiments/EXP-010-optimization/results_newdata/Q_bs_focal` | 3 | 88.28±0.22 | 2.15 | 需谨慎 |

## 4 配对 t 检验（新数据版本，仅作补充审计）

| 对比 | seeds | 均值差/pp | t | p | 解释 |
|---|---|---:|---:|---:|---|
| `Q_bs_focal` - `CRF_nodict_bertwwm` | 42,43,44 | 2.80 | 16.518 | 0.0036 | 达到 0.05 水平 |
| `Q_bs_focal` - `CRF_nodict` | 42,43,44 | 2.71 | 55.268 | 0.0003 | 达到 0.05 水平 |
| `Q_bs_focal` - `G_bilstm_baseline` | 42,43,44 | 9.60 | 51.431 | 0.0004 | 达到 0.05 水平 |
| `Q_bs_focal` - `H_bs_baseline` | 42,43,44 | 0.62 | 9.688 | 0.0105 | 达到 0.05 水平 |
| `H_bs_baseline` - `CRF_nodict` | 42,43,44 | 2.10 | 21.907 | 0.0021 | 达到 0.05 水平 |

## 5 对投稿稿的处理建议

1. 主结果继续采用三种子均值 `88.28%±0.22%`，并在结果注册表中补入 raw seed 路径。
2. 可在正文主结果段落加入配对 t 检验句，但应仅针对三种子强基线，不扩展到单种子模块。
3. 可在补充材料或答审材料中加入本文件，说明已进行多配置鲁棒性分析。
4. 若继续新增训练，优先补 `Q_bs_focal_attnv1` 的 seed 43/44 独立复核或 K 折验证；不要直接用单种子最高值替换主结论。

## 6 生成文件

- `docs/paper/experiment_sufficiency_groups_2026-05-23.csv`
- `docs/paper/experiment_sufficiency_seeds_2026-05-23.csv`
