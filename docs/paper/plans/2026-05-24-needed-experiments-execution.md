# 2026-05-24 补实验执行计划

## 目标

补齐 `docs/paper/needed_experiment_results.md` 中当前最容易落地的 RJND/RedJujube
同口径结果，并为暂不能直接训练的经典 lattice 基线记录前置条件。

## 统一口径

- 数据：`datasets/raw/RedJujube`
- 随机种子：`42`
- 评价：同一训练脚本输出的测试集 P/R/F1，优先记录 `results.json`
- 主模型参数：EDBP，`sb_epsilon=0.1`，`sb_size=2`，`fl_gamma=2.0`，
  `no_fgm`，`no_ema`
- 训练环境：`conda run -n eznlp11`，GPU 0

## 执行顺序

1. 查询已有结果，确认是否可直接登记。
2. 重跑 `EDBP min_freq=1`，保存到
   `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current/`。
3. 重跑 `EDBP min_freq=3`，保存到
   `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/`。
4. 训练完成后记录：
   `model_name`、`seed`、`dataset_split`、`precision`、`recall`、`f1`、
   `result_path`、`config_path`、`lexicon`、`bert_backbone`、`note`。
5. 对 `LatticeLSTM` 暂不直接启动训练；对 `NFLAT` 先做 RedJujube
   adapter/依赖/smoke 验证，再等待 GPU 可用后启动全量训练。
6. 旧路径 `Boundary Smoothing` 结果不能证明当前 `datasets/raw/RedJujube`
   口径时，补跑无专家词典的 `BoundarySelectionDecoder` seed=42 基线。

## 命令

长实验统一放到 tmux 中运行，便于断开终端后继续训练和随时查询。
2026-05-24 第一组 `min_freq=1` 已在普通执行会话启动，后续实验从
`min_freq=3` 开始使用 tmux。

```bash
conda run -n eznlp11 env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python research/training/train_redjujube_expert_boundary.py \
  --data_dir datasets/raw/RedJujube \
  --bert_arch hfl/chinese-macbert-base \
  --save_dir experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current \
  --seed 42 \
  --num_epochs 30 \
  --batch_size 16 \
  --sb_epsilon 0.1 \
  --sb_size 2 \
  --min_freq 1 \
  --fl_gamma 2.0 \
  --no_fgm \
  --no_ema
```

```bash
tmux new-session -d -s rjnd-mf3-20260524 -c /home/shiwenlong/NERlabs/eznlp \
  'conda run -n eznlp11 env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python research/training/train_redjujube_expert_boundary.py \
  --data_dir datasets/raw/RedJujube \
  --bert_arch hfl/chinese-macbert-base \
  --save_dir experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current \
  --seed 42 \
  --num_epochs 30 \
  --batch_size 16 \
  --sb_epsilon 0.1 \
  --sb_size 2 \
  --min_freq 3 \
  --fl_gamma 2.0 \
  --no_fgm \
  --no_ema'
```

```bash
tmux new-session -d -s rjnd-bs-nodict-20260524 -c /home/shiwenlong/NERlabs/eznlp \
  'conda run -n eznlp11 env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/shiwenlong/NERlabs/eznlp \
python research/training/train_redjujube_expert_boundary.py \
  --data_dir datasets/raw/RedJujube \
  --bert_arch hfl/chinese-macbert-base \
  --save_dir experiments/EXP-010-optimization/results_needed_20260524/BS_nodict_seed42_current \
  --seed 42 \
  --num_epochs 30 \
  --batch_size 16 \
  --sb_epsilon 0.1 \
  --sb_size 2 \
  --fl_gamma 0.0 \
  --no_expert_dict \
  --no_fgm \
  --no_ema'
```

## 查询命令

```bash
tmux ls
tmux capture-pane -pt rjnd-mf3-20260524 -S -80
tail -n 80 experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/*/training.log
find experiments/EXP-010-optimization/results_needed_20260524 -path '*results.json' -print
```

P/R/F1 复评命令：

```bash
conda run -n eznlp11 env PYTHONPATH=/home/shiwenlong/NERlabs/eznlp \
python research/evaluation/test_redjujube_baseline.py \
  --save_dir <result_dir> \
  --model_type expert_dict_auto \
  --expert_dict_auto_path <result_dir>/auto_lexicon.txt \
  --no_export_predictions
```

## 执行结果

2026-05-24 已完成 `EDBP min_freq=1`、`EDBP min_freq=3`、`EDBP min_freq=4`
、`Boundary Smoothing`、`SoftLexicon-TrainLex`、`SoftLexicon-External`
、`AdaSeq BERT-CRF` 与 `BERT-MRC+DSC` 当前路径 RJND/RedJujube seed=42 实验。
eznlp 内部补跑的 `results.json` 只保存 loss/F1，Precision 和
Recall 由 `research/evaluation/test_redjujube_baseline.py` 加载最佳模型复评得到；
`BERT-MRC+DSC` 指标来自外部 `dice_loss_for_NLP` 的 `eval_result_log.txt`。

| 模型 | seed | 词典规模 | 测试 P/% | 测试 R/% | 测试 F1/% | best dev F1/% | best epoch | 结果目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EDBP min_freq=1 | 42 | 5 317 | 86.94 | 81.92 | 84.36 | 83.47 | 19 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current/expert_boundary_20260524-124317/` |
| EDBP min_freq=3 | 42 | 1 087 | 88.56 | 86.49 | 87.51 | 86.30 | 22 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/expert_boundary_20260524-125722/` |
| EDBP min_freq=4 | 42 | 786 | 87.79 | 86.12 | 86.95 | 85.90 | 22 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq4_seed42_current/expert_boundary_20260524-132104/` |
| Boundary Smoothing | 42 | 0 | 87.36 | 85.61 | 86.48 | 85.54 | 18 | `experiments/EXP-010-optimization/results_needed_20260524/BS_nodict_seed42_current/bert_bs_pure_20260524-164044/` |
| SoftLexicon-TrainLex | 42 | 198 437 | 84.42 | 86.71 | 85.55 | 84.02 | 28 | `experiments/EXP-010-optimization/results_needed_20260524/SoftLexicon_trainlex_seed42_current/softlexicon_trainlex_20260524-171342/` |
| SoftLexicon-External | 42 | CTB 50d vector vocab | 84.32 | 85.65 | 84.98 | 83.78 | 28 | `experiments/EXP-010-optimization/results_needed_20260524/SoftLexicon_external_seed42_current/softlexicon_20260524-173106/` |
| AdaSeq BERT-CRF | 42 | 0 | 84.42 | 85.90 | 85.16 | - | 30 | `experiments/EXP-010-optimization/results_needed_20260524/AdaSeq_bert_crf_seed42_current/metrics_summary.json` |
| BERT-MRC+DSC | 42 | 0 | 83.06 | 77.77 | 80.33 | 80.19 | 8 | `experiments/EXP-010-optimization/results_needed_20260524/MRC_DSC_current_redjujube_status_20260524.json` |

校验记录：

- `min_freq=1`：`results.json` 中 `test_f1=0.8435502068446783`；
  `auto_lexicon.txt` 共 5 317 行。
- `min_freq=3`：`results.json` 中 `test_f1=0.8750923872875093`；
  `auto_lexicon.txt` 共 1 087 行。
- `min_freq=4`：`results.json` 中 `test_f1=0.8694690265486725`；
  `auto_lexicon.txt` 共 786 行。
- `Boundary Smoothing`：`results.json` 中 `test_f1=0.8647850950009224`；
  `args.use_expert_dict=false`、`args.fl_gamma=0.0`，复评 Micro P/R/F1 为
  0.8736/0.8561/0.8648。
- `SoftLexicon-TrainLex`：`results.json` 中
  `test_precision=0.8442389758179232`、`test_recall=0.8670562454346238`、
  `test_f1=0.8554954954954955`；匹配词表为
  `datasets/raw/RedJujube/softlexicon_train.txt`。
- `SoftLexicon-External`：`results.json` 中
  `test_precision=0.8432218626393384`、`test_recall=0.8564645726807889`、
  `test_f1=0.8497916289182824`；匹配词表来自 `assets/vectors/ctb.50d.vec`。
- `AdaSeq BERT-CRF`：BIO 转换后，源结果
  `/home/shiwenlong/NERlabs/AdaSeq/experiments/redjujube_bert_crf/260421215039.098706/metrics.json`
  的 test P/R/F1=0.84422/0.85902/0.85156；本仓库保存 summary JSON。
- `BERT-MRC+DSC`：旧 `ValueError` 与随机矩阵设备问题已修复，`dice_ohem=0`
  + `train_batch_size=4` 的 20-step GPU smoke 已通过。完整 10 epoch 长实验已在
  tmux `rjnd-mrc-dsc-20260524` 中完成，输出目录为
  `/home/shiwenlong/NERlabs/dice_loss_for_NLP/_9LOG/mrc_ner/redjujube_current_dice_noohem_bs4_seed42_20260524`。
  `eval_result_log.txt` 中 test P/R/F1=0.830589/0.777723/0.803287，
  best dev F1=0.801879。

## 优先基线查询与阻塞

- `Boundary Smoothing` 旧候选位于
  `experiments/EXP-010-optimization/results_newdata/BS_nodict/`、
  `experiments/EXP-010-optimization/results_newdata/BS_focal_nodict/` 和
  `experiments/EXP-010-optimization/results_newdata/H_bs_baseline/`，但这些结果的
  `args.data_dir` 为 `_2DATA/RedJujube`。当前工作树中该目录只有 `.orig`/`.bak`
  备份，且与 `datasets/raw/RedJujube` 的行数和 sha256 不一致，因此不直接登记为
  当前路径证据。
- `SoftLexicon` 旧结果
  `experiments/EXP-010-optimization/results_newdata/SoftLexicon_baseline/seed_42/softlexicon_20260421-212809/results.json`
  只能作历史筛查；当前路径证据已由上表两个 2026-05-24 补跑结果替代。
- `BERT-MRC` / `BERT-MRC+DSC`：`_9LOGS/dice_loss_redjujube_train.log`
  中的旧 `ValueError` 已由外部工程补丁解除；当前路径 `BERT-MRC+DSC`
  已得到 test P/R/F1=83.06/77.77/80.33，纯 `BERT-MRC` 尚未单独补跑。
  `dice_ohem=0.3` 仍受 PyTorch `nonzero` INT_MAX 与显存限制影响，因此登记的是
  关闭 OHEM 的 Dice/DSC 结果。
- `RA_NER / AdaSeq BERT-CRF`：旧失败来自直接使用 BMES 标签；当前 BIO 转换后
  已完成 seed=42 test 评估。

## LatticeLSTM/NFLAT 前置条件

- `LatticeLSTM` 暂不直接启动训练：`projects/LatticeLSTM-master/main.py` 为
  Python 2/PyTorch 0.3 风格，`seed_num=100` 写死；服务器 `/usr/bin/python2`
  无 `torch`，当前未发现可直接使用的 Python 2.7 + torch 环境。需要先恢复旧环境，
  或将代码迁移到 Python 3/PyTorch 当前接口，并将 seed 改为 42。
- `NFLAT` 暂不直接启动全量训练：`projects/NFLAT4CNER-main/main.py` 旧版仅支持
  `weibo/resume/ontonotes/msra`，seed 写死为 2022；需要先补 RedJujube 数据配置、
  seed 参数和 Python 3.7 依赖，再在 GPU 可用时训练。
  2026-05-24 已在 `references/external_projects/NFLAT4CNER-main` 中补充
  `redjujube`、`seed`、`n_epochs`、`refresh_data`、`smoke_samples`、CPU 设备解析
  与当前向量路径；`flat37` 已补 `prettytable` 并可 import FastNLP。16 条样本
  CPU smoke 已完成，退出码 0，覆盖数据读取、词典装备、模型构建、训练和验证/测试
  回调。该 smoke 不作为论文指标；MRC/DSC 已完成，但完整训练仍需等待当前 GPU 上
  其他 Python 进程利用率下降。

## 旧结果参考

已有旧路径结果可作为交叉检查，但不直接替代本次当前路径口径：

- `experiments/EXP-010-optimization/results/Q_bs_focal_minfreq1_seed42/.../results.json`
- `experiments/EXP-010-optimization/results/Q_bs_focal_minfreq3_seed42/.../results.json`
