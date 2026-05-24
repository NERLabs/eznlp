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
5. 对 `LatticeLSTM` 和 `NFLAT` 暂不直接启动训练；先记录阻塞项：
   旧代码环境、RedJujube 数据适配、seed 参数、依赖和 GPU 可用性。

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

2026-05-24 已完成 `EDBP min_freq=1`、`EDBP min_freq=3` 与 `EDBP min_freq=4` 三组当前路径
RJND/RedJujube seed=42 实验。`results.json` 只保存 loss/F1，Precision 和
Recall 由 `research/evaluation/test_redjujube_baseline.py` 加载最佳模型复评得到。

| 模型 | seed | 词典规模 | 测试 P/% | 测试 R/% | 测试 F1/% | best dev F1/% | best epoch | 结果目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EDBP min_freq=1 | 42 | 5 317 | 86.94 | 81.92 | 84.36 | 83.47 | 19 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current/expert_boundary_20260524-124317/` |
| EDBP min_freq=3 | 42 | 1 087 | 88.56 | 86.49 | 87.51 | 86.30 | 22 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/expert_boundary_20260524-125722/` |
| EDBP min_freq=4 | 42 | 786 | 87.79 | 86.12 | 86.95 | 85.90 | 22 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq4_seed42_current/expert_boundary_20260524-132104/` |

校验记录：

- `min_freq=1`：`results.json` 中 `test_f1=0.8435502068446783`；
  `auto_lexicon.txt` 共 5 317 行。
- `min_freq=3`：`results.json` 中 `test_f1=0.8750923872875093`；
  `auto_lexicon.txt` 共 1 087 行。
- `min_freq=4`：`results.json` 中 `test_f1=0.8694690265486725`；
  `auto_lexicon.txt` 共 786 行。
- `tmux ls` 返回 `no server running on /tmp/tmux-1010/default`，当前没有仍在运行的
  补实验会话。

## LatticeLSTM/NFLAT 前置条件

- `LatticeLSTM` 暂不直接启动训练：`projects/LatticeLSTM-master/main.py` 为
  Python 2/PyTorch 0.3 风格，`seed_num=100` 写死；服务器 `/usr/bin/python2`
  无 `torch`，当前未发现可直接使用的 Python 2.7 + torch 环境。需要先恢复旧环境，
  或将代码迁移到 Python 3/PyTorch 当前接口，并将 seed 改为 42。
- `NFLAT` 暂不直接启动训练：`projects/NFLAT4CNER-main/main.py` 仅支持
  `weibo/resume/ontonotes/msra`，seed 写死为 2022；需要先补 RedJujube 数据配置、
  seed 参数和 Python 3.7 依赖，再在 GPU 可用时训练。

## 旧结果参考

已有旧路径结果可作为交叉检查，但不直接替代本次当前路径口径：

- `experiments/EXP-010-optimization/results/Q_bs_focal_minfreq1_seed42/.../results.json`
- `experiments/EXP-010-optimization/results/Q_bs_focal_minfreq3_seed42/.../results.json`
