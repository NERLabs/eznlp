#!/bin/bash
# v1 通道注意力补全 3 种子（seed=43, 44）
set -e

cd /home/shiwenlong/NERlabs/eznlp
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eznlp11

DATA_DIR="datasets/raw/RedJujube"
BERT_ARCH="hfl/chinese-macbert-base"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results_newdata"
COMMON_ARGS="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 --sb_epsilon 0.1 --sb_size 2 --no_fgm --no_ema --fl_gamma 2.0"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] ===== v1 三种子补全启动 ====="

# E: v1 + seed=43
echo "[$(ts)] >>> E: attnv1 seed=43"
python -u research/training/train_redjujube_expert_boundary.py \
    $COMMON_ARGS \
    --save_dir ${BASE_SAVE_DIR}/Q_bs_focal_attnv1_s43 \
    --seed 43 \
    --use_channel_attention --channel_attn_version v1 \
    --channel_attn_heads 4 --channel_attn_dropout 0.1
echo "[$(ts)] <<< E 完成"

# F: v1 + seed=44
echo "[$(ts)] >>> F: attnv1 seed=44"
python -u research/training/train_redjujube_expert_boundary.py \
    $COMMON_ARGS \
    --save_dir ${BASE_SAVE_DIR}/Q_bs_focal_attnv1_s44 \
    --seed 44 \
    --use_channel_attention --channel_attn_version v1 \
    --channel_attn_heads 4 --channel_attn_dropout 0.1
echo "[$(ts)] <<< F 完成"

echo "[$(ts)] ===== v1 三种子全部完成 ====="
