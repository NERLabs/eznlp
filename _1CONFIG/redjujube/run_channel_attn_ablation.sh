#!/bin/bash
# 通道注意力消融串行执行脚本
# 共 4 次：v2+s43, v2+s44, v1+s42, v2+bmes_aux+s42
set -e

cd /home/shiwenlong/NERlabs/eznlp
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eznlp11

DATA_DIR="_2DATA/RedJujube"
BERT_ARCH="hfl/chinese-macbert-base"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results_newdata"
COMMON_ARGS="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 --sb_epsilon 0.1 --sb_size 2 --no_fgm --no_ema --fl_gamma 2.0"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] ===== 通道注意力消融实验启动 ====="

# A: v2 + seed=43
echo "[$(ts)] >>> A: attnv2 seed=43"
python -u _5TRAIN/train_redjujube_expert_boundary.py \
    $COMMON_ARGS \
    --save_dir ${BASE_SAVE_DIR}/Q_bs_focal_attnv2_s43 \
    --seed 43 \
    --use_channel_attention --channel_attn_version v2 \
    --channel_attn_heads 4 --channel_attn_dropout 0.1
echo "[$(ts)] <<< A 完成"

# B: v2 + seed=44
echo "[$(ts)] >>> B: attnv2 seed=44"
python -u _5TRAIN/train_redjujube_expert_boundary.py \
    $COMMON_ARGS \
    --save_dir ${BASE_SAVE_DIR}/Q_bs_focal_attnv2_s44 \
    --seed 44 \
    --use_channel_attention --channel_attn_version v2 \
    --channel_attn_heads 4 --channel_attn_dropout 0.1
echo "[$(ts)] <<< B 完成"

# C: v1 + seed=42
echo "[$(ts)] >>> C: attnv1 seed=42"
python -u _5TRAIN/train_redjujube_expert_boundary.py \
    $COMMON_ARGS \
    --save_dir ${BASE_SAVE_DIR}/Q_bs_focal_attnv1_s42 \
    --seed 42 \
    --use_channel_attention --channel_attn_version v1 \
    --channel_attn_heads 4 --channel_attn_dropout 0.1
echo "[$(ts)] <<< C 完成"

# D: v2 + bmes aux loss + seed=42
echo "[$(ts)] >>> D: attnv2 + bmes_aux seed=42"
python -u _5TRAIN/train_redjujube_expert_boundary.py \
    $COMMON_ARGS \
    --save_dir ${BASE_SAVE_DIR}/Q_bs_focal_attnv2_bmesaux_s42 \
    --seed 42 \
    --use_channel_attention --channel_attn_version v2 \
    --channel_attn_heads 4 --channel_attn_dropout 0.1 \
    --bmes_aux_lambda 0.1 --bmes_label_aux_lambda 0.1
echo "[$(ts)] <<< D 完成"

echo "[$(ts)] ===== 全部完成 ====="
