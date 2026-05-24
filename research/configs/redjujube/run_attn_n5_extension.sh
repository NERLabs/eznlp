#!/bin/bash
# 扩展到 n=5/6 种子，做配对 t 检验
# 顺序：基线 s45/46/47, v1 s45/46/47
set -e

cd /home/shiwenlong/NERlabs/eznlp
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eznlp11

DATA_DIR="datasets/raw/RedJujube"
BERT_ARCH="hfl/chinese-macbert-base"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results_newdata"
COMMON="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 --sb_epsilon 0.1 --sb_size 2 --no_fgm --no_ema --fl_gamma 2.0 --eval_test_each_epoch"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] ===== n=6 扩展实验启动（6 个 run，约 90 min）====="

# ===== 基线 三种子 =====
for SEED in 45 46 47; do
    echo "[$(ts)] >>> 基线 seed=$SEED"
    python -u research/training/train_redjujube_expert_boundary.py $COMMON \
        --save_dir ${BASE_SAVE_DIR}/Q_bs_focal_s${SEED} \
        --seed ${SEED}
    echo "[$(ts)] <<< 基线 s${SEED} 完成"
done

# ===== v1 注意力 三种子 =====
for SEED in 45 46 47; do
    echo "[$(ts)] >>> v1 seed=$SEED"
    python -u research/training/train_redjujube_expert_boundary.py $COMMON \
        --save_dir ${BASE_SAVE_DIR}/Q_bs_focal_attnv1_s${SEED} \
        --seed ${SEED} \
        --use_channel_attention --channel_attn_version v1 \
        --channel_attn_heads 4 --channel_attn_dropout 0.1
    echo "[$(ts)] <<< v1 s${SEED} 完成"
done

echo "[$(ts)] ===== 全部完成 ====="
