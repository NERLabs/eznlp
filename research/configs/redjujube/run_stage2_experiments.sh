#!/bin/bash
# RedJujube NER 阶段1-3优化实验（M-T组）
set -e

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

DATA_DIR="datasets/raw/RedJujube"
BERT_ARCH="hfl/chinese-macbert-base"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results"
COMMON_ARGS="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 --sb_epsilon 0.1 --sb_size 2"

echo "============================================================"
echo "RedJujube NER 阶段1-3优化实验（M-T组）"
echo "============================================================"

# 实验 M: BS + sb_size=3
echo ""
echo "============================================================"
echo "实验 M: BS 基线 + sb_size=3"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[M] 运行 seed=$SEED ..."
    python research/training/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --sb_size 3 \
        --save_dir ${BASE_SAVE_DIR}/M_bs_sbsize3 \
        --seed $SEED \
        --no_fgm --no_ema
    echo "[M] seed=$SEED 完成"
done
echo "[M] 实验 M 全部完成"

# 实验 N: BS + min_freq=1
echo ""
echo "============================================================"
echo "实验 N: BS 基线 + min_freq=1"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[N] 运行 seed=$SEED ..."
    python research/training/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --min_freq 1 \
        --save_dir ${BASE_SAVE_DIR}/N_bs_minfreq1 \
        --seed $SEED \
        --no_fgm --no_ema
    echo "[N] seed=$SEED 完成"
done
echo "[N] 实验 N 全部完成"

# 实验 O: BS + sb_size=3 + min_freq=1
echo ""
echo "============================================================"
echo "实验 O: BS + sb_size=3 + min_freq=1"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[O] 运行 seed=$SEED ..."
    python research/training/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --sb_size 3 \
        --min_freq 1 \
        --save_dir ${BASE_SAVE_DIR}/O_bs_sbsize3_minfreq1 \
        --seed $SEED \
        --no_fgm --no_ema
    echo "[O] seed=$SEED 完成"
done
echo "[O] 实验 O 全部完成"

# 实验 P: BS + BMES Aux Loss
echo ""
echo "============================================================"
echo "实验 P: BS + BMES Aux Loss"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[P] 运行 seed=$SEED ..."
    python research/training/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/P_bs_bmes_aux \
        --seed $SEED \
        --no_fgm --no_ema \
        --use_channel_attention \
        --bmes_aux_lambda 0.1 \
        --bmes_label_aux_lambda 0.1
    echo "[P] seed=$SEED 完成"
done
echo "[P] 实验 P 全部完成"

# 实验 Q: BS + Focal Loss (gamma=2.0)
echo ""
echo "============================================================"
echo "实验 Q: BS + Focal Loss (gamma=2.0)"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[Q] 运行 seed=$SEED ..."
    python research/training/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/Q_bs_focal \
        --seed $SEED \
        --no_fgm --no_ema \
        --fl_gamma 2.0
    echo "[Q] seed=$SEED 完成"
done
echo "[Q] 实验 Q 全部完成"

# 实验 R: BS + BMES Aux Loss + Focal Loss
echo ""
echo "============================================================"
echo "实验 R: BS + BMES Aux Loss + Focal Loss"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[R] 运行 seed=$SEED ..."
    python research/training/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/R_bs_bmes_focal \
        --seed $SEED \
        --no_fgm --no_ema \
        --use_channel_attention \
        --bmes_aux_lambda 0.1 \
        --bmes_label_aux_lambda 0.1 \
        --fl_gamma 2.0
    echo "[R] seed=$SEED 完成"
done
echo "[R] 实验 R 全部完成"

# 实验 S: BS + SRG
echo ""
echo "============================================================"
echo "实验 S: BS + SRG"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[S] 运行 seed=$SEED ..."
    python research/training/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/S_bs_srg \
        --seed $SEED \
        --no_fgm --no_ema \
        --use_srg --srg_hid_dim 128 --srg_dropout 0.2
    echo "[S] seed=$SEED 完成"
done
echo "[S] 实验 S 全部完成"

# 实验 T: BS + BMES Aux Loss + Focal Loss + SRG
echo ""
echo "============================================================"
echo "实验 T: BS + BMES Aux Loss + Focal Loss + SRG"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[T] 运行 seed=$SEED ..."
    python research/training/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/T_bs_full \
        --seed $SEED \
        --no_fgm --no_ema \
        --use_channel_attention \
        --bmes_aux_lambda 0.1 \
        --bmes_label_aux_lambda 0.1 \
        --fl_gamma 2.0 \
        --use_srg --srg_hid_dim 128 --srg_dropout 0.2
    echo "[T] seed=$SEED 完成"
done
echo "[T] 实验 T 全部完成"

echo ""
echo "============================================================"
echo "阶段1-3所有实验完成！"
echo "============================================================"
