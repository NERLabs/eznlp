#!/bin/bash
# =============================================================================
# M-T 阶段剩余实验 + 类型自适应 sb_size 新实验
# 包含: P/Q/R/S/T (重跑) + U/V (新增)
# =============================================================================
set -e

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 公共参数
DATA_DIR="_2DATA/RedJujube"
BERT_ARCH="hfl/chinese-macbert-base"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results"
COMMON_ARGS="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 --sb_epsilon 0.1 --sb_size 2"

echo "============================================================"
echo "开始运行 M-T 阶段剩余实验 + 类型自适应 sb_size 实验"
echo "实验组: P/Q/R/S/T (重跑) + U/V (新增)"
echo "每组运行 3 个种子: 42, 43, 44"
echo "============================================================"

# =============================================================================
# P 组: BS + BMES Aux Loss
# =============================================================================
echo ""
echo "============================================================"
echo "实验 P: BS + BMES Aux Loss"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[P] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/P_bs_bmes_aux \
        --no_fgm --no_ema \
        --use_channel_attention \
        --bmes_aux_lambda 0.1 \
        --bmes_label_aux_lambda 0.1 \
        --seed $SEED
    echo "[P] seed=$SEED 完成"
done
echo "[P] 实验 P 全部完成"

# =============================================================================
# Q 组: BS + Focal Loss
# =============================================================================
echo ""
echo "============================================================"
echo "实验 Q: BS + Focal Loss"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[Q] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/Q_bs_focal \
        --no_fgm --no_ema \
        --fl_gamma 2.0 \
        --seed $SEED
    echo "[Q] seed=$SEED 完成"
done
echo "[Q] 实验 Q 全部完成"

# =============================================================================
# R 组: BS + BMES Aux + Focal
# =============================================================================
echo ""
echo "============================================================"
echo "实验 R: BS + BMES Aux + Focal Loss"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[R] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/R_bs_bmes_focal \
        --no_fgm --no_ema \
        --use_channel_attention \
        --bmes_aux_lambda 0.1 \
        --bmes_label_aux_lambda 0.1 \
        --fl_gamma 2.0 \
        --seed $SEED
    echo "[R] seed=$SEED 完成"
done
echo "[R] 实验 R 全部完成"

# =============================================================================
# S 组: BS + SRG
# =============================================================================
echo ""
echo "============================================================"
echo "实验 S: BS + SRG (Span Refinement Gate)"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[S] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/S_bs_srg \
        --no_fgm --no_ema \
        --use_srg \
        --srg_hid_dim 128 \
        --srg_dropout 0.2 \
        --seed $SEED
    echo "[S] seed=$SEED 完成"
done
echo "[S] 实验 S 全部完成"

# =============================================================================
# T 组: BS + BMES + Focal + SRG 全组合
# =============================================================================
echo ""
echo "============================================================"
echo "实验 T: BS + BMES Aux + Focal + SRG (全组合)"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[T] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/T_bs_full \
        --no_fgm --no_ema \
        --use_channel_attention \
        --bmes_aux_lambda 0.1 \
        --bmes_label_aux_lambda 0.1 \
        --fl_gamma 2.0 \
        --use_srg \
        --srg_hid_dim 128 \
        --srg_dropout 0.2 \
        --seed $SEED
    echo "[T] seed=$SEED 完成"
done
echo "[T] 实验 T 全部完成"

# =============================================================================
# U 组: BS + 类型自适应 sb_size (新增)
# 长实体类型(AvgLen>=3.5): DRU/LOC/FER/TAX/EQU → sb_size=3
# 短实体类型(AvgLen<=2.5): PER/PRO/PAR/WED → sb_size=1
# 其他类型: DIS/CUL/AGR/PES/NUT → sb_size=2 (默认)
# =============================================================================
echo ""
echo "============================================================"
echo "实验 U: BS + 类型自适应 sb_size (新增)"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[U] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/U_bs_adaptive_sbsize \
        --no_fgm --no_ema \
        --sb_size_map "DRU:3,LOC:3,FER:3,TAX:3,EQU:3,PER:1,PRO:1,PAR:1,WED:1" \
        --seed $SEED
    echo "[U] seed=$SEED 完成"
done
echo "[U] 实验 U 全部完成"

# =============================================================================
# V 组: BS + 类型自适应 sb_size + BMES Aux + Focal (新增)
# 最佳组合: 自适应边界大小 + BMES辅助损失 + Focal Loss
# =============================================================================
echo ""
echo "============================================================"
echo "实验 V: BS + 类型自适应 sb_size + BMES Aux + Focal (最佳组合)"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[V] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/V_bs_adaptive_bmes_focal \
        --no_fgm --no_ema \
        --sb_size_map "DRU:3,LOC:3,FER:3,TAX:3,EQU:3,PER:1,PRO:1,PAR:1,WED:1" \
        --use_channel_attention \
        --bmes_aux_lambda 0.1 \
        --bmes_label_aux_lambda 0.1 \
        --fl_gamma 2.0 \
        --seed $SEED
    echo "[V] seed=$SEED 完成"
done
echo "[V] 实验 V 全部完成"

# =============================================================================
# 完成提示
# =============================================================================
echo ""
echo "============================================================"
echo "所有实验已完成!"
echo "============================================================"
echo "实验结果保存在: ${BASE_SAVE_DIR}/"
echo "  - P_bs_bmes_aux/          : BS + BMES Aux Loss"
echo "  - Q_bs_focal/             : BS + Focal Loss"
echo "  - R_bs_bmes_focal/        : BS + BMES Aux + Focal"
echo "  - S_bs_srg/               : BS + SRG"
echo "  - T_bs_full/              : BS + BMES + Focal + SRG (全组合)"
echo "  - U_bs_adaptive_sbsize/   : BS + 类型自适应 sb_size"
echo "  - V_bs_adaptive_bmes_focal/: BS + 自适应 + BMES + Focal (最佳组合)"
echo ""
echo "共 7 组实验 × 3 种子 = 21 次训练"
echo "============================================================"
