#!/bin/bash
# RedJujube NER BS（Boundary Selection）补充优化实验
# 在 CRF 优化实验 C 组之后运行
set -e

DATA_DIR="_2DATA/RedJujube"
BERT_ARCH="hfl/chinese-macbert-base"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results"

COMMON_ARGS="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 --sb_epsilon 0.1 --sb_size 2"

echo "============================================================"
echo "RedJujube NER BS 补充优化实验"
echo "============================================================"
echo "数据目录: $DATA_DIR"
echo "BERT模型: $BERT_ARCH"
echo "BS参数: sb_epsilon=0.1, sb_size=2"
echo "保存目录: $BASE_SAVE_DIR"
echo "种子列表: 42 43 44"
echo "============================================================"

mkdir -p ${BASE_SAVE_DIR}

# ============================================================
# 实验 H: BS 基线（无 FGM/EMA）
# ============================================================
echo ""
echo "============================================================"
echo "实验 H: BS 基线（禁用 FGM/EMA）"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[H] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/H_bs_baseline \
        --seed $SEED \
        --no_fgm --no_ema
    echo "[H] seed=$SEED 完成"
done
echo "[H] 实验 H 全部完成"

# ============================================================
# 实验 I: BS + FGM
# ============================================================
echo ""
echo "============================================================"
echo "实验 I: BS + FGM"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[I] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/I_bs_fgm \
        --seed $SEED \
        --use_fgm --fgm_epsilon 1.0 \
        --no_ema
    echo "[I] seed=$SEED 完成"
done
echo "[I] 实验 I 全部完成"

# ============================================================
# 实验 J: BS + FGM + EMA
# ============================================================
echo ""
echo "============================================================"
echo "实验 J: BS + FGM + EMA"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[J] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/J_bs_fgm_ema \
        --seed $SEED \
        --use_fgm --fgm_epsilon 1.0 \
        --use_ema --ema_decay 0.995
    echo "[J] seed=$SEED 完成"
done
echo "[J] 实验 J 全部完成"

# ============================================================
# 实验 K: BS + FGM + EMA + 数据增强
# ============================================================
echo ""
echo "============================================================"
echo "实验 K: BS + FGM + EMA + 数据增强"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[K] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/K_bs_fgm_ema_aug \
        --seed $SEED \
        --use_fgm --fgm_epsilon 1.0 \
        --use_ema --ema_decay 0.995 \
        --use_augment --aug_ratio 2 --aug_prob 0.3
    echo "[K] seed=$SEED 完成"
done
echo "[K] 实验 K 全部完成"

# ============================================================
# 实验 L: BS + FGM + EMA + R-Drop
# ============================================================
echo ""
echo "============================================================"
echo "实验 L: BS + FGM + EMA + R-Drop"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[L] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/L_bs_fgm_ema_rdrop \
        --seed $SEED \
        --use_fgm --fgm_epsilon 1.0 \
        --use_ema --ema_decay 0.995 \
        --use_rdrop --rdrop_alpha 0.5
    echo "[L] seed=$SEED 完成"
done
echo "[L] 实验 L 全部完成"

# ============================================================
# 实验 M: BS + sb_size=3（解决长实体问题）
# ============================================================
echo ""
echo "============================================================"
echo "实验 M: BS 基线 + sb_size=3"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[M] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --sb_size 3 \
        --save_dir ${BASE_SAVE_DIR}/M_bs_sbsize3 \
        --seed $SEED \
        --no_fgm --no_ema
    echo "[M] seed=$SEED 完成"
done
echo "[M] 实验 M 全部完成"

# ============================================================
# 实验 N: BS + min_freq=1（更大词典覆盖）
# ============================================================
echo ""
echo "============================================================"
echo "实验 N: BS 基线 + min_freq=1"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[N] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --min_freq 1 \
        --save_dir ${BASE_SAVE_DIR}/N_bs_minfreq1 \
        --seed $SEED \
        --no_fgm --no_ema
    echo "[N] seed=$SEED 完成"
done
echo "[N] 实验 N 全部完成"

# ============================================================
# 实验 O: BS + sb_size=3 + min_freq=1（组合）
# ============================================================
echo ""
echo "============================================================"
echo "实验 O: BS + sb_size=3 + min_freq=1"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[O] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --sb_size 3 \
        --min_freq 1 \
        --save_dir ${BASE_SAVE_DIR}/O_bs_sbsize3_minfreq1 \
        --seed $SEED \
        --no_fgm --no_ema
    echo "[O] seed=$SEED 完成"
done
echo "[O] 实验 O 全部完成"

# ============================================================
# 阶段2: BMES 辅助损失 + Focal Loss 实验组
# ============================================================

# ============================================================
# 实验 P: BS + BMES Aux Loss（通道注意力 + 辅助损失）
# ============================================================
echo ""
echo "============================================================"
echo "实验 P: BS + BMES Aux Loss"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[P] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
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

# ============================================================
# 实验 Q: BS + Focal Loss (gamma=2.0)
# ============================================================
echo ""
echo "============================================================"
echo "实验 Q: BS + Focal Loss (gamma=2.0)"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[Q] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/Q_bs_focal \
        --seed $SEED \
        --no_fgm --no_ema \
        --fl_gamma 2.0
    echo "[Q] seed=$SEED 完成"
done
echo "[Q] 实验 Q 全部完成"

# ============================================================
# 实验 R: BS + BMES Aux Loss + Focal Loss（组合）
# ============================================================
echo ""
echo "============================================================"
echo "实验 R: BS + BMES Aux Loss + Focal Loss"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[R] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
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

# ============================================================
# 阶段3: SRG (Self-Rectified Gate) 实验组
# ============================================================

# ============================================================
# 实验 S: BS + SRG
# ============================================================
echo ""
echo "============================================================"
echo "实验 S: BS + SRG"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[S] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/S_bs_srg \
        --seed $SEED \
        --no_fgm --no_ema \
        --use_srg --srg_hid_dim 128 --srg_dropout 0.2
    echo "[S] seed=$SEED 完成"
done
echo "[S] 实验 S 全部完成"

# ============================================================
# 实验 T: BS + BMES Aux Loss + Focal Loss + SRG（全组合）
# ============================================================
echo ""
echo "============================================================"
echo "实验 T: BS + BMES Aux Loss + Focal Loss + SRG"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[T] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
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
echo "所有 BS 补充实验完成！"
echo "============================================================"
echo "使用以下命令收集结果："
echo "python _6EVALUATE/collect_optimization_results.py --detailed"
