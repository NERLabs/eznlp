#!/bin/bash
# =============================================================================
# W-Z 组实验：长实体优化技术测试
# 基于论文启发的技术：Enhanced Size Embedding, Focal Loss, Span Width Limit, LogN-Scaling
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
echo "开始运行 W-Z 组实验：长实体优化技术测试"
echo "实验组: W/X/Y/Z"
echo "每组运行 3 个种子: 42, 43, 44"
echo "============================================================"

# =============================================================================
# W 组: Enhanced Size Embedding
# 使用参数化的 Size Embedding，增强对长实体的边界感知能力
# =============================================================================
echo ""
echo "============================================================"
echo "实验 W: Enhanced Size Embedding"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[W] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/W_enhanced_size_emb \
        --no_fgm --no_ema \
        --enhanced_size_emb \
        --seed $SEED
    echo "[W] seed=$SEED 完成"
done
echo "[W] 实验 W 全部完成"

# =============================================================================
# X 组: Enhanced Size Emb + Focal Loss
# 组合 W 组技术与当前最优的 Q 组（Focal Loss）
# =============================================================================
echo ""
echo "============================================================"
echo "实验 X: Enhanced Size Emb + Focal Loss"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[X] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/X_enhanced_size_focal \
        --no_fgm --no_ema \
        --enhanced_size_emb \
        --fl_gamma 2.0 \
        --seed $SEED
    echo "[X] seed=$SEED 完成"
done
echo "[X] 实验 X 全部完成"

# =============================================================================
# Y 组: Enhanced Size Emb + Focal Loss + Span Width Limit
# 添加最大跨度宽度限制，控制解码时的最大实体长度
# =============================================================================
echo ""
echo "============================================================"
echo "实验 Y: Enhanced Size Emb + Focal Loss + Span Width Limit"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[Y] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/Y_size_focal_spanwidth \
        --no_fgm --no_ema \
        --enhanced_size_emb \
        --fl_gamma 2.0 \
        --max_span_width 50 \
        --seed $SEED
    echo "[Y] seed=$SEED 完成"
done
echo "[Y] 实验 Y 全部完成"

# =============================================================================
# Z 组: 全技术组合 (Enhanced Size Emb + Focal Loss + LogN-Scaling)
# 整合所有长实体优化技术
# =============================================================================
echo ""
echo "============================================================"
echo "实验 Z: 全技术组合 (Enhanced Size Emb + Focal Loss + LogN-Scaling)"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[Z] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/Z_full_long_entity_opt \
        --no_fgm --no_ema \
        --enhanced_size_emb \
        --fl_gamma 2.0 \
        --use_lognscaling \
        --seed $SEED
    echo "[Z] seed=$SEED 完成"
done
echo "[Z] 实验 Z 全部完成"

# =============================================================================
# 完成提示
# =============================================================================
echo ""
echo "============================================================"
echo "W-Z 组实验全部完成!"
echo "============================================================"
echo "实验结果保存在: ${BASE_SAVE_DIR}/"
echo "  - W_enhanced_size_emb/     : Enhanced Size Embedding"
echo "  - X_enhanced_size_focal/   : Enhanced Size Emb + Focal Loss"
echo "  - Y_size_focal_spanwidth/  : Enhanced Size Emb + Focal + Span Width Limit"
echo "  - Z_full_long_entity_opt/  : 全技术组合 (Size Emb + Focal + LogN-Scaling)"
echo ""
echo "共 4 组实验 × 3 种子 = 12 次训练"
echo "============================================================"
