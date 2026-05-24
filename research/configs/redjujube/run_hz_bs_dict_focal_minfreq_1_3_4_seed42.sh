#!/bin/bash
# ==============================================================================
# 红枣 HZ：BS+Dict+Focal（Boundary Selection + ExpertDict + Focal Loss）
# 补齐 min_freq=1/3/4 的严格对齐实验（只跑 seed=42）
# 用于解释不同阈值“为什么会涨”以及支持通用抽取策略选择
# ==============================================================================

set -e

PROJECT_ROOT="/home/shiwenlong/NERlabs/eznlp"
cd "$PROJECT_ROOT"

DATA_DIR="$PROJECT_ROOT/datasets/raw/RedJujube"
BERT_ARCH="hfl/chinese-macbert-base"

COMMON_ARGS="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 --sb_epsilon 0.1 --sb_size 2"

SEED=42
FL_GAMMA=2.0

run_one() {
  local min_freq=$1
  local save_base="experiments/EXP-010-optimization/results/Q_bs_focal_minfreq${min_freq}_seed${SEED}"

  echo ""
  echo "======================================================================"
  echo "[HZ] BS+Dict+Focal seed=${SEED} min_freq=${min_freq}"
  echo "save_base=${save_base}"
  echo "======================================================================"

  conda run -n eznlp11 python "research/training/train_redjujube_expert_boundary.py" \
    $COMMON_ARGS \
    --save_dir "$save_base" \
    --seed "$SEED" \
    --no_fgm --no_ema \
    --fl_gamma "$FL_GAMMA" \
    --min_freq "$min_freq"

  echo "✅ [HZ] done: min_freq=${min_freq}"
}

run_one 1
run_one 3
run_one 4

echo ""
echo "############################################################################"
echo "[HZ] all补齐完成"
echo "############################################################################"

