#!/bin/bash
# =============================================================================
# Z 组补跑实验：在旧版数据上运行全技术组合
# 目的：与 A-Y 组实验保持数据一致性（之前 Z 组意外使用了重新划分的新数据）
# =============================================================================
set -e

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# =============================================================================
# 1. 检查旧数据文件是否存在
# =============================================================================
echo "============================================================"
echo "检查旧版数据文件..."
echo "============================================================"

ORIG_DATA_DIR="datasets/raw/RedJujube"
OLD_DATA_DIR="datasets/raw/RedJujube_old"

TRAIN_ORIG="$ORIG_DATA_DIR/redjujube_train.bmes.orig"
DEV_ORIG="$ORIG_DATA_DIR/redjujube_dev.bmes.orig"
TEST_ORIG="$ORIG_DATA_DIR/redjujube_test.bmes.orig"

if [[ ! -f "$TRAIN_ORIG" ]]; then
    echo "错误: 找不到旧版训练集 $TRAIN_ORIG"
    exit 1
fi

if [[ ! -f "$DEV_ORIG" ]]; then
    echo "错误: 找不到旧版验证集 $DEV_ORIG"
    exit 1
fi

if [[ ! -f "$TEST_ORIG" ]]; then
    echo "错误: 找不到旧版测试集 $TEST_ORIG"
    exit 1
fi

echo "旧版数据文件检查通过:"
echo "  - $TRAIN_ORIG"
echo "  - $DEV_ORIG"
echo "  - $TEST_ORIG"

# =============================================================================
# 2. 创建临时目录并复制旧数据（避免影响当前运行的 Z seed44）
# =============================================================================
echo ""
echo "============================================================"
echo "创建临时数据目录: $OLD_DATA_DIR"
echo "============================================================"

mkdir -p "$OLD_DATA_DIR"

# 复制旧版数据到临时目录
cp "$TRAIN_ORIG" "$OLD_DATA_DIR/redjujube_train.bmes"
cp "$DEV_ORIG" "$OLD_DATA_DIR/redjujube_dev.bmes"
cp "$TEST_ORIG" "$OLD_DATA_DIR/redjujube_test.bmes"

echo "已复制旧版数据文件:"
ls -la "$OLD_DATA_DIR/"*.bmes

# 复制辅助文件（专家词典、软词典等）
echo ""
echo "复制辅助文件..."
cp "$ORIG_DATA_DIR"/expert_lexicon*.txt "$OLD_DATA_DIR/" 2>/dev/null && echo "  - 已复制 expert_lexicon*.txt" || echo "  - 无 expert_lexicon*.txt 文件"
cp "$ORIG_DATA_DIR"/softlexicon*.txt "$OLD_DATA_DIR/" 2>/dev/null && echo "  - 已复制 softlexicon*.txt" || echo "  - 无 softlexicon*.txt 文件"

echo ""
echo "临时数据目录内容:"
ls -la "$OLD_DATA_DIR/"

# =============================================================================
# 3. 运行 Z 组实验（全技术组合 - 旧数据）
# =============================================================================
echo ""
echo "============================================================"
echo "Z 组补跑实验：全技术组合 (Enhanced Size Emb + Focal Loss + LogN-Scaling)"
echo "数据目录: $OLD_DATA_DIR (旧版数据)"
echo "============================================================"

BERT_ARCH="hfl/chinese-macbert-base"
RESULTS_BASE="experiments/EXP-010-optimization/results"
COMMON_ARGS="--data_dir $OLD_DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 --sb_epsilon 0.1 --sb_size 2"

SAVE_DIR="$RESULTS_BASE/Z_old_full_long_entity_opt"

for SEED in 42 43 44; do
    echo ""
    echo "===== Z_old 组 (全技术组合 - 旧数据) seed=$SEED ====="
    conda run -n eznlp11 python research/training/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir $SAVE_DIR \
        --no_fgm --no_ema \
        --enhanced_size_emb \
        --fl_gamma 2.0 \
        --use_lognscaling \
        --seed $SEED
    echo "[Z_old] seed=$SEED 完成"
done

# =============================================================================
# 4. 完成提示
# =============================================================================
echo ""
echo "============================================================"
echo "Z 组补跑实验完成!"
echo "============================================================"
echo "实验结果保存在: $SAVE_DIR/"
echo "使用数据: $OLD_DATA_DIR/ (旧版数据)"
echo ""
echo "实验配置:"
echo "  - Enhanced Size Embedding: 启用"
echo "  - Focal Loss (gamma=2.0): 启用"
echo "  - LogN-Scaling: 启用"
echo "  - FGM/EMA: 禁用 (--no_fgm --no_ema)"
echo ""
echo "共 3 个种子 (42, 43, 44) 完成训练"
echo "============================================================"
