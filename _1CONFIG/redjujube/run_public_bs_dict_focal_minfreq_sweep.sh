#!/bin/bash
# ==============================================================================
# 公开数据集 BS+Dict+Focal 词典阈值扫参实验
# 数据集: Boson / CLUE / MSRA
# 阈值: min_freq = 1, 3
# ==============================================================================

set -e

PROJECT_ROOT="/home/shiwenlong/NERlabs/eznlp"
cd "$PROJECT_ROOT"

RESULTS_DIR="experiments/EXP-010-optimization/results_public"
mkdir -p "$RESULTS_DIR"

SEEDS=(42 43 44)
MIN_FREQS=(1 3)
BERT_ARCH="hfl/chinese-macbert-base"

# Boson
BOSON_TRAIN="$PROJECT_ROOT/_2DATA/boson/boson.train.bmes"
BOSON_DEV="$PROJECT_ROOT/_2DATA/boson/boson.dev.bmes"
BOSON_TEST="$PROJECT_ROOT/_2DATA/boson/boson.test.bmes"

# CLUE
CLUE_TRAIN="$PROJECT_ROOT/_2DATA/clue/train.char.bmes"
CLUE_DEV="$PROJECT_ROOT/_2DATA/clue/dev.char.bmes"
CLUE_TEST="$PROJECT_ROOT/_2DATA/clue/test.char.bmes"

# MSRA
MSRA_TRAIN="$PROJECT_ROOT/_2DATA/MSRA/train.char.bmes"
MSRA_DEV="$PROJECT_ROOT/_2DATA/MSRA/dev.char.bmes"
MSRA_TEST="$PROJECT_ROOT/_2DATA/MSRA/test.char.bmes"

run_full_model_with_min_freq() {
    local dataset_name=$1
    local train_file=$2
    local dev_file=$3
    local test_file=$4
    local seed=$5
    local min_freq=$6

    # 用单独目录区分不同阈值，避免覆盖已有结果
    local save_dir="$RESULTS_DIR/${dataset_name}_bs_dict_focal_mf${min_freq}/seed_${seed}"

    echo ""
    echo "======================================================================"
    echo "运行完整模型 (BS+Dict+Focal): $dataset_name (seed=$seed, min_freq=$min_freq)"
    echo "训练集: $train_file"
    echo "验证集: $dev_file"
    echo "测试集: $test_file"
    echo "保存目录: $save_dir"
    echo "======================================================================"

    conda run -n eznlp11 python "$PROJECT_ROOT/_5TRAIN/train_general_expert_boundary.py" \
        --train_file "$train_file" \
        --dev_file "$dev_file" \
        --test_file "$test_file" \
        --bert_arch "$BERT_ARCH" \
        --save_dir "$save_dir" \
        --seed "$seed" \
        --num_epochs 30 \
        --batch_size 16 \
        --sb_epsilon 0.1 \
        --sb_size 2 \
        --no_fgm \
        --no_ema \
        --fl_gamma 2.0 \
        --min_freq "$min_freq"

    echo "✅ $dataset_name BS+Dict+Focal (seed=$seed, min_freq=$min_freq) 完成"
}

echo ""
echo "############################################################################"
echo "#         公开数据集 BS+Dict+Focal 词典阈值扫参实验开始                     #"
echo "#  数据集: Boson, CLUE, MSRA                                              #"
echo "#  配置: BS+Dict+Focal                                                    #"
echo "#  min_freq: ${MIN_FREQS[*]}                                              #"
echo "#  种子: ${SEEDS[*]}                                                       #"
echo "############################################################################"
echo ""

for min_freq in "${MIN_FREQS[@]}"; do
    echo ""
    echo "#################### min_freq=$min_freq ####################"

    # Boson
    for seed in "${SEEDS[@]}"; do
        run_full_model_with_min_freq "boson" "$BOSON_TRAIN" "$BOSON_DEV" "$BOSON_TEST" "$seed" "$min_freq"
    done

    # CLUE
    for seed in "${SEEDS[@]}"; do
        run_full_model_with_min_freq "clue" "$CLUE_TRAIN" "$CLUE_DEV" "$CLUE_TEST" "$seed" "$min_freq"
    done

    # MSRA
    for seed in "${SEEDS[@]}"; do
        run_full_model_with_min_freq "msra" "$MSRA_TRAIN" "$MSRA_DEV" "$MSRA_TEST" "$seed" "$min_freq"
    done
done

echo ""
echo "############################################################################"
echo "#                    阈值扫参实验完成!                                      #"
echo "#  结果保存在: $RESULTS_DIR                                                 #"
echo "############################################################################"
