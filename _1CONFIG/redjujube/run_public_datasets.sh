#!/bin/bash
# ==============================================================================
# 公开数据集实验脚本（全量重跑）
# 在 Boson、CLUE、MSRA 三个数据集上运行完整模型和纯基线
# ==============================================================================

set -e  # 遇到错误立即退出

# 项目根目录
PROJECT_ROOT="/home/shiwenlong/NERlabs/eznlp"
cd "$PROJECT_ROOT"

# 结果保存目录
RESULTS_DIR="experiments/EXP-010-optimization/results_public"
mkdir -p "$RESULTS_DIR"

# 随机种子
SEEDS=(42 43 44)

# BERT 模型
BERT_ARCH="hfl/chinese-macbert-base"

# 创建临时目录用于 pure 基线（软链接数据文件）
TEMP_DIR=$(mktemp -d -p "$PROJECT_ROOT" tmp.public_datasets.XXXXXX)
echo "临时目录: $TEMP_DIR"

# 清理函数
cleanup() {
    echo ""
    echo "======================================================================"
    echo "清理临时目录..."
    rm -rf "$TEMP_DIR"
    echo "清理完成"
    echo "======================================================================"
}

# 脚本退出时自动清理
trap cleanup EXIT

# ==============================================================================
# 数据集路径定义
# ==============================================================================

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

# ==============================================================================
# 辅助函数：创建临时软链接目录
# ==============================================================================
setup_temp_links() {
    local dataset_name=$1
    local train_file=$2
    local dev_file=$3
    local test_file=$4
    
    local link_dir="$TEMP_DIR/$dataset_name"
    mkdir -p "$link_dir"
    
    # 创建软链接，命名为 redjujube_*.bmes
    ln -sf "$train_file" "$link_dir/redjujube_train.bmes"
    ln -sf "$dev_file" "$link_dir/redjujube_dev.bmes"
    ln -sf "$test_file" "$link_dir/redjujube_test.bmes"
    
    echo "$link_dir"
}

# ==============================================================================
# 运行纯基线 (BERT+BiLSTM+CRF)
# ==============================================================================
run_pure_baseline() {
    local dataset_name=$1
    local data_dir=$2
    local seed=$3
    
    local save_dir="$RESULTS_DIR/${dataset_name}_pure_baseline/seed_${seed}"
    
    echo ""
    echo "======================================================================"
    echo "运行纯基线 (BERT+BiLSTM+CRF): $dataset_name (seed=$seed)"
    echo "数据目录: $data_dir"
    echo "保存目录: $save_dir"
    echo "======================================================================"
    
    conda run -n eznlp11 python "$PROJECT_ROOT/_5TRAIN/train_redjujube_ner.py" \
        --data_dir "$data_dir" \
        --bert_arch "$BERT_ARCH" \
        --save_dir "$save_dir" \
        --seed "$seed" \
        --num_epochs 30 \
        --batch_size 16 \
        --no_fgm \
        --no_ema \
        --bmes_aux_lambda 0 \
        --bmes_label_aux_lambda 0
    
    echo "✅ $dataset_name 纯基线 (seed=$seed) 完成"
}

# ==============================================================================
# 运行完整模型 (BS+Dict+Focal)
# ==============================================================================
run_full_model() {
    local dataset_name=$1
    local train_file=$2
    local dev_file=$3
    local test_file=$4
    local seed=$5
    
    local save_dir="$RESULTS_DIR/${dataset_name}_bs_dict_focal/seed_${seed}"
    
    echo ""
    echo "======================================================================"
    echo "运行完整模型 (BS+Dict+Focal): $dataset_name (seed=$seed)"
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
        --fl_gamma 2.0
    
    echo "✅ $dataset_name BS+Dict+Focal (seed=$seed) 完成"
}

# ==============================================================================
# 主流程
# ==============================================================================

echo ""
echo "############################################################################"
echo "#                    公开数据集实验开始                                     #"
echo "#  数据集: Boson, CLUE, MSRA                                              #"
echo "#  配置: Pure BERT+BiLSTM+CRF (基线) vs BS+Dict+Focal (完整模型)            #"
echo "#  种子: ${SEEDS[*]}                                                       #"
echo "############################################################################"
echo ""

# ==============================================================================
# Boson 数据集
# ==============================================================================
echo ""
echo "#################### Boson 数据集 ####################"

# 创建 Boson 临时链接目录
BOSON_LINK_DIR=$(setup_temp_links "boson" "$BOSON_TRAIN" "$BOSON_DEV" "$BOSON_TEST")
echo "Boson 临时目录: $BOSON_LINK_DIR"

# Boson 纯基线
for seed in "${SEEDS[@]}"; do
    run_pure_baseline "boson" "$BOSON_LINK_DIR" "$seed"
done

# Boson 完整模型
for seed in "${SEEDS[@]}"; do
    run_full_model "boson" "$BOSON_TRAIN" "$BOSON_DEV" "$BOSON_TEST" "$seed"
done

# ==============================================================================
# CLUE 数据集
# ==============================================================================
echo ""
echo "#################### CLUE 数据集 ####################"

# 创建 CLUE 临时链接目录
CLUE_LINK_DIR=$(setup_temp_links "clue" "$CLUE_TRAIN" "$CLUE_DEV" "$CLUE_TEST")
echo "CLUE 临时目录: $CLUE_LINK_DIR"

# CLUE 纯基线
for seed in "${SEEDS[@]}"; do
    run_pure_baseline "clue" "$CLUE_LINK_DIR" "$seed"
done

# CLUE 完整模型
for seed in "${SEEDS[@]}"; do
    run_full_model "clue" "$CLUE_TRAIN" "$CLUE_DEV" "$CLUE_TEST" "$seed"
done

# ==============================================================================
# MSRA 数据集
# ==============================================================================
echo ""
echo "#################### MSRA 数据集 ####################"

# 创建 MSRA 临时链接目录
MSRA_LINK_DIR=$(setup_temp_links "msra" "$MSRA_TRAIN" "$MSRA_DEV" "$MSRA_TEST")
echo "MSRA 临时目录: $MSRA_LINK_DIR"

# MSRA 纯基线
for seed in "${SEEDS[@]}"; do
    run_pure_baseline "msra" "$MSRA_LINK_DIR" "$seed"
done

# MSRA 完整模型
for seed in "${SEEDS[@]}"; do
    run_full_model "msra" "$MSRA_TRAIN" "$MSRA_DEV" "$MSRA_TEST" "$seed"
done

# ==============================================================================
# 完成
# ==============================================================================
echo ""
echo "############################################################################"
echo "#                    所有实验完成!                                          #"
echo "#  结果保存在: $RESULTS_DIR                                                 #"
echo "############################################################################"
echo ""
echo "结果目录结构:"
echo "  $RESULTS_DIR/"
echo "    ├── boson_pure_baseline/seed_{42,43,44}/"
echo "    ├── boson_bs_dict_focal/seed_{42,43,44}/"
echo "    ├── clue_pure_baseline/seed_{42,43,44}/"
echo "    ├── clue_bs_dict_focal/seed_{42,43,44}/"
echo "    ├── msra_pure_baseline/seed_{42,43,44}/"
echo "    └── msra_bs_dict_focal/seed_{42,43,44}/"
