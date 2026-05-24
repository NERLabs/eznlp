#!/bin/bash
set -e

cd /home/shiwenlong/NERlabs/eznlp

RESULT_DIR="experiments/EXP-010-optimization/results_public"
DATA_DIR="datasets/raw/boson_as_redjujube"

# 等待当前 Boson CRF seed_43 完成
while kill -0 1700190 2>/dev/null; do
    echo "等待 Boson CRF seed_43 完成..."
    sleep 30
done
echo "Boson CRF seed_43 已完成，继续执行剩余实验..."

# ==================== Boson CRF seed_44 ====================
echo ""
echo "=========================================="
echo "开始: Boson CRF seed_44"
echo "=========================================="
conda run -n eznlp11 python research/training/train_redjujube_ner.py \
    --data_dir ${DATA_DIR} \
    --bert_arch hfl/chinese-macbert-base \
    --save_dir ${RESULT_DIR}/boson_crf_baseline/seed_44 \
    --seed 44 \
    --num_epochs 30 \
    --batch_size 16 \
    --no_fgm \
    --no_ema \
    --bmes_aux_lambda 0 \
    --bmes_label_aux_lambda 0
echo "完成: Boson CRF seed_44"

# ==================== Boson BS+Dict+Focal seed_42 ====================
echo ""
echo "=========================================="
echo "开始: Boson BS+Dict+Focal seed_42"
echo "=========================================="
conda run -n eznlp11 python research/training/train_general_expert_boundary.py \
    --train_file ${DATA_DIR}/redjujube_train.bmes \
    --dev_file ${DATA_DIR}/redjujube_dev.bmes \
    --test_file ${DATA_DIR}/redjujube_test.bmes \
    --bert_arch hfl/chinese-macbert-base \
    --save_dir ${RESULT_DIR}/boson_bs_dict_focal/seed_42 \
    --seed 42 \
    --num_epochs 30 \
    --batch_size 16 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --no_fgm \
    --no_ema \
    --fl_gamma 2.0
echo "完成: Boson BS+Dict+Focal seed_42"

# ==================== Boson BS+Dict+Focal seed_43 ====================
echo ""
echo "=========================================="
echo "开始: Boson BS+Dict+Focal seed_43"
echo "=========================================="
conda run -n eznlp11 python research/training/train_general_expert_boundary.py \
    --train_file ${DATA_DIR}/redjujube_train.bmes \
    --dev_file ${DATA_DIR}/redjujube_dev.bmes \
    --test_file ${DATA_DIR}/redjujube_test.bmes \
    --bert_arch hfl/chinese-macbert-base \
    --save_dir ${RESULT_DIR}/boson_bs_dict_focal/seed_43 \
    --seed 43 \
    --num_epochs 30 \
    --batch_size 16 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --no_fgm \
    --no_ema \
    --fl_gamma 2.0
echo "完成: Boson BS+Dict+Focal seed_43"

# ==================== Boson BS+Dict+Focal seed_44 ====================
echo ""
echo "=========================================="
echo "开始: Boson BS+Dict+Focal seed_44"
echo "=========================================="
conda run -n eznlp11 python research/training/train_general_expert_boundary.py \
    --train_file ${DATA_DIR}/redjujube_train.bmes \
    --dev_file ${DATA_DIR}/redjujube_dev.bmes \
    --test_file ${DATA_DIR}/redjujube_test.bmes \
    --bert_arch hfl/chinese-macbert-base \
    --save_dir ${RESULT_DIR}/boson_bs_dict_focal/seed_44 \
    --seed 44 \
    --num_epochs 30 \
    --batch_size 16 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --no_fgm \
    --no_ema \
    --fl_gamma 2.0
echo "完成: Boson BS+Dict+Focal seed_44"

echo ""
echo "=========================================="
echo "Boson 全部实验完成！"
echo "=========================================="
