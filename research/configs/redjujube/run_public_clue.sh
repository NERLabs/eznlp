#!/bin/bash
set -e

cd /home/shiwenlong/NERlabs/eznlp

RESULT_DIR="experiments/EXP-010-optimization/results_public"
DATA_DIR="datasets/raw/clue_as_redjujube"

# ==================== CLUE CRF seed_42 ====================
echo ""
echo "=========================================="
echo "开始: CLUE CRF seed_42"
echo "=========================================="
conda run -n eznlp11 python research/training/train_redjujube_ner.py \
    --data_dir ${DATA_DIR} \
    --bert_arch hfl/chinese-macbert-base \
    --save_dir ${RESULT_DIR}/clue_crf_baseline/seed_42 \
    --seed 42 \
    --num_epochs 30 \
    --batch_size 16 \
    --no_fgm \
    --no_ema \
    --bmes_aux_lambda 0 \
    --bmes_label_aux_lambda 0
echo "完成: CLUE CRF seed_42"

# ==================== CLUE CRF seed_43 ====================
echo ""
echo "=========================================="
echo "开始: CLUE CRF seed_43"
echo "=========================================="
conda run -n eznlp11 python research/training/train_redjujube_ner.py \
    --data_dir ${DATA_DIR} \
    --bert_arch hfl/chinese-macbert-base \
    --save_dir ${RESULT_DIR}/clue_crf_baseline/seed_43 \
    --seed 43 \
    --num_epochs 30 \
    --batch_size 16 \
    --no_fgm \
    --no_ema \
    --bmes_aux_lambda 0 \
    --bmes_label_aux_lambda 0
echo "完成: CLUE CRF seed_43"

# ==================== CLUE CRF seed_44 ====================
echo ""
echo "=========================================="
echo "开始: CLUE CRF seed_44"
echo "=========================================="
conda run -n eznlp11 python research/training/train_redjujube_ner.py \
    --data_dir ${DATA_DIR} \
    --bert_arch hfl/chinese-macbert-base \
    --save_dir ${RESULT_DIR}/clue_crf_baseline/seed_44 \
    --seed 44 \
    --num_epochs 30 \
    --batch_size 16 \
    --no_fgm \
    --no_ema \
    --bmes_aux_lambda 0 \
    --bmes_label_aux_lambda 0
echo "完成: CLUE CRF seed_44"

# ==================== CLUE BS+Dict+Focal seed_42 ====================
echo ""
echo "=========================================="
echo "开始: CLUE BS+Dict+Focal seed_42"
echo "=========================================="
conda run -n eznlp11 python research/training/train_general_expert_boundary.py \
    --train_file ${DATA_DIR}/redjujube_train.bmes \
    --dev_file ${DATA_DIR}/redjujube_dev.bmes \
    --test_file ${DATA_DIR}/redjujube_test.bmes \
    --bert_arch hfl/chinese-macbert-base \
    --save_dir ${RESULT_DIR}/clue_bs_dict_focal/seed_42 \
    --seed 42 \
    --num_epochs 30 \
    --batch_size 16 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --no_fgm \
    --no_ema \
    --fl_gamma 2.0
echo "完成: CLUE BS+Dict+Focal seed_42"

# ==================== CLUE BS+Dict+Focal seed_43 ====================
echo ""
echo "=========================================="
echo "开始: CLUE BS+Dict+Focal seed_43"
echo "=========================================="
conda run -n eznlp11 python research/training/train_general_expert_boundary.py \
    --train_file ${DATA_DIR}/redjujube_train.bmes \
    --dev_file ${DATA_DIR}/redjujube_dev.bmes \
    --test_file ${DATA_DIR}/redjujube_test.bmes \
    --bert_arch hfl/chinese-macbert-base \
    --save_dir ${RESULT_DIR}/clue_bs_dict_focal/seed_43 \
    --seed 43 \
    --num_epochs 30 \
    --batch_size 16 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --no_fgm \
    --no_ema \
    --fl_gamma 2.0
echo "完成: CLUE BS+Dict+Focal seed_43"

# ==================== CLUE BS+Dict+Focal seed_44 ====================
echo ""
echo "=========================================="
echo "开始: CLUE BS+Dict+Focal seed_44"
echo "=========================================="
conda run -n eznlp11 python research/training/train_general_expert_boundary.py \
    --train_file ${DATA_DIR}/redjujube_train.bmes \
    --dev_file ${DATA_DIR}/redjujube_dev.bmes \
    --test_file ${DATA_DIR}/redjujube_test.bmes \
    --bert_arch hfl/chinese-macbert-base \
    --save_dir ${RESULT_DIR}/clue_bs_dict_focal/seed_44 \
    --seed 44 \
    --num_epochs 30 \
    --batch_size 16 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --no_fgm \
    --no_ema \
    --fl_gamma 2.0
echo "完成: CLUE BS+Dict+Focal seed_44"

echo ""
echo "=========================================="
echo "CLUE 全部实验完成！"
echo "=========================================="
