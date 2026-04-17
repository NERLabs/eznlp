#!/bin/bash
# =============================================================================
# G' BiLSTM-CRF 独立并行脚本（可与BERT系列实验同时运行）
# =============================================================================
# GPU占用约1.7G，可与BERT实验（~6G）并行
# =============================================================================
set -e

DATA_DIR="_2DATA/RedJujube"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results_newdata"
SEEDS=(42 43 44)

echo "============================================================"
echo "实验 G': BiLSTM-CRF（无BERT）- 独立并行"
echo "参数: emb_dim=100, hid_dim=256, num_layers=2, epochs=80"
echo "============================================================"

mkdir -p ${BASE_SAVE_DIR}/G_bilstm_baseline

for SEED in "${SEEDS[@]}"; do
    echo "[G'] 运行 seed=$SEED ..."
    conda run -n eznlp11 python _5TRAIN/train_redjujube_bilstm_crf.py \
        --data_dir $DATA_DIR \
        --save_dir ${BASE_SAVE_DIR}/G_bilstm_baseline \
        --seed $SEED \
        --num_epochs 80 --batch_size 16 \
        --emb_dim 100 --hid_dim 256 --num_layers 2
    echo "[G'] seed=$SEED 完成"
done
echo "[G'] BiLSTM-CRF 全部完成"
