#!/bin/bash
# 公开数据集 FLAT + BERT 泛化基线
# 4 数据集 x seed=42，复用 train_flat_complete.py，在 FLAT 基础上加入 BERT 预训练特征
set -e

# 等待前置任务（可选）
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ] && ps -p "$WAIT_PID" > /dev/null 2>&1; then
    echo "=== Waiting for PID ${WAIT_PID} to finish ==="
    while ps -p "$WAIT_PID" > /dev/null 2>&1; do sleep 60; done
    echo "=== PID ${WAIT_PID} finished, starting FLAT+BERT ==="
fi

source /home/shiwenlong/miniconda3/etc/profile.d/conda.sh
conda activate eznlp11
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /home/shiwenlong/NERlabs/eznlp

SEED=42
RESULT_BASE="experiments/EXP-010-optimization/results_public"
DATASETS=(msra peopledaily resume boson)

for ds in "${DATASETS[@]}"; do
    data_dir="_2DATA/${ds}_as_redjujube"
    save_dir="${RESULT_BASE}/${ds}_flat_bert/seed_${SEED}"
    echo "=== [$(date +%H:%M:%S)] Training FLAT+BERT on ${ds} ==="
    python _5TRAIN/train_flat_complete.py \
        --data_dir "${data_dir}" \
        --word_file assets/vectors/ctb.50d.vec \
        --save_dir "${save_dir}" \
        --model_type flat \
        --use_bert \
        --bert_model hfl/chinese-macbert-base \
        --hidden_size 256 \
        --embed_size 50 \
        --num_heads 4 \
        --max_seq_len 256 \
        --dropout 0.15 \
        --four_pos_fusion ff \
        --seed ${SEED} \
        --num_epochs 30 \
        --batch_size 16 \
        --lr 1e-3 \
        --finetune_lr 2e-5 >> logs/flat_bert_public.log 2>&1
    echo "=== [$(date +%H:%M:%S)] Done: ${ds} ==="
done

echo "=== ALL FLAT+BERT DONE [$(date +%H:%M:%S)] ==="
