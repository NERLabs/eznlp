#!/bin/bash
# 公开数据集 BiLSTM-CRF（无 BERT）泛化基线
# 4 数据集 × seed=42，复用 train_redjujube_bilstm_crf.py + *_as_redjujube 数据
set -e

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
    save_dir="${RESULT_BASE}/${ds}_bilstm_crf_no_bert/seed_${SEED}"
    echo "=== [$(date +%H:%M:%S)] Training BiLSTM-CRF (no BERT) on ${ds} ==="
    python _5TRAIN/train_redjujube_bilstm_crf.py \
        --data_dir "${data_dir}" \
        --save_dir "${save_dir}" \
        --seed ${SEED} \
        --num_epochs 30 \
        --batch_size 32 \
        --emb_dim 100 \
        --hid_dim 256 \
        --num_layers 2 \
        --dropout 0.5 \
        --lr 1e-3 \
        --no_tensorboard 2>&1 | tee -a logs/bilstm_crf_no_bert_public.log
    echo "=== [$(date +%H:%M:%S)] Done: ${ds} ==="
done

echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
