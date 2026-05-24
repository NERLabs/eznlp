#!/usr/bin/env bash
# ============================================================
# Weibo CRF Baseline 修复版（使用 _clean 数据集）
# 原因：原版 weibo_as_redjujube 训练集有 1237 行单列损坏行
# ============================================================

# 1. conda 激活（nohup 必须显式 source）
source /home/shiwenlong/miniconda3/etc/profile.d/conda.sh
conda activate eznlp11

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 2. 参数
CONDA_ENV="eznlp11"
BERT_ARCH="hfl/chinese-macbert-base"
SEEDS=(42 43 44)
DATA_DIR="datasets/raw/weibo_as_redjujube_clean"
RESULT_BASE="experiments/EXP-010-optimization/results_public"

# 3. 串行执行
idx=1
total=${#SEEDS[@]}
for seed in "${SEEDS[@]}"; do
    echo "======================================"
    echo ">>> [$idx/$total] CRF baseline | weibo_fix | seed=${seed}  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================"
    
    save_dir="${RESULT_BASE}/weibo_crf_baseline/seed_${seed}"
    
    conda run --no-capture-output -n ${CONDA_ENV} python research/training/train_redjujube_ner.py \
        --data_dir ${DATA_DIR} \
        --bert_arch ${BERT_ARCH} \
        --save_dir ${save_dir} \
        --seed ${seed} \
        --num_epochs 30 \
        --batch_size 16 \
        --max_len 510 \
        --no_fgm \
        --no_ema \
        --bmes_aux_lambda 0 \
        --bmes_label_aux_lambda 0
    
    echo "<<< [$idx/$total] done at $(date '+%Y-%m-%d %H:%M:%S')"
    idx=$((idx + 1))
done

echo "所有 Weibo 修复版实验完成！"
