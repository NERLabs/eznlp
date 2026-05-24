#!/bin/bash
# =============================================================================
# 公开数据集 CRF Baseline 补全脚本
# 仅补 Resume / Weibo / PeopleDaily（这三个之前只跑了 bs_dict_focal，缺 baseline）
# Boson / CLUE / MSRA 已有 crf_baseline，无需重跑
# =============================================================================
# 不使用 set -e，单次失败不中断后续

# 加载 conda（nohup 后台执行环境必需）
source /home/shiwenlong/miniconda3/etc/profile.d/conda.sh
conda activate eznlp11

# 启用离线模式，强制走本地 HF 缓存，避免 huggingface.co 联网超时
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

CONDA_ENV="eznlp11"
BERT_ARCH="hfl/chinese-macbert-base"
SEEDS=(42 43 44)
NUM_EPOCHS=30
BATCH_SIZE=16

DATASETS=("resume" "weibo" "peopledaily")
DATA_DIRS=(
    "datasets/raw/resume_as_redjujube"
    "datasets/raw/weibo_as_redjujube"
    "datasets/raw/peopledaily_as_redjujube"
)

RESULT_BASE="experiments/EXP-010-optimization/results_public"

START_TIME=$(date +%s)
echo "============================================================"
echo "公开数据集 CRF Baseline 补全（Resume/Weibo/PeopleDaily × 3 seeds = 9 runs）"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

CURRENT=0
TOTAL=9

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    data_dir="${DATA_DIRS[$i]}"

    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        save_dir="${RESULT_BASE}/${dataset}_crf_baseline/seed_${seed}"

        echo ""
        echo ">>> [${CURRENT}/${TOTAL}] CRF baseline | ${dataset} | seed=${seed}"
        echo "    save_dir: ${save_dir}"
        echo "    start: $(date '+%Y-%m-%d %H:%M:%S')"

        conda run --no-capture-output -n ${CONDA_ENV} python research/training/train_redjujube_ner.py \
            --data_dir ${data_dir} \
            --bert_arch ${BERT_ARCH} \
            --save_dir ${save_dir} \
            --seed ${seed} \
            --num_epochs ${NUM_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --max_len 510 \
            --no_fgm \
            --no_ema \
            --bmes_aux_lambda 0 \
            --bmes_label_aux_lambda 0

        echo "<<< [${CURRENT}/${TOTAL}] done at $(date '+%Y-%m-%d %H:%M:%S')"
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "============================================================"
echo "全部完成！总耗时: $((ELAPSED / 60)) 分 $((ELAPSED % 60)) 秒"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "结果目录："
for ds in "${DATASETS[@]}"; do
    echo "  ${RESULT_BASE}/${ds}_crf_baseline/seed_{42,43,44}/"
done
