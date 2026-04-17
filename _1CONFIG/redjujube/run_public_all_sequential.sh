#!/bin/bash
# =============================================================================
# 泛化实验串行执行脚本
# 在5个公开数据集上验证方法泛化性
# 每个数据集：CRF基线 + BS+Dict（禁用Focal Loss）
# =============================================================================

# 注意：不使用 set -e，确保单个实验失败不会中断后续实验

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
CONDA_ENV="eznlp11"
BERT_ARCH="hfl/chinese-macbert-base"
SEEDS=(42 43 44)
NUM_EPOCHS=30
BATCH_SIZE=16
SB_EPSILON=0.1
SB_SIZE=2

# 数据集配置
DATASETS=("msra" "resume" "weibo" "peopledaily" "pingguo")
DATA_DIRS=(
    "_2DATA/msra_as_redjujube"
    "_2DATA/resume_as_redjujube"
    "_2DATA/weibo_as_redjujube"
    "_2DATA/peopledaily_as_redjujube"
    "_2DATA/pingguo_as_redjujube"
)

# 结果目录前缀
RESULT_BASE="experiments/EXP-010-optimization/results_public"

# 统计
TOTAL_DATASETS=5
# MSRA(6) + Resume(6) + Weibo(6) + PeopleDaily(6) + pingguo(3) = 27
TOTAL_EXPERIMENTS=27
CURRENT_EXPERIMENT=0

# 开始时间
START_TIME=$(date +%s)

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}                    泛化实验串行执行脚本                                    ${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${YELLOW}总数据集: ${TOTAL_DATASETS}${NC}"
echo -e "${YELLOW}总实验数: ${TOTAL_EXPERIMENTS}${NC}"
echo -e "${YELLOW}开始时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# 函数：运行CRF基线实验
run_crf_baseline() {
    local dataset=$1
    local data_dir=$2
    local seed=$3
    local save_dir="${RESULT_BASE}/${dataset}_crf_baseline/seed_${seed}"
    
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    
    echo -e "${GREEN}>>> [实验 ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] CRF基线 - ${dataset} - seed ${seed}${NC}"
    echo -e "    保存目录: ${save_dir}"
    echo -e "    开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    conda run -n ${CONDA_ENV} python _5TRAIN/train_redjujube_ner.py \
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
    
    echo -e "${GREEN}<<< [实验 ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] 完成 - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
}

# 函数：运行BS+Dict实验（无Focal Loss）
run_bs_dict() {
    local dataset=$1
    local data_dir=$2
    local seed=$3
    local save_dir="${RESULT_BASE}/${dataset}_bs_dict/seed_${seed}"
    
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    
    echo -e "${GREEN}>>> [实验 ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] BS+Dict - ${dataset} - seed ${seed}${NC}"
    echo -e "    保存目录: ${save_dir}"
    echo -e "    开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    conda run -n ${CONDA_ENV} python _5TRAIN/train_general_expert_boundary.py \
        --train_file ${data_dir}/redjujube_train.bmes \
        --dev_file ${data_dir}/redjujube_dev.bmes \
        --test_file ${data_dir}/redjujube_test.bmes \
        --bert_arch ${BERT_ARCH} \
        --save_dir ${save_dir} \
        --seed ${seed} \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --sb_epsilon ${SB_EPSILON} \
        --sb_size ${SB_SIZE} \
        --no_fgm \
        --no_ema
    
    echo -e "${GREEN}<<< [实验 ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] 完成 - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
}

# =============================================================================
# 数据集 1: MSRA (6个实验)
# =============================================================================
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}[数据集 1/5] MSRA - 开始${NC}"
echo -e "${BLUE}=============================================================================${NC}"

# CRF基线 x 3 seeds
for seed in "${SEEDS[@]}"; do
    run_crf_baseline "msra" "_2DATA/msra_as_redjujube" ${seed}
done

# BS+Dict x 3 seeds
for seed in "${SEEDS[@]}"; do
    run_bs_dict "msra" "_2DATA/msra_as_redjujube" ${seed}
done

echo -e "${BLUE}[数据集 1/5] MSRA - 完成${NC}"
echo ""

# =============================================================================
# 数据集 2: ResumeNER (6个实验)
# =============================================================================
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}[数据集 2/5] ResumeNER - 开始${NC}"
echo -e "${BLUE}=============================================================================${NC}"

# CRF基线 x 3 seeds
for seed in "${SEEDS[@]}"; do
    run_crf_baseline "resume" "_2DATA/resume_as_redjujube" ${seed}
done

# BS+Dict x 3 seeds
for seed in "${SEEDS[@]}"; do
    run_bs_dict "resume" "_2DATA/resume_as_redjujube" ${seed}
done

echo -e "${BLUE}[数据集 2/5] ResumeNER - 完成${NC}"
echo ""

# =============================================================================
# 数据集 3: WeiboNER (6个实验)
# =============================================================================
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}[数据集 3/5] WeiboNER - 开始${NC}"
echo -e "${BLUE}=============================================================================${NC}"

# CRF基线 x 3 seeds
for seed in "${SEEDS[@]}"; do
    run_crf_baseline "weibo" "_2DATA/weibo_as_redjujube" ${seed}
done

# BS+Dict x 3 seeds
for seed in "${SEEDS[@]}"; do
    run_bs_dict "weibo" "_2DATA/weibo_as_redjujube" ${seed}
done

echo -e "${BLUE}[数据集 3/5] WeiboNER - 完成${NC}"
echo ""

# =============================================================================
# 数据集 4: People's Daily (6个实验)
# =============================================================================
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}[数据集 4/5] People's Daily - 开始${NC}"
echo -e "${BLUE}=============================================================================${NC}"

# CRF基线 x 3 seeds
for seed in "${SEEDS[@]}"; do
    run_crf_baseline "peopledaily" "_2DATA/peopledaily_as_redjujube" ${seed}
done

# BS+Dict x 3 seeds
for seed in "${SEEDS[@]}"; do
    run_bs_dict "peopledaily" "_2DATA/peopledaily_as_redjujube" ${seed}
done

echo -e "${BLUE}[数据集 4/5] People's Daily - 完成${NC}"
echo ""

# =============================================================================
# 数据集 5: pingguo (3个实验 - 仅CRF基线)
# =============================================================================
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}[数据集 5/5] pingguo - 开始 (仅CRF基线)${NC}"
echo -e "${BLUE}=============================================================================${NC}"

# CRF基线 x 3 seeds (BS+Dict已有旧实验数据，跳过)
for seed in "${SEEDS[@]}"; do
    run_crf_baseline "pingguo" "_2DATA/pingguo_as_redjujube" ${seed}
done

echo -e "${BLUE}[数据集 5/5] pingguo - 完成${NC}"
echo ""

# =============================================================================
# 总结
# =============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${GREEN}                    所有泛化实验完成！                                      ${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${YELLOW}完成时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${YELLOW}总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒${NC}"
echo -e "${YELLOW}实验总数: ${TOTAL_EXPERIMENTS}${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo "实验结果保存在: ${RESULT_BASE}/"
echo "  - msra_crf_baseline/seed_{42,43,44}"
echo "  - msra_bs_dict/seed_{42,43,44}"
echo "  - resume_crf_baseline/seed_{42,43,44}"
echo "  - resume_bs_dict/seed_{42,43,44}"
echo "  - weibo_crf_baseline/seed_{42,43,44}"
echo "  - weibo_bs_dict/seed_{42,43,44}"
echo "  - peopledaily_crf_baseline/seed_{42,43,44}"
echo "  - peopledaily_bs_dict/seed_{42,43,44}"
echo "  - pingguo_crf_baseline/seed_{42,43,44}"
