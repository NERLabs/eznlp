#!/bin/bash
# =============================================================================
# 批量评估脚本 - 评估所有模型组的最佳 seed
# =============================================================================

set -e

# 项目根目录
PROJECT_ROOT="/home/shiwenlong/NERlabs/eznlp"
cd "$PROJECT_ROOT"

# 日志文件
LOG_FILE="logs/research_logs/evaluation_results.log"
mkdir -p "$(dirname "$LOG_FILE")"

# 专家词典路径
EXPERT_DICT_PATH="datasets/raw/RedJujube/expert_lexicon_auto_min1.txt"

# 模型组配置
declare -A MODEL_GROUPS
MODEL_GROUPS["A_CRF+Dict"]="experiments/EXP-010-optimization/results_newdata/A_baseline"
MODEL_GROUPS["H_BS+Dict"]="experiments/EXP-010-optimization/results_newdata/H_bs_baseline"
MODEL_GROUPS["Q_BS+Dict+Focal"]="experiments/EXP-010-optimization/results_newdata/Q_bs_focal"
MODEL_GROUPS["CRF_nodict"]="experiments/EXP-010-optimization/results_newdata/CRF_nodict"
MODEL_GROUPS["BS_nodict"]="experiments/EXP-010-optimization/results_newdata/BS_nodict"
MODEL_GROUPS["BS_focal_nodict"]="experiments/EXP-010-optimization/results_newdata/BS_focal_nodict"

# 函数：从 results.json 获取 best_dev_f1
get_best_dev_f1() {
    local dir="$1"
    local results_file="$dir/results.json"
    if [[ -f "$results_file" ]]; then
        python3 -c "import json; f=open('$results_file'); d=json.load(f); print(d.get('best_dev_f1', 0))"
    else
        echo "0"
    fi
}

# 函数：找到最佳 seed 目录
find_best_seed_dir() {
    local group_dir="$1"
    local best_dir=""
    local best_f1=0
    
    for subdir in "$group_dir"/*; do
        if [[ -d "$subdir" ]]; then
            f1=$(get_best_dev_f1 "$subdir")
            if (( $(echo "$f1 > $best_f1" | bc -l) )); then
                best_f1=$f1
                best_dir=$subdir
            fi
        fi
    done
    
    echo "$best_dir"
}

# 开始评估
echo "=================================================================="
echo "批量模型评估 - $(date)"
echo "=================================================================="
echo "" 

# 清空/初始化日志文件
{
    echo "=================================================================="
    echo "批量模型评估结果"
    echo "评估时间: $(date)"
    echo "=================================================================="
    echo ""
} > "$LOG_FILE"

# 按顺序定义模型组
GROUP_ORDER=("A_CRF+Dict" "H_BS+Dict" "Q_BS+Dict+Focal" "CRF_nodict" "BS_nodict" "BS_focal_nodict")

for group_name in "${GROUP_ORDER[@]}"; do
    group_dir="${MODEL_GROUPS[$group_name]}"
    
    echo "----------------------------------------------------------------"
    echo "处理模型组: $group_name"
    echo "目录: $group_dir"
    
    # 找到最佳 seed 目录
    best_dir=$(find_best_seed_dir "$group_dir")
    
    if [[ -z "$best_dir" ]]; then
        echo "警告: 未找到有效的模型目录，跳过 $group_name"
        continue
    fi
    
    # 获取最佳验证 F1
    best_f1=$(get_best_dev_f1 "$best_dir")
    echo "最佳目录: $best_dir"
    echo "最佳验证 F1: $best_f1"
    
    # 判断是否需要传入专家词典路径
    # 对于有词典的模型（A', H', Q'），需要传入专家词典路径
    EXTRA_ARGS=""
    if [[ "$group_name" == "A_CRF+Dict" ]] || [[ "$group_name" == "H_BS+Dict" ]] || [[ "$group_name" == "Q_BS+Dict+Focal" ]]; then
        EXTRA_ARGS="--expert_dict_path $EXPERT_DICT_PATH"
    fi
    
    # 运行评估
    echo ""
    echo "运行评估..."
    
    {
        echo "=================================================================="
        echo "模型组: $group_name"
        echo "模型目录: $best_dir"
        echo "最佳验证 F1: $best_f1"
        echo "=================================================================="
        echo ""
    } >> "$LOG_FILE"
    
    conda run -n eznlp11 python research/evaluation/evaluate_all_models.py \
        --save_dir "$best_dir" \
        --model_type auto \
        $EXTRA_ARGS \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    echo "完成: $group_name"
    echo ""
done

echo "=================================================================="
echo "所有评估完成！"
echo "结果保存在: $LOG_FILE"
echo "=================================================================="
