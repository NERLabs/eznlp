#!/bin/bash
# MSRA数据集晚间实验计划
# 时间：2025-12-13 晚上
# 目标：验证ExpertDict + 超参数探索

set -e

BASE_DIR=/home/shiwenlong/NERlabs/eznlp
cd ${BASE_DIR}

# 创建实验记录目录
EXPERIMENT_LOG=experiments/hz_lexicon/results/12-2_softlexicon/msra_experiments_$(date +%Y%m%d).log
mkdir -p experiments/hz_lexicon/results/12-2_softlexicon/

echo "========================================" | tee -a ${EXPERIMENT_LOG}
echo "MSRA 晚间实验计划 - $(date)" | tee -a ${EXPERIMENT_LOG}
echo "========================================" | tee -a ${EXPERIMENT_LOG}
echo "" | tee -a ${EXPERIMENT_LOG}

# ============================================
# 实验1：Baseline vs ExpertDict（基础对比）
# 预计时间：2-3小时
# ============================================
echo "实验1：Baseline vs ExpertDict 基础对比" | tee -a ${EXPERIMENT_LOG}
echo "开始时间：$(date)" | tee -a ${EXPERIMENT_LOG}

python _1CONFIG/msra/train_msra_ner_baseline_vs_expert_dict.py \
    --run_baseline \
    --run_expert_dict \
    --save_dir cache/msra_experiments/exp1_baseline_vs_expert \
    --num_epochs 30 \
    --batch_size 16 \
    --expert_dict_dim 50 \
    --min_freq 2 \
    2>&1 | tee -a ${EXPERIMENT_LOG}

echo "完成时间：$(date)" | tee -a ${EXPERIMENT_LOG}
echo "" | tee -a ${EXPERIMENT_LOG}

# ============================================
# 实验2：emb_dim 超参数探索
# 预计时间：3-4小时
# ============================================
echo "实验2：ExpertDict emb_dim 超参数探索" | tee -a ${EXPERIMENT_LOG}

for dim in 50 100 200; do
    echo "  - 测试 emb_dim=${dim}" | tee -a ${EXPERIMENT_LOG}
    echo "    开始时间：$(date)" | tee -a ${EXPERIMENT_LOG}
    
    python _1CONFIG/msra/train_msra_ner_baseline_vs_expert_dict.py \
        --run_expert_dict \
        --save_dir cache/msra_experiments/exp2_emb_dim_${dim} \
        --num_epochs 30 \
        --batch_size 16 \
        --expert_dict_dim ${dim} \
        --min_freq 2 \
        2>&1 | tee -a ${EXPERIMENT_LOG}
    
    echo "    完成时间：$(date)" | tee -a ${EXPERIMENT_LOG}
done

echo "" | tee -a ${EXPERIMENT_LOG}

# ============================================
# 实验3：min_freq 超参数探索
# 预计时间：3-4小时
# ============================================
echo "实验3：ExpertDict min_freq 超参数探索" | tee -a ${EXPERIMENT_LOG}

for freq in 2 3 5; do
    echo "  - 测试 min_freq=${freq}" | tee -a ${EXPERIMENT_LOG}
    echo "    开始时间：$(date)" | tee -a ${EXPERIMENT_LOG}
    
    python _1CONFIG/msra/train_msra_ner_baseline_vs_expert_dict.py \
        --run_expert_dict \
        --save_dir cache/msra_experiments/exp3_min_freq_${freq} \
        --num_epochs 30 \
        --batch_size 16 \
        --expert_dict_dim 50 \
        --min_freq ${freq} \
        2>&1 | tee -a ${EXPERIMENT_LOG}
    
    echo "    完成时间：$(date)" | tee -a ${EXPERIMENT_LOG}
done

echo "" | tee -a ${EXPERIMENT_LOG}

# ============================================
# 实验4：最优配置验证（如果时间充足）
# 预计时间：1-2小时
# ============================================
echo "实验4：最优配置验证（emb_dim=100, min_freq=3）" | tee -a ${EXPERIMENT_LOG}
echo "开始时间：$(date)" | tee -a ${EXPERIMENT_LOG}

python _1CONFIG/msra/train_msra_ner_baseline_vs_expert_dict.py \
    --run_expert_dict \
    --save_dir cache/msra_experiments/exp4_best_config \
    --num_epochs 30 \
    --batch_size 16 \
    --expert_dict_dim 100 \
    --min_freq 3 \
    2>&1 | tee -a ${EXPERIMENT_LOG}

echo "完成时间：$(date)" | tee -a ${EXPERIMENT_LOG}
echo "" | tee -a ${EXPERIMENT_LOG}

# ============================================
# 汇总实验结果
# ============================================
echo "========================================" | tee -a ${EXPERIMENT_LOG}
echo "所有实验完成！" | tee -a ${EXPERIMENT_LOG}
echo "结束时间：$(date)" | tee -a ${EXPERIMENT_LOG}
echo "========================================" | tee -a ${EXPERIMENT_LOG}
echo "" | tee -a ${EXPERIMENT_LOG}

echo "实验结果保存在：" | tee -a ${EXPERIMENT_LOG}
echo "  - cache/msra_experiments/" | tee -a ${EXPERIMENT_LOG}
echo "  - ${EXPERIMENT_LOG}" | tee -a ${EXPERIMENT_LOG}

# 自动提取关键结果
echo "" | tee -a ${EXPERIMENT_LOG}
echo "快速结果汇总：" | tee -a ${EXPERIMENT_LOG}
find cache/msra_experiments -name "results.json" | while read file; do
    exp_name=$(dirname ${file} | xargs basename)
    f1=$(python -c "import json; print(json.load(open('${file}'))['test']['f1'])" 2>/dev/null || echo "N/A")
    echo "  ${exp_name}: F1 = ${f1}" | tee -a ${EXPERIMENT_LOG}
done

echo "" | tee -a ${EXPERIMENT_LOG}
echo "🎉 晚间实验计划执行完毕！" | tee -a ${EXPERIMENT_LOG}
