#!/bin/bash
# HZ 数据集 NER 对比实验：使用自动提取的专家词典

set -e

echo "========================================================================"
echo "HZ NER 对比实验：Baseline vs 专家词典（自动提取）"
echo "========================================================================"
echo ""

# 步骤 1: 从训练数据提取专家词典
echo "步骤 1/2: 从训练数据提取专家词典"
echo "------------------------------------------------------------------------"
python scripts/extract_lexicon_from_training.py \
    --train_path data/HZ/hz_train.bmes \
    --output_path data/HZ/expert_lexicon_auto.txt \
    --min_freq 2

echo ""
echo "✅ 词典提取完成！"
echo ""

# 步骤 2: 运行对比实验
echo "步骤 2/2: 运行对比实验"
echo "------------------------------------------------------------------------"
python scripts/train_hz_ner_baseline_vs_expert_dict.py \
    --run_both \
    --data_dir data/HZ \
    --expert_dict_path data/HZ/expert_lexicon_auto.txt \
    --save_dir cache/hz_comparison_auto_lexicon \
    --num_epochs 30 \
    --batch_size 16 \
    --seed 42

echo ""
echo "========================================================================"
echo "✅ 实验完成！"
echo "========================================================================"
echo ""
echo "结果保存在: cache/hz_comparison_auto_lexicon/"
echo ""
