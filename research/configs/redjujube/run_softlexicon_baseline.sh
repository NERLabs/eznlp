#!/bin/bash
# =============================================================================
# SoftLexicon 基线实验 - RedJujube 新数据集 (RJND)
# 目的：对比 SoftLexicon（经典软词典方法）与 EDBP 方法的性能差异
# =============================================================================
set -e

PROJECT_ROOT="/home/shiwenlong/NERlabs/eznlp"
DATA_DIR="datasets/raw/RedJujube"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results_newdata/SoftLexicon_baseline"
BERT_ARCH="/home/shiwenlong/.cache/huggingface/hub/models--hfl--chinese-macbert-base/snapshots/a986e004d2a7f2a1c2f5a3edef4e20604a974ed1"
SEEDS=(42 43 44)

echo "============================================================"
echo "SoftLexicon 基线实验 - RedJujube 新数据集"
echo "配置: MacBERT + BiLSTM + CRF + SoftLexicon (CTB词表)"
echo "============================================================"

mkdir -p ${BASE_SAVE_DIR}

cd $PROJECT_ROOT

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[SoftLexicon] 运行 seed=$SEED ..."
    
    # 使用 hz 训练脚本（已支持 SoftLexicon），修改数据路径
    SAVE_DIR="${BASE_SAVE_DIR}/seed_${SEED}"
    
    PYTHON="/home/shiwenlong/miniconda3/envs/eznlp11/bin/python"
    
    $PYTHON research/configs/hz/train_hz_ner_baseline_vs_expert_dict.py \
        --data_dir $DATA_DIR \
        --bert_arch $BERT_ARCH \
        --save_dir $SAVE_DIR \
        --seed $SEED \
        --run_softlexicon \
        --num_epochs 30 \
        --batch_size 16 \
        --hid_dim 256 \
        --num_layers 1 \
        --dropout 0.5
    
    echo "[SoftLexicon] seed=$SEED 完成"
done

echo ""
echo "============================================================"
echo "SoftLexicon 基线实验全部完成！"
echo "结果保存在: $BASE_SAVE_DIR"
echo "============================================================"
echo ""
echo "使用以下命令查看结果："
echo "python research/evaluation/collect_optimization_results.py --results_dir ${BASE_SAVE_DIR}"