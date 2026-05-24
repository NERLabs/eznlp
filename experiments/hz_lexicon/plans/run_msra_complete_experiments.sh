#!/bin/bash
# MSRA数据集完整实验（对标RedJujube 14个实验）
# 优化版：使用自动抽取词典，避免跨数据集词典污染
set -e

cd /home/shiwenlong/NERlabs/eznlp

SCRIPT=research/configs/msra/train_msra_ner_all_methods.py
BASE_DIR=cache/msra_complete_experiments_$(date +%Y%m%d)

echo "=========================================="
echo "MSRA完整实验 - 14个实验配置"
echo "开始时间: $(date)"
echo "优化: 使用MSRA训练集自动提取词典"
echo "=========================================="
echo ""

# 1. Baseline
echo "[1/14] Baseline (MacBERT + BiLSTM + CRF)..."
python ${SCRIPT} \
    --run_baseline \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/baseline

# 2. ExpertDict (自动提取)
echo "[2/14] ExpertDict (MSRA训练集自动提取)..."
python ${SCRIPT} \
    --run_expert_dict \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/expert_dict

# 3-5. SoftLexicon实验
echo ""
echo "实验3-5: SoftLexicon词典优化"
echo "-----------------------------------"

echo "[3/14] SoftLexicon-v1 (CTB原版)..."
python ${SCRIPT} \
    --run_softlexicon \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/softlex_v1

echo "[4/14] SoftLexicon-v2 (去标点)..."
python ${SCRIPT} \
    --run_softlexicon_v2 \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/softlex_v2

echo "[5/14] SoftLexicon-Balanced (均衡版)..."
python ${SCRIPT} \
    --run_softlexicon_balanced \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/softlex_balanced

# 6-10. 融合方法
echo ""
echo "实验6-10: Soft+Expert融合策略"
echo "-----------------------------------"

echo "[6/14] Concat融合..."
python ${SCRIPT} \
    --run_softlexicon_expert_concat \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/fusion_concat

echo "[7/14] Concat融合 (重复验证)..."
python ${SCRIPT} \
    --run_softlexicon_expert_concat \
    --seed 123 \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/fusion_concat_run2

echo "[8/14] Weighted融合..."
python ${SCRIPT} \
    --run_softlexicon_expert_weighted \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/fusion_weighted

echo "[9/14] Attention融合..."
python ${SCRIPT} \
    --run_softlexicon_expert_attention \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/fusion_attention

echo "[10/14] Gated融合..."
python ${SCRIPT} \
    --run_softlexicon_expert_gated \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/fusion_gated

# 11-14. ExpertDict稳定性验证 (4次运行)
echo ""
echo "实验11-14: ExpertDict稳定性验证"
echo "-----------------------------------"

echo "[11/14] ExpertDict Run2 (seed=123)..."
python ${SCRIPT} \
    --run_expert_dict \
    --seed 123 \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/expert_run2

echo "[12/14] ExpertDict Run3 (seed=456)..."
python ${SCRIPT} \
    --run_expert_dict \
    --seed 456 \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/expert_run3

echo "[13/14] ExpertDict Run4 (seed=789)..."
python ${SCRIPT} \
    --run_expert_dict \
    --seed 789 \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/expert_run4

echo "[14/14] ExpertDict Run5 (seed=999)..."
python ${SCRIPT} \
    --run_expert_dict \
    --seed 999 \
    --num_epochs 30 \
    --batch_size 16 \
    --save_dir ${BASE_DIR}/expert_run5

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "结束时间: $(date)"
echo "=========================================="
echo ""
echo "结果保存在: ${BASE_DIR}"
echo ""

# 快速汇总结果
echo "快速结果汇总:"
echo "-----------------------------------"
find ${BASE_DIR} -name "results.json" 2>/dev/null | while read file; do
    exp_name=$(dirname ${file} | xargs basename)
    f1=$(python -c "import json; data=json.load(open('${file}')); print(f\"{data.get('test', {}).get('f1', 'N/A'):.3f}\")" 2>/dev/null || echo "N/A")
    echo "  ${exp_name}: F1 = ${f1}"
done

echo ""
echo "🎉 MSRA完整实验执行完毕！"
