#!/bin/bash
# MSRA剩余12个实验（排除已完成的Baseline和ExpertDict）
set -e

cd /home/shiwenlong/NERlabs/eznlp

SCRIPT=research/configs/msra/train_msra_ner_all_methods.py
BASE_DIR=cache/msra_complete_experiments_$(date +%Y%m%d)

echo "=========================================="
echo "MSRA剩余实验 - 12个实验"
echo "开始时间: $(date)"
echo "已跳过: Baseline, ExpertDict (正在运行)"
echo "=========================================="
echo ""

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

# 11-14. ExpertDict稳定性验证（跳过第一次，已在运行）
echo ""
echo "实验11-14: ExpertDict稳定性验证 (跳过run1)"
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
echo "剩余12个实验完成！"
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
echo "🎉 MSRA剩余实验执行完毕！"
