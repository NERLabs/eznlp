#!/bin/bash
# 顺序运行所有四种融合方案的实验脚本
# 每个方案完成后自动启动下一个

set -e  # 遇到错误立即退出

cd /home/shiwenlong/NERlabs/eznlp

PYTHON_BIN="/home/shiwenlong/miniconda3/envs/eznlp11/bin/python"

# 公共参数
COMMON_ARGS="--data_dir data/RedJujube \
  --save_dir cache/redjujube_softlexicon_expert \
  --expert_dict_auto_path data/RedJujube/expert_lexicon_auto.txt \
  --softlex_train_path data/RedJujube/softlexicon_train.txt \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 256 \
  --num_layers 1 \
  --dropout 0.5 \
  --num_epochs 30 \
  --batch_size 16 \
  --lr 2e-3 \
  --finetune_lr 2e-5 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --seed 42"

echo "========================================================================"
echo "  RedJujube Soft+Expert 联合模型实验 - 顺序执行所有方案"
echo "  开始时间: $(date)"
echo "========================================================================"
echo ""

# 方案A: 直接拼接
echo "[1/4] 运行方案A: 直接拼接融合..."
echo "开始时间: $(date)"
$PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS --run_softlexicon_expert_concat
echo "完成时间: $(date)"
echo "✅ 方案A完成!"
echo ""

# 方案B: 加权求和
echo "[2/4] 运行方案B: 加权求和融合..."
echo "开始时间: $(date)"
$PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS --run_softlexicon_expert_weighted
echo "完成时间: $(date)"
echo "✅ 方案B完成!"
echo ""

# 方案C: 门控机制
echo "[3/4] 运行方案C: 门控机制融合..."
echo "开始时间: $(date)"
$PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS --run_softlexicon_expert_gated
echo "完成时间: $(date)"
echo "✅ 方案C完成!"
echo ""

# 方案D: 注意力融合
echo "[4/4] 运行方案D: 注意力融合..."
echo "开始时间: $(date)"
$PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS --run_softlexicon_expert_attention
echo "完成时间: $(date)"
echo "✅ 方案D完成!"
echo ""

echo "========================================================================"
echo "  全部实验完成!"
echo "  结束时间: $(date)"
echo "========================================================================"
echo ""
echo "结果保存在: cache/redjujube_softlexicon_expert/"
echo ""
echo "查看结果:"
echo "  方案A (拼接):   cache/redjujube_softlexicon_expert/softlexicon_expert_concat_*/results.json"
echo "  方案B (加权):   cache/redjujube_softlexicon_expert/softlexicon_expert_weighted_*/results.json"
echo "  方案C (门控):   cache/redjujube_softlexicon_expert/softlexicon_expert_gated_*/results.json"
echo "  方案D (注意力): cache/redjujube_softlexicon_expert/softlexicon_expert_attention_*/results.json"
echo ""
