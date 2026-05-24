#!/bin/bash
# RedJujube 数据集 - 改进融合实验
# 
# 策略：ExpertDict(2k精选) + SoftLex(18k过滤版)
# 避免：低质量n-gram干扰
# 目标：突破 97% F1

set -e

echo "======================================================================"
echo "RedJujube - 改进融合实验"
echo "======================================================================"
echo ""
echo "融合策略:"
echo "  - ExpertDict: 2,078词（自动提取，高质量）"
echo "  - SoftLex-Filtered: 18,678词（过滤版，去除低质量词）"
echo "  - 融合方式: Concat（已验证最优）"
echo ""
echo "改进点:"
echo "  ✓ 使用过滤版SoftLex（18k vs 原198k）"
echo "  ✓ 减少特征冲突"
echo "  ✓ 降低过拟合风险"
echo ""
echo "预期性能: > 97.0% F1"
echo "======================================================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# 运行改进的融合实验
python research/training/train_redjujube_ner.py \
  --data_dir datasets/raw/RedJujube \
  --expert_dict_auto_path datasets/raw/RedJujube/expert_lexicon_auto.txt \
  --softlex_train_path datasets/raw/RedJujube/softlexicon_filtered.txt \
  --save_dir cache/redjujube_fusion_improved \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 256 \
  --num_layers 1 \
  --dropout 0.5 \
  --expert_dict_dim 50 \
  --num_epochs 30 \
  --batch_size 16 \
  --lr 2e-3 \
  --finetune_lr 2e-5 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --disp_every_steps 50 \
  --eval_every_steps 200 \
  --seed 42 \
  --run_softlexicon_expert_concat

echo ""
echo "======================================================================"
echo "✅ 训练完成！"
echo "======================================================================"
