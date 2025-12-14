#!/bin/bash
# RedJujube 数据集 - 测试过滤版 SoftLexicon
# 
# 对比实验：
# - SoftLexicon(原版)：198,437词 → F1: 96.07%
# - SoftLexicon(过滤版)：18,678词 → F1: ?
#
# 目标：用更少的词达到相同或更好的性能

set -e

echo "======================================================================"
echo "RedJujube - SoftLexicon 过滤版实验"
echo "======================================================================"
echo ""
echo "词典对比:"
echo "  - 原版: 198,437词 (全部n-gram)"
echo "  - 过滤版: 18,678词 (高质量过滤)"
echo ""
echo "过滤策略:"
echo "  ✓ 实体优先权重 3.0x"
echo "  ✓ n-gram最小频次 10"
echo "  ✓ 长度限制 2-4字"
echo "  ✓ PMI互信息过滤"
echo "  ✓ 停用词过滤"
echo ""
echo "======================================================================"
echo ""

# 进入项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# 运行训练
python _5TRAIN/train_redjujube_ner.py \
  --data_dir _2DATA/RedJujube \
  --softlex_train_path _2DATA/RedJujube/softlexicon_filtered.txt \
  --save_dir cache/redjujube_softlexicon_filtered \
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
  --disp_every_steps 50 \
  --eval_every_steps 200 \
  --seed 42 \
  --run_softlexicon

echo ""
echo "======================================================================"
echo "✅ 训练完成！"
echo "======================================================================"
echo ""
echo "查看结果:"
echo "  - 训练日志: cache/redjujube_softlexicon_filtered/*/training.log"
echo "  - 测试结果: cache/redjujube_softlexicon_filtered/*/results.json"
echo ""
