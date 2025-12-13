#!/bin/bash
# RedJujube - 超参数调优实验
# 
# 基于：ExpertDict(自动) - 当前最优 96.99%
# 目标：通过超参数优化突破 97.2%

set -e

echo "======================================================================"
echo "RedJujube - ExpertDict 超参数调优"
echo "======================================================================"
echo ""
echo "当前最优配置: F1=96.99%"
echo "  - hid_dim: 256"
echo "  - num_layers: 1"
echo "  - dropout: 0.5"
echo "  - lr: 2e-3"
echo "  - batch_size: 16"
echo ""
echo "调优方向:"
echo "  1. 尝试更大隐藏层: 256 → 384/512"
echo "  2. 尝试2层BiLSTM"
echo "  3. 调整Dropout: 0.5 → 0.3/0.4"
echo "  4. 尝试更大批次: 16 → 32"
echo "======================================================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# 实验1: 更大隐藏层 (hid_dim=384)
echo "🔬 实验1: hid_dim=384"
python _1CONFIG/redjujube/train_redjujube_ner_comparison.py \
  --data_dir _2DATA/RedJujube \
  --expert_dict_auto_path _2DATA/RedJujube/expert_lexicon_auto.txt \
  --save_dir cache/redjujube_expert_tuning/exp1_hid384 \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 384 \
  --num_layers 1 \
  --dropout 0.5 \
  --expert_dict_dim 50 \
  --num_epochs 30 \
  --batch_size 16 \
  --lr 2e-3 \
  --finetune_lr 2e-5 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --seed 42 \
  --run_expert_dict_auto

echo ""
echo "✅ 实验1完成"
echo ""

# 实验2: 2层BiLSTM (num_layers=2)
echo "🔬 实验2: num_layers=2"
python _1CONFIG/redjujube/train_redjujube_ner_comparison.py \
  --data_dir _2DATA/RedJujube \
  --expert_dict_auto_path _2DATA/RedJujube/expert_lexicon_auto.txt \
  --save_dir cache/redjujube_expert_tuning/exp2_layers2 \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 256 \
  --num_layers 2 \
  --dropout 0.5 \
  --expert_dict_dim 50 \
  --num_epochs 30 \
  --batch_size 16 \
  --lr 2e-3 \
  --finetune_lr 2e-5 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --seed 42 \
  --run_expert_dict_auto

echo ""
echo "✅ 实验2完成"
echo ""

# 实验3: 降低Dropout (dropout=0.3)
echo "🔬 实验3: dropout=0.3"
python _1CONFIG/redjujube/train_redjujube_ner_comparison.py \
  --data_dir _2DATA/RedJujube \
  --expert_dict_auto_path _2DATA/RedJujube/expert_lexicon_auto.txt \
  --save_dir cache/redjujube_expert_tuning/exp3_drop03 \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 256 \
  --num_layers 1 \
  --dropout 0.3 \
  --expert_dict_dim 50 \
  --num_epochs 30 \
  --batch_size 16 \
  --lr 2e-3 \
  --finetune_lr 2e-5 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --seed 42 \
  --run_expert_dict_auto

echo ""
echo "✅ 实验3完成"
echo ""

# 实验4: 更大批次 (batch_size=32)
echo "🔬 实验4: batch_size=32"
python _1CONFIG/redjujube/train_redjujube_ner_comparison.py \
  --data_dir _2DATA/RedJujube \
  --expert_dict_auto_path _2DATA/RedJujube/expert_lexicon_auto.txt \
  --save_dir cache/redjujube_expert_tuning/exp4_bs32 \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 256 \
  --num_layers 1 \
  --dropout 0.5 \
  --expert_dict_dim 50 \
  --num_epochs 30 \
  --batch_size 32 \
  --lr 2e-3 \
  --finetune_lr 2e-5 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --seed 42 \
  --run_expert_dict_auto

echo ""
echo "======================================================================"
echo "✅ 所有超参数调优实验完成！"
echo "======================================================================"
echo ""
echo "结果汇总："
echo "  实验1: cache/redjujube_expert_tuning/exp1_hid384/*/results.json"
echo "  实验2: cache/redjujube_expert_tuning/exp2_layers2/*/results.json"
echo "  实验3: cache/redjujube_expert_tuning/exp3_drop03/*/results.json"
echo "  实验4: cache/redjujube_expert_tuning/exp4_bs32/*/results.json"
echo ""
