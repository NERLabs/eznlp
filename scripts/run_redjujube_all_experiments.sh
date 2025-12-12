#!/bin/bash
# RedJujube 数据集 NER 四组对比实验
# 
# 实验组：
# 1. Baseline
# 2. SoftLexicon (TrainLex)
# 3. ExpertDict (自动)
# 4. ExpertDict (手动)

set -e

echo "======================================================================"
echo "RedJujube 数据集 NER 四组对比实验"
echo "======================================================================"
echo ""

# 进入项目根目录
cd "$(dirname "$0")/.."

# 运行所有四组实验
python scripts/train_redjujube_ner_comparison.py \
  --data_dir data/RedJujube \
  --expert_dict_path data/RedJujube/expert_lexicon.txt \
  --expert_dict_auto_path data/RedJujube/expert_lexicon_auto.txt \
  --softlex_train_path data/RedJujube/softlexicon_train.txt \
  --save_dir cache/redjujube_ner_comparison \
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
  --run_all

echo ""
echo "======================================================================"
echo "✅ 所有实验完成！"
echo "======================================================================"
