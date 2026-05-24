#!/bin/bash
# 仅运行方案A进行快速验证

set -e  # 遇到错误立即退出

cd /home/shiwenlong/NERlabs/eznlp

echo "========================================="
echo "  RedJujube Soft+Expert 联合模型实验"
echo "  仅运行方案A: 直接拼接融合"
echo "========================================="
echo ""

# 激活环境
source ~/.bashrc
conda activate eznlp11

# 运行方案A
echo "运行方案A: 直接拼接融合..."
python research/training/train_redjujube_ner.py \
  --data_dir data/RedJujube \
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
  --seed 42 \
  --run_softlexicon_expert_concat

echo ""
echo "✅ 方案A完成!"
echo ""
echo "结果保存在: cache/redjujube_softlexicon_expert/softlexicon_expert_concat_*/"
echo ""
