#!/bin/bash

# Balanced版本软词典实验（52,487词）
# 使用eznlp11环境

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eznlp11

cd /home/shiwenlong/NERlabs/eznlp

python research/training/train_redjujube_ner.py \
    --data_dir datasets/raw/RedJujube \
    --save_dir cache/redjujube_softlexicon_balanced \
    --bert_arch hfl/chinese-macbert-base \
    --num_epochs 30 \
    --batch_size 16 \
    --lr 3e-5 \
    --run_softlexicon_trainlex \
    --softlex_train_path datasets/raw/RedJujube/softlexicon_balanced.txt \
    --seed 42

echo "✅ Balanced版本（52,487词）训练完成"
