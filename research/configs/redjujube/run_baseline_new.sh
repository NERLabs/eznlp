#!/bin/bash

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eznlp11

cd /home/shiwenlong/NERlabs/eznlp

# Baseline 模型训练
python research/training/train_redjujube_ner.py \
    --data_dir datasets/raw/RedJujube \
    --save_dir cache/redjujube_baseline_new \
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
    --run_baseline \
    --seed 42
