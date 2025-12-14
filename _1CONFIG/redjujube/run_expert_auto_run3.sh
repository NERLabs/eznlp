#!/bin/bash

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eznlp11

cd /home/shiwenlong/NERlabs/eznlp

# ExpertDict(自动) 重复实验3 - seed=456
python _5TRAIN/train_redjujube_ner.py \
    --data_dir _2DATA/RedJujube \
    --expert_dict_auto_path _2DATA/RedJujube/expert_lexicon_auto.txt \
    --save_dir cache/redjujube_expert_auto_run3 \
    --bert_arch hfl/chinese-macbert-base \
    --num_epochs 30 \
    --batch_size 16 \
    --lr 2e-3 \
    --finetune_lr 2e-5 \
    --run_expert_dict_auto \
    --seed 456
