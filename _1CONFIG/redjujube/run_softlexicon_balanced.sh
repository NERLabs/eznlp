#!/bin/bash

# Balanced版本软词典实验（52,487词）

python train_redjujube_ner_comparison.py \
    --data_dir ../../_2DATA/RedJujube \
    --save_dir ../../cache/redjujube_softlexicon_balanced \
    --bert_arch hfl/chinese-macbert-base \
    --num_epochs 30 \
    --batch_size 16 \
    --lr 3e-5 \
    --run_softlexicon_trainlex \
    --softlex_train_path ../../_2DATA/RedJujube/softlexicon_balanced.txt \
    --seed 42

echo "✅ Balanced版本（52,487词）训练完成"
