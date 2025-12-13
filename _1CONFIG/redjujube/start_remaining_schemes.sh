#!/bin/bash

cd /home/shiwenlong/NERlabs/eznlp

PYTHON_BIN="/home/shiwenlong/miniconda3/envs/eznlp11/bin/python"

# 公共参数
COMMON_ARGS="--data_dir data/RedJujube \
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

echo "启动方案B: 加权求和融合..."
nohup $PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS \
    --save_dir cache/redjujube_softlexicon_expert_weighted \
    --run_softlexicon_expert_weighted \
    > logs/scheme_B_weighted_new.log 2>&1 &
echo "方案B已启动，PID: $!"

sleep 20

echo "启动方案C: 门控机制融合..."
nohup $PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS \
    --save_dir cache/redjujube_softlexicon_expert_gated \
    --run_softlexicon_expert_gated \
    > logs/scheme_C_gated_new.log 2>&1 &
echo "方案C已启动，PID: $!"

sleep 20

echo "启动方案D: 注意力融合..."
nohup $PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS \
    --save_dir cache/redjujube_softlexicon_expert_attention \
    --run_softlexicon_expert_attention \
    > logs/scheme_D_attention_new.log 2>&1 &
echo "方案D已启动，PID: $!"

echo "所有方案已启动完成！"
