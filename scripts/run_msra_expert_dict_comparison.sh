#!/bin/bash
# MSRA NER 对比实验：Baseline vs 自动专家词典 (ExpertDict)

set -e

DATASET=msra
LANG=chinese
BERT_ARCH=hfl/chinese-macbert-base
SAVE_ROOT=cache/msra_ner_expert_dict

mkdir -p "${SAVE_ROOT}"

echo "======================================================================"
echo "Step 1: 从 MSRA 训练集自动抽取专家词典"
echo "======================================================================"
python scripts/extract_lexicon_from_training.py \
    --train_path data/MSRA/train.char.bmes \
    --output_path data/MSRA/expert_lexicon_auto.txt \
    --min_freq 2

echo

echo "======================================================================"
echo "Step 2: Baseline (不使用专家词典)"
echo "======================================================================"
python scripts/entity_recognition.py \
    --dataset ${DATASET} \
    --language ${LANG} \
    --ck_decoder sequence_tagging \
    --scheme BIOES \
    --bert_arch ${BERT_ARCH} \
    --enc_arch LSTM \
    --hid_dim 256 \
    --num_layers 1 \
    --batch_size 16 \
    --num_epochs 30 \
    --lr 2e-3 \
    --finetune_lr 2e-5 \
    --save_dir ${SAVE_ROOT}/baseline

echo

echo "======================================================================"
echo "Step 3: +ExpertDict (使用自动抽取的专家词典特征)"
echo "======================================================================"
python scripts/entity_recognition.py \
    --dataset ${DATASET} \
    --language ${LANG} \
    --ck_decoder sequence_tagging \
    --scheme BIOES \
    --bert_arch ${BERT_ARCH} \
    --enc_arch LSTM \
    --hid_dim 256 \
    --num_layers 1 \
    --batch_size 16 \
    --num_epochs 30 \
    --lr 2e-3 \
    --finetune_lr 2e-5 \
    --save_dir ${SAVE_ROOT}/with_expert_dict \
    --use_expert_dict \
    --expert_dict_path data/MSRA/expert_lexicon_auto.txt \
    --expert_dict_dim 50

echo

echo "======================================================================"
echo "对比实验完成！"
echo "Baseline 结果目录:      ${SAVE_ROOT}/baseline"
echo "+ExpertDict 结果目录:   ${SAVE_ROOT}/with_expert_dict"
echo "======================================================================"