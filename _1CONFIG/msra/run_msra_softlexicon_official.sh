#!/bin/bash
# MSRA数据集 - 使用官方脚本运行SoftLexicon实验
# 结构: MacBERT + BiLSTM + CRF + SoftLexicon (与自定义脚本一致)

set -e

cd /home/shiwenlong/NERlabs/eznlp

# 设置Python路径，确保可以导入_8TOOL模块
export PYTHONPATH="/home/shiwenlong/NERlabs/eznlp:$PYTHONPATH"

echo "=========================================="
echo "MSRA SoftLexicon - 官方脚本"
echo "开始时间: $(date)"
echo "模型结构: MacBERT + BiLSTM + CRF + SoftLexicon"
echo "=========================================="
echo ""

# 参数说明:
# --dataset MSRA           - 使用MSRA数据集
# --bert_arch MacBERT_base - 使用MacBERT预训练模型
# --use_softlexicon        - 启用SoftLexicon特征
# --ck_decoder sequence_tagging - 序列标注解码器
# --enc_arch LSTM          - 使用LSTM编码器
# --hid_dim 256            - LSTM隐藏层维度256 (与自定义脚本一致)
# --num_layers 1           - LSTM层数1 (与自定义脚本一致)
# --batch_size 16          - 批次大小16 (与自定义脚本一致)
# --num_epochs 30          - 训练30轮 (与自定义脚本一致)
# --lr 2e-3                - 学习率0.002 (与自定义脚本一致)
# --finetune_lr 2e-5       - BERT微调学习率 (与自定义脚本一致)

python _5TRAIN/entity_recognition.py \
    --dataset MSRA \
    --bert_arch MacBERT_base \
    --use_softlexicon \
    --ck_decoder sequence_tagging \
    --enc_arch LSTM \
    --hid_dim 256 \
    --num_layers 1 \
    --batch_size 16 \
    --num_epochs 30 \
    --lr 2e-3 \
    --finetune_lr 2e-5 \
    --emb_dim 0

echo ""
echo "=========================================="
echo "训练完成！"
echo "结束时间: $(date)"
echo "=========================================="
