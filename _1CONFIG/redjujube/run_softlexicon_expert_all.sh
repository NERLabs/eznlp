#!/bin/bash
# 运行所有四种融合方案的实验脚本

set -e  # 遇到错误立即退出

cd /home/shiwenlong/NERlabs/eznlp

echo "========================================="
echo "  RedJujube Soft+Expert 联合模型实验"
echo "  执行四种融合方案(A/B/C/D)"
echo "========================================="
echo ""

# 激活环境
source ~/.bashrc
conda activate eznlp11

# 公共参数
DATA_DIR="data/RedJujube"
SAVE_DIR="cache/redjujube_softlexicon_expert"
EXPERT_DICT_AUTO="data/RedJujube/expert_lexicon_auto.txt"
SOFTLEX_TRAIN="data/RedJujube/softlexicon_train.txt"
BERT_ARCH="hfl/chinese-macbert-base"
HID_DIM=256
NUM_LAYERS=1
DROPOUT=0.5
NUM_EPOCHS=30
BATCH_SIZE=16
LR=2e-3
FINETUNE_LR=2e-5
WEIGHT_DECAY=1e-4
GRAD_CLIP=5.0
SEED=42

# 方案A: 直接拼接
echo "[1/4] 运行方案A: 直接拼接融合..."
python scripts/train_redjujube_ner_comparison.py \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --expert_dict_auto_path "${EXPERT_DICT_AUTO}" \
  --softlex_train_path "${SOFTLEX_TRAIN}" \
  --bert_arch "${BERT_ARCH}" \
  --hid_dim ${HID_DIM} \
  --num_layers ${NUM_LAYERS} \
  --dropout ${DROPOUT} \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --finetune_lr ${FINETUNE_LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --grad_clip ${GRAD_CLIP} \
  --seed ${SEED} \
  --run_softlexicon_expert_concat

echo ""
echo "✅ 方案A完成!"
echo ""

# 方案B: 加权求和
echo "[2/4] 运行方案B: 加权求和融合..."
python scripts/train_redjujube_ner_comparison.py \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --expert_dict_auto_path "${EXPERT_DICT_AUTO}" \
  --softlex_train_path "${SOFTLEX_TRAIN}" \
  --bert_arch "${BERT_ARCH}" \
  --hid_dim ${HID_DIM} \
  --num_layers ${NUM_LAYERS} \
  --dropout ${DROPOUT} \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --finetune_lr ${FINETUNE_LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --grad_clip ${GRAD_CLIP} \
  --seed ${SEED} \
  --run_softlexicon_expert_weighted

echo ""
echo "✅ 方案B完成!"
echo ""

# 方案C: 门控机制
echo "[3/4] 运行方案C: 门控机制融合..."
python scripts/train_redjujube_ner_comparison.py \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --expert_dict_auto_path "${EXPERT_DICT_AUTO}" \
  --softlex_train_path "${SOFTLEX_TRAIN}" \
  --bert_arch "${BERT_ARCH}" \
  --hid_dim ${HID_DIM} \
  --num_layers ${NUM_LAYERS} \
  --dropout ${DROPOUT} \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --finetune_lr ${FINETUNE_LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --grad_clip ${GRAD_CLIP} \
  --seed ${SEED} \
  --run_softlexicon_expert_gated

echo ""
echo "✅ 方案C完成!"
echo ""

# 方案D: 注意力融合
echo "[4/4] 运行方案D: 注意力融合..."
python scripts/train_redjujube_ner_comparison.py \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --expert_dict_auto_path "${EXPERT_DICT_AUTO}" \
  --softlex_train_path "${SOFTLEX_TRAIN}" \
  --bert_arch "${BERT_ARCH}" \
  --hid_dim ${HID_DIM} \
  --num_layers ${NUM_LAYERS} \
  --dropout ${DROPOUT} \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --finetune_lr ${FINETUNE_LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --grad_clip ${GRAD_CLIP} \
  --seed ${SEED} \
  --run_softlexicon_expert_attention

echo ""
echo "✅ 方案D完成!"
echo ""

echo "========================================="
echo "  全部实验完成!"
echo "========================================="
echo ""
echo "结果保存在: ${SAVE_DIR}/"
echo ""
echo "查看结果:"
echo "  方案A (拼接):   ${SAVE_DIR}/softlexicon_expert_concat_*/results.json"
echo "  方案B (加权):   ${SAVE_DIR}/softlexicon_expert_weighted_*/results.json"
echo "  方案C (门控):   ${SAVE_DIR}/softlexicon_expert_gated_*/results.json"
echo "  方案D (注意力): ${SAVE_DIR}/softlexicon_expert_attention_*/results.json"
echo ""
