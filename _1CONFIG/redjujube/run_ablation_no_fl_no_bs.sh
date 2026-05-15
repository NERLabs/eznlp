#!/bin/bash
# 消融实验 #2 + #3：补全消融表
# #2: CA + BS + CE       (w/o Focal Loss) - 6 个 run × 3 种子 = 3 run
# #3: CA + CRF           (w/o BS-Decoder) - CRF 不能接 FL，自然不带 FL - 3 run
# 共 6 run × ~13 min ≈ 80 min
set -e

cd /home/shiwenlong/NERlabs/eznlp
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eznlp11

DATA_DIR="_2DATA/RedJujube"
BERT_ARCH="hfl/chinese-macbert-base"
EXPERT_DICT="_2DATA/RedJujube/expert_lexicon_auto_min1.txt"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results_newdata"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] ===== 消融实验 #2 + #3 启动（6 个 run，约 80 min）====="

# ===== #2: w/o Focal Loss (CA + BS + CE) =====
# 与 Q_bs_focal_attnv1 唯一差异: --fl_gamma 0
COMMON_2="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 \
          --sb_epsilon 0.1 --sb_size 2 --no_fgm --no_ema --fl_gamma 0.0 \
          --use_channel_attention --channel_attn_version v1 \
          --channel_attn_heads 4 --channel_attn_dropout 0.1"

for SEED in 42 43 44; do
    echo "[$(ts)] >>> #2 w/o FL  seed=$SEED  (CA + BS + CE)"
    python -u _5TRAIN/train_redjujube_expert_boundary.py $COMMON_2 \
        --save_dir ${BASE_SAVE_DIR}/Q_bs_ce_attnv1_s${SEED} \
        --seed ${SEED}
    echo "[$(ts)] <<< #2 s${SEED} 完成"
done

# ===== #3: w/o BS-Decoder (CA + CRF) =====
# 用 train_redjujube_ner.py（CRF 解码），需显式传 --expert_dict_path
# FFN 对齐: --encoder_arch FFN --hid_dim 150 --num_layers 1，与 #1 BS 路线编码层完全一致
# 关闭 FGM/EMA 以与 #1 (--no_fgm --no_ema) 对齐，保证唯一差异 = 解码器 (CRF vs BS-FL)
COMMON_3="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 \
          --expert_dict_path $EXPERT_DICT \
          --use_channel_attention --channel_attn_version v1 \
          --channel_attn_heads 4 --channel_attn_dropout 0.1 \
          --encoder_arch FFN --hid_dim 150 --num_layers 1 \
          --no_fgm --no_ema"

for SEED in 42 43 44; do
    echo "[$(ts)] >>> #3 w/o BS  seed=$SEED  (CA + CRF, FFN-aligned)"
    python -u _5TRAIN/train_redjujube_ner.py $COMMON_3 \
        --save_dir ${BASE_SAVE_DIR}/Q_crf_attnv1_ffn_s${SEED} \
        --seed ${SEED}
    echo "[$(ts)] <<< #3 s${SEED} 完成"
done

echo "[$(ts)] ===== 消融实验 #2 + #3 全部完成 ====="
