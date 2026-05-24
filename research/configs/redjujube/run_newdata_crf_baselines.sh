#!/bin/bash
# =============================================================================
# RedJujube NER 新数据集 完整消融实验脚本（BERT系列）
# =============================================================================
# 说明：三个创新点（词典特征、边界选择BS、Focal Loss）的完整消融矩阵
# 数据：datasets/raw/RedJujube（新分层划分的数据）
# 
# 本脚本仅包含BERT系列实验（可与G' BiLSTM并行）：
#   CRF_nodict - 纯CRF基线（无词典）
#   A' - BERT+LSTM+CRF（标准CRF基线，带专家词典）
#   BS_nodict - BS解码器无词典
#   BS_focal_nodict - BS+Focal无词典
# 
# G' BiLSTM-CRF 由 run_newdata_bilstm.sh 独立并行运行
# =============================================================================
set -e

# =============================================================================
# 基础配置
# =============================================================================
DATA_DIR="datasets/raw/RedJujube"
EXPERT_DICT="datasets/raw/RedJujube/expert_lexicon_auto_min1.txt"
BERT_ARCH="hfl/chinese-macbert-base"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results_newdata"

SEEDS=(42 43 44)

echo "============================================================"
echo "RedJujube NER 新数据集 CRF 基线实验"
echo "============================================================"
echo "数据目录: $DATA_DIR（分层划分后的新数据）"
echo "专家词典: $EXPERT_DICT"
echo "BERT模型: $BERT_ARCH"
echo "保存目录: $BASE_SAVE_DIR"
echo "种子列表: ${SEEDS[*]}"
echo "============================================================"
echo ""
echo "实验组（BERT系列，与G' BiLSTM并行）："
echo "  CRF_nodict - 纯CRF基线（无词典）"
echo "  A' - BERT+LSTM+CRF（标准CRF基线，带专家词典）"
echo "  BS_nodict - BS解码器无词典"
echo "  BS_focal_nodict - BS+Focal无词典"
echo "============================================================"

mkdir -p ${BASE_SAVE_DIR}

# G' BiLSTM-CRF 已移至 run_newdata_bilstm.sh 独立并行运行

# =============================================================================
# 实验 CRF_nodict: 纯CRF基线（无词典）
# BERT+LSTM+CRF，不使用专家词典特征
# =============================================================================
echo ""
echo "============================================================"
echo "实验 CRF_nodict: 纯CRF基线（无词典）"
echo "参数: epochs=30, batch_size=16, no_fgm, no_ema, bmes_aux=0, 无词典"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[CRF_nodict] 运行 seed=$SEED ..."
    conda run -n eznlp11 python research/training/train_redjujube_ner.py \
        --data_dir $DATA_DIR \
        --bert_arch $BERT_ARCH \
        --save_dir ${BASE_SAVE_DIR}/CRF_nodict \
        --seed $SEED \
        --num_epochs 30 --batch_size 16 \
        --no_fgm --no_ema \
        --bmes_aux_lambda 0 --bmes_label_aux_lambda 0
    echo "[CRF_nodict] seed=$SEED 完成"
done
echo "[CRF_nodict] 实验 CRF_nodict 全部完成"

# =============================================================================
# 实验 A': BERT+LSTM+CRF（标准CRF基线，带专家词典）
# 标准 BERT-based CRF 基线，关闭 FGM/EMA 和辅助任务
# =============================================================================
echo ""
echo "============================================================"
echo "实验 A': BERT+LSTM+CRF（标准CRF基线，带专家词典）"
echo "参数: epochs=30, batch_size=16, no_fgm, no_ema, bmes_aux=0"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[A'] 运行 seed=$SEED ..."
    conda run -n eznlp11 python research/training/train_redjujube_ner.py \
        --data_dir $DATA_DIR \
        --expert_dict_path $EXPERT_DICT \
        --bert_arch $BERT_ARCH \
        --save_dir ${BASE_SAVE_DIR}/A_baseline \
        --seed $SEED \
        --num_epochs 30 --batch_size 16 \
        --no_fgm --no_ema \
        --bmes_aux_lambda 0 --bmes_label_aux_lambda 0
    echo "[A'] seed=$SEED 完成"
done
echo "[A'] 实验 A' 全部完成"

# =============================================================================
# 实验 BS_nodict: BS解码器无词典
# 边界选择解码器，不使用专家词典特征
# =============================================================================
echo ""
echo "============================================================"
echo "实验 BS_nodict: BS解码器无词典"
echo "参数: epochs=30, batch_size=16, sb_epsilon=0.1, sb_size=2, 无词典"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[BS_nodict] 运行 seed=$SEED ..."
    conda run -n eznlp11 python research/training/train_redjujube_expert_boundary.py \
        --data_dir $DATA_DIR \
        --bert_arch $BERT_ARCH \
        --num_epochs 30 --batch_size 16 \
        --sb_epsilon 0.1 --sb_size 2 \
        --no_fgm --no_ema \
        --no_expert_dict \
        --save_dir ${BASE_SAVE_DIR}/BS_nodict \
        --seed $SEED
    echo "[BS_nodict] seed=$SEED 完成"
done
echo "[BS_nodict] 实验 BS_nodict 全部完成"

# =============================================================================
# 实验 BS_focal_nodict: BS+Focal无词典
# 边界选择解码器 + Focal Loss，不使用专家词典特征
# =============================================================================
echo ""
echo "============================================================"
echo "实验 BS_focal_nodict: BS+Focal无词典"
echo "参数: epochs=30, batch_size=16, sb_epsilon=0.1, sb_size=2, fl_gamma=2.0, 无词典"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[BS_focal_nodict] 运行 seed=$SEED ..."
    conda run -n eznlp11 python research/training/train_redjujube_expert_boundary.py \
        --data_dir $DATA_DIR \
        --bert_arch $BERT_ARCH \
        --num_epochs 30 --batch_size 16 \
        --sb_epsilon 0.1 --sb_size 2 \
        --no_fgm --no_ema \
        --no_expert_dict \
        --fl_gamma 2.0 \
        --save_dir ${BASE_SAVE_DIR}/BS_focal_nodict \
        --seed $SEED
    echo "[BS_focal_nodict] seed=$SEED 完成"
done
echo "[BS_focal_nodict] 实验 BS_focal_nodict 全部完成"

# =============================================================================
# 实验完成总结
# =============================================================================
echo ""
echo "============================================================"
echo "新数据集完整消融实验全部完成！"
echo "============================================================"
echo ""
echo "实验组汇总："
echo "  G' - BiLSTM-CRF（无BERT）  : ${BASE_SAVE_DIR}/G_bilstm_baseline （独立并行脚本）"
echo "  CRF_nodict - 纯CRF（无词典）: ${BASE_SAVE_DIR}/CRF_nodict"
echo "  A' - BERT+LSTM+CRF（带词典）: ${BASE_SAVE_DIR}/A_baseline"
echo "  BS_nodict - BS（无词典）    : ${BASE_SAVE_DIR}/BS_nodict"
echo "  BS_focal_nodict - BS+Focal  : ${BASE_SAVE_DIR}/BS_focal_nodict"
echo ""
echo "使用以下命令收集结果："
echo "python research/evaluation/collect_optimization_results.py --results_dir ${BASE_SAVE_DIR} --detailed"
echo "============================================================"
