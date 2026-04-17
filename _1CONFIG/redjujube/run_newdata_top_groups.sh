#!/bin/bash
# =============================================================================
# RedJujube NER 新数据集（分层划分）Top 5 实验组重跑脚本
# =============================================================================
# 说明：在分层划分后的新数据集上重跑超过基线的5组实验配置
# 数据：_2DATA/RedJujube（新分层划分的数据）
# 基线参考：Z' 组（原始BS基线）已在新数据上完成，F1=88.20%
# =============================================================================
set -e

# =============================================================================
# 基础配置
# =============================================================================
DATA_DIR="_2DATA/RedJujube"
BERT_ARCH="hfl/chinese-macbert-base"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results_newdata"

# 公共参数：epochs=30, batch_size=16, sb_epsilon=0.1, sb_size=2
COMMON_ARGS="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16 --sb_epsilon 0.1 --sb_size 2 --no_fgm --no_ema"

echo "============================================================"
echo "RedJujube NER 新数据集 Top 5 实验组重跑"
echo "============================================================"
echo "数据目录: $DATA_DIR（分层划分后的新数据）"
echo "BERT模型: $BERT_ARCH"
echo "BS参数: sb_epsilon=0.1, sb_size=2"
echo "保存目录: $BASE_SAVE_DIR"
echo "种子列表: 42 43 44"
echo "============================================================"
echo "注意：Z' 组（原始BS基线）已完成，F1=88.20%，作为对比基准"
echo "============================================================"

mkdir -p ${BASE_SAVE_DIR}

# =============================================================================
# 实验 H': BS 基线（无额外参数）
# 对应原 H 组，作为新数据集的基线
# =============================================================================
echo ""
echo "============================================================"
echo "实验 H': BS 基线"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[H'] 运行 seed=$SEED ..."
    conda run -n eznlp11 python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/H_bs_baseline \
        --seed $SEED
    echo "[H'] seed=$SEED 完成"
done
echo "[H'] 实验 H' 全部完成"

# =============================================================================
# 实验 Q': BS + Focal Loss (gamma=2.0)
# 对应原 Q 组，使用 Focal Loss 处理类别不平衡
# =============================================================================
echo ""
echo "============================================================"
echo "实验 Q': BS + Focal Loss (gamma=2.0)"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[Q'] 运行 seed=$SEED ..."
    conda run -n eznlp11 python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/Q_bs_focal \
        --seed $SEED \
        --fl_gamma 2.0
    echo "[Q'] seed=$SEED 完成"
done
echo "[Q'] 实验 Q' 全部完成"

# =============================================================================
# 实验 S': BS + SRG 门控
# 对应原 S 组，使用 Self-Rectified Gate 增强边界选择
# =============================================================================
echo ""
echo "============================================================"
echo "实验 S': BS + SRG 门控"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[S'] 运行 seed=$SEED ..."
    conda run -n eznlp11 python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/S_bs_srg \
        --seed $SEED \
        --use_srg --srg_hid_dim 128 --srg_dropout 0.2
    echo "[S'] seed=$SEED 完成"
done
echo "[S'] 实验 S' 全部完成"

# =============================================================================
# 实验 W': Enhanced Size Embedding
# 对应原 W 组，使用增强的 size embedding
# =============================================================================
echo ""
echo "============================================================"
echo "实验 W': Enhanced Size Embedding"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[W'] 运行 seed=$SEED ..."
    conda run -n eznlp11 python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/W_enhanced_size_emb \
        --seed $SEED \
        --enhanced_size_emb
    echo "[W'] seed=$SEED 完成"
done
echo "[W'] 实验 W' 全部完成"

# =============================================================================
# 实验 U': 自适应 sb_size（按实体类型分配）
# 对应原 U 组，长实体类型使用 sb_size=3，短实体使用 sb_size=1
# =============================================================================
echo ""
echo "============================================================"
echo "实验 U': 自适应 sb_size（按实体类型）"
echo "============================================================"
for SEED in 42 43 44; do
    echo "[U'] 运行 seed=$SEED ..."
    conda run -n eznlp11 python _5TRAIN/train_redjujube_expert_boundary.py \
        $COMMON_ARGS \
        --save_dir ${BASE_SAVE_DIR}/U_bs_adaptive_sbsize \
        --seed $SEED \
        --sb_size_map "DRU:3,LOC:3,FER:3,TAX:3,EQU:3,PER:1,PRO:1,PAR:1,WED:1"
    echo "[U'] seed=$SEED 完成"
done
echo "[U'] 实验 U' 全部完成"

# =============================================================================
# 实验完成总结
# =============================================================================
echo ""
echo "============================================================"
echo "新数据集 Top 5 实验组全部完成！"
echo "============================================================"
echo ""
echo "实验组汇总："
echo "  H' - BS 基线         : ${BASE_SAVE_DIR}/H_bs_baseline"
echo "  Q' - Focal Loss      : ${BASE_SAVE_DIR}/Q_bs_focal"
echo "  S' - SRG 门控        : ${BASE_SAVE_DIR}/S_bs_srg"
echo "  W' - Enhanced Size   : ${BASE_SAVE_DIR}/W_enhanced_size_emb"
echo "  U' - Adaptive sb_size: ${BASE_SAVE_DIR}/U_bs_adaptive_sbsize"
echo ""
echo "对比基准：Z' 组 F1=88.20%"
echo ""
echo "使用以下命令收集结果："
echo "python _6EVALUATE/collect_optimization_results.py --results_dir ${BASE_SAVE_DIR} --detailed"
echo "============================================================"
