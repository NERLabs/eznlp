#!/bin/bash
# ============================================================
# RedJujube NER 优化实验矩阵
# 每组实验运行 3 个种子 (42, 43, 44)
# ============================================================
# 作者: Qoder
# 日期: 2026-03-18
# 
# 实验组说明:
#   A - 基线复现（禁用FGM/EMA）
#   B - 仅 FGM 对抗训练
#   C - FGM + EMA
#   D - FGM + EMA + 数据增强
#   E - FGM + EMA + R-Drop
#   F - 全部优化（FGM + EMA + 数据增强 + R-Drop）
#   G - 无BERT基线（BiLSTM-CRF）
# ============================================================

set -e  # 出错时停止执行

# 配置路径（相对于项目根目录）
DATA_DIR="_2DATA/RedJujube"
DICT_PATH="_2DATA/RedJujube/expert_lexicon_auto_min1.txt"
BERT_ARCH="hfl/chinese-macbert-base"
BASE_SAVE_DIR="experiments/EXP-010-optimization/results"

# 公共参数
COMMON_ARGS="--data_dir $DATA_DIR --bert_arch $BERT_ARCH --num_epochs 30 --batch_size 16"

# 种子列表
SEEDS=(42 43 44)

# 确保保存目录存在
mkdir -p "$BASE_SAVE_DIR"

echo "============================================================"
echo "RedJujube NER 优化实验矩阵"
echo "============================================================"
echo "数据目录: $DATA_DIR"
echo "词典路径: $DICT_PATH"
echo "BERT模型: $BERT_ARCH"
echo "保存目录: $BASE_SAVE_DIR"
echo "种子列表: ${SEEDS[*]}"
echo "============================================================"
echo ""

# ============================================================
# 实验 A: 基线复现（无优化，禁用FGM/EMA）
# ============================================================
echo "============================================================"
echo "实验 A: 基线复现（禁用 FGM/EMA）"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[A] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_ner.py \
        $COMMON_ARGS \
        --expert_dict_path "$DICT_PATH" \
        --save_dir "${BASE_SAVE_DIR}/A_baseline" \
        --seed "$SEED" \
        --no_fgm --no_ema \
        --bmes_aux_lambda 0 --bmes_label_aux_lambda 0
    echo "[A] seed=$SEED 完成"
    echo ""
done

# ============================================================
# 实验 B: 仅 FGM 对抗训练
# ============================================================
echo "============================================================"
echo "实验 B: 仅 FGM 对抗训练"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[B] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_ner.py \
        $COMMON_ARGS \
        --expert_dict_path "$DICT_PATH" \
        --save_dir "${BASE_SAVE_DIR}/B_fgm" \
        --seed "$SEED" \
        --use_fgm --fgm_epsilon 1.0 \
        --no_ema \
        --bmes_aux_lambda 0 --bmes_label_aux_lambda 0
    echo "[B] seed=$SEED 完成"
    echo ""
done

# ============================================================
# 实验 C: FGM + EMA
# ============================================================
echo "============================================================"
echo "实验 C: FGM + EMA"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[C] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_ner.py \
        $COMMON_ARGS \
        --expert_dict_path "$DICT_PATH" \
        --save_dir "${BASE_SAVE_DIR}/C_fgm_ema" \
        --seed "$SEED" \
        --use_fgm --fgm_epsilon 1.0 \
        --use_ema --ema_decay 0.999 \
        --bmes_aux_lambda 0 --bmes_label_aux_lambda 0
    echo "[C] seed=$SEED 完成"
    echo ""
done

# ============================================================
# 实验 D: FGM + EMA + 数据增强
# ============================================================
echo "============================================================"
echo "实验 D: FGM + EMA + 数据增强"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[D] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_ner.py \
        $COMMON_ARGS \
        --expert_dict_path "$DICT_PATH" \
        --save_dir "${BASE_SAVE_DIR}/D_fgm_ema_aug" \
        --seed "$SEED" \
        --use_fgm --fgm_epsilon 1.0 \
        --use_ema --ema_decay 0.999 \
        --use_augment --aug_ratio 2 --aug_prob 0.3 \
        --bmes_aux_lambda 0 --bmes_label_aux_lambda 0
    echo "[D] seed=$SEED 完成"
    echo ""
done

# ============================================================
# 实验 E: FGM + EMA + R-Drop
# ============================================================
echo "============================================================"
echo "实验 E: FGM + EMA + R-Drop"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[E] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_ner.py \
        $COMMON_ARGS \
        --expert_dict_path "$DICT_PATH" \
        --save_dir "${BASE_SAVE_DIR}/E_fgm_ema_rdrop" \
        --seed "$SEED" \
        --use_fgm --fgm_epsilon 1.0 \
        --use_ema --ema_decay 0.999 \
        --use_rdrop --rdrop_alpha 0.5 \
        --bmes_aux_lambda 0 --bmes_label_aux_lambda 0
    echo "[E] seed=$SEED 完成"
    echo ""
done

# ============================================================
# 实验 F: 全部优化（FGM + EMA + 数据增强 + R-Drop）
# ============================================================
echo "============================================================"
echo "实验 F: 全部优化（FGM + EMA + 数据增强 + R-Drop）"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[F] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_ner.py \
        $COMMON_ARGS \
        --expert_dict_path "$DICT_PATH" \
        --save_dir "${BASE_SAVE_DIR}/F_all_optimizations" \
        --seed "$SEED" \
        --use_fgm --fgm_epsilon 1.0 \
        --use_ema --ema_decay 0.999 \
        --use_rdrop --rdrop_alpha 0.5 \
        --use_augment --aug_ratio 2 --aug_prob 0.3 \
        --bmes_aux_lambda 0 --bmes_label_aux_lambda 0
    echo "[F] seed=$SEED 完成"
    echo ""
done

# ============================================================
# 实验 G: 无BERT基线（BiLSTM-CRF）
# 注意：train_redjujube_bilstm_crf.py 不支持 FGM/EMA 参数
# ============================================================
echo "============================================================"
echo "实验 G: 无BERT基线（BiLSTM-CRF）"
echo "============================================================"
for SEED in "${SEEDS[@]}"; do
    echo "[G] 运行 seed=$SEED ..."
    python _5TRAIN/train_redjujube_bilstm_crf.py \
        --data_dir "$DATA_DIR" \
        --save_dir "${BASE_SAVE_DIR}/G_bilstm_baseline" \
        --seed "$SEED" \
        --num_epochs 50 --batch_size 16 \
        --emb_dim 100 --hid_dim 256 --num_layers 2
    echo "[G] seed=$SEED 完成"
    echo ""
done

# ============================================================
# 实验完成
# ============================================================
echo ""
echo "============================================================"
echo "🎉 所有实验完成！"
echo "============================================================"
echo "结果保存在: $BASE_SAVE_DIR"
echo ""
echo "实验汇总:"
echo "  A_baseline          - 基线（禁用FGM/EMA）"
echo "  B_fgm               - 仅FGM对抗训练"
echo "  C_fgm_ema           - FGM + EMA"
echo "  D_fgm_ema_aug       - FGM + EMA + 数据增强"
echo "  E_fgm_ema_rdrop     - FGM + EMA + R-Drop"
echo "  F_all_optimizations - 全部优化"
echo "  G_bilstm_baseline   - BiLSTM-CRF基线"
echo ""
echo "运行结果收集脚本:"
echo "  python _6EVALUATE/collect_optimization_results.py"
echo "============================================================"
