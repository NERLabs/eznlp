#!/bin/bash

# 并行执行策略：同时运行方案B、C、D
# 方案A已经在运行中，我们充分利用剩余的17GB显存

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit

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

echo "=========================================="
echo "并行执行策略启动"
echo "=========================================="
echo "方案A：已在运行（PID 1018542）"
echo "现在同时启动方案B、C、D"
echo "预计1-2小时内完成所有实验"
echo "=========================================="

# 方案B: 加权求和融合 - 后台运行
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动方案B: 加权求和融合..."
nohup $PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS \
    --save_dir cache/redjujube_softlexicon_expert_weighted \
    --run_softlexicon_expert_weighted \
    > logs/scheme_B_weighted_$(date +%Y%m%d_%H%M%S).log 2>&1 &
SCHEME_B_PID=$!
echo "方案B已启动，PID: $SCHEME_B_PID"
sleep 30  # 等待30秒让方案B加载模型

# 方案C: 门控机制融合 - 后台运行
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动方案C: 门控机制融合..."
nohup $PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS \
    --save_dir cache/redjujube_softlexicon_expert_gated \
    --run_softlexicon_expert_gated \
    > logs/scheme_C_gated_$(date +%Y%m%d_%H%M%S).log 2>&1 &
SCHEME_C_PID=$!
echo "方案C已启动，PID: $SCHEME_C_PID"
sleep 30  # 等待30秒让方案C加载模型

# 方案D: 注意力融合 - 后台运行
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动方案D: 注意力融合..."
nohup $PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS \
    --save_dir cache/redjujube_softlexicon_expert_attention \
    --run_softlexicon_expert_attention \
    > logs/scheme_D_attention_$(date +%Y%m%d_%H%M%S).log 2>&1 &
SCHEME_D_PID=$!
echo "方案D已启动，PID: $SCHEME_D_PID"

echo ""
echo "=========================================="
echo "所有方案已启动！"
echo "=========================================="
echo "方案A (Concat)    - PID: 1018542 (已运行中)"
echo "方案B (Weighted)  - PID: $SCHEME_B_PID"
echo "方案C (Gated)     - PID: $SCHEME_C_PID"
echo "方案D (Attention) - PID: $SCHEME_D_PID"
echo "=========================================="
echo "日志位置: logs/"
echo "查看GPU使用: nvidia-smi"
echo "查看进度: tail -f logs/scheme_*.log"
echo "=========================================="
