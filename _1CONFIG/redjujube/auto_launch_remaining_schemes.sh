#!/bin/bash

# 自动监控方案A完成并启动方案B和方案C的脚本

WORK_DIR="/home/shiwenlong/NERlabs/eznlp"
SCHEME_A_LOG="$WORK_DIR/cache/redjujube_softlexicon_expert/softlexicon_expert_concat_20251213-172348/training.log"
CONDA_ENV="eznlp11"

cd "$WORK_DIR"

echo "=========================================="
echo "⏳ 等待方案A (Concat) 完成..."
echo "=========================================="
echo ""

# 监控方案A是否完成
while true; do
    # 检查方案A进程是否还在运行
    if ! ps -p 1018542 > /dev/null 2>&1; then
        echo "✅ 方案A进程已结束"
        break
    fi
    
    # 检查日志中是否有训练完成的标志
    if grep -q "Training completed" "$SCHEME_A_LOG" 2>/dev/null; then
        echo "✅ 方案A训练已完成"
        sleep 5  # 等待进程清理
        break
    fi
    
    # 显示当前进度
    CURRENT_PROGRESS=$(tail -1 "$SCHEME_A_LOG" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] 方案A进度: $CURRENT_PROGRESS"
    
    sleep 30  # 每30秒检查一次
done

echo ""
echo "=========================================="
echo "🚀 启动方案B (Weighted)..."
echo "=========================================="
echo ""

# 启动方案B - 加权求和融合（控制台输出）
conda run -n "$CONDA_ENV" python scripts/train_redjujube_ner_comparison.py \
  --data_dir data/RedJujube \
  --run_softlexicon_expert_weighted \
  --save_dir cache/redjujube_ner_comparison \
  --expert_dict_auto_path data/RedJujube/expert_lexicon_auto.txt \
  --softlex_train_path data/RedJujube/softlexicon_train.txt \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 256 --num_layers 1 --dropout 0.5 \
  --num_epochs 30 --batch_size 16 \
  --lr 2e-3 --finetune_lr 2e-5 \
  --weight_decay 1e-4 --grad_clip 5.0 --seed 42

SCHEME_B_EXIT_CODE=$?

if [ $SCHEME_B_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 方案B训练完成"
    echo "=========================================="
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 方案B训练失败 (退出码: $SCHEME_B_EXIT_CODE)"
    echo "=========================================="
    echo ""
    exit 1
fi

echo ""
echo "=========================================="
echo "🚀 启动方案C (Gated)..."
echo "=========================================="
echo ""

# 启动方案C - 门控融合（控制台输出）
conda run -n "$CONDA_ENV" python scripts/train_redjujube_ner_comparison.py \
  --data_dir data/RedJujube \
  --run_softlexicon_expert_gated \
  --save_dir cache/redjujube_ner_comparison \
  --expert_dict_auto_path data/RedJujube/expert_lexicon_auto.txt \
  --softlex_train_path data/RedJujube/softlexicon_train.txt \
  --bert_arch hfl/chinese-macbert-base \
  --hid_dim 256 --num_layers 1 --dropout 0.5 \
  --num_epochs 30 --batch_size 16 \
  --lr 2e-3 --finetune_lr 2e-5 \
  --weight_decay 1e-4 --grad_clip 5.0 --seed 42

SCHEME_C_EXIT_CODE=$?

if [ $SCHEME_C_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 方案C训练完成"
    echo "=========================================="
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 方案C训练失败 (退出码: $SCHEME_C_EXIT_CODE)"
    echo "=========================================="
    echo ""
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 所有方案训练完成！"
echo "=========================================="
echo ""
echo "方案A (Concat): cache/redjujube_softlexicon_expert/softlexicon_expert_concat_20251213-172348/"
echo "方案D (Attention): cache/redjujube_ner_comparison/softlexicon_expert_attention_20251213-175441/"
echo "方案B (Weighted): 查看最新目录"
echo "方案C (Gated): 查看最新目录"
echo ""
