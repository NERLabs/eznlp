#!/bin/bash
# 在方案A完成后自动运行剩余方案(B/C/D)
# 使用nohup确保后台持续运行

set -e

cd /home/shiwenlong/NERlabs/eznlp

PYTHON_BIN="/home/shiwenlong/miniconda3/envs/eznlp11/bin/python"
LOG_DIR="cache/redjujube_softlexicon_expert/logs"
mkdir -p "$LOG_DIR"

COMMON_ARGS="--data_dir data/RedJujube \
  --save_dir cache/redjujube_softlexicon_expert \
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

echo "========================================================================"
echo "  等待方案A完成后运行剩余方案(B/C/D)"
echo "  开始时间: $(date)"
echo "========================================================================"
echo ""

# 等待方案A完成（检查results.json是否生成）
echo "等待方案A完成..."
while true; do
    if ls cache/redjujube_softlexicon_expert/softlexicon_expert_concat_*/results.json 1> /dev/null 2>&1; then
        echo "✅ 方案A已完成！"
        break
    fi
    echo "方案A仍在运行... $(date)"
    sleep 300  # 每5分钟检查一次
done

echo ""
echo "========================================================================"
echo "  开始运行剩余方案"
echo "========================================================================"
echo ""

# 方案B: 加权求和
echo "[1/3] 运行方案B: 加权求和融合..."
echo "开始时间: $(date)"
$PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS --run_softlexicon_expert_weighted 2>&1 | tee "$LOG_DIR/scheme_B.log"
echo "完成时间: $(date)"
echo "✅ 方案B完成!"
echo ""

# 方案C: 门控机制
echo "[2/3] 运行方案C: 门控机制融合..."
echo "开始时间: $(date)"
$PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS --run_softlexicon_expert_gated 2>&1 | tee "$LOG_DIR/scheme_C.log"
echo "完成时间: $(date)"
echo "✅ 方案C完成!"
echo ""

# 方案D: 注意力融合
echo "[3/3] 运行方案D: 注意力融合..."
echo "开始时间: $(date)"
$PYTHON_BIN scripts/train_redjujube_ner_comparison.py $COMMON_ARGS --run_softlexicon_expert_attention 2>&1 | tee "$LOG_DIR/scheme_D.log"
echo "完成时间: $(date)"
echo "✅ 方案D完成!"
echo ""

echo "========================================================================"
echo "  全部4个方案执行完成！"
echo "  结束时间: $(date)"
echo "========================================================================"
echo ""

# 汇总结果
echo "实验结果汇总:"
echo "----------------------------------------"
for scheme in concat weighted gated attention; do
    result_file=$(ls cache/redjujube_softlexicon_expert/softlexicon_expert_${scheme}_*/results.json 2>/dev/null | head -1)
    if [ -f "$result_file" ]; then
        echo "方案 $scheme: $result_file"
    else
        echo "方案 $scheme: 未找到结果文件"
    fi
done
echo ""
