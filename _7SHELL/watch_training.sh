#!/bin/bash
# 定时监控训练进度脚本

WORK_DIR="/home/shiwenlong/NERlabs/eznlp"
LOG_FILE="$WORK_DIR/training_watch.log"

cd "$WORK_DIR"

echo "======================================" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - 训练进度检查" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"

# 检查GPU状态
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader,nounits >> "$LOG_FILE" 2>&1

echo "" >> "$LOG_FILE"

# 检查运行中的训练进程
ps aux | grep "train_redjujube_ner_comparison.py" | grep -v grep >> "$LOG_FILE" 2>&1

echo "" >> "$LOG_FILE"

# 检查最新的训练日志
python3 scripts/monitor_training.py --once --log-lines 2 >> "$LOG_FILE" 2>&1

echo "" >> "$LOG_FILE"
echo "下次检查: 5分钟后" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
