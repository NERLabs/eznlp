#!/bin/bash

echo "========================================="
echo "Soft+Expert 融合方案训练监控"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo ""

echo "GPU 状态:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""

echo "运行中的训练进程:"
ps aux | grep -E "(concat|weighted|gated|attention)" | grep python | grep train_redjujube | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    time=$(echo $line | awk '{print $10}')
    
    if echo $line | grep -q "concat"; then
        scheme="方案A-Concat"
    elif echo $line | grep -q "weighted"; then
        scheme="方案B-Weighted"
    elif echo $line | grep -q "gated"; then
        scheme="方案C-Gated"
    elif echo $line | grep -q "attention"; then
        scheme="方案D-Attention"
    else
        scheme="Unknown"
    fi
    
    echo "  [$scheme] PID:$pid CPU:${cpu}% MEM:${mem}% 运行时间:$time"
done
echo ""

echo "最新训练日志（最后10行）:"
echo ""
for log in /home/shiwenlong/NERlabs/eznlp/logs/scheme_*_fixed.log /home/shiwenlong/NERlabs/eznlp/cache/redjujube_softlexicon_expert/*/training.log; do
    if [ -f "$log" ]; then
        scheme_name=$(basename $(dirname $log))
        if echo $log | grep -q "scheme_A\|concat"; then
            echo "--- 方案A (Concat) ---"
        elif echo $log | grep -q "scheme_B\|weighted"; then
            echo "--- 方案B (Weighted) ---"
        elif echo $log | grep -q "scheme_C\|gated"; then
            echo "--- 方案C (Gated) ---"
        elif echo $log | grep -q "scheme_D\|attention"; then
            echo "--- 方案D (Attention) ---"
        fi
        tail -5 $log 2>/dev/null | grep -E "epoch|INFO|训练|Loss|F1" | head -3
        echo ""
    fi
done
