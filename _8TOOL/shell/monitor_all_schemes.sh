#!/bin/bash

# 监控所有4个方案的训练进度

echo "=========================================="
echo "4个融合方案实时监控"
echo "=========================================="
echo ""

# 检查GPU使用情况
echo "【GPU资源使用】"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""

# 检查所有训练进程
echo "【训练进程状态】"
ps aux | grep "train_redjujube_ner_comparison.py" | grep -v grep | awk '{print "PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "运行时间:", $10}'
echo ""

# 检查日志文件最新进度
echo "【训练进度】"
echo ""

# 方案A
if [ -d "cache/redjujube_softlexicon_expert" ]; then
    echo "方案A (Concat) - 直接拼接融合:"
    LATEST_LOG=$(ls -t cache/redjujube_softlexicon_expert/log.txt 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        tail -20 "$LATEST_LOG" | grep -E "(Epoch|Test F1)" | tail -5
    else
        echo "  未找到日志文件"
    fi
    echo ""
fi

# 方案B
if [ -d "cache/redjujube_softlexicon_expert_weighted" ]; then
    echo "方案B (Weighted) - 加权求和融合:"
    LATEST_LOG=$(ls -t cache/redjujube_softlexicon_expert_weighted/log.txt 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        tail -20 "$LATEST_LOG" | grep -E "(Epoch|Test F1)" | tail -5
    else
        echo "  未找到日志文件"
    fi
    echo ""
fi

# 方案C
if [ -d "cache/redjujube_softlexicon_expert_gated" ]; then
    echo "方案C (Gated) - 门控机制融合:"
    LATEST_LOG=$(ls -t cache/redjujube_softlexicon_expert_gated/log.txt 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        tail -20 "$LATEST_LOG" | grep -E "(Epoch|Test F1)" | tail -5
    else
        echo "  未找到日志文件"
    fi
    echo ""
fi

# 方案D
if [ -d "cache/redjujube_softlexicon_expert_attention" ]; then
    echo "方案D (Attention) - 注意力融合:"
    LATEST_LOG=$(ls -t cache/redjujube_softlexicon_expert_attention/log.txt 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        tail -20 "$LATEST_LOG" | grep -E "(Epoch|Test F1)" | tail -5
    else
        echo "  未找到日志文件"
    fi
    echo ""
fi

echo "=========================================="
echo "监控脚本说明："
echo "- 每隔1分钟运行: watch -n 60 bash scripts/monitor_all_schemes.sh"
echo "- 查看实时日志: tail -f logs/scheme_*.log"
echo "=========================================="
