#!/bin/bash
# 快速查看训练状态

echo "================================"
echo "📊 快速状态检查"
echo "================================"
echo ""

# 检查GPU
echo "🖥️  GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1
echo ""

# 检查训练进程
echo "🔄 训练进程:"
ps aux | grep train_redjujube_ner_comparison | grep -v grep | wc -l | xargs -I {} echo "  运行中: {} 个"
echo ""

# 检查有效训练任务的最新进度
echo "📋 训练进度:"
echo ""

# 方案D
echo "🏆 方案D (Attention):"
tail -30 cache/redjujube_ner_comparison/softlexicon_expert_attention_20251213-175441/training.log 2>/dev/null | grep -E "Epoch:|Dev.*Metrics:" | tail -2
echo ""

# 方案A
echo "📌 方案A (Concat):"
tail -30 cache/redjujube_softlexicon_expert_concat/softlexicon_expert_concat_20251213-181422/training.log 2>/dev/null | grep -E "Epoch:|Dev.*Metrics:" | tail -2
echo ""

# 方案B
echo "📌 方案B (Weighted):"
tail -30 cache/redjujube_softlexicon_expert_weighted/softlexicon_expert_weighted_20251213-181422/training.log 2>/dev/null | grep -E "Epoch:|Dev.*Metrics:" | tail -2
echo ""

# 检查自动监视器
echo "🤖 自动监视器:"
if ps aux | grep auto_collect_when_complete | grep -v grep > /dev/null; then
    echo "  ✅ 运行中 (PID: $(ps aux | grep auto_collect_when_complete | grep -v grep | awk '{print $2}'))"
else
    echo "  ❌ 未运行"
fi
echo ""

echo "================================"
echo "💡 提示: 使用以下命令查看详细信息"
echo "   python scripts/monitor_training.py --once"
echo "================================"
