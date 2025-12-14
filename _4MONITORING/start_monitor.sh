#!/bin/bash
# NFLAT训练监控启动脚本

# ============ 配置区 ============
# 邮箱配置
SENDER_EMAIL="shiwenlongyes@163.com"        # 发送者邮箱
SMTP_PASSWORD="KNXRm35vHjizvjcS"         # SMTP密码(已设置)
RECEIVER_EMAIL="shiwenlongyes@163.com"   # 接收者邮箱

# 监控配置
CHECK_INTERVAL=120  # 检查间隔(秒) 默认2分钟
# ================================

cd /home/shiwenlong/NERlabs/eznlp

# 查找NFLAT训练进程PID
NFLAT_PID=$(ps aux | grep "python main.py --dataset msra" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$NFLAT_PID" ]; then
    echo "[$(date)] 未找到NFLAT训练进程"
    echo "监控器将持续等待进程启动..."
    nohup python _4MONITORING/email_notifier.py \
        --email "$RECEIVER_EMAIL" \
        --sender "$SENDER_EMAIL" \
        --password "$SMTP_PASSWORD" \
        --interval $CHECK_INTERVAL \
        > _4MONITORING/monitor.log 2>&1 &
else
    echo "[$(date)] 找到NFLAT训练进程 PID: $NFLAT_PID"
    nohup python _4MONITORING/email_notifier.py \
        --pid $NFLAT_PID \
        --email "$RECEIVER_EMAIL" \
        --sender "$SENDER_EMAIL" \
        --password "$SMTP_PASSWORD" \
        --interval $CHECK_INTERVAL \
        > _4MONITORING/monitor.log 2>&1 &
fi

MONITOR_PID=$!
echo "[$(date)] 监控程序已启动 PID: $MONITOR_PID"
echo "日志文件: _4MONITORING/monitor.log"
echo ""
echo "查看监控日志: tail -f _4MONITORING/monitor.log"
echo "停止监控: kill $MONITOR_PID"
