#!/bin/bash
# 批量运行多个实验流水线

set -e

WORK_DIR="/home/shiwenlong/NERlabs/eznlp"
cd "$WORK_DIR"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 邮件配置（可选）
EMAIL_SENDER="shiwenlongyes@163.com"
EMAIL_RECEIVER="shiwenlongyes@163.com"
EMAIL_PASSWORD="KNXRm35vHjizvjcS"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}批量实验流水线启动${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 定义要运行的实验
experiments=(
    "RedJujube_Baseline|research/pipelines/config_templates/baseline_example.json"
    "RedJujube_SoftLexicon|research/pipelines/config_templates/softlexicon_example.json"
    "RedJujube_ExpertDict|research/pipelines/config_templates/expert_dict_example.json"
)

# 存储进程PID
declare -a PIDS=()

# 启动所有实验
for exp in "${experiments[@]}"; do
    IFS='|' read -r name config <<< "$exp"
    
    echo -e "${YELLOW}启动实验: $name${NC}"
    
    # 构建命令
    cmd="python research/pipelines/experiment_pipeline.py \
        --name \"$name\" \
        --config \"$config\" \
        --skip-test \
        --monitor-interval 300"
    
    # 如果配置了邮件，添加邮件参数
    if [ -n "$EMAIL_SENDER" ] && [ -n "$EMAIL_PASSWORD" ]; then
        cmd="$cmd \
            --email-sender \"$EMAIL_SENDER\" \
            --email-receiver \"$EMAIL_RECEIVER\" \
            --email-password \"$EMAIL_PASSWORD\""
    fi
    
    # 后台运行
    eval "$cmd" > "pipeline_runs/${name}_output.log" 2>&1 &
    PIDS+=($!)
    
    echo -e "${GREEN}✓ 已启动 (PID: ${PIDS[-1]})${NC}"
    echo ""
    
    # 等待5秒避免资源冲突
    sleep 5
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验已启动${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "进程PID列表:"
for i in "${!PIDS[@]}"; do
    echo "  实验 $((i+1)): PID ${PIDS[$i]}"
done
echo ""
echo -e "${YELLOW}监控命令:${NC}"
echo "  ps -p $(IFS=,; echo "${PIDS[*]}") -o pid,cmd"
echo ""
echo -e "${YELLOW}停止所有实验:${NC}"
echo "  kill ${PIDS[@]}"
echo ""

# 等待所有进程完成（可选）
read -p "是否等待所有实验完成? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}等待实验完成...${NC}"
    
    for pid in "${PIDS[@]}"; do
        wait $pid
        echo -e "${GREEN}✓ 进程 $pid 已完成${NC}"
    done
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}🎉 所有实验已完成!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${YELLOW}后台运行中，使用 ps 命令监控进度${NC}"
fi
