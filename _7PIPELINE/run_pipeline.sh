#!/bin/bash
# 实验流水线快速启动脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认配置
WORK_DIR="/home/shiwenlong/NERlabs/eznlp"
PIPELINE_SCRIPT="_7PIPELINE/experiment_pipeline.py"

# 邮件配置（可选）
EMAIL_SENDER=""
EMAIL_RECEIVER=""
EMAIL_PASSWORD=""
SMTP_SERVER="smtp.163.com"
SMTP_PORT="465"

# 打印帮助信息
print_help() {
    cat << EOF
实验自动化流水线快速启动脚本

用法:
    bash $0 --name <实验名称> --config <配置文件> [选项]

必需参数:
    --name NAME              实验名称
    --config PATH            实验配置文件路径

可选参数:
    --skip-test              跳过测试运行阶段
    --monitor-interval SEC   监控间隔（秒），默认300
    --work-dir PATH          工作目录，默认自动生成
    
邮件通知参数（可选）:
    --email-sender EMAIL     发件人邮箱
    --email-receiver EMAIL   收件人邮箱
    --email-password PASS    SMTP密码/授权码
    --smtp-server SERVER     SMTP服务器，默认smtp.163.com
    --smtp-port PORT         SMTP端口，默认465

预设配置:
    --baseline               使用Baseline配置模板
    --softlexicon            使用SoftLexicon配置模板
    --expert-dict            使用ExpertDict配置模板

示例:
    # 使用自定义配置
    bash $0 --name "我的实验" --config my_config.json
    
    # 使用预设配置
    bash $0 --name "Baseline测试" --baseline
    
    # 完整配置（含邮件通知）
    bash $0 --name "实验" --config config.json \\
      --email-sender "your@163.com" \\
      --email-receiver "target@example.com" \\
      --email-password "auth_code"

EOF
}

# 解析参数
EXP_NAME=""
CONFIG_FILE=""
SKIP_TEST=""
MONITOR_INTERVAL="300"
CUSTOM_WORK_DIR=""
USE_PRESET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            EXP_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --skip-test)
            SKIP_TEST="--skip-test"
            shift
            ;;
        --monitor-interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        --work-dir)
            CUSTOM_WORK_DIR="$2"
            shift 2
            ;;
        --email-sender)
            EMAIL_SENDER="$2"
            shift 2
            ;;
        --email-receiver)
            EMAIL_RECEIVER="$2"
            shift 2
            ;;
        --email-password)
            EMAIL_PASSWORD="$2"
            shift 2
            ;;
        --smtp-server)
            SMTP_SERVER="$2"
            shift 2
            ;;
        --smtp-port)
            SMTP_PORT="$2"
            shift 2
            ;;
        --baseline)
            USE_PRESET="baseline"
            shift
            ;;
        --softlexicon)
            USE_PRESET="softlexicon"
            shift
            ;;
        --expert-dict)
            USE_PRESET="expert_dict"
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# 进入工作目录
cd "$WORK_DIR"

# 检查必需参数
if [ -z "$EXP_NAME" ]; then
    echo -e "${RED}错误: 必须指定实验名称 (--name)${NC}"
    print_help
    exit 1
fi

# 处理预设配置
if [ -n "$USE_PRESET" ]; then
    case $USE_PRESET in
        baseline)
            CONFIG_FILE="_7PIPELINE/config_templates/baseline_example.json"
            ;;
        softlexicon)
            CONFIG_FILE="_7PIPELINE/config_templates/softlexicon_example.json"
            ;;
        expert_dict)
            CONFIG_FILE="_7PIPELINE/config_templates/expert_dict_example.json"
            ;;
    esac
    echo -e "${YELLOW}使用预设配置: $CONFIG_FILE${NC}"
fi

# 检查配置文件
if [ -z "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 必须指定配置文件 (--config) 或使用预设配置${NC}"
    print_help
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

# 构建命令
CMD="python $PIPELINE_SCRIPT --name \"$EXP_NAME\" --config \"$CONFIG_FILE\""

# 添加可选参数
if [ -n "$SKIP_TEST" ]; then
    CMD="$CMD $SKIP_TEST"
fi

CMD="$CMD --monitor-interval $MONITOR_INTERVAL"

if [ -n "$CUSTOM_WORK_DIR" ]; then
    CMD="$CMD --work-dir \"$CUSTOM_WORK_DIR\""
fi

# 添加邮件配置
if [ -n "$EMAIL_SENDER" ] && [ -n "$EMAIL_RECEIVER" ] && [ -n "$EMAIL_PASSWORD" ]; then
    CMD="$CMD --email-sender \"$EMAIL_SENDER\""
    CMD="$CMD --email-receiver \"$EMAIL_RECEIVER\""
    CMD="$CMD --email-password \"$EMAIL_PASSWORD\""
    CMD="$CMD --smtp-server \"$SMTP_SERVER\""
    CMD="$CMD --smtp-port \"$SMTP_PORT\""
    echo -e "${GREEN}✓ 已启用邮件通知${NC}"
fi

# 显示配置摘要
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}实验自动化流水线${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "实验名称: ${YELLOW}$EXP_NAME${NC}"
echo -e "配置文件: ${YELLOW}$CONFIG_FILE${NC}"
echo -e "监控间隔: ${YELLOW}${MONITOR_INTERVAL}秒${NC}"
echo -e "跳过测试: ${YELLOW}$([ -n "$SKIP_TEST" ] && echo "是" || echo "否")${NC}"

if [ -n "$EMAIL_SENDER" ]; then
    echo -e "邮件通知: ${YELLOW}已启用 ($EMAIL_SENDER -> $EMAIL_RECEIVER)${NC}"
else
    echo -e "邮件通知: ${YELLOW}未启用${NC}"
fi

echo -e "${GREEN}========================================${NC}"
echo ""

# 确认启动
read -p "是否启动流水线? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}已取消${NC}"
    exit 0
fi

# 执行命令
echo -e "${GREEN}🚀 启动实验流水线...${NC}"
echo -e "${YELLOW}执行命令:${NC}"
echo "$CMD"
echo ""

eval $CMD

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ 流水线执行成功!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ 流水线执行失败!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
