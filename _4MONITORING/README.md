# 训练监控与邮件通知系统

自动监控NFLAT训练进程,训练完成后发送结果到指定邮箱。

## 📋 功能特性

- ✅ 自动检测训练进程启动/结束
- ✅ 提取最佳F1分数和超参数
- ✅ 生成精美的HTML格式邮件
- ✅ 性能对比表格(NFLAT vs 你的方法 vs FLAT论文)
- ✅ 后台运行,不影响训练
- ✅ 详细日志记录

## 🚀 快速开始

### 1. 获取QQ邮箱SMTP授权码

1. 登录QQ邮箱网页版
2. 设置 → 账户 → POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV服务
3. 开启"POP3/SMTP服务"
4. 生成授权码(16位字符)
5. **保存授权码**(这不是邮箱登录密码!)

### 2. 配置邮箱信息

编辑 `start_monitor.sh`:

```bash
SENDER_EMAIL="your_email@qq.com"        # 你的QQ邮箱
SMTP_PASSWORD="abcd1234efgh5678"        # 刚才获取的16位授权码
RECEIVER_EMAIL="target@example.com"     # 接收邮件的邮箱
```

### 3. 启动监控

```bash
cd /home/shiwenlong/NERlabs/eznlp
chmod +x _4MONITORING/start_monitor.sh
./_4MONITORING/start_monitor.sh
```

### 4. 查看监控日志

```bash
tail -f _4MONITORING/monitor.log
```

## 📧 邮件内容示例

邮件标题:
```
🎉 NFLAT训练完成 - F1: 0.9612
```

邮件内容包含:
- 📊 训练结果(最佳F1、epoch、时长)
- ⚙️ 超参数配置表
- 📁 输出文件路径
- 🔗 性能对比表(vs 你的方法 vs FLAT论文)

## 🔧 高级用法

### 手动指定PID

```bash
python _4MONITORING/email_notifier.py \
    --pid 1234567 \
    --email receiver@example.com \
    --sender your@qq.com \
    --password "your_auth_code"
```

### 自定义检查间隔

```bash
# 每5分钟检查一次
python _4MONITORING/email_notifier.py \
    --email receiver@example.com \
    --sender your@qq.com \
    --password "your_auth_code" \
    --interval 300
```

### 停止监控

```bash
# 查找监控进程
ps aux | grep email_notifier

# 停止
kill <PID>
```

## 📝 文件说明

```
_4MONITORING/
├── email_notifier.py      # 核心监控程序
├── start_monitor.sh       # 启动脚本
├── monitor.log           # 监控日志(自动生成)
└── README.md             # 本文档
```

## ⚠️ 注意事项

1. **SMTP授权码 ≠ 邮箱登录密码**
2. QQ邮箱SMTP服务器: `smtp.qq.com:465`
3. 首次发送可能需要验证,检查QQ邮箱安全中心
4. 建议使用专门的监控邮箱,避免泄露主邮箱授权码
5. 监控程序会自动后台运行,重启服务器后需重新启动

## 🐛 常见问题

### Q1: 提示"SMTPAuthenticationError"
A: 检查SMTP授权码是否正确,不是邮箱密码!

### Q2: 邮件发送失败
A: 
- 确认QQ邮箱已开启SMTP服务
- 检查授权码是否过期
- 尝试重新生成授权码

### Q3: 监控程序找不到进程
A: 确保训练命令包含 `python main.py --dataset msra`

### Q4: 如何同时监控多个实验?
A: 为每个实验启动独立的监控进程,使用不同的PID

## 📊 邮件预览

训练完成后,你会收到类似这样的邮件:

```
🎉 NFLAT训练完成

📊 训练结果
最佳F1分数: 0.9612
最佳Epoch: 45
训练时长: 2:15:33
结束时间: 2025-12-14 14:30:00

⚙️ 超参数配置
model: NFLAT
batch_size: 16
lr: 0.002
...

🔗 性能对比
┌────────────────┬──────────┬────────┐
│ 模型           │ F1 Score │ 差距   │
├────────────────┼──────────┼────────┤
│ NFLAT (本次)   │ 0.9612   │ -      │
│ 你的ExpertDict │ 0.9542   │ +0.70% │
│ FLAT论文       │ 0.9609   │ +0.03% │
└────────────────┴──────────┴────────┘
```

## 🔄 更新日志

- 2025-12-14: 初始版本
  - 支持NFLAT训练监控
  - HTML格式邮件
  - 性能对比表
