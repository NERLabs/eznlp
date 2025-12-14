#!/bin/bash
export SMTP_PASSWORD="swfhttbpzsrprqeu"

python3 << 'EOF'
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

SENDER = "xxs9331@outlook.com"
RECEIVER = "xxs9331@outlook.com"
PASSWORD = os.environ.get('SMTP_PASSWORD')

msg = MIMEMultipart('alternative')
msg['Subject'] = "🧪 NFLAT监控测试"
msg['From'] = SENDER
msg['To'] = RECEIVER

html = f"""
<html>
<body style="font-family: Arial;">
    <h2 style="color: #4CAF50;">✅ 邮件配置成功!</h2>
    <p>发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>NFLAT训练监控系统已就绪</p>
</body>
</html>
"""

msg.attach(MIMEText(html, 'html', 'utf-8'))

try:
    with smtplib.SMTP("smtp-mail.outlook.com", 587, timeout=30) as server:
        server.starttls()
        server.login(SENDER, PASSWORD)
        server.send_message(msg)
    print("✅ 邮件发送成功! 请检查收件箱")
except Exception as e:
    print(f"❌ 失败: {e}")
EOF
