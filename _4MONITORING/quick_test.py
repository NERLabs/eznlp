#!/usr/bin/env python3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import socket

print("开始测试...")
print(f"时间: {datetime.now()}")

# 测试网络连接
try:
    print("测试SMTP服务器连接...")
    sock = socket.create_connection(("smtp-mail.outlook.com", 587), timeout=10)
    sock.close()
    print("✓ 网络连接正常")
except Exception as e:
    print(f"✗ 网络连接失败: {e}")
    exit(1)

# 发送邮件
try:
    msg = MIMEText('测试成功 ' + datetime.now().strftime('%H:%M:%S'), 'plain', 'utf-8')
    msg['Subject'] = 'NFLAT Test'
    msg['From'] = 'xxs9331@outlook.com'
    msg['To'] = 'xxs9331@outlook.com'
    
    print("连接SMTP...")
    server = smtplib.SMTP('smtp-mail.outlook.com', 587, timeout=60)
    print("启动TLS...")
    server.starttls()
    print("登录...")
    server.login('xxs9331@outlook.com', 'swfhttbpzsrprqeu')
    print("发送邮件...")
    server.send_message(msg)
    server.quit()
    print("✓ 邮件发送成功!")
except Exception as e:
    print(f"✗ 失败: {e}")
