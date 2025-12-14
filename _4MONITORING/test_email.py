#!/usr/bin/env python3
"""测试Outlook邮件发送"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import getpass

# 邮箱配置
SENDER_EMAIL = "xxs9331@outlook.com"
RECEIVER_EMAIL = "xxs9331@outlook.com"
SMTP_SERVER = "smtp-mail.outlook.com"
SMTP_PORT = 587

# 获取应用密码
print("=" * 50)
print("Outlook邮件发送测试")
print("=" * 50)
print(f"\n发件人: {SENDER_EMAIL}")
print(f"收件人: {RECEIVER_EMAIL}")
print(f"SMTP服务器: {SMTP_SERVER}:{SMTP_PORT}")
print("\n请输入Outlook应用密码 (16位):")
password = getpass.getpass("密码: ")

# 构建测试邮件
msg = MIMEMultipart('alternative')
msg['Subject'] = "🧪 NFLAT监控系统测试邮件"
msg['From'] = SENDER_EMAIL
msg['To'] = RECEIVER_EMAIL

html_body = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; }}
        .content {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
        .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
        .success {{ color: #4CAF50; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🧪 NFLAT训练监控系统</h2>
        <p>邮件功能测试</p>
    </div>
    
    <div class="content">
        <p class="success">✅ 如果您收到此邮件，说明配置成功！</p>
        
        <h3>📋 配置信息</h3>
        <ul>
            <li><b>SMTP服务器:</b> {SMTP_SERVER}:{SMTP_PORT}</li>
            <li><b>加密方式:</b> STARTTLS</li>
            <li><b>发送时间:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
        </ul>
        
        <h3>📊 监控系统功能</h3>
        <ul>
            <li>✅ 自动监控训练进程</li>
            <li>✅ 提取训练结果(F1/Precision/Recall)</li>
            <li>✅ 训练完成自动发送通知</li>
            <li>✅ HTML格式美化报告</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>此邮件由NFLAT训练监控系统自动发送</p>
        <p>工作目录: /home/shiwenlong/NERlabs/eznlp/examples/NFLAT4CNER-main</p>
    </div>
</body>
</html>
"""

html_part = MIMEText(html_body, 'html', 'utf-8')
msg.attach(html_part)

# 发送邮件
print("\n正在发送测试邮件...")
try:
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
        server.set_debuglevel(0)  # 设置为1可看详细日志
        print("连接SMTP服务器...")
        server.starttls()
        print("启动TLS加密...")
        server.login(SENDER_EMAIL, password)
        print("登录成功...")
        server.send_message(msg)
        print("\n" + "=" * 50)
        print("✅ 测试邮件发送成功!")
        print("=" * 50)
        print(f"\n请检查 {RECEIVER_EMAIL} 收件箱")
        print("(可能在垃圾邮件中)")
except smtplib.SMTPAuthenticationError:
    print("\n❌ 认证失败! 请检查:")
    print("1. 应用密码是否正确(16位)")
    print("2. 是否启用了两步验证")
    print("3. 应用密码是否已创建")
except Exception as e:
    print(f"\n❌ 发送失败: {e}")
    print("\n可能的原因:")
    print("1. 网络连接问题")
    print("2. SMTP服务器无法访问")
    print("3. 防火墙阻止587端口")
