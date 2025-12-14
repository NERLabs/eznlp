#!/usr/bin/env python3
"""发送原始实验结果邮件"""
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

# 实验路径
exp_dir = Path("/home/shiwenlong/NERlabs/eznlp/experiments/hz_lexicon/results/12-3_expert_optimization/msra_softlex_v1")
result_file = exp_dir / "results.json"
log_file = exp_dir / "training.log"

# 读取results.json
with open(result_file) as f:
    result_content = f.read()

# 读取training.log (取最后50行)
with open(log_file) as f:
    log_lines = f.readlines()
    log_content = ''.join(log_lines[-50:])

# 解析F1分数
result_data = json.loads(result_content)
f1_score = result_data['test_metrics'][0]

# 构建邮件
html = f"""
<html>
<head>
    <style>
        body {{ font-family: 'Courier New', monospace; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2196F3; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .section {{ background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        pre {{ background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 12px; }}
        .label {{ color: #666; font-weight: bold; margin-bottom: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>📋 实验原始结果</h2>
        <p>MSRA-SoftLexicon | F1={f1_score:.4f}</p>
    </div>
    
    <div class="section">
        <div class="label">📄 results.json</div>
        <pre>{result_content}</pre>
    </div>
    
    <div class="section">
        <div class="label">📝 training.log (最后50行)</div>
        <pre>{log_content}</pre>
    </div>
    
    <div style="margin-top: 20px; padding: 10px; background-color: white; border-radius: 5px; color: #666; font-size: 12px;">
        <strong>发送时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <strong>实验路径:</strong> {exp_dir}
    </div>
</body>
</html>
"""

# 创建邮件
msg = MIMEMultipart('alternative')
msg['Subject'] = f"[实验进度] MSRA-SoftLexicon原始结果 | F1={f1_score:.4f}"
msg['From'] = 'shiwenlongyes@163.com'
msg['To'] = 'shiwenlongyes@163.com'

msg.attach(MIMEText(html, 'html', 'utf-8'))

# 发送
try:
    server = smtplib.SMTP_SSL('smtp.163.com', 465, timeout=30)
    server.login('shiwenlongyes@163.com', 'KNXRm35vHjizvjcS')
    server.send_message(msg)
    server.quit()
    print(f"✅ 原始结果邮件已发送 | F1={f1_score:.4f}")
except Exception as e:
    print(f"❌ 发送失败: {e}")
