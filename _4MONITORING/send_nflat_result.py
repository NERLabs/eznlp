#!/usr/bin/env python3
"""发送NFLAT训练完成邮件"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

# 读取结果文件
result_file = Path("/home/shiwenlong/NERlabs/eznlp/examples/NFLAT4CNER-main/nflat_msra_result.txt")
with open(result_file) as f:
    result_content = f.read()

# 读取训练日志(终端输出)
terminal_log = """
Best test performance(may not correspond to the best dev performance):{'SpanFPreRecMetric': {'f': 0.945171, 'pre': 0.949103, 'rec': 0.941272}, 'label_acc': {'acc': 0.990718}} achieved at Epoch:66.
Best test performance(correspond to the best dev performance):{'SpanFPreRecMetric': {'f': 0.945171, 'pre': 0.949103, 'rec': 0.941272}, 'label_acc': {'acc': 0.990718}} achieved at Epoch:66.

In Epoch:66/Step:191730, got best dev performance:
SpanFPreRecMetric: f=0.945171, pre=0.949103, rec=0.941272
label_acc: acc=0.990718

训练完成: 100/100 epochs
最终F1: 0.942454 (Epoch 100)
最佳F1: 0.945171 (Epoch 66)
"""

# 构建HTML邮件
html = f"""
<html>
<head>
    <style>
        body {{ font-family: 'Courier New', monospace; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #4CAF50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .section {{ background-color: white; padding: 20px; margin: 15px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .best {{ background-color: #fff9c4; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        pre {{ background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 12px; }}
        .metric {{ font-size: 24px; color: #2196F3; font-weight: bold; }}
        .label {{ color: #666; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🎉 NFLAT训练完成</h2>
        <p>MSRA数据集 | 100 Epochs</p>
        <p style="font-size: 28px; margin: 10px 0;">F1 = <span class="metric">94.52%</span></p>
    </div>
    
    <div class="section">
        <div class="best">
            <h3 style="margin-top: 0;">⭐ 最佳性能 (Epoch 66/100)</h3>
            <table>
                <tr><th>指标</th><th>数值</th></tr>
                <tr><td><b>F1 Score</b></td><td><span style="color:#4CAF50; font-size:18px;">0.945171 (94.52%)</span></td></tr>
                <tr><td>Precision</td><td>0.949103 (94.91%)</td></tr>
                <tr><td>Recall</td><td>0.941272 (94.13%)</td></tr>
                <tr><td>Accuracy</td><td>0.990718 (99.07%)</td></tr>
            </table>
        </div>
    </div>
    
    <div class="section">
        <h3>📈 训练曲线</h3>
        <table>
            <tr><th>Epoch</th><th>F1</th><th>Precision</th><th>Recall</th></tr>
            <tr><td>1</td><td>55.6%</td><td>57.7%</td><td>53.6%</td></tr>
            <tr><td>10</td><td>91.1%</td><td>91.1%</td><td>91.1%</td></tr>
            <tr><td>20</td><td>92.8%</td><td>93.5%</td><td>92.1%</td></tr>
            <tr><td>30</td><td>93.8%</td><td>94.4%</td><td>93.2%</td></tr>
            <tr><td>40</td><td>94.1%</td><td>94.5%</td><td>93.7%</td></tr>
            <tr><td>50</td><td>94.0%</td><td>94.1%</td><td>93.9%</td></tr>
            <tr><td>60</td><td>94.1%</td><td>94.1%</td><td>94.1%</td></tr>
            <tr style="background-color:#fff9c4;"><td><b>66 ★</b></td><td><b>94.5%</b></td><td><b>94.9%</b></td><td><b>94.1%</b></td></tr>
            <tr><td>70</td><td>94.1%</td><td>94.3%</td><td>93.8%</td></tr>
            <tr><td>80</td><td>94.3%</td><td>94.8%</td><td>93.9%</td></tr>
            <tr><td>90</td><td>94.4%</td><td>94.6%</td><td>94.2%</td></tr>
            <tr><td>100</td><td>94.2%</td><td>94.4%</td><td>94.1%</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h3>📊 性能对比</h3>
        <table>
            <tr><th>模型</th><th>F1分数</th><th>备注</th></tr>
            <tr style="background-color:#e8f5e9;"><td><b>NFLAT (本次)</b></td><td><b>94.52%</b></td><td>我们的复现</td></tr>
            <tr><td>FLAT原文</td><td>96.09%</td><td>2020年论文</td></tr>
            <tr><td>NFLAT原文</td><td>96.5-96.7%</td><td>2022年论文</td></tr>
        </table>
        <p style="color:#666; margin-top:10px;">差距: 约1.5-2.2个百分点</p>
    </div>
    
    <div class="section">
        <h3>📝 完整结果报告</h3>
        <pre>{result_content}</pre>
    </div>
    
    <div class="section">
        <h3>💻 终端输出</h3>
        <pre>{terminal_log}</pre>
    </div>
    
    <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 5px; color: #666; font-size: 12px;">
        <strong>发送时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <strong>训练耗时:</strong> 约5-6小时<br>
        <strong>自动发送:</strong> NFLAT训练监控系统
    </div>
</body>
</html>
"""

# 创建邮件
msg = MIMEMultipart('alternative')
msg['Subject'] = "[实验进度] NFLAT-MSRA训练完成 | F1=0.9452"
msg['From'] = 'shiwenlongyes@163.com'
msg['To'] = 'shiwenlongyes@163.com'

msg.attach(MIMEText(html, 'html', 'utf-8'))

# 发送
try:
    server = smtplib.SMTP_SSL('smtp.163.com', 465, timeout=30)
    server.login('shiwenlongyes@163.com', 'KNXRm35vHjizvjcS')
    server.send_message(msg)
    server.quit()
    print(f"✅ NFLAT训练结果邮件已发送 | F1=0.9452")
except Exception as e:
    print(f"❌ 发送失败: {e}")
