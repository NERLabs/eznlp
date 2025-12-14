#!/usr/bin/env python3
"""发送实验结果邮件"""
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# 读取实验结果
result_file = "/home/shiwenlong/NERlabs/eznlp/experiments/hz_lexicon/results/12-3_expert_optimization/msra_softlex_v1/results.json"
with open(result_file) as f:
    data = json.load(f)

# 提取关键信息
f1_score = data['test_metrics'][0]
model_type = data['model_type']
params = data['args']

# 构建HTML邮件
html = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background-color: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .param-table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        .param-table th, .param-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .param-table th {{ background-color: #4CAF50; color: white; }}
        .value {{ color: #2196F3; font-weight: bold; }}
        .label {{ color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🎯 NER实验结果报告</h2>
        <p>MSRA数据集 - SoftLexicon方法</p>
    </div>
    
    <div class="section">
        <h3>📊 核心指标</h3>
        <div class="metric">
            <span class="label">测试集F1分数:</span> 
            <span class="value" style="font-size: 24px;">{f1_score:.4f} ({f1_score*100:.2f}%)</span>
        </div>
        <div class="metric">
            <span class="label">模型类型:</span> <span class="value">{model_type}</span>
        </div>
        <div class="metric">
            <span class="label">总参数量:</span> <span class="value">{data['total_params']:,}</span>
        </div>
    </div>
    
    <div class="section">
        <h3>⚙️ 训练参数</h3>
        <table class="param-table">
            <tr><th>参数名称</th><th>参数值</th></tr>
            <tr><td>数据集路径</td><td>{params['data_dir']}</td></tr>
            <tr><td>BERT模型</td><td>{params['bert_arch']}</td></tr>
            <tr><td>训练轮数</td><td>{params['num_epochs']}</td></tr>
            <tr><td>批次大小</td><td>{params['batch_size']}</td></tr>
            <tr><td>学习率</td><td>{params['lr']}</td></tr>
            <tr><td>微调学习率</td><td>{params['finetune_lr']}</td></tr>
            <tr><td>权重衰减</td><td>{params['weight_decay']}</td></tr>
            <tr><td>梯度裁剪</td><td>{params['grad_clip']}</td></tr>
            <tr><td>隐藏层维度</td><td>{params['hid_dim']}</td></tr>
            <tr><td>LSTM层数</td><td>{params['num_layers']}</td></tr>
            <tr><td>Dropout率</td><td>{params['dropout']}</td></tr>
            <tr><td>BERT Dropout</td><td>{params['bert_drop_rate']}</td></tr>
            <tr><td>专家词典维度</td><td>{params['expert_dict_dim']}</td></tr>
            <tr><td>软词典路径</td><td>{params['softlex_train_path']}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h3>📁 实验配置</h3>
        <table class="param-table">
            <tr><th>配置项</th><th>值</th></tr>
            <tr><td>运行基线</td><td>{'✓' if params['run_baseline'] else '✗'}</td></tr>
            <tr><td>运行专家词典</td><td>{'✓' if params['run_expert_dict'] else '✗'}</td></tr>
            <tr><td>运行SoftLexicon</td><td>{'✓' if params['run_softlexicon'] else '✗'}</td></tr>
            <tr><td>运行TrainLex</td><td>{'✓' if params['run_softlexicon_trainlex'] else '✗'}</td></tr>
            <tr><td>随机种子</td><td>{params['seed']}</td></tr>
        </table>
    </div>
    
    <div style="margin-top: 30px; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
        <p style="color: #666; font-size: 12px;">
            <strong>实验时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>结果文件:</strong> {result_file}<br>
            <strong>自动发送:</strong> NFLAT训练监控系统
        </p>
    </div>
</body>
</html>
"""

# 创建邮件
msg = MIMEMultipart('alternative')
msg['Subject'] = f"[实验进度] MSRA-SoftLexicon实验完成 | F1={f1_score:.4f}"
msg['From'] = 'shiwenlongyes@163.com'
msg['To'] = 'shiwenlongyes@163.com'

msg.attach(MIMEText(html, 'html', 'utf-8'))

# 发送
try:
    server = smtplib.SMTP_SSL('smtp.163.com', 465, timeout=30)
    server.login('shiwenlongyes@163.com', 'KNXRm35vHjizvjcS')
    server.send_message(msg)
    server.quit()
    print(f"✅ 实验结果邮件已发送")
    print(f"   F1分数: {f1_score:.4f}")
    print(f"   模型: {model_type}")
except Exception as e:
    print(f"❌ 发送失败: {e}")
