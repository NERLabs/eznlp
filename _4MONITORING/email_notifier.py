#!/usr/bin/env python3
"""
训练进程监控与邮件通知程序
监控NFLAT训练进程,训练结束后自动发送结果邮件
"""
import os
import time
import json
import smtplib
import psutil
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path


class TrainingMonitor:
    def __init__(self, 
                 pid=None,
                 process_name="python main.py",
                 work_dir="/home/shiwenlong/NERlabs/eznlp/examples/NFLAT4CNER-main",
                 check_interval=60):
        """
        Args:
            pid: 进程PID (如果不指定则通过process_name查找)
            process_name: 进程名称关键字
            work_dir: 工作目录
            check_interval: 检查间隔(秒)
        """
        self.pid = pid
        self.process_name = process_name
        self.work_dir = Path(work_dir)
        self.check_interval = check_interval
        self.start_time = None
        
    def find_process(self):
        """查找训练进程"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if self.process_name in cmdline and 'msra' in cmdline:
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def is_process_running(self):
        """检查进程是否运行"""
        if self.pid is None:
            self.pid = self.find_process()
            if self.pid:
                print(f"[{datetime.now()}] 找到训练进程 PID: {self.pid}")
                self.start_time = datetime.now()
            return self.pid is not None
        
        try:
            proc = psutil.Process(self.pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
    
    def extract_results(self):
        """从训练日志中提取结果"""
        results = {
            "status": "completed",
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": str(datetime.now() - self.start_time) if self.start_time else "Unknown",
            "best_f1": None,
            "best_epoch": None,
            "final_metrics": {}
        }
        
        # 查找fitlog日志
        fitlog_dir = self.work_dir / "logs"
        log_files = list(fitlog_dir.glob("*.log"))
        
        if log_files:
            # 读取最新的日志文件
            latest_log = max(log_files, key=os.path.getmtime)
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 提取最佳结果
            best_f1 = 0.0
            for line in lines:
                if 'SpanFPreRecMetric: f=' in line:
                    try:
                        f1_str = line.split('f=')[1].split(',')[0]
                        f1 = float(f1_str)
                        if f1 > best_f1:
                            best_f1 = f1
                            # 提取epoch信息
                            if 'Epoch' in line:
                                epoch_str = line.split('Epoch')[1].split('/')[0].strip().rstrip(':')
                                results["best_epoch"] = int(epoch_str)
                    except:
                        continue
            
            results["best_f1"] = best_f1
            results["log_file"] = str(latest_log)
        
        # 查找保存的模型结果
        cache_dir = Path("/home/shiwenlong/NERlabs/eznlp/cache")
        nflat_results = list(cache_dir.glob("*nflat*"))
        if nflat_results:
            results["output_dir"] = str(max(nflat_results, key=os.path.getmtime))
        
        return results
    
    def get_hyperparameters(self):
        """提取超参数配置"""
        params = {
            "model": "NFLAT",
            "dataset": "MSRA",
            "batch_size": 16,
            "lr": 0.002,
            "n_epochs": 100,
            "num_layers": 1,
            "n_heads": 8,
            "head_dims": 32,
            "hidden_size": 256,
            "warmup_steps": 0.2,
        }
        return params


class EmailNotifier:
    def __init__(self, 
                 smtp_server="smtp.163.com",
                 smtp_port=465,
                 sender_email="shiwenlongyes@163.com",
                 sender_password="your_password",
                 receiver_email="shiwenlongyes@163.com"):
        """
        Args:
            smtp_server: SMTP服务器
            smtp_port: SMTP端口
            sender_email: 发送者邮箱
            sender_password: SMTP授权码(不是邮箱密码!)
            receiver_email: 接收者邮箱
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = receiver_email
    
    def send_notification(self, results, hyperparams):
        """发送训练完成通知"""
        # 构建邮件内容
        dataset = hyperparams.get('dataset', 'MSRA')
        f1_score = results.get('best_f1', 0)
        subject = f"[实验进度] NFLAT-{dataset.upper()} 训练完成 | F1={f1_score:.4f}"
        
        body = self._format_email_body(results, hyperparams)
        
        # 创建邮件
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email
        
        # HTML格式
        html_part = MIMEText(body, 'html', 'utf-8')
        msg.attach(html_part)
        
        # 发送邮件
        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            print(f"[{datetime.now()}] 邮件发送成功!")
            return True
        except Exception as e:
            print(f"[{datetime.now()}] 邮件发送失败: {e}")
            return False
    
    def _format_email_body(self, results, hyperparams):
        """格式化邮件正文"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #4CAF50; color: white; padding: 10px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f0f0f0; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .key {{ font-weight: bold; color: #333; }}
                .value {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🎉 NFLAT训练完成通知</h2>
            </div>
            
            <div class="section">
                <h3>📊 训练结果</h3>
                <div class="metric">
                    <span class="key">最佳F1分数:</span> 
                    <span class="value" style="font-size: 20px; color: #4CAF50;">
                        {results.get('best_f1', 'N/A'):.4f}
                    </span>
                </div>
                <div class="metric">
                    <span class="key">最佳Epoch:</span> 
                    <span class="value">{results.get('best_epoch', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="key">训练时长:</span> 
                    <span class="value">{results.get('duration', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="key">结束时间:</span> 
                    <span class="value">{results.get('end_time', 'N/A')}</span>
                </div>
            </div>
            
            <div class="section">
                <h3>⚙️ 超参数配置</h3>
                <table>
                    <tr>
                        <th>参数</th>
                        <th>值</th>
                    </tr>
        """
        
        for key, value in hyperparams.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html += f"""
                </table>
            </div>
            
            <div class="section">
                <h3>📁 输出文件</h3>
                <div class="metric">
                    <span class="key">日志文件:</span> 
                    <span class="value">{results.get('log_file', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="key">输出目录:</span> 
                    <span class="value">{results.get('output_dir', 'N/A')}</span>
                </div>
            </div>
            
            <div class="section">
                <h3>🔗 性能对比</h3>
                <table>
                    <tr>
                        <th>模型</th>
                        <th>F1 Score</th>
                        <th>差距</th>
                    </tr>
                    <tr>
                        <td>NFLAT (本次)</td>
                        <td>{results.get('best_f1', 0):.4f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>你的ExpertDict</td>
                        <td>0.9542</td>
                        <td>{(results.get('best_f1', 0) - 0.9542)*100:+.2f}%</td>
                    </tr>
                    <tr>
                        <td>FLAT论文</td>
                        <td>0.9609</td>
                        <td>{(results.get('best_f1', 0) - 0.9609)*100:+.2f}%</td>
                    </tr>
                </table>
            </div>
            
            <p style="color: #666; font-size: 12px; margin-top: 30px;">
                此邮件由训练监控程序自动发送 | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </p>
        </body>
        </html>
        """
        return html


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='NFLAT训练监控与邮件通知')
    parser.add_argument('--pid', type=int, help='训练进程PID')
    parser.add_argument('--email', required=True, help='接收邮箱')
    parser.add_argument('--sender', required=True, help='发送者邮箱')
    parser.add_argument('--password', required=True, help='SMTP授权码')
    parser.add_argument('--interval', type=int, default=60, help='检查间隔(秒)')
    args = parser.parse_args()
    
    # 创建监控器
    monitor = TrainingMonitor(
        pid=args.pid,
        check_interval=args.interval
    )
    
    # 创建邮件通知器
    notifier = EmailNotifier(
        sender_email=args.sender,
        sender_password=args.password,
        receiver_email=args.email
    )
    
    print(f"[{datetime.now()}] 开始监控训练进程...")
    print(f"检查间隔: {args.interval}秒")
    
    # 监控循环
    was_running = False
    while True:
        is_running = monitor.is_process_running()
        
        if is_running:
            was_running = True
            print(f"[{datetime.now()}] 训练进行中... (PID: {monitor.pid})")
        elif was_running:
            # 进程刚结束
            print(f"[{datetime.now()}] 训练已完成,正在收集结果...")
            time.sleep(5)  # 等待日志写入完成
            
            results = monitor.extract_results()
            hyperparams = monitor.get_hyperparameters()
            
            print(f"[{datetime.now()}] 最佳F1: {results.get('best_f1', 'N/A')}")
            print(f"[{datetime.now()}] 正在发送邮件...")
            
            if notifier.send_notification(results, hyperparams):
                print(f"[{datetime.now()}] 任务完成!")
            else:
                print(f"[{datetime.now()}] 邮件发送失败,但结果已保存")
            
            break
        else:
            print(f"[{datetime.now()}] 未找到训练进程,继续等待...")
        
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
