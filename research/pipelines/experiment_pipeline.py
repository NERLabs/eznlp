#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验自动化流水线系统
支持完整生命周期：参数修改 -> 测试运行 -> 正式运行 -> 资源监控 -> 结果对比 -> 报告生成 -> 邮件通知
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import psutil


class PipelineStage:
    """流水线阶段基类"""
    
    def __init__(self, name: str, pipeline):
        self.name = name
        self.pipeline = pipeline
        self.start_time = None
        self.end_time = None
        self.status = "PENDING"  # PENDING, RUNNING, SUCCESS, FAILED, SKIPPED
        self.error_msg = None
    
    def execute(self) -> bool:
        """执行阶段，返回是否成功"""
        raise NotImplementedError
    
    def on_start(self):
        """阶段开始时的钩子"""
        self.start_time = datetime.now()
        self.status = "RUNNING"
        self.log(f"🚀 开始执行: {self.name}")
    
    def on_success(self):
        """阶段成功时的钩子"""
        self.end_time = datetime.now()
        self.status = "SUCCESS"
        duration = (self.end_time - self.start_time).total_seconds()
        self.log(f"✅ 完成: {self.name} (耗时: {duration:.1f}秒)")
    
    def on_failure(self, error: str):
        """阶段失败时的钩子"""
        self.end_time = datetime.now()
        self.status = "FAILED"
        self.error_msg = error
        self.log(f"❌ 失败: {self.name} - {error}")
    
    def on_skip(self, reason: str):
        """阶段跳过时的钩子"""
        self.status = "SKIPPED"
        self.log(f"⏭️  跳过: {self.name} - {reason}")
    
    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{self.name}] {message}"
        print(log_msg)
        if self.pipeline.log_file:
            with open(self.pipeline.log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')


class ParameterModificationStage(PipelineStage):
    """1. 参数修改阶段"""
    
    def __init__(self, pipeline):
        super().__init__("参数修改", pipeline)
    
    def execute(self) -> bool:
        self.on_start()
        try:
            exp_config = self.pipeline.exp_config
            
            # 如果提供了参数覆盖配置，应用它们
            if self.pipeline.param_overrides:
                self.log(f"应用参数覆盖: {self.pipeline.param_overrides}")
                exp_config.update(self.pipeline.param_overrides)
            
            # 保存修改后的配置
            config_path = Path(self.pipeline.work_dir) / "pipeline_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(exp_config, f, indent=2, ensure_ascii=False)
            
            self.log(f"配置已保存: {config_path}")
            self.on_success()
            return True
            
        except Exception as e:
            self.on_failure(str(e))
            return False


class TestRunStage(PipelineStage):
    """2. 测试运行阶段"""
    
    def __init__(self, pipeline):
        super().__init__("测试运行", pipeline)
    
    def execute(self) -> bool:
        self.on_start()
        try:
            # 运行1个epoch的快速测试
            test_config = self.pipeline.exp_config.copy()
            test_config['num_epochs'] = 1
            test_config['save_dir'] = str(Path(self.pipeline.work_dir) / "test_run")
            
            # 构建测试命令
            cmd = self._build_command(test_config)
            self.log(f"测试命令: {cmd}")
            
            # 执行测试
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=600  # 10分钟超时
            )
            
            if result.returncode != 0:
                self.on_failure(f"测试运行失败: {result.stderr[:200]}")
                return False
            
            self.log("测试运行通过，配置有效")
            self.on_success()
            return True
            
        except subprocess.TimeoutExpired:
            self.on_failure("测试运行超时")
            return False
        except Exception as e:
            self.on_failure(str(e))
            return False
    
    def _build_command(self, config: dict) -> str:
        """构建训练命令"""
        script = config.get('script', 'python research/training/text2text.py')
        save_dir = config.get('save_dir', 'cache')
        args = []
        
        # 处理特殊参数
        special_keys = ['script', 'experiment_name', 'dataset']
        
        for key, value in config.items():
            if key in special_keys:
                continue
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.append(f"--{key} {value}")
        
        return f"{script} {' '.join(args)}"


class BackgroundTrainingStage(PipelineStage):
    """3. 后台正式运行阶段"""
    
    def __init__(self, pipeline):
        super().__init__("后台正式运行", pipeline)
        self.process = None
        self.pid = None
    
    def execute(self) -> bool:
        self.on_start()
        try:
            # 构建正式训练命令
            cmd = self._build_command()
            self.log(f"训练命令: {cmd}")
            
            # 后台启动
            log_file = Path(self.pipeline.work_dir) / "training_output.log"
            with open(log_file, 'w') as f:
                self.process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            
            self.pid = self.process.pid
            self.pipeline.training_pid = self.pid
            self.log(f"训练进程已启动 (PID: {self.pid})")
            self.log(f"日志文件: {log_file}")
            
            # 等待几秒确认进程正常启动
            time.sleep(5)
            if self.process.poll() is not None:
                self.on_failure("进程启动后立即退出")
                return False
            
            self.on_success()
            return True
            
        except Exception as e:
            self.on_failure(str(e))
            return False
    
    def _build_command(self) -> str:
        """构建训练命令"""
        config = self.pipeline.exp_config
        script = config.get('script', 'python research/training/text2text.py')
        args = []
        
        # 处理特殊参数
        special_keys = ['script', 'experiment_name', 'dataset']
        
        for key, value in config.items():
            if key in special_keys:
                continue
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.append(f"--{key} {value}")
        
        return f"{script} {' '.join(args)}"


class ResourceMonitoringStage(PipelineStage):
    """4. 资源监控阶段"""
    
    def __init__(self, pipeline):
        super().__init__("资源监控", pipeline)
        self.monitoring = True
    
    def execute(self) -> bool:
        self.on_start()
        try:
            pid = self.pipeline.training_pid
            if not pid:
                self.on_failure("没有找到训练进程PID")
                return False
            
            self.log(f"开始监控进程 PID: {pid}")
            check_interval = self.pipeline.monitor_interval
            
            while self.monitoring:
                # 检查进程是否还在运行
                if not self._is_process_running(pid):
                    self.log("训练进程已结束")
                    break
                
                # 获取资源使用情况
                cpu, mem, gpu_mem = self._get_resource_usage(pid)
                self.log(f"资源使用 - CPU: {cpu:.1f}%, Memory: {mem:.1f}MB, GPU: {gpu_mem}")
                
                # 检查训练进度
                progress = self._check_training_progress()
                if progress:
                    self.log(f"训练进度 - {progress}")
                
                time.sleep(check_interval)
            
            self.on_success()
            return True
            
        except Exception as e:
            self.on_failure(str(e))
            return False
    
    def _is_process_running(self, pid: int) -> bool:
        """检查进程是否运行"""
        try:
            return psutil.pid_exists(pid)
        except:
            return False
    
    def _get_resource_usage(self, pid: int) -> Tuple[float, float, str]:
        """获取资源使用情况"""
        try:
            process = psutil.Process(pid)
            cpu = process.cpu_percent(interval=1)
            mem = process.memory_info().rss / 1024 / 1024  # MB
            
            # 尝试获取GPU使用情况
            gpu_mem = "N/A"
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.strip().split('\n'):
                    if line and str(pid) in line:
                        gpu_mem = f"{line.split(',')[1].strip()}MB"
                        break
            except:
                pass
            
            return cpu, mem, gpu_mem
        except:
            return 0.0, 0.0, "N/A"
    
    def _check_training_progress(self) -> Optional[str]:
        """检查训练进度"""
        try:
            log_file = Path(self.pipeline.work_dir) / self.pipeline.exp_config.get('save_dir', 'cache') / "training.log"
            if not log_file.exists():
                return None
            
            # 读取最后几行
            result = subprocess.run(
                f"tail -10 {log_file}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            for line in reversed(result.stdout.split('\n')):
                if 'Epoch' in line and 'F1' in line:
                    return line.strip()
            
            return None
        except:
            return None


class ResultComparisonStage(PipelineStage):
    """5. 结果对比阶段"""
    
    def __init__(self, pipeline):
        super().__init__("结果对比", pipeline)
    
    def execute(self) -> bool:
        self.on_start()
        try:
            dataset = self.pipeline.exp_config.get('dataset', 'unknown')
            save_dir = self.pipeline.exp_config.get('save_dir', 'cache')
            
            # 查找同一数据集的所有实验结果
            results = self._collect_results(dataset, save_dir)
            
            if not results:
                self.on_failure("没有找到可对比的实验结果")
                return False
            
            # 生成对比数据
            comparison = self._generate_comparison(results)
            
            # 保存对比结果
            output_file = Path(self.pipeline.work_dir) / "comparison_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            
            self.log(f"对比结果已保存: {output_file}")
            self.pipeline.comparison_data = comparison
            
            self.on_success()
            return True
            
        except Exception as e:
            self.on_failure(str(e))
            return False
    
    def _collect_results(self, dataset: str, save_dir: str) -> List[dict]:
        """收集实验结果"""
        results = []
        base_dir = Path(save_dir)
        
        # 查找所有results.json文件
        for result_file in base_dir.rglob("results.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('dataset') == dataset or dataset in str(result_file):
                        results.append({
                            'path': str(result_file),
                            'data': data
                        })
            except:
                continue
        
        return results
    
    def _generate_comparison(self, results: List[dict]) -> dict:
        """生成对比数据"""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'experiments': []
        }
        
        for result in results:
            data = result['data']
            comparison['experiments'].append({
                'path': result['path'],
                'model': data.get('model_type', 'Unknown'),
                'test_f1': data.get('test_metrics', [0])[0],
                'params': data.get('total_params', 0)
            })
        
        # 按F1排序
        comparison['experiments'].sort(key=lambda x: x['test_f1'], reverse=True)
        
        return comparison


class ReportGenerationStage(PipelineStage):
    """6. 生成结果总结报告阶段"""
    
    def __init__(self, pipeline):
        super().__init__("生成结果总结报告", pipeline)
    
    def execute(self) -> bool:
        self.on_start()
        try:
            # 使用format_redjujube_results.py的逻辑或自定义报告生成
            report_content = self._generate_report()
            
            # 保存报告
            report_file = Path(self.pipeline.work_dir) / "experiment_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.log(f"报告已生成: {report_file}")
            self.pipeline.report_file = str(report_file)
            
            self.on_success()
            return True
            
        except Exception as e:
            self.on_failure(str(e))
            return False
    
    def _generate_report(self) -> str:
        """生成Markdown报告"""
        lines = []
        lines.append(f"# 实验自动化流水线报告")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**实验名称**: {self.pipeline.exp_name}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # 流水线执行概况
        lines.append("## 📊 流水线执行概况")
        lines.append("")
        lines.append("| 阶段 | 状态 | 耗时 | 备注 |")
        lines.append("|------|------|------|------|")
        
        for stage in self.pipeline.stages:
            duration = ""
            if stage.start_time and stage.end_time:
                duration = f"{(stage.end_time - stage.start_time).total_seconds():.1f}s"
            
            status_emoji = {
                'SUCCESS': '✅',
                'FAILED': '❌',
                'SKIPPED': '⏭️',
                'RUNNING': '🔄',
                'PENDING': '⏸️'
            }.get(stage.status, '❓')
            
            note = stage.error_msg if stage.error_msg else "-"
            lines.append(f"| {stage.name} | {status_emoji} {stage.status} | {duration} | {note} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # 实验配置
        lines.append("## ⚙️ 实验配置")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(self.pipeline.exp_config, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # 结果对比
        if self.pipeline.comparison_data:
            lines.append("## 🎯 结果对比")
            lines.append("")
            lines.append("| 排名 | 模型 | Test F1 | 参数量 | 路径 |")
            lines.append("|------|------|---------|--------|------|")
            
            for idx, exp in enumerate(self.pipeline.comparison_data.get('experiments', []), 1):
                f1_pct = f"{exp['test_f1']*100:.2f}%"
                params = f"{exp['params']:,}"
                path_short = Path(exp['path']).parent.name
                lines.append(f"| {idx} | {exp['model']} | {f1_pct} | {params} | {path_short} |")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)


class EmailNotificationStage(PipelineStage):
    """7. 自动发送邮件通知阶段"""
    
    def __init__(self, pipeline):
        super().__init__("自动发送邮件通知", pipeline)
    
    def execute(self) -> bool:
        self.on_start()
        try:
            if not self.pipeline.email_config:
                self.on_skip("未配置邮件发送")
                return True
            
            # 发送邮件
            self._send_email()
            
            self.on_success()
            return True
            
        except Exception as e:
            self.on_failure(str(e))
            return False
    
    def _send_email(self):
        """发送邮件通知"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        email_cfg = self.pipeline.email_config
        
        # 构建邮件内容
        html = self._build_email_html()
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[实验完成] {self.pipeline.exp_name}"
        msg['From'] = email_cfg['sender']
        msg['To'] = email_cfg['receiver']
        
        msg.attach(MIMEText(html, 'html', 'utf-8'))
        
        # 发送邮件
        if 'smtp.163.com' in email_cfg.get('smtp_server', ''):
            server = smtplib.SMTP_SSL(email_cfg['smtp_server'], email_cfg.get('smtp_port', 465), timeout=30)
        else:
            server = smtplib.SMTP(email_cfg['smtp_server'], email_cfg.get('smtp_port', 587), timeout=30)
            server.starttls()
        
        server.login(email_cfg['sender'], email_cfg['password'])
        server.send_message(msg)
        server.quit()
        
        self.log(f"邮件已发送至: {email_cfg['receiver']}")
    
    def _build_email_html(self) -> str:
        """构建HTML邮件内容"""
        # 获取最佳结果
        best_f1 = 0
        if self.pipeline.comparison_data and self.pipeline.comparison_data.get('experiments'):
            best_f1 = self.pipeline.comparison_data['experiments'][0]['test_f1'] * 100
        
        html = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background-color: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🎯 实验自动化流水线完成通知</h2>
        <p>{self.pipeline.exp_name}</p>
    </div>
    
    <div class="section">
        <h3>📊 核心指标</h3>
        <div class="metric">
            <strong>最佳 F1 Score:</strong> {best_f1:.2f}%
        </div>
    </div>
    
    <div class="section">
        <h3>✅ 流水线执行状态</h3>
        <table>
            <tr><th>阶段</th><th>状态</th></tr>
"""
        
        for stage in self.pipeline.stages:
            status_emoji = {
                'SUCCESS': '✅',
                'FAILED': '❌',
                'SKIPPED': '⏭️'
            }.get(stage.status, '❓')
            html += f"            <tr><td>{stage.name}</td><td>{status_emoji} {stage.status}</td></tr>\n"
        
        html += f"""
        </table>
    </div>
    
    <div class="section">
        <p><strong>完成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>报告文件:</strong> {self.pipeline.report_file}</p>
    </div>
</body>
</html>
"""
        return html


class ExperimentPipeline:
    """实验自动化流水线"""
    
    def __init__(self, 
                 exp_name: str,
                 exp_config: dict,
                 work_dir: str = None,
                 param_overrides: dict = None,
                 email_config: dict = None,
                 skip_test: bool = False,
                 monitor_interval: int = 300):
        
        self.exp_name = exp_name
        self.exp_config = exp_config
        self.work_dir = work_dir or f"pipeline_runs/{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.param_overrides = param_overrides or {}
        self.email_config = email_config
        self.skip_test = skip_test
        self.monitor_interval = monitor_interval
        
        # 创建工作目录
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = Path(self.work_dir) / "pipeline.log"
        
        # 流水线状态
        self.training_pid = None
        self.comparison_data = None
        self.report_file = None
        
        # 初始化各阶段
        self.stages = [
            ParameterModificationStage(self),
            TestRunStage(self) if not skip_test else None,
            BackgroundTrainingStage(self),
            ResourceMonitoringStage(self),
            ResultComparisonStage(self),
            ReportGenerationStage(self),
            EmailNotificationStage(self)
        ]
        self.stages = [s for s in self.stages if s is not None]
    
    def run(self) -> bool:
        """运行完整流水线"""
        print("=" * 80)
        print(f"🚀 启动实验自动化流水线: {self.exp_name}")
        print("=" * 80)
        print(f"工作目录: {self.work_dir}")
        print(f"日志文件: {self.log_file}")
        print("=" * 80)
        print("")
        
        for stage in self.stages:
            success = stage.execute()
            
            if not success and stage.status == "FAILED":
                print("")
                print("=" * 80)
                print(f"❌ 流水线在阶段 [{stage.name}] 失败，终止执行")
                print("=" * 80)
                return False
            
            print("")
        
        print("=" * 80)
        print("🎉 实验自动化流水线全部完成！")
        print("=" * 80)
        print(f"📊 报告文件: {self.report_file}")
        print(f"📁 工作目录: {self.work_dir}")
        print("=" * 80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='实验自动化流水线')
    parser.add_argument('--name', type=str, required=True, help='实验名称')
    parser.add_argument('--config', type=str, required=True, help='实验配置文件(JSON)')
    parser.add_argument('--work-dir', type=str, help='工作目录')
    parser.add_argument('--skip-test', action='store_true', help='跳过测试运行')
    parser.add_argument('--monitor-interval', type=int, default=300, help='监控间隔(秒)')
    
    # 邮件配置
    parser.add_argument('--email-sender', type=str, help='发件人邮箱')
    parser.add_argument('--email-receiver', type=str, help='收件人邮箱')
    parser.add_argument('--email-password', type=str, help='邮箱密码/授权码')
    parser.add_argument('--smtp-server', type=str, default='smtp.163.com', help='SMTP服务器')
    parser.add_argument('--smtp-port', type=int, default=465, help='SMTP端口')
    
    args = parser.parse_args()
    
    # 加载实验配置
    with open(args.config, 'r', encoding='utf-8') as f:
        exp_config = json.load(f)
    
    # 邮件配置
    email_config = None
    if args.email_sender and args.email_receiver and args.email_password:
        email_config = {
            'sender': args.email_sender,
            'receiver': args.email_receiver,
            'password': args.email_password,
            'smtp_server': args.smtp_server,
            'smtp_port': args.smtp_port
        }
    
    # 创建并运行流水线
    pipeline = ExperimentPipeline(
        exp_name=args.name,
        exp_config=exp_config,
        work_dir=args.work_dir,
        email_config=email_config,
        skip_test=args.skip_test,
        monitor_interval=args.monitor_interval
    )
    
    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
