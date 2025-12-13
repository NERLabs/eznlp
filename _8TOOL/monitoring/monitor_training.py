#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用深度学习训练任务监控脚本

功能：
1. 实时监控GPU使用情况
2. 跟踪多个训练进程的状态和进度
3. 显示日志摘要（最近N行）
4. 自动检测并报告异常情况
5. 生成简洁的训练概览

使用方法：
    python scripts/monitor_training.py                    # 监控所有训练进程
    python scripts/monitor_training.py --pids 1234 5678   # 监控指定PID
    python scripts/monitor_training.py --log-dir cache    # 监控指定目录的训练日志
    python scripts/monitor_training.py --interval 10      # 自定义刷新间隔（秒）
"""

import argparse
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class GPUMonitor:
    """GPU监控器"""
    
    @staticmethod
    def get_gpu_info() -> List[Dict]:
        """获取GPU信息"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return []
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'memory_total': int(parts[4]),
                        'temperature': int(parts[5])
                    })
            return gpus
        except Exception as e:
            print(f"{Colors.WARNING}⚠️ 无法获取GPU信息: {e}{Colors.ENDC}")
            return []
    
    @staticmethod
    def get_gpu_processes() -> List[Dict]:
        """获取GPU进程信息"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return []
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    processes.append({
                        'pid': int(parts[0]),
                        'gpu_memory': int(parts[1])
                    })
            return processes
        except Exception:
            return []
    
    @staticmethod
    def format_gpu_summary(gpus: List[Dict]) -> str:
        """格式化GPU摘要"""
        if not gpus:
            return f"{Colors.WARNING}⚠️ 未检测到GPU{Colors.ENDC}"
        
        lines = [f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}"]
        lines.append(f"{Colors.HEADER}🖥️  GPU 状态{Colors.ENDC}")
        lines.append(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        
        for gpu in gpus:
            util_color = Colors.OKGREEN if gpu['utilization'] < 50 else Colors.WARNING if gpu['utilization'] < 80 else Colors.FAIL
            mem_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            mem_color = Colors.OKGREEN if mem_percent < 50 else Colors.WARNING if mem_percent < 80 else Colors.FAIL
            temp_color = Colors.OKGREEN if gpu['temperature'] < 70 else Colors.WARNING if gpu['temperature'] < 85 else Colors.FAIL
            
            lines.append(f"GPU {gpu['index']}: {gpu['name']}")
            lines.append(f"  利用率: {util_color}{gpu['utilization']:3d}%{Colors.ENDC}  "
                        f"显存: {mem_color}{gpu['memory_used']:5d}/{gpu['memory_total']:5d} MiB ({mem_percent:.1f}%){Colors.ENDC}  "
                        f"温度: {temp_color}{gpu['temperature']:2d}°C{Colors.ENDC}")
        
        lines.append("")
        return '\n'.join(lines)


class ProcessMonitor:
    """进程监控器"""
    
    @staticmethod
    def get_process_info(pid: int) -> Optional[Dict]:
        """获取进程信息"""
        try:
            # 检查进程是否存在
            result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,etime,cmd'],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                return None
            
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return None
            
            # 解析输出
            parts = lines[1].split(None, 2)
            if len(parts) < 3:
                return None
            
            return {
                'pid': int(parts[0]),
                'elapsed_time': parts[1],
                'command': parts[2]
            }
        except Exception:
            return None
    
    @staticmethod
    def find_training_processes(pattern: str = r"train.*\.py") -> List[int]:
        """查找训练进程"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            pids = []
            for line in result.stdout.split('\n'):
                if re.search(pattern, line) and 'grep' not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pids.append(int(parts[1]))
                        except ValueError:
                            continue
            return pids
        except Exception:
            return []


class LogAnalyzer:
    """日志分析器"""
    
    @staticmethod
    def find_training_logs(base_dir: str, recent_hours: int = 24, only_valid: bool = True) -> List[Path]:
        """查找最近的训练日志
        
        Args:
            base_dir: 基础目录
            recent_hours: 最近几小时的日志
            only_valid: 是否只显示有最佳模型保存的日志（过滤失败任务）
        """
        logs = []
        cutoff_time = time.time() - (recent_hours * 3600)
        
        for log_file in Path(base_dir).rglob("training.log"):
            if log_file.stat().st_mtime > cutoff_time:
                # 如果启用过滤，检查是否有最佳模型保存记录
                if only_valid:
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # 必须包含"保存最佳模型"或"✅ 保存"
                            if "保存最佳模型" not in content and "✅ 保存" not in content:
                                continue
                    except:
                        continue
                
                logs.append(log_file)
        
        return sorted(logs, key=lambda x: x.stat().st_mtime, reverse=True)
    
    @staticmethod
    def parse_training_progress(log_file: Path, tail_lines: int = 30) -> Dict:
        """解析训练进度"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                return {'status': 'empty', 'lines': []}
            
            # 取最后N行
            recent_lines = lines[-tail_lines:]
            
            # 提取关键信息
            info = {
                'log_file': str(log_file),
                'total_lines': len(lines),
                'recent_lines': [l.strip() for l in recent_lines if l.strip()],
                'status': 'running'
            }
            
            # 解析最后一条Epoch信息
            for line in reversed(recent_lines):
                # 匹配 Epoch 信息
                epoch_match = re.search(r'Epoch:\s*(\d+)\s*\|\s*Step:\s*(\d+)', line)
                if epoch_match:
                    info['current_epoch'] = int(epoch_match.group(1))
                    info['current_step'] = int(epoch_match.group(2))
                
                # 匹配 Loss 和 Metrics
                loss_match = re.search(r'Loss:\s*([\d.]+)', line)
                if loss_match:
                    info['loss'] = float(loss_match.group(1))
                
                metrics_match = re.search(r'Metrics:\s*([\d.]+)%?', line)
                if metrics_match:
                    info['metrics'] = float(metrics_match.group(1))
                
                # 匹配 Dev F1
                dev_match = re.search(r'Dev.*Metrics:\s*([\d.]+)%?', line)
                if dev_match:
                    info['dev_metrics'] = float(dev_match.group(1))
                
                if 'current_epoch' in info and 'loss' in info:
                    break
            
            # 检查是否完成
            last_10_lines = '\n'.join(lines[-10:])
            if any(keyword in last_10_lines for keyword in ['completed', '完成', 'Best dev', 'Test']):
                info['status'] = 'completed'
            
            # 检查是否有错误
            if any(keyword in last_10_lines.lower() for keyword in ['error', 'exception', 'traceback']):
                info['status'] = 'error'
            
            return info
        
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'lines': []}
    
    @staticmethod
    def format_log_summary(log_info: Dict, max_lines: int = 5) -> str:
        """格式化日志摘要"""
        lines = []
        
        # 状态标识
        status_icons = {
            'running': f"{Colors.OKGREEN}🟢{Colors.ENDC}",
            'completed': f"{Colors.OKBLUE}✅{Colors.ENDC}",
            'error': f"{Colors.FAIL}❌{Colors.ENDC}",
            'empty': f"{Colors.WARNING}⚠️{Colors.ENDC}"
        }
        status = log_info.get('status', 'unknown')
        icon = status_icons.get(status, '❓')
        
        # 日志文件路径
        log_path = log_info.get('log_file', 'Unknown')
        lines.append(f"{icon} {Colors.BOLD}{log_path}{Colors.ENDC}")
        
        # 进度信息
        if 'current_epoch' in log_info:
            progress = f"  Epoch {log_info['current_epoch']} | Step {log_info['current_step']}"
            if 'loss' in log_info:
                progress += f" | Loss {log_info['loss']:.3f}"
            if 'metrics' in log_info:
                progress += f" | Train {log_info['metrics']:.2f}%"
            if 'dev_metrics' in log_info:
                progress += f" | Dev {log_info['dev_metrics']:.2f}%"
            lines.append(f"{Colors.OKCYAN}{progress}{Colors.ENDC}")
        
        # 最近日志行（显示文件名和行号范围）
        recent = log_info.get('recent_lines', [])
        if recent and max_lines > 0:
            total_lines = log_info.get('total_lines', 0)
            start_line = max(1, total_lines - len(recent) + 1)
            end_line = total_lines
            
            lines.append(f"  {Colors.UNDERLINE}@{Path(log_path).name} {start_line}-{end_line}{Colors.ENDC}")
            
            display_lines = recent[-max_lines:]
            for line in display_lines:
                # 截断过长的行
                if len(line) > 100:
                    line = line[:97] + '...'
                lines.append(f"    {line}")
        
        lines.append("")
        return '\n'.join(lines)


class TrainingMonitor:
    """训练监控主类"""
    
    def __init__(self, pids: Optional[List[int]] = None, 
                 log_dir: Optional[str] = None,
                 interval: int = 30,
                 log_tail_lines: int = 5,
                 only_valid_logs: bool = True):
        self.pids = pids or []
        self.log_dir = log_dir
        self.interval = interval
        self.log_tail_lines = log_tail_lines
        self.only_valid_logs = only_valid_logs
        
        self.gpu_monitor = GPUMonitor()
        self.process_monitor = ProcessMonitor()
        self.log_analyzer = LogAnalyzer()
    
    def run_once(self):
        """运行一次监控"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}📊 训练任务监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
        
        # GPU状态
        gpus = self.gpu_monitor.get_gpu_info()
        print(self.gpu_monitor.format_gpu_summary(gpus))
        
        # 进程状态
        if not self.pids:
            self.pids = self.process_monitor.find_training_processes()
        
        if self.pids:
            print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
            print(f"{Colors.HEADER}🔄 运行中的进程 ({len(self.pids)}){Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
            
            gpu_processes = {p['pid']: p for p in self.gpu_monitor.get_gpu_processes()}
            
            for pid in self.pids:
                proc_info = self.process_monitor.get_process_info(pid)
                if proc_info:
                    gpu_mem = gpu_processes.get(pid, {}).get('gpu_memory', 0)
                    print(f"{Colors.OKGREEN}PID {pid}{Colors.ENDC} | "
                          f"运行时间: {proc_info['elapsed_time']} | "
                          f"GPU显存: {gpu_mem} MiB")
                    
                    # 显示命令（截断）
                    cmd = proc_info['command']
                    if len(cmd) > 80:
                        cmd = cmd[:77] + '...'
                    print(f"  命令: {cmd}\n")
        
        # 日志分析
        if self.log_dir:
            logs = self.log_analyzer.find_training_logs(self.log_dir, only_valid=self.only_valid_logs)
            
            if logs:
                filter_note = " (仅显示有效训练)" if self.only_valid_logs else ""
                print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
                print(f"{Colors.HEADER}📋 训练日志 ({len(logs)}){filter_note}{Colors.ENDC}")
                print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
                
                for log_file in logs[:10]:  # 最多显示10个
                    log_info = self.log_analyzer.parse_training_progress(
                        log_file, tail_lines=self.log_tail_lines
                    )
                    print(self.log_analyzer.format_log_summary(log_info, max_lines=3))
        
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}⏱️  下次刷新: {self.interval}秒后{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
    
    def run(self):
        """持续监控"""
        try:
            while True:
                self.run_once()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}⏹️  监控已停止{Colors.ENDC}\n")


def main():
    parser = argparse.ArgumentParser(
        description='通用深度学习训练任务监控脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--pids', type=int, nargs='+',
                       help='指定要监控的进程PID（默认自动检测）')
    parser.add_argument('--log-dir', type=str, default='cache',
                       help='训练日志目录（默认: cache）')
    parser.add_argument('--interval', type=int, default=30,
                       help='刷新间隔（秒，默认: 30）')
    parser.add_argument('--log-lines', type=int, default=5,
                       help='每个日志显示的行数（默认: 5）')
    parser.add_argument('--once', action='store_true',
                       help='只运行一次，不持续监控')
    parser.add_argument('--show-all', action='store_true',
                       help='显示所有日志（包括失败的任务，默认只显示有效训练）')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(
        pids=args.pids,
        log_dir=args.log_dir,
        interval=args.interval,
        log_tail_lines=args.log_lines,
        only_valid_logs=not args.show_all  # 默认只显示有效训练
    )
    
    if args.once:
        monitor.run_once()
    else:
        monitor.run()


if __name__ == '__main__':
    main()
