#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动监控训练进度，所有任务完成后自动收集结果
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

class TrainingWatcher:
    """训练完成监视器"""
    
    def __init__(self):
        self.valid_trainings = {
            "Attention": "cache/redjujube_ner_comparison/softlexicon_expert_attention_20251213-175441",
            "Weighted": "cache/redjujube_softlexicon_expert_weighted/softlexicon_expert_weighted_20251213-181422",
            "Concat": "cache/redjujube_softlexicon_expert_concat/softlexicon_expert_concat_20251213-181422"
        }
        
    def check_training_complete(self, log_path):
        """检查训练是否完成"""
        log_file = Path(log_path) / "training.log"
        if not log_file.exists():
            return False, None, None
            
        try:
            # 读取最后50行
            cmd = f"tail -50 {log_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            lines = result.stdout
            
            # 检查是否完成（包含"测试集结果"或"加载最佳模型"）
            is_complete = "测试集结果" in lines or ("加载最佳模型" in lines and "Epoch: 30" in lines)
            
            # 提取当前epoch和Dev F1
            current_epoch = None
            dev_f1 = None
            
            for line in reversed(lines.split('\n')):
                if "Epoch:" in line and current_epoch is None:
                    try:
                        current_epoch = int(line.split("Epoch:")[1].split("|")[0].strip().split()[0])
                    except:
                        pass
                        
                if "Dev.  Metrics:" in line and dev_f1 is None:
                    try:
                        dev_f1 = float(line.split("Dev.  Metrics:")[1].split("%")[0].strip())
                    except:
                        pass
            
            return is_complete, current_epoch, dev_f1
            
        except Exception as e:
            print(f"  ❌ 检查失败: {e}")
            return False, None, None
    
    def get_all_status(self):
        """获取所有训练状态"""
        status = {}
        for name, path in self.valid_trainings.items():
            complete, epoch, dev_f1 = self.check_training_complete(path)
            status[name] = {
                "path": path,
                "complete": complete,
                "epoch": epoch,
                "dev_f1": dev_f1
            }
        return status
    
    def display_status(self, status):
        """显示状态"""
        print(f"\n{'='*70}")
        print(f"📊 训练状态检查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        all_complete = True
        for name, info in status.items():
            emoji = "✅" if info["complete"] else "🔄"
            epoch_str = f"Epoch {info['epoch']}/30" if info['epoch'] else "未开始"
            f1_str = f"Dev F1={info['dev_f1']:.2f}%" if info['dev_f1'] else "N/A"
            status_str = "已完成" if info["complete"] else "训练中"
            
            print(f"{emoji} 方案{name:12s}: {epoch_str:15s} | {f1_str:15s} | {status_str}")
            
            if not info["complete"]:
                all_complete = False
        
        print(f"\n{'='*70}")
        return all_complete
    
    def collect_results(self):
        """收集结果"""
        print("\n🎉 所有训练已完成！开始收集结果...\n")
        
        cmd = "python scripts/collect_fusion_results.py"
        result = subprocess.run(cmd, shell=True, cwd="/home/shiwenlong/NERlabs/eznlp")
        
        if result.returncode == 0:
            print("\n✅ 结果收集完成！")
        else:
            print("\n❌ 结果收集失败！")
    
    def run(self, check_interval=300):
        """运行监控"""
        print("🚀 启动训练完成监视器...\n")
        print(f"⏰ 检查间隔: {check_interval}秒 ({check_interval/60:.1f}分钟)")
        print(f"📁 监控任务: {len(self.valid_trainings)}个\n")
        
        iteration = 0
        while True:
            iteration += 1
            print(f"\n{'#'*70}")
            print(f"# 第 {iteration} 次检查")
            print(f"{'#'*70}")
            
            status = self.get_all_status()
            all_complete = self.display_status(status)
            
            if all_complete:
                self.collect_results()
                print("\n🎊 任务完成！监视器退出。\n")
                break
            else:
                print(f"\n⏳ 等待 {check_interval} 秒后再次检查...\n")
                time.sleep(check_interval)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='自动监控训练并在完成后收集结果')
    parser.add_argument('--interval', type=int, default=300, 
                       help='检查间隔（秒）, 默认300秒=5分钟')
    parser.add_argument('--once', action='store_true',
                       help='只检查一次状态，不循环等待')
    
    args = parser.parse_args()
    
    watcher = TrainingWatcher()
    
    if args.once:
        status = watcher.get_all_status()
        all_complete = watcher.display_status(status)
        if all_complete:
            print("\n✅ 所有训练已完成！")
            print("💡 可以运行以下命令收集结果:")
            print("   python scripts/collect_fusion_results.py")
        else:
            print("\n⏳ 训练尚未全部完成，请稍后再检查。")
    else:
        watcher.run(check_interval=args.interval)

if __name__ == "__main__":
    main()
