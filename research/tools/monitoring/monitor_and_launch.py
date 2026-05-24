#!/usr/bin/env python3
"""监控方案A完成并自动启动方案B和C"""

import subprocess
import time
import os
import sys
from datetime import datetime

WORK_DIR = "/home/shiwenlong/NERlabs/eznlp"
SCHEME_A_LOG = f"{WORK_DIR}/cache/redjujube_softlexicon_expert/softlexicon_expert_concat_20251213-172348/training.log"
SCHEME_A_PID = 1018542
CONDA_ENV = "eznlp11"

def check_process_running(pid):
    """检查进程是否在运行"""
    try:
        result = subprocess.run(['ps', '-p', str(pid)], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def get_latest_log_line(log_file):
    """获取日志文件最后一行"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-1].strip() if lines else ""
    except:
        return ""

def check_training_completed(log_file):
    """检查训练是否完成"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            return "Training completed" in content or "Best dev" in content[-2000:]
    except:
        return False

def run_training(scheme_name, run_flag):
    """运行训练任务"""
    print(f"\n{'='*50}")
    print(f"🚀 启动{scheme_name}...")
    print(f"{'='*50}\n")
    
    cmd = [
        "conda", "run", "-n", CONDA_ENV,
        "python", "scripts/train_redjujube_ner_comparison.py",
        "--data_dir", "data/RedJujube",
        run_flag,
        "--save_dir", "cache/redjujube_ner_comparison",
        "--expert_dict_auto_path", "data/RedJujube/expert_lexicon_auto.txt",
        "--softlex_train_path", "data/RedJujube/softlexicon_train.txt",
        "--bert_arch", "hfl/chinese-macbert-base",
        "--hid_dim", "256",
        "--num_layers", "1",
        "--dropout", "0.5",
        "--num_epochs", "30",
        "--batch_size", "16",
        "--lr", "2e-3",
        "--finetune_lr", "2e-5",
        "--weight_decay", "1e-4",
        "--grad_clip", "5.0",
        "--seed", "42"
    ]
    
    os.chdir(WORK_DIR)
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n{'='*50}")
        print(f"✅ {scheme_name}训练完成")
        print(f"{'='*50}\n")
        return True
    else:
        print(f"\n{'='*50}")
        print(f"❌ {scheme_name}训练失败 (退出码: {result.returncode})")
        print(f"{'='*50}\n")
        return False

def main():
    print(f"{'='*50}")
    print(f"⏳ 等待方案A (Concat) 完成...")
    print(f"{'='*50}\n")
    
    # 监控方案A
    while True:
        # 检查进程是否还在运行
        if not check_process_running(SCHEME_A_PID):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 方案A进程已结束")
            break
        
        # 获取最新进度
        latest_line = get_latest_log_line(SCHEME_A_LOG)
        if latest_line:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {latest_line}")
        
        time.sleep(30)  # 每30秒检查一次
    
    # 等待5秒确保资源释放
    print("\n等待5秒以释放GPU资源...\n")
    time.sleep(5)
    
    # 启动方案B
    success_b = run_training("方案B (Weighted)", "--run_softlexicon_expert_weighted")
    if not success_b:
        print("方案B失败，终止后续任务")
        sys.exit(1)
    
    # 等待5秒
    print("\n等待5秒以释放GPU资源...\n")
    time.sleep(5)
    
    # 启动方案C
    success_c = run_training("方案C (Gated)", "--run_softlexicon_expert_gated")
    if not success_c:
        print("方案C失败")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print(f"🎉 所有方案训练完成！")
    print(f"{'='*50}\n")
    print("方案A (Concat): cache/redjujube_softlexicon_expert/softlexicon_expert_concat_20251213-172348/")
    print("方案D (Attention): cache/redjujube_ner_comparison/softlexicon_expert_attention_20251213-175441/")
    print("方案B (Weighted): 查看cache/redjujube_ner_comparison/最新目录")
    print("方案C (Gated): 查看cache/redjujube_ner_comparison/最新目录\n")

if __name__ == "__main__":
    main()
