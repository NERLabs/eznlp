#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试训练脚本（用小数据集验证代码正确性）
"""

import subprocess
import sys

print("🧪 快速测试训练脚本...")
print("=" * 70)

# 运行快速测试（只训练 2 个 epoch，小批次）
cmd = [
    sys.executable,
    "scripts/train_hz_ner_baseline_vs_expert_dict.py",
    "--run_both",
    "--data_dir", "data/HZ",
    "--expert_dict_path", "data/HZ/expert_lexicon.txt",
    "--save_dir", "cache/hz_ner_test",
    "--bert_arch", "hfl/chinese-macbert-base",
    "--num_epochs", "2",
    "--batch_size", "8",
    "--disp_every_steps", "10",
    "--eval_every_steps", "20",
    "--seed", "42"
]

print(f"运行命令: {' '.join(cmd)}\n")

try:
    subprocess.run(cmd, check=True)
    print("\n✅ 测试通过！脚本运行正常。")
except subprocess.CalledProcessError as e:
    print(f"\n❌ 测试失败！错误码: {e.returncode}")
    sys.exit(1)
