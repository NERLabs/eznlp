#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集MSRA官方脚本实验结果并集成到12-3周实验目录
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache" / "MSRA-ER"
RESULT_DIR = PROJECT_ROOT / "experiments" / "hz_lexicon" / "results" / "12-3_expert_optimization"


def find_latest_experiment():
    """查找最新的MSRA实验目录"""
    if not CACHE_DIR.exists():
        print(f"❌ Cache目录不存在: {CACHE_DIR}")
        return None
    
    # 查找今天的实验目录
    today = datetime.now().strftime("%Y%m%d")
    exp_dirs = list(CACHE_DIR.glob(f"{today}-*"))
    
    if not exp_dirs:
        print(f"❌ 未找到今天的实验目录 ({today})")
        return None
    
    # 按时间戳排序，取最新的
    exp_dirs.sort(key=lambda x: x.name, reverse=True)
    latest_dir = exp_dirs[0]
    
    print(f"✅ 找到最新实验目录: {latest_dir.name}")
    return latest_dir


def extract_results(exp_dir):
    """从实验目录提取结果"""
    results = {
        "exp_dir": exp_dir.name,
        "model_type": "SoftLexicon (官方脚本)",
        "timestamp": None,
        "test_f1": None,
        "dev_f1": None,
        "params": None,
        "config": {}
    }
    
    # 查找训练日志
    log_file = exp_dir / "training.log"
    if log_file.exists():
        print(f"✅ 找到训练日志: {log_file.name}")
        
        # 解析日志获取最终结果
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 查找测试集结果
            for i, line in enumerate(lines):
                if "Test.  Metrics:" in line:
                    # 提取F1
                    parts = line.split("Test.  Metrics:")
                    if len(parts) > 1:
                        f1_str = parts[1].strip().rstrip('%')
                        try:
                            results["test_f1"] = float(f1_str)
                        except:
                            pass
                
                # 查找Dev最佳结果
                if "Dev.  Metrics:" in line:
                    parts = line.split("Dev.  Metrics:")
                    if len(parts) > 1:
                        f1_str = parts[1].strip().split()[0].rstrip('%')
                        try:
                            dev_f1 = float(f1_str)
                            if results["dev_f1"] is None or dev_f1 > results["dev_f1"]:
                                results["dev_f1"] = dev_f1
                        except:
                            pass
                
                # 查找参数量
                if "The model has" in line and "parameters" in line:
                    parts = line.split("has")[1].split("parameters")[0]
                    try:
                        params_str = parts.strip().replace(',', '')
                        results["params"] = int(params_str)
                    except:
                        pass
    
    # 查找配置信息(从日志开头)
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "'batch_size':" in content:
                # 提取关键配置
                for key in ['batch_size', 'lr', 'finetune_lr', 'num_epochs', 'hid_dim', 'num_layers']:
                    if f"'{key}':" in content:
                        try:
                            start = content.find(f"'{key}':")
                            end = content.find(',', start)
                            value_str = content[start:end].split(':')[1].strip()
                            results["config"][key] = eval(value_str)
                        except:
                            pass
    
    return results


def save_to_result_dir(results, exp_dir):
    """保存结果到12-3周目录"""
    # 创建结果子目录
    result_subdir = RESULT_DIR / "msra_softlexicon_official"
    result_subdir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 创建结果目录: {result_subdir.relative_to(PROJECT_ROOT)}")
    
    # 复制关键文件
    files_to_copy = [
        ("training.log", "training.log"),
    ]
    
    for src_name, dst_name in files_to_copy:
        src = exp_dir / src_name
        dst = result_subdir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✅ {src_name}")
    
    # 保存结果JSON
    result_json = {
        "experiment": "MSRA SoftLexicon (官方脚本)",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model_type": results["model_type"],
        "cache_dir": results["exp_dir"],
        "results": {
            "test_f1": results["test_f1"],
            "dev_f1_best": results["dev_f1"],
        },
        "model_config": {
            "bert": "hfl/chinese-macbert-base",
            "encoder": "BiLSTM",
            "decoder": "CRF",
            "features": "SoftLexicon (CTB词典)",
            "params": results["params"]
        },
        "training_config": results["config"]
    }
    
    with open(result_subdir / "results.json", 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"  ✅ results.json")
    
    return result_json


def update_report(result_json):
    """更新实验报告"""
    report_file = RESULT_DIR / "MSRA_ER_Experiments_20251213.md"
    
    if not report_file.exists():
        print(f"\n⚠️  报告文件不存在: {report_file.name}")
        return
    
    # 读取现有报告
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加新实验结果
    new_section = f"""

---

## 实验5: SoftLexicon (官方脚本验证)

**日期**: {result_json['date']}  
**实验类型**: 官方脚本验证实验

### 配置
- **脚本**: `_5TRAIN/entity_recognition.py` (官方)
- **BERT**: {result_json['model_config']['bert']}
- **编码器**: {result_json['model_config']['encoder']} (hidden={result_json['training_config'].get('hid_dim', 256)}, layers={result_json['training_config'].get('num_layers', 1)})
- **解码器**: {result_json['model_config']['decoder']}
- **特征**: {result_json['model_config']['features']}

### 训练参数
- Epochs: {result_json['training_config'].get('num_epochs', 30)}
- Batch Size: {result_json['training_config'].get('batch_size', 16)}
- Learning Rate: {result_json['training_config'].get('lr', 0.002)}
- Finetune LR: {result_json['training_config'].get('finetune_lr', 2e-5)}

### 结果
- **测试集F1**: **{result_json['results']['test_f1']:.2f}%**
- **验证集F1 (最佳)**: {result_json['results']['dev_f1_best']:.2f}%
- **总参数量**: {result_json['model_config']['params']:,}

### 说明
使用官方脚本验证SoftLexicon在MSRA数据集上的性能，与自定义脚本结果对比。
"""
    
    # 追加到文件末尾
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(content)
        f.write(new_section)
    
    print(f"\n✅ 已更新实验报告: {report_file.name}")


def main():
    print("=" * 70)
    print("MSRA官方脚本实验结果收集")
    print("=" * 70)
    
    # 1. 查找最新实验
    exp_dir = find_latest_experiment()
    if not exp_dir:
        return 1
    
    # 2. 提取结果
    print(f"\n📊 提取实验结果...")
    results = extract_results(exp_dir)
    
    if results["test_f1"] is None:
        print(f"⚠️  警告: 未找到测试集F1结果，实验可能未完成")
        print(f"   请等待训练完成后再运行此脚本")
        return 1
    
    print(f"\n结果摘要:")
    print(f"  测试集F1: {results['test_f1']:.2f}%")
    print(f"  验证集F1: {results['dev_f1']:.2f}%")
    print(f"  参数量: {results['params']:,}")
    
    # 3. 保存到结果目录
    print(f"\n💾 保存结果到12-3周目录...")
    result_json = save_to_result_dir(results, exp_dir)
    
    # 4. 更新实验报告
    print(f"\n📝 更新实验报告...")
    update_report(result_json)
    
    print(f"\n{'=' * 70}")
    print(f"✅ 结果集成完成!")
    print(f"{'=' * 70}")
    print(f"\n保存位置: {RESULT_DIR.relative_to(PROJECT_ROOT)}/msra_softlexicon_official/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
