#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube NER 优化实验结果收集脚本

功能：
- 扫描 experiments/EXP-010-optimization/results/ 下所有子目录
- 读取每个子目录中的 results.json 文件
- 提取 model_type, best_dev_f1, test_f1, seed 等信息
- 按实验组计算3个种子的平均值和标准差
- 输出一个汇总表格（打印到终端）
"""

import os
import sys
import json
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# 结果目录路径
DEFAULT_RESULTS_DIR = "experiments/EXP-010-optimization/results"


def find_result_files(base_dir: str) -> Dict[str, List[str]]:
    """
    扫描结果目录，按实验组收集 results.json 文件路径
    
    Args:
        base_dir: 结果目录根路径
        
    Returns:
        字典 {实验组名称: [results.json路径列表]}
    """
    results_by_group = defaultdict(list)
    
    if not os.path.exists(base_dir):
        print(f"❌ 结果目录不存在: {base_dir}")
        return results_by_group
    
    # 遍历所有一级子目录（实验组）
    for group_name in os.listdir(base_dir):
        group_path = os.path.join(base_dir, group_name)
        if not os.path.isdir(group_path):
            continue
        
        # 在实验组目录下搜索 results.json（可能在子目录中）
        pattern = os.path.join(group_path, "**", "results.json")
        for result_file in glob.glob(pattern, recursive=True):
            results_by_group[group_name].append(result_file)
    
    return dict(results_by_group)


def load_result_file(file_path: str) -> Optional[Dict]:
    """
    加载单个 results.json 文件
    
    Args:
        file_path: results.json 文件路径
        
    Returns:
        解析后的字典，失败返回 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 无法加载 {file_path}: {e}")
        return None


def extract_metrics(result: Dict) -> Dict:
    """
    从结果字典中提取关键指标
    
    Args:
        result: results.json 解析后的字典
        
    Returns:
        提取的指标字典
    """
    metrics = {
        'model_type': result.get('model_type', 'unknown'),
        'seed': result.get('seed', -1),
        'best_dev_f1': result.get('best_dev_f1', 0.0),
        'test_f1': result.get('test_f1', 0.0),
        'test_precision': result.get('test_precision', 0.0),
        'test_recall': result.get('test_recall', 0.0),
        'best_epoch': result.get('best_epoch', -1),
        'use_fgm': result.get('use_fgm', False),
        'use_ema': result.get('use_ema', False),
        'use_rdrop': result.get('use_rdrop', False),
        'use_augment': result.get('use_augment', False),
    }
    return metrics


def calculate_statistics(metrics_list: List[Dict]) -> Dict:
    """
    计算一组实验结果的统计信息
    
    Args:
        metrics_list: 指标字典列表
        
    Returns:
        统计信息字典
    """
    if not metrics_list:
        return {
            'count': 0,
            'dev_f1_mean': 0.0, 'dev_f1_std': 0.0,
            'test_f1_mean': 0.0, 'test_f1_std': 0.0,
        }
    
    import numpy as np
    
    dev_f1_values = [m['best_dev_f1'] for m in metrics_list]
    test_f1_values = [m['test_f1'] for m in metrics_list]
    seeds = [m['seed'] for m in metrics_list]
    
    return {
        'count': len(metrics_list),
        'seeds': seeds,
        'dev_f1_mean': np.mean(dev_f1_values),
        'dev_f1_std': np.std(dev_f1_values),
        'test_f1_mean': np.mean(test_f1_values),
        'test_f1_std': np.std(test_f1_values),
        'dev_f1_max': max(dev_f1_values),
        'test_f1_max': max(test_f1_values),
    }


def get_experiment_description(group_name: str) -> str:
    """
    获取实验组的描述
    
    Args:
        group_name: 实验组名称
        
    Returns:
        实验描述
    """
    descriptions = {
        'A_baseline': '基线（禁用FGM/EMA）',
        'B_fgm': '仅FGM对抗训练',
        'C_fgm_ema': 'FGM + EMA',
        'D_fgm_ema_aug': 'FGM + EMA + 数据增强',
        'E_fgm_ema_rdrop': 'FGM + EMA + R-Drop',
        'F_all_optimizations': '全部优化',
        'G_bilstm_baseline': 'BiLSTM-CRF基线',
    }
    return descriptions.get(group_name, group_name)


def print_summary_table(results_by_group: Dict[str, Dict]):
    """
    打印汇总表格
    
    Args:
        results_by_group: {实验组名称: 统计信息字典}
    """
    print("\n" + "=" * 100)
    print("RedJujube NER 优化实验结果汇总")
    print("=" * 100)
    
    # 表头
    header = f"{'实验组':<25} {'描述':<20} {'#':<3} {'Dev F1 (mean±std)':<20} {'Test F1 (mean±std)':<20}"
    print(header)
    print("-" * 100)
    
    # 按实验组名称排序
    sorted_groups = sorted(results_by_group.keys())
    
    for group_name in sorted_groups:
        stats = results_by_group[group_name]
        desc = get_experiment_description(group_name)
        
        if stats['count'] > 0:
            dev_f1_str = f"{stats['dev_f1_mean']*100:.2f}±{stats['dev_f1_std']*100:.2f}"
            test_f1_str = f"{stats['test_f1_mean']*100:.2f}±{stats['test_f1_std']*100:.2f}"
        else:
            dev_f1_str = "N/A"
            test_f1_str = "N/A"
        
        row = f"{group_name:<25} {desc:<20} {stats['count']:<3} {dev_f1_str:<20} {test_f1_str:<20}"
        print(row)
    
    print("-" * 100)
    
    # 打印最佳结果
    print("\n📊 最佳结果:")
    best_dev_f1 = 0.0
    best_dev_group = ""
    best_test_f1 = 0.0
    best_test_group = ""
    
    for group_name, stats in results_by_group.items():
        if stats['count'] > 0:
            if stats['dev_f1_mean'] > best_dev_f1:
                best_dev_f1 = stats['dev_f1_mean']
                best_dev_group = group_name
            if stats['test_f1_mean'] > best_test_f1:
                best_test_f1 = stats['test_f1_mean']
                best_test_group = group_name
    
    if best_dev_group:
        print(f"  最高平均 Dev F1:  {best_dev_f1*100:.2f}% ({best_dev_group})")
    if best_test_group:
        print(f"  最高平均 Test F1: {best_test_f1*100:.2f}% ({best_test_group})")
    
    print("=" * 100)


def print_detailed_results(all_results: Dict[str, List[Dict]]):
    """
    打印详细结果（每个实验的具体数值）
    
    Args:
        all_results: {实验组名称: [指标字典列表]}
    """
    print("\n" + "=" * 100)
    print("详细实验结果")
    print("=" * 100)
    
    for group_name in sorted(all_results.keys()):
        metrics_list = all_results[group_name]
        print(f"\n📁 {group_name} ({get_experiment_description(group_name)})")
        print("-" * 60)
        
        if not metrics_list:
            print("  无有效结果")
            continue
        
        # 按 seed 排序
        sorted_metrics = sorted(metrics_list, key=lambda x: x.get('seed', 0))
        
        for m in sorted_metrics:
            seed = m.get('seed', '?')
            dev_f1 = m.get('best_dev_f1', 0.0) * 100
            test_f1 = m.get('test_f1', 0.0) * 100
            epoch = m.get('best_epoch', '?')
            
            # 优化标志
            flags = []
            if m.get('use_fgm'):
                flags.append('FGM')
            if m.get('use_ema'):
                flags.append('EMA')
            if m.get('use_rdrop'):
                flags.append('R-Drop')
            if m.get('use_augment'):
                flags.append('Aug')
            flags_str = '+'.join(flags) if flags else '无'
            
            print(f"  seed={seed}: Dev F1={dev_f1:.2f}%, Test F1={test_f1:.2f}%, "
                  f"Epoch={epoch}, 优化=[{flags_str}]")
    
    print("\n" + "=" * 100)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="收集 RedJujube NER 优化实验结果")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="结果目录路径")
    parser.add_argument("--detailed", action="store_true", default=False,
                        help="显示详细结果")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 文件路径（可选）")
    
    args = parser.parse_args()
    
    print(f"📂 扫描结果目录: {args.results_dir}")
    
    # 查找所有结果文件
    result_files = find_result_files(args.results_dir)
    
    if not result_files:
        print("❌ 未找到任何实验结果")
        sys.exit(1)
    
    print(f"✅ 找到 {len(result_files)} 个实验组")
    
    # 加载并提取所有结果
    all_results = {}
    statistics = {}
    
    for group_name, file_paths in result_files.items():
        metrics_list = []
        for fp in file_paths:
            result = load_result_file(fp)
            if result:
                metrics = extract_metrics(result)
                metrics_list.append(metrics)
        
        all_results[group_name] = metrics_list
        statistics[group_name] = calculate_statistics(metrics_list)
    
    # 打印汇总表格
    print_summary_table(statistics)
    
    # 打印详细结果（可选）
    if args.detailed:
        print_detailed_results(all_results)
    
    # 输出 JSON 文件（可选）
    if args.output:
        output_data = {
            'results_dir': args.results_dir,
            'statistics': {},
            'details': {}
        }
        for group_name, stats in statistics.items():
            # 转换为可序列化格式
            output_data['statistics'][group_name] = {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in stats.items()
            }
        for group_name, metrics_list in all_results.items():
            output_data['details'][group_name] = metrics_list
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n📄 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
