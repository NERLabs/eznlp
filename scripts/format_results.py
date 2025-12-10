#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
格式化并美化实验结果展示
"""
import json
import argparse
from pathlib import Path


def format_percentage(value):
    """格式化百分比"""
    if value is None:
        return 'N/A'
    return f"{value * 100:.2f}%"


def format_number(value, precision=2):
    """格式化数字"""
    if value is None:
        return 'N/A'
    return f"{value:.{precision}f}"


def display_comparison_results(json_file):
    """显示对比实验结果"""
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"{'HZ 数据集 NER 对比实验结果汇总':^76}")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        exp_dir = result.get('experiment_dir', 'N/A')
        timestamp = result.get('timestamp', 'N/A')
        
        print(f"实验 {i}: {exp_dir}")
        print(f"{'─'*80}")
        print(f"时间戳: {timestamp}")
        
        # Baseline 结果
        baseline_loss = result.get('baseline_test_loss')
        baseline_f1 = result.get('baseline_f1')
        baseline_precision = result.get('baseline_precision')
        baseline_recall = result.get('baseline_recall')
        baseline_params = result.get('baseline_params')
        
        # ExpertDict 结果
        expert_loss = result.get('expert_test_loss')
        expert_f1 = result.get('expert_f1')
        expert_precision = result.get('expert_precision')
        expert_recall = result.get('expert_recall')
        expert_params = result.get('expert_params')
        
        # 显示 Baseline
        print(f"\n📊 Baseline 模型:")
        if baseline_f1 is not None:
            print(f"   F1-Score:  {format_percentage(baseline_f1)}")
        if baseline_precision is not None:
            print(f"   Precision: {format_percentage(baseline_precision)}")
        if baseline_recall is not None:
            print(f"   Recall:    {format_percentage(baseline_recall)}")
        if baseline_loss is not None:
            print(f"   Test Loss: {format_number(baseline_loss, 4)}")
        if baseline_params is not None:
            print(f"   参数量:    {baseline_params:,}")
        
        # 显示 ExpertDict
        print(f"\n📊 +ExpertDict 模型:")
        if expert_f1 is not None:
            print(f"   F1-Score:  {format_percentage(expert_f1)}")
        if expert_precision is not None:
            print(f"   Precision: {format_percentage(expert_precision)}")
        if expert_recall is not None:
            print(f"   Recall:    {format_percentage(expert_recall)}")
        if expert_loss is not None:
            print(f"   Test Loss: {format_number(expert_loss, 4)}")
        if expert_params is not None:
            print(f"   参数量:    {expert_params:,}")
        
        # 计算提升
        if baseline_f1 is not None and expert_f1 is not None:
            improvement_f1 = (expert_f1 - baseline_f1) * 100
            improvement_pct = (expert_f1 - baseline_f1) / baseline_f1 * 100
            
            print(f"\n🎯 性能提升:")
            print(f"   F1 提升:   {improvement_f1:+.2f}% (绝对值)")
            print(f"   相对提升:  {improvement_pct:+.2f}%")
        
        if baseline_loss is not None and expert_loss is not None:
            loss_reduction = (baseline_loss - expert_loss) / baseline_loss * 100
            print(f"   Loss 降低: {loss_reduction:+.2f}%")
        
        print(f"\n{'='*80}\n")


def display_individual_results(json_file):
    """显示单独实验结果"""
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"{'单独实验结果汇总':^76}")
    print(f"{'='*80}\n")
    
    for result in results:
        exp_name = result.get('experiment', 'N/A')
        model_type = result.get('model_type', 'N/A')
        
        print(f"实验: {exp_name}")
        print(f"类型: {model_type}")
        
        f1 = result.get('f1')
        precision = result.get('precision')
        recall = result.get('recall')
        test_loss = result.get('test_loss')
        
        if f1 is not None:
            print(f"  F1:        {format_percentage(f1)}")
        if precision is not None:
            print(f"  Precision: {format_percentage(precision)}")
        if recall is not None:
            print(f"  Recall:    {format_percentage(recall)}")
        if test_loss is not None:
            print(f"  Loss:      {format_number(test_loss, 4)}")
        
        print()


def main():
    parser = argparse.ArgumentParser(description='格式化显示实验结果')
    parser.add_argument('--comparison', type=str, 
                        default='cache/experiment_results_comparison.json',
                        help='对比实验结果JSON文件')
    parser.add_argument('--individual', type=str,
                        default='cache/experiment_results_individual.json',
                        help='单独实验结果JSON文件')
    parser.add_argument('--mode', type=str, default='comparison',
                        choices=['comparison', 'individual', 'both'],
                        help='显示模式')
    
    args = parser.parse_args()
    
    if args.mode in ['comparison', 'both']:
        if Path(args.comparison).exists():
            display_comparison_results(args.comparison)
        else:
            print(f"⚠️  文件不存在: {args.comparison}")
    
    if args.mode in ['individual', 'both']:
        if Path(args.individual).exists():
            display_individual_results(args.individual)
        else:
            print(f"⚠️  文件不存在: {args.individual}")


if __name__ == '__main__':
    main()
