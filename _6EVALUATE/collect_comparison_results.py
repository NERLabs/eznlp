#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验结果收集工具 - 专门用于对比实验
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd


def collect_comparison_results(base_dir: str) -> List[Dict]:
    """收集对比实验结果"""
    results = []
    
    # 查找所有 comparison_*.json 文件
    for comparison_file in Path(base_dir).rglob("comparison_*.json"):
        try:
            with open(comparison_file, 'r') as f:
                data = json.load(f)
                
            # 提取实验信息
            result = {
                'experiment_dir': comparison_file.parent.name,
                'timestamp': comparison_file.stem.split('_', 1)[1],
            }
            
            # 提取 baseline 和 expert_dict 结果
            for key in ['baseline', 'expert_dict']:
                if key in data:
                    prefix = 'baseline' if key == 'baseline' else 'expert'
                    result[f'{prefix}_test_loss'] = data[key].get('test_loss', None)
                    
                    metrics = data[key].get('test_metrics', [])
                    if len(metrics) >= 3:
                        result[f'{prefix}_precision'] = metrics[0]
                        result[f'{prefix}_recall'] = metrics[1]
                        result[f'{prefix}_f1'] = metrics[2]
                    elif len(metrics) == 1:
                        # 只有F1分数
                        result[f'{prefix}_f1'] = metrics[0]
                    
                    result[f'{prefix}_params'] = data[key].get('total_params', None)
            
            # 提取元信息
            if 'summary' in data:
                result['data_dir'] = data['summary'].get('data_dir', '')
                result['expert_dict_path'] = data['summary'].get('expert_dict_path', '')
                result['improvement_f1'] = data['summary'].get('improvement_f1', 0)
            
            results.append(result)
            
        except Exception as e:
            print(f"⚠️  跳过 {comparison_file}: {e}")
    
    return results


def collect_individual_results(base_dir: str) -> List[Dict]:
    """收集单独实验结果"""
    results = []
    
    # 查找所有 results.json 文件
    for result_file in Path(base_dir).rglob("results.json"):
        # 跳过在对比实验目录中的
        if 'comparison' in str(result_file.parent.parent):
            parent_comparison = result_file.parent.parent / f"comparison_{result_file.parent.name.split('_', 1)[1]}.json"
            if parent_comparison.exists():
                continue
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            result = {
                'experiment': result_file.parent.name,
                'model_type': data.get('model_type', ''),
                'test_loss': data.get('test_loss', None),
            }
            
            metrics = data.get('test_metrics', [])
            if len(metrics) >= 3:
                result['precision'] = metrics[0]
                result['recall'] = metrics[1]
                result['f1'] = metrics[2]
            
            result['total_params'] = data.get('total_params', None)
            result['trainable_params'] = data.get('trainable_params', None)
            
            # 提取配置信息
            args = data.get('args', {})
            result['data_dir'] = args.get('data_dir', '')
            result['num_epochs'] = args.get('num_epochs', None)
            result['batch_size'] = args.get('batch_size', None)
            result['learning_rate'] = args.get('learning_rate', None)
            
            results.append(result)
            
        except Exception as e:
            print(f"⚠️  跳过 {result_file}: {e}")
    
    return results


def format_percentage(value):
    """格式化百分比"""
    if value is None:
        return 'N/A'
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description='收集实验结果')
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='缓存目录路径')
    parser.add_argument('--output', type=str, default='experiment_results',
                        help='输出文件路径(不含扩展名)')
    parser.add_argument('--mode', type=str, default='comparison',
                        choices=['comparison', 'individual', 'both'],
                        help='收集模式')
    parser.add_argument('--format', type=str, default='json',
                        choices=['json', 'excel', 'both'],
                        help='输出格式')
    
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print(f"📊 实验结果收集")
    print(f"{'='*70}\n")
    
    # 收集对比实验结果
    if args.mode in ['comparison', 'both']:
        print("📦 收集对比实验结果...")
        comparison_results = collect_comparison_results(args.cache_dir)
        
        if comparison_results:
            df_comparison = pd.DataFrame(comparison_results)
            
            # 排序
            df_comparison = df_comparison.sort_values('timestamp', ascending=False)
            
            print(f"   找到 {len(df_comparison)} 个对比实验\n")
            
            # 显示摘要
            print("📈 对比实验摘要:")
            for _, row in df_comparison.iterrows():
                print(f"\n实验: {row['experiment_dir']}")
                print(f"  时间: {row['timestamp']}")
                print(f"  数据集: {row.get('data_dir', 'N/A')}")
                
                baseline_f1 = row.get('baseline_f1')
                expert_f1 = row.get('expert_f1')
                
                if baseline_f1 is not None:
                    print(f"  Baseline F1: {format_percentage(baseline_f1)}")
                if expert_f1 is not None:
                    print(f"  +ExpertDict F1: {format_percentage(expert_f1)}")
                if baseline_f1 is not None and expert_f1 is not None:
                    improvement = (expert_f1 - baseline_f1) * 100
                    print(f"  提升: {improvement:+.2f}%")
            
            # 保存结果
            if args.format in ['json', 'both']:
                output_json = f"{args.output}_comparison.json"
                df_comparison.to_json(output_json, orient='records', indent=2, force_ascii=False)
                print(f"\n✅ 对比实验结果已保存到: {output_json}")
            
            if args.format in ['excel', 'both']:
                try:
                    output_excel = f"{args.output}_comparison.xlsx"
                    df_comparison.to_excel(output_excel, index=False)
                    print(f"✅ Excel结果已保存到: {output_excel}")
                except ImportError:
                    print("⚠️  未安装openpyxl,跳过Excel导出")
    
    # 收集单独实验结果
    if args.mode in ['individual', 'both']:
        print("\n📦 收集单独实验结果...")
        individual_results = collect_individual_results(args.cache_dir)
        
        if individual_results:
            df_individual = pd.DataFrame(individual_results)
            
            # 排序
            df_individual = df_individual.sort_values('experiment', ascending=False)
            
            print(f"   找到 {len(df_individual)} 个单独实验\n")
            
            # 显示摘要
            print("📈 单独实验摘要:")
            for _, row in df_individual.iterrows():
                print(f"\n实验: {row['experiment']}")
                print(f"  类型: {row.get('model_type', 'N/A')}")
                print(f"  F1: {format_percentage(row.get('f1'))}")
            
            # 保存结果
            if args.format in ['json', 'both']:
                output_json = f"{args.output}_individual.json"
                df_individual.to_json(output_json, orient='records', indent=2, force_ascii=False)
                print(f"\n✅ 单独实验结果已保存到: {output_json}")
            
            if args.format in ['excel', 'both']:
                try:
                    output_excel = f"{args.output}_individual.xlsx"
                    df_individual.to_excel(output_excel, index=False)
                    print(f"✅ Excel结果已保存到: {output_excel}")
                except ImportError:
                    print("⚠️  未安装openpyxl,跳过Excel导出")
    
    print(f"\n{'='*70}")
    print("✅ 结果收集完成")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
