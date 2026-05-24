#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨项目实验结果收集脚本
收集多个NER项目在RedJujube数据集上的实验结果
"""

import os
import json
import re
import glob
from pathlib import Path
from datetime import datetime


def collect_eznlp_results():
    """收集eznlp实验结果"""
    results_dir = "experiments/EXP-010-optimization/results_newdata/SoftLexicon_baseline"
    results = []
    
    if os.path.exists(results_dir):
        for seed_dir in glob.glob(os.path.join(results_dir, "seed_*")):
            seed = os.path.basename(seed_dir).split("_")[1]
            for exp_dir in glob.glob(os.path.join(seed_dir, "softlexicon_*")):
                # 查找metrics.json或训练日志
                metrics_file = os.path.join(exp_dir, "metrics.json")
                log_file = glob.glob(os.path.join(exp_dir, "*.log"))
                
                result = {
                    "project": "eznlp",
                    "model": "SoftLexicon",
                    "seed": seed,
                    "exp_dir": exp_dir
                }
                
                if os.path.exists(metrics_file):
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                        result["precision"] = metrics.get("precision", 0)
                        result["recall"] = metrics.get("recall", 0)
                        result["f1"] = metrics.get("f1", 0)
                
                # 从日志中提取最终结果
                if log_file:
                    with open(log_file[0], "r") as f:
                        content = f.read()
                        # 查找最终测试结果
                        match = re.search(r"Test\.?\s+Metrics:\s+(\d+\.\d+)%", content)
                        if match:
                            result["test_f1"] = float(match.group(1))
                
                results.append(result)
    
    return results


def collect_adaseq_results():
    """收集AdaSeq实验结果"""
    results_dir = "/home/shiwenlong/NERlabs/AdaSeq/experiments/redjujube_bert_crf"
    results = []
    
    if os.path.exists(results_dir):
        for exp_dir in glob.glob(os.path.join(results_dir, "*")):
            metrics_file = os.path.join(exp_dir, "metrics.json")
            
            result = {
                "project": "AdaSeq",
                "model": "BERT-CRF",
                "exp_dir": exp_dir
            }
            
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                    # AdaSeq metrics format
                    if "ner_metric" in metrics:
                        ner_metrics = metrics["ner_metric"]
                        result["precision"] = ner_metrics.get("precision", 0)
                        result["recall"] = ner_metrics.get("recall", 0)
                        result["f1"] = ner_metrics.get("f1", 0)
            
            results.append(result)
    
    return results


def collect_dice_loss_results():
    """收集dice_loss实验结果"""
    results_dir = "/home/shiwenlong/NERlabs/dice_loss_for_NLP/output/redjujube"
    results = []
    
    if os.path.exists(results_dir):
        # 查找eval_result_log.txt
        log_file = os.path.join(results_dir, "eval_result_log.txt")
        
        result = {
            "project": "dice_loss",
            "model": "DiceLoss-MRC-NER",
            "exp_dir": results_dir
        }
        
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                # 提取Span-F1
                match = re.search(r"Span-F1:\s+(\d+\.\d+)", content)
                if match:
                    result["f1"] = float(match.group(1))
        
        results.append(result)
    
    return results


def collect_piqn_results():
    """收集piqn实验结果"""
    results_dir = "/home/shiwenlong/NERlabs/piqn/_9LOG/checkpoints/redjujube"
    results = []
    
    if os.path.exists(results_dir):
        # 查找metrics.json或结果文件
        for f in glob.glob(os.path.join(results_dir, "*.json")):
            result = {
                "project": "piqn",
                "model": "PIQN",
                "exp_dir": results_dir
            }
            
            with open(f, "r") as fp:
                metrics = json.load(fp)
                if "f1" in metrics:
                    result["f1"] = metrics["f1"]
            
            results.append(result)
    
    return results


def collect_flat_results():
    """收集FLAT实验结果"""
    results_dir = "/home/shiwenlong/NERlabs/Flat-Lattice-Transformer/V0/logs"
    results = []
    
    if os.path.exists(results_dir):
        for log_dir in glob.glob(os.path.join(results_dir, "*")):
            result = {
                "project": "FLAT",
                "model": "Flat-Lattice-Transformer",
                "exp_dir": log_dir
            }
            
            # FLAT使用fitlog，查找best_performance.json
            perf_file = os.path.join(log_dir, "best_performance.json")
            if os.path.exists(perf_file):
                with open(perf_file, "r") as f:
                    perf = json.load(f)
                    if "f1" in perf:
                        result["f1"] = perf["f1"]
            
            results.append(result)
    
    return results


def format_results_table(results):
    """格式化结果为表格"""
    print("\n" + "=" * 80)
    print("RedJujube数据集跨项目实验结果对比")
    print("=" * 80)
    print(f"| 项目 | 模型 | Precision | Recall | F1 |")
    print(f"|------|------|-----------|--------|-----|")
    
    for r in results:
        project = r.get("project", "N/A")
        model = r.get("model", "N/A")
        precision = r.get("precision", r.get("test_f1", 0))
        recall = r.get("recall", 0)
        f1 = r.get("f1", r.get("test_f1", 0))
        
        print(f"| {project} | {model} | {precision:.2f}% | {recall:.2f}% | {f1:.2f}% |")
    
    print("=" * 80)


def main():
    """主函数"""
    all_results = []
    
    # 收集各项目结果
    print("收集eznlp结果...")
    all_results.extend(collect_eznlp_results())
    
    print("收集AdaSeq结果...")
    all_results.extend(collect_adaseq_results())
    
    print("收集dice_loss结果...")
    all_results.extend(collect_dice_loss_results())
    
    print("收集piqn结果...")
    all_results.extend(collect_piqn_results())
    
    print("收集FLAT结果...")
    all_results.extend(collect_flat_results())
    
    # 格式化输出
    format_results_table(all_results)
    
    # 保存到JSON文件
    output_file = "research/evaluation/cross_project_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()