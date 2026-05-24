#!/usr/bin/env python3
"""
收集各项目RedJujube数据集实验结果
"""
import json
import os
import re

def collect_results():
    results = {}
    
    # 1. eznlp SoftLexicon
    eznlp_dir = "/home/shiwenlong/NERlabs/eznlp/experiments/EXP-010-optimization/results_newdata/SoftLexicon_baseline"
    eznlp_results = []
    for seed_dir in ["seed_42", "seed_43", "seed_44"]:
        seed_path = os.path.join(eznlp_dir, seed_dir)
        if os.path.exists(seed_path):
            for subdir in os.listdir(seed_path):
                log_file = os.path.join(seed_path, subdir, "training.log")
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        content = f.read()
                        match = re.search(r'Metric 0: (\d+\.\d+)', content)
                        if match:
                            f1 = float(match.group(1))
                            eznlp_results.append({"seed": seed_dir.replace("seed_", ""), "f1": f1})
    if eznlp_results:
        avg_f1 = sum(r["f1"] for r in eznlp_results) / len(eznlp_results)
        results["eznlp_SoftLexicon"] = {"results": eznlp_results, "avg_f1": avg_f1}
    
    # 2. AdaSeq
    adaseq_dir = "/home/shiwenlong/NERlabs/AdaSeq/experiments/redjujube_bert_crf"
    if os.path.exists(adaseq_dir):
        for run_dir in os.listdir(adaseq_dir):
            metrics_file = os.path.join(adaseq_dir, run_dir, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    # Find test results
                    for key in metrics:
                        if "test" in key and "f1" in metrics[key]:
                            results["AdaSeq_BERT_CRF"] = {"f1": metrics[key].get("f1", metrics[key].get("f1-score", 0))}
                            break
    
    # 3. dice_loss
    dice_dir = "/home/shiwenlong/NERlabs/dice_loss_for_NLP/output/redjujube"
    if os.path.exists(dice_dir):
        eval_file = os.path.join(dice_dir, "eval_result_log.txt")
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                content = f.read()
                match = re.search(r'valid_f1 is: (\d+\.\d+)', content)
                if match:
                    results["dice_loss"] = {"status": "in_progress", "current_f1": float(match.group(1))}
    
    # 4. piqn
    piqn_dir = "/home/shiwenlong/NERlabs/piqn/_9LOG/redjujube"
    if os.path.exists(piqn_dir):
        results["piqn"] = {"status": "in_progress"}
    
    return results

if __name__ == "__main__":
    results = collect_results()
    print("=" * 60)
    print("RedJujube Cross-Project Results Summary")
    print("=" * 60)
    for project, data in results.items():
        if "avg_f1" in data:
            print(f"\n{project}:")
            for r in data["results"]:
                print(f"  seed{r['seed']}: {r['f1']:.2f}%")
            print(f"  Average: {data['avg_f1']:.2f}%")
        elif "f1" in data:
            print(f"\n{project}: {data['f1']:.2f}%")
        else:
            print(f"\n{project}: {data.get('status', 'pending')}")
    print("=" * 60)
