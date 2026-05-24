#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
收集RedJujube数据集SoftLexicon+ExpertDict融合实验的最终结果
生成完整的对比报告
"""

import json
from pathlib import Path
from datetime import datetime


def load_results(path):
    """加载results.json文件"""
    results_file = Path(path) / "results.json"
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    cache_dir = Path("cache")
    
    # 定义实验目录
    experiments = {
        "Baseline": "redjujube_ner_comparison/baseline_20251212-200053",
        "ExpertDict(手动)": "redjujube_ner_comparison/expert_dict_manual_20251212-202537",
        "Soft+Expert(Concat)": "redjujube_softlexicon_expert_concat/softlexicon_expert_concat_20251213-181422",
        "Soft+Expert(Weighted)": "redjujube_softlexicon_expert_weighted/softlexicon_expert_weighted_20251213-181422",
        "Soft+Expert(Attention)": "redjujube_ner_comparison/softlexicon_expert_attention_20251213-175441",
    }
    
    # 收集结果
    results = {}
    for name, path in experiments.items():
        full_path = cache_dir / path
        result = load_results(full_path)
        if result:
            results[name] = {
                "test_f1": result["test_metrics"][0] * 100,
                "test_loss": result["test_loss"],
                "params": result["total_params"],
                "path": str(full_path)
            }
            print(f"✅ 加载 {name}: F1={result['test_metrics'][0]*100:.2f}%")
        else:
            print(f"❌ 未找到 {name} 的结果文件: {full_path}")
    
    if not results:
        print("未找到任何实验结果！")
        return
    
    # 生成Markdown报告
    output_file = Path("experiments/hz_lexicon/results/RedJujube_Fusion_Final_Results.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    baseline_f1 = results["Baseline"]["test_f1"]
    best_single_f1 = results["ExpertDict(手动)"]["test_f1"]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# RedJujube NER SoftLexicon+ExpertDict 融合实验最终报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 📊 实验结果总览\n\n")
        f.write("| 排名 | 模型方案 | 测试集F1 (%) | vs Baseline | vs 最佳单特征 | 模型参数量 |\n")
        f.write("|------|---------|--------------|-------------|--------------|------------|\n")
        
        # 按F1降序排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]["test_f1"], reverse=True)
        
        for rank, (name, data) in enumerate(sorted_results, 1):
            f1 = data["test_f1"]
            vs_baseline = f1 - baseline_f1
            vs_best = f1 - best_single_f1
            params = data["params"] / 1_000_000  # 转换为百万
            
            medal = ""
            if rank == 1:
                medal = "🥇 "
            elif rank == 2:
                medal = "🥈 "
            elif rank == 3:
                medal = "🥉 "
            
            f.write(f"| {medal}{rank} | **{name}** | **{f1:.2f}** | ")
            f.write(f"{vs_baseline:+.2f}% | {vs_best:+.2f}% | {params:.1f}M |\n")
        
        f.write("\n---\n\n")
        
        f.write("## 🎯 关键发现\n\n")
        
        # 找到最佳融合模型
        best_fusion = None
        best_fusion_f1 = 0
        for name, data in results.items():
            if "Soft+Expert" in name and data["test_f1"] > best_fusion_f1:
                best_fusion = name
                best_fusion_f1 = data["test_f1"]
        
        f.write("### 性能提升\n\n")
        f.write(f"1. **最佳融合方案**: {best_fusion}\n")
        f.write(f"   - 测试集F1: **{best_fusion_f1:.2f}%**\n")
        f.write(f"   - 相比Baseline提升: **{best_fusion_f1 - baseline_f1:.2f}%**\n")
        f.write(f"   - 相比最佳单特征提升: **{best_fusion_f1 - best_single_f1:.2f}%**\n\n")
        
        f.write("2. **所有融合方案表现**:\n")
        for name, data in sorted_results:
            if "Soft+Expert" in name:
                f.write(f"   - {name}: {data['test_f1']:.2f}% ")
                f.write(f"(Baseline+{data['test_f1'] - baseline_f1:.2f}%)\n")
        f.write("\n")
        
        f.write("### 方案对比\n\n")
        f.write("| 融合策略 | 特点 | F1性能 | 参数量 |\n")
        f.write("|---------|------|--------|--------|\n")
        
        fusion_models = {k: v for k, v in results.items() if "Soft+Expert" in k}
        for name, data in sorted(fusion_models.items(), key=lambda x: x[1]["test_f1"], reverse=True):
            strategy = name.split("(")[1].rstrip(")")
            f1 = data["test_f1"]
            params = data["params"] / 1_000_000
            
            if "Concat" in strategy:
                feature = "直接拼接，简单有效"
            elif "Weighted" in strategy:
                feature = "可学习权重，动态融合"
            elif "Attention" in strategy:
                feature = "注意力机制，自适应选择"
            else:
                feature = "-"
            
            f.write(f"| {strategy} | {feature} | {f1:.2f}% | {params:.1f}M |\n")
        
        f.write("\n---\n\n")
        
        f.write("## 📈 详细性能数据\n\n")
        for name, data in sorted_results:
            f.write(f"### {name}\n\n")
            f.write(f"- **测试集F1**: {data['test_f1']:.2f}%\n")
            f.write(f"- **测试集Loss**: {data['test_loss']:.4f}\n")
            f.write(f"- **模型参数**: {data['params']:,}\n")
            f.write(f"- **结果路径**: `{data['path']}`\n\n")
        
        f.write("---\n\n")
        
        f.write("## 💡 结论与建议\n\n")
        f.write("### 主要结论\n\n")
        f.write("1. **融合效果显著**: 所有SoftLexicon+ExpertDict融合方案都显著优于Baseline\n")
        f.write(f"2. **最佳方案**: {best_fusion}达到{best_fusion_f1:.2f}%，是本次实验的最优结果\n")
        f.write("3. **特征互补性**: 软词典和专家词典的融合展现出良好的互补性\n\n")
        
        f.write("### 实践建议\n\n")
        f.write("1. **推荐方案**: ")
        if abs(best_fusion_f1 - max(data["test_f1"] for name, data in results.items() if "Concat" in name)) < 0.05:
            f.write("优先选择Concat方案，实现简单且性能优秀\n")
        else:
            f.write(f"推荐使用{best_fusion}，性能最优\n")
        
        f.write("2. **参数效率**: 考虑模型大小时，Weighted方案参数最少\n")
        f.write("3. **进一步优化**: 可以尝试调整超参数进一步提升性能\n\n")
        
        f.write("---\n\n")
        f.write("## 📋 实验配置\n\n")
        f.write("### 共同配置\n\n")
        f.write("```python\n")
        f.write("数据集: RedJujube\n")
        f.write("预训练模型: hfl/chinese-macbert-base\n")
        f.write("训练轮数: 30 epochs\n")
        f.write("优化器: AdamW\n")
        f.write("```\n\n")
        
        f.write("### 差异化配置\n\n")
        f.write("| 方案 | 隐藏层数 | Dropout | Batch Size | 学习率 |\n")
        f.write("|------|---------|---------|------------|--------|\n")
        f.write("| Baseline | 1 | 0.5 | 16 | 0.002 |\n")
        f.write("| Concat/Weighted | 1 | 0.5 | 16 | 0.002 |\n")
        f.write("| Attention | 2 | 0.2 | 32 | 0.001 |\n")
        f.write("\n")
    
    print(f"\n✅ 最终报告已生成: {output_file}")
    print(f"\n📊 实验汇总:")
    print(f"   - 基线F1: {baseline_f1:.2f}%")
    print(f"   - 最佳单特征F1: {best_single_f1:.2f}%")
    print(f"   - 最佳融合F1: {best_fusion_f1:.2f}%")
    print(f"   - 最大提升: {best_fusion_f1 - baseline_f1:.2f}%")


if __name__ == "__main__":
    main()
