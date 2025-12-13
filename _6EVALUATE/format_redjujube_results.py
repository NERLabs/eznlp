#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
格式化 RedJujube NER 实验结果
"""
import json
import argparse
from pathlib import Path
from datetime import datetime


def generate_markdown_report(json_file, output_file=None):
    """生成 Markdown 格式的实验报告"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构建 Markdown 内容
    lines = []
    lines.append(f"# {data['experiment_name']}")
    lines.append("")
    lines.append(f"**实验日期**: {data['date']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 数据集信息
    lines.append("## 📊 数据集信息")
    lines.append("")
    ds = data['dataset']
    lines.append(f"- **数据集名称**: {ds['name']}")
    lines.append(f"- **训练集样本数**: {ds['train_samples']:,}")
    lines.append(f"- **验证集样本数**: {ds['dev_samples']:,}")
    lines.append(f"- **测试集样本数**: {ds['test_samples']:,}")
    lines.append(f"- **实体总数**: {ds['total_entities']:,}")
    lines.append(f"- **实体类型数**: {ds['entity_types']}")
    lines.append(f"- **平均句长**: {ds['avg_length']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 实验结果表格
    lines.append("## 🎯 实验结果对比")
    lines.append("")
    lines.append("| 方法 | 测试集 F1 | 测试Loss | 参数量 | 提升幅度 | 词典大小 | 词典来源 |")
    lines.append("|------|----------|---------|--------|---------|---------|---------|")
    
    results = data['results']
    for key in ['baseline', 'softlexicon_trainlex', 'expert_dict_auto', 'expert_dict_manual']:
        if key in results:
            r = results[key]
            lines.append(f"| {r['model_type']} | {r['test_f1_percent']} | {r['test_loss']:.3f} | {r['params']:,} | {r['improvement']} | {r['lexicon_size']:,} | {r['lexicon_source']} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 性能排名
    lines.append("## 🏆 性能排名")
    lines.append("")
    lines.append("| 排名 | 方法 | F1 Score | 相对提升 |")
    lines.append("|-----|------|----------|---------|")
    
    for item in data['ranking']:
        lines.append(f"| {item['rank']} | {item['method']} | {item['f1']} | {item['improvement']} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 关键发现
    lines.append("## 💡 关键发现")
    lines.append("")
    for i, finding in enumerate(data['key_findings'], 1):
        lines.append(f"{i}. {finding}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 推荐方案
    lines.append("## 📋 推荐方案")
    lines.append("")
    rec = data['recommendations']
    lines.append(f"- **最佳性能**: {rec['best_performance']}")
    lines.append(f"- **最佳实践**: {rec['best_practice']}")
    lines.append(f"- **快速部署**: {rec['quick_deploy']}")
    lines.append(f"- **不推荐**: {rec['not_recommended']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 模型配置
    lines.append("## ⚙️ 模型配置")
    lines.append("")
    cfg = data['configuration']
    lines.append(f"- **基础模型**: {cfg['base_model']}")
    lines.append(f"- **模型架构**: {cfg['architecture']}")
    lines.append(f"- **训练轮数**: {cfg['epochs']}")
    lines.append(f"- **批次大小**: {cfg['batch_size']}")
    lines.append(f"- **学习率**: {cfg['learning_rate']}")
    lines.append(f"- **微调学习率**: {cfg['finetune_lr']}")
    lines.append(f"- **随机种子**: {cfg['seed']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 实验文件
    lines.append("## 📁 实验文件")
    lines.append("")
    lines.append("- **实验缓存目录**: `cache/redjujube_ner_comparison/`")
    lines.append("- **结果JSON**: `cache/redjujube_ner_comparison/comparison_results.json`")
    lines.append("- **训练脚本**: `scripts/run_redjujube_all_experiments.sh`")
    lines.append("")
    lines.append("### 各实验模型目录")
    lines.append("")
    lines.append("```")
    lines.append("cache/redjujube_ner_comparison/")
    lines.append("├── baseline_20251212-200053/")
    lines.append("│   └── results.json")
    lines.append("├── expert_dict_auto_20251212-202537/")
    lines.append("│   └── results.json")
    lines.append("├── expert_dict_manual_20251212-202537/")
    lines.append("│   └── results.json")
    lines.append("├── softlexicon_trainlex_20251212-202537/")
    lines.append("│   └── results.json")
    lines.append("└── comparison_results.json")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 结论
    lines.append("## 📝 结论")
    lines.append("")
    lines.append("本次 RedJujube 数据集 NER 实验系统对比了多种词典特征方法，主要结论如下：")
    lines.append("")
    lines.append("1. **ExpertDict 方法整体优于 SoftLexicon**")
    lines.append("   - ExpertDict（手动）达到 97.04% F1，提升 +1.53%")
    lines.append("   - ExpertDict（自动）达到 96.99% F1，提升 +1.48%")
    lines.append("   - SoftLexicon（TrainLex）达到 96.07% F1，提升 +0.56%")
    lines.append("")
    lines.append("2. **词典质量比规模更重要**")
    lines.append("   - ExpertDict 用 2,078-3,389 词达到 97%+ F1")
    lines.append("   - SoftLexicon 用 198,437 词仅达到 96.07% F1")
    lines.append("   - 精选专家词典效率是大规模词表的 2.6 倍")
    lines.append("")
    lines.append("3. **自动词典提取策略有效**")
    lines.append("   - 自动提取（min_freq=2）性能接近手动标注")
    lines.append("   - 仅差 0.05% F1，但完全避免数据泄露")
    lines.append("   - 推荐作为最佳实践方案")
    lines.append("")
    lines.append("4. **实践建议**")
    lines.append("   - 生产环境推荐：ExpertDict（自动），平衡性能与安全")
    lines.append("   - 研究实验推荐：ExpertDict（手动），追求最高性能")
    lines.append("   - 快速部署推荐：Baseline，95.51% F1 已足够优秀")
    lines.append("")
    
    # 生成时间
    lines.append("---")
    lines.append("")
    lines.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    content = "\n".join(lines)
    
    # 输出到文件或打印
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 报告已生成: {output_file}")
    else:
        print(content)
    
    return content


def main():
    parser = argparse.ArgumentParser(description='格式化 RedJujube NER 实验结果')
    parser.add_argument('--input', type=str, 
                        default='cache/redjujube_ner_comparison/comparison_results.json',
                        help='实验结果JSON文件')
    parser.add_argument('--output', type=str,
                        default='experiments/hz_lexicon/results/RedJujube_NER_实验报告_20251212.md',
                        help='输出Markdown文件路径')
    parser.add_argument('--print', action='store_true',
                        help='打印到控制台而不保存文件')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ 文件不存在: {args.input}")
        return
    
    if args.print:
        generate_markdown_report(args.input, output_file=None)
    else:
        generate_markdown_report(args.input, output_file=args.output)


if __name__ == '__main__':
    main()
