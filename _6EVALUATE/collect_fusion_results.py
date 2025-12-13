#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集和汇总Soft+Expert融合实验结果

自动查找所有融合方案的结果文件并生成对比报告
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class FusionResultsCollector:
    """融合实验结果收集器"""
    
    def __init__(self, base_dir: str = "cache"):
        self.base_dir = Path(base_dir)
        self.results = {}
        
    def find_result_files(self) -> Dict[str, Path]:
        """查找所有融合方案的结果文件"""
        result_files = {}
        
        # 定义方案名称模式
        patterns = {
            "concat": "*concat*",
            "weighted": "*weighted*",
            "gated": "*gated*",
            "attention": "*attention*"
        }
        
        for scheme_name, pattern in patterns.items():
            # 查找最新的结果文件
            matching_files = list(self.base_dir.rglob(f"{pattern}/results.json"))
            if matching_files:
                # 按修改时间排序，取最新的
                latest = max(matching_files, key=lambda p: p.stat().st_mtime)
                result_files[scheme_name] = latest
                
        return result_files
    
    def load_result(self, result_file: Path) -> Optional[Dict]:
        """加载单个结果文件"""
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ 加载结果文件失败 {result_file}: {e}")
            return None
    
    def collect_all_results(self):
        """收集所有结果"""
        print("🔍 查找融合实验结果文件...\n")
        
        result_files = self.find_result_files()
        
        if not result_files:
            print("⚠️ 未找到任何结果文件")
            return
        
        for scheme_name, file_path in result_files.items():
            print(f"📄 {scheme_name}: {file_path}")
            result = self.load_result(file_path)
            if result:
                self.results[scheme_name] = {
                    'file': str(file_path),
                    'data': result
                }
        
        print(f"\n✅ 成功加载 {len(self.results)} 个结果文件\n")
    
    def generate_comparison_table(self) -> str:
        """生成对比表格"""
        if not self.results:
            return "无可用结果"
        
        lines = []
        lines.append("## 📊 融合方案性能对比\n")
        lines.append("| 方案 | 测试F1 | 测试Loss | 总参数量 | 训练参数量 | 结果文件 |")
        lines.append("|------|--------|----------|---------|-----------|---------|")
        
        # 方案顺序
        scheme_order = ["concat", "weighted", "gated", "attention"]
        scheme_names = {
            "concat": "方案A (Concat)",
            "weighted": "方案B (Weighted)",
            "gated": "方案C (Gated)",
            "attention": "方案D (Attention)"
        }
        
        for scheme_key in scheme_order:
            if scheme_key not in self.results:
                continue
                
            result = self.results[scheme_key]['data']
            file_path = self.results[scheme_key]['file']
            
            # 提取关键指标
            test_f1 = result.get('test_metrics', [0])[0] * 100 if result.get('test_metrics') else 0
            test_loss = result.get('test_loss', 0)
            total_params = result.get('total_params', 0)
            trainable_params = result.get('trainable_params', 0)
            
            # 格式化参数量
            total_params_str = f"{total_params:,}" if total_params else "-"
            trainable_params_str = f"{trainable_params:,}" if trainable_params else "-"
            
            lines.append(f"| {scheme_names[scheme_key]} | {test_f1:.2f}% | {test_loss:.4f} | "
                        f"{total_params_str} | {trainable_params_str} | `{Path(file_path).parent.name}` |")
        
        lines.append("")
        return "\n".join(lines)
    
    def generate_analysis(self) -> str:
        """生成结果分析"""
        if not self.results:
            return ""
        
        lines = []
        lines.append("## 📈 结果分析\n")
        
        # 提取所有F1分数
        f1_scores = {}
        for scheme_key, result_data in self.results.items():
            result = result_data['data']
            f1 = result.get('test_metrics', [0])[0] * 100 if result.get('test_metrics') else 0
            f1_scores[scheme_key] = f1
        
        if f1_scores:
            # 找出最佳方案
            best_scheme = max(f1_scores.items(), key=lambda x: x[1])
            worst_scheme = min(f1_scores.items(), key=lambda x: x[1])
            
            scheme_names = {
                "concat": "方案A (Concat)",
                "weighted": "方案B (Weighted)",
                "gated": "方案C (Gated)",
                "attention": "方案D (Attention)"
            }
            
            lines.append(f"### 🏆 最佳方案")
            lines.append(f"- **{scheme_names.get(best_scheme[0], best_scheme[0])}**: {best_scheme[1]:.2f}% F1\n")
            
            lines.append(f"### 📊 性能排名")
            sorted_schemes = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (scheme_key, f1) in enumerate(sorted_schemes, 1):
                lines.append(f"{i}. {scheme_names.get(scheme_key, scheme_key)}: {f1:.2f}% F1")
            
            lines.append("")
            
            # 与基线对比
            baseline_f1 = 95.51
            expert_dict_f1 = 96.99
            
            lines.append("### 📌 与基线对比\n")
            lines.append(f"- Baseline F1: {baseline_f1:.2f}%")
            lines.append(f"- ExpertDict F1: {expert_dict_f1:.2f}%")
            lines.append(f"- 最佳融合方案: {best_scheme[1]:.2f}%")
            
            improvement = best_scheme[1] - baseline_f1
            expert_improvement = best_scheme[1] - expert_dict_f1
            
            lines.append(f"- 相比Baseline提升: **{improvement:+.2f}%**")
            lines.append(f"- 相比ExpertDict提升: **{expert_improvement:+.2f}%**")
            
            lines.append("")
            
            # 成功标准检查
            lines.append("### ✅ 成功标准检查\n")
            target_f1 = 97.04
            if best_scheme[1] > target_f1:
                lines.append(f"- ✅ **达到目标**: 最佳F1 ({best_scheme[1]:.2f}%) > 目标 ({target_f1:.2f}%)")
            else:
                lines.append(f"- ⚠️ **未达目标**: 最佳F1 ({best_scheme[1]:.2f}%) < 目标 ({target_f1:.2f}%)")
                lines.append(f"  - 差距: {target_f1 - best_scheme[1]:.2f}%")
                lines.append(f"  - 建议: 进行超参数调优或特征工程优化")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_recommendations(self) -> str:
        """生成建议"""
        if not self.results:
            return ""
        
        lines = []
        lines.append("## 💡 建议与下一步\n")
        
        # 提取F1分数
        f1_scores = {}
        for scheme_key, result_data in self.results.items():
            result = result_data['data']
            f1 = result.get('test_metrics', [0])[0] * 100 if result.get('test_metrics') else 0
            f1_scores[scheme_key] = f1
        
        if f1_scores:
            best_f1 = max(f1_scores.values())
            
            if best_f1 > 97.04:
                lines.append("### ✅ 性能达标")
                lines.append("- 联合特征模型成功超越单独方法")
                lines.append("- 建议进行消融实验，分析各组件贡献")
                lines.append("- 可以进行实体类型详细分析")
            else:
                lines.append("### ⚠️ 性能待优化")
                lines.append("- 当前最佳方案未达到预期目标")
                lines.append("- 建议调整超参数：")
                lines.append("  - 增加训练轮数 (epochs > 30)")
                lines.append("  - 调整学习率 (尝试不同的lr)")
                lines.append("  - 调整特征维度 (emb_dim)")
                lines.append("  - 尝试不同的融合策略")
            
            lines.append("")
            lines.append("### 📋 后续工作")
            lines.append("1. 消融实验：分析SoftLex和Expert各自的贡献度")
            lines.append("2. 错误分析：对比不同方案的预测差异")
            lines.append("3. 实体类型分析：查看各方案在不同实体类型上的表现")
            lines.append("4. 超参数调优：如果性能未达标")
            lines.append("5. 生成完整实验报告")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def save_report(self, output_file: str = "experiments/hz_lexicon/results/Fusion_Comparison_Report.md"):
        """保存报告"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        report_lines = []
        report_lines.append(f"# Soft+Expert 融合方案对比报告\n")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append("---\n")
        
        report_lines.append(self.generate_comparison_table())
        report_lines.append(self.generate_analysis())
        report_lines.append(self.generate_recommendations())
        
        report_lines.append("---\n")
        report_lines.append("**数据集**: RedJujube")
        report_lines.append("**实验脚本**: `scripts/train_redjujube_ner_comparison.py`")
        report_lines.append("**结果目录**: `cache/`\n")
        
        report_content = "\n".join(report_lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 报告已保存到: {output_file}\n")
        
        # 同时打印到控制台
        print("=" * 70)
        print(report_content)
        print("=" * 70)
        
        return output_file


def main():
    """主函数"""
    print("=" * 70)
    print("Soft+Expert 融合实验结果收集器")
    print("=" * 70)
    print()
    
    collector = FusionResultsCollector()
    collector.collect_all_results()
    
    if collector.results:
        report_file = collector.save_report()
        print(f"\n📄 详细报告: {report_file}")
    else:
        print("⚠️ 未找到任何实验结果，请确保训练已完成")
        sys.exit(1)


if __name__ == "__main__":
    main()
