#\!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从已有的结果 JSON 文件对比分析长实体性能
不需要重新加载模型预测
"""

import json
import os

# ====================== 配置 ======================

MODEL_CONFIGS = {
    'H': {
        'name': 'H 基线 (sb_size=2)',
        'result': 'experiments/EXP-010-optimization/results/H_bs_baseline/expert_boundary_20260318-163821/results.json',
    },
    'Q': {
        'name': 'Q Focal Loss (fl_gamma=2.0)',
        'result': 'experiments/EXP-010-optimization/results/Q_bs_focal/expert_boundary_20260319-103810/results.json',
    },
    'W': {
        'name': 'W Enhanced Size Emb (size_emb_dim=50)',
        'result': 'experiments/EXP-010-optimization/results/W_enhanced_size_emb/expert_boundary_20260319-143927/results.json',
    },
    'Y': {
        'name': 'Y Size+Focal+SpanWidth (max_span=50)',
        'result': 'experiments/EXP-010-optimization/results/Y_size_focal_spanwidth/expert_boundary_20260319-154017/results.json',
    },
}

def load_results(result_path):
    with open(result_path, 'r') as f:
        return json.load(f)

def main():
    print("=" * 90)
    print("  W-Y 组（长实体优化）vs H 组（基线）和 Q 组（Focal Loss）性能对比")
    print("  基于已有的训练结果数据")
    print("=" * 90)
    
    results = {}
    for group_id, config in MODEL_CONFIGS.items():
        result_path = config['result']
        if os.path.exists(result_path):
            try:
                results[group_id] = load_results(result_path)
                print(f"\n✓ {group_id} 组: {config['name']}")
                print(f"  Test F1: {results[group_id].get('test_f1', 'N/A'):.4f}")
            except Exception as e:
                print(f"\n✗ {group_id} 组: 加载失败 - {e}")
        else:
            print(f"\n✗ {group_id} 组: 文件不存在 - {result_path}")
    
    # ==================== 输出分析结果 ====================
    
    print("\n")
    print("#" * 90)
    print("#" + " " * 28 + "对比分析结果报告" + " " * 28 + "#")
    print("#" * 90)
    
    print("\n" + "=" * 90)
    print("  1. 总体指标对比")
    print("=" * 90)
    
    print(f"\n  {'组别':<12} {'说明':<30} {'Test F1':>12}")
    print("  " + "-" * 65)
    
    for group_id in ['H', 'Q', 'W', 'Y']:
        if group_id not in results:
            print(f"  {group_id:<12} {MODEL_CONFIGS[group_id]['name']:<30} {'N/A':>12}")
            continue
        
        test_f1 = results[group_id].get('test_f1', 0)
        name = MODEL_CONFIGS[group_id]['name']
        print(f"  {group_id:<12} {name:<30} {test_f1:>12.4f}")
    
    # ==================== 对比分析 ====================
    print("\n" + "=" * 90)
    print("  2. 长实体优化技术效果分析（基于总体 F1）")
    print("=" * 90)
    
    if all(g in results for g in ['H', 'Q', 'W', 'Y']):
        h_f1 = results['H'].get('test_f1', 0)
        q_f1 = results['Q'].get('test_f1', 0)
        w_f1 = results['W'].get('test_f1', 0)
        y_f1 = results['Y'].get('test_f1', 0)
        
        print(f"""
  分析基于完整测试集 F1 分数（注：无法获取长实体子集评估）
  
  总体表现对比:
    - H 基线: F1 = {h_f1:.4f}
    - Q Focal: F1 = {q_f1:.4f} (vs H: {q_f1 - h_f1:+.4f})
    - W 增强大小嵌入: F1 = {w_f1:.4f} (vs H: {w_f1 - h_f1:+.4f})
    - Y Size+Focal+SpanWidth: F1 = {y_f1:.4f} (vs H: {y_f1 - h_f1:+.4f})
  
  结论:
    Q Focal Loss 的实现在总体表现上: {'优于' if q_f1 > h_f1 else '劣于'} H 基线
    W 增强大小嵌入在总体表现上: {'优于' if w_f1 > h_f1 else '劣于'} H 基线
    Y Size+Focal+SpanWidth 在总体表现上: {'优于' if y_f1 > h_f1 else '劣于'} H 基线
    
  注意: 由于训练日志中未包含长实体子集评估，此报告仅基于完整测试集数据。
        要获得长实体专项评估，需要：
        1. 重新运行预测并按长度分桶
        2. 或修改模型配置支持按长度的详细指标输出
    """)
    
    print("=" * 90)
    print("  分析完成")
    print("=" * 90)
    
    # ==================== 技术参数差异分析 ====================
    print("\n" + "=" * 90)
    print("  3. 实验组配置差异说明")
    print("=" * 90)
    
    print("""
    H 组 - 基线 (Boundary Selection 解码器)
      - sb_size: 2
      - 无 Focal Loss
      
    Q 组 - Focal Loss (fl_gamma=2.0)
      - sb_size: 2
      - 添加 Focal Loss 处理困难样例
      
    W 组 - 增强大小嵌入 (size_emb_dim=50, 默认为 25)
      - sb_size: 2
      - 增大大小嵌入维度，为长实体提供更好的表示
      
    Y 组 - Size+Focal+SpanWidth (max_span=50)
      - sb_size: 2
      - 增强大小嵌入 + Focal Loss + Span 宽度限制
      - max_span_width: 50 (限制实体最大宽度，可能对长实体不友好)
    """)
    
    print("=" * 90)


if __name__ == "__main__":
    main()
