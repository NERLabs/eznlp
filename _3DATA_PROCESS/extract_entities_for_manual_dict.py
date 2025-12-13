#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 BMES 标注文件中提取实体，生成手动专家词典
"""

import argparse
from collections import Counter


def extract_entities_from_bmes(file_path):
    """从 BMES 格式文件提取实体"""
    entities = []
    current_entity = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 空行表示句子结束
            if not line:
                # 保存未完成的实体
                if current_entity:
                    entities.append(''.join(current_entity))
                    current_entity = []
                continue
            
            parts = line.split()
            if len(parts) != 2:
                continue
            
            char, label = parts
            
            # S: 单字实体
            if label.startswith('S-'):
                if current_entity:
                    entities.append(''.join(current_entity))
                    current_entity = []
                entities.append(char)
            
            # B: 实体开始
            elif label.startswith('B-'):
                if current_entity:
                    entities.append(''.join(current_entity))
                current_entity = [char]
            
            # M: 实体中间
            elif label.startswith('M-'):
                current_entity.append(char)
            
            # E: 实体结束
            elif label.startswith('E-'):
                current_entity.append(char)
                entities.append(''.join(current_entity))
                current_entity = []
            
            # O: 非实体
            else:
                if current_entity:
                    entities.append(''.join(current_entity))
                    current_entity = []
    
    # 处理文件末尾未完成的实体
    if current_entity:
        entities.append(''.join(current_entity))
    
    return entities


def main():
    parser = argparse.ArgumentParser(description='从 BMES 标注提取实体生成手动专家词典')
    parser.add_argument('--train_path', type=str, required=True,
                        help='训练集路径')
    parser.add_argument('--dev_path', type=str, default=None,
                        help='验证集路径（可选）')
    parser.add_argument('--test_path', type=str, default=None,
                        help='测试集路径（可选）')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出词典路径')
    parser.add_argument('--min_freq', type=int, default=1,
                        help='最小频次过滤（默认=1，保留所有实体）')
    parser.add_argument('--include_all_sets', action='store_true',
                        help='包含所有数据集（训练+验证+测试）的实体')
    
    args = parser.parse_args()
    
    print("="*70)
    print("从 BMES 标注提取实体生成手动专家词典")
    print("="*70)
    
    # 提取训练集实体
    print(f"\n提取训练集实体: {args.train_path}")
    train_entities = extract_entities_from_bmes(args.train_path)
    print(f"  - 提取到 {len(train_entities)} 个实体（包含重复）")
    
    all_entities = train_entities.copy()
    
    # 可选：提取验证集实体
    if args.include_all_sets and args.dev_path:
        print(f"\n提取验证集实体: {args.dev_path}")
        dev_entities = extract_entities_from_bmes(args.dev_path)
        print(f"  - 提取到 {len(dev_entities)} 个实体（包含重复）")
        all_entities.extend(dev_entities)
    
    # 可选：提取测试集实体
    if args.include_all_sets and args.test_path:
        print(f"\n提取测试集实体: {args.test_path}")
        test_entities = extract_entities_from_bmes(args.test_path)
        print(f"  - 提取到 {len(test_entities)} 个实体（包含重复）")
        all_entities.extend(test_entities)
    
    # 统计频次
    print(f"\n统计实体频次...")
    entity_counter = Counter(all_entities)
    
    # 频次过滤
    if args.min_freq > 1:
        print(f"应用频次过滤: min_freq={args.min_freq}")
        filtered_entities = {entity: freq for entity, freq in entity_counter.items() 
                           if freq >= args.min_freq}
    else:
        filtered_entities = dict(entity_counter)
    
    # 按频次降序排序
    sorted_entities = sorted(filtered_entities.items(), key=lambda x: x[1], reverse=True)
    
    # 保存词典
    print(f"\n保存词典到: {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for entity, freq in sorted_entities:
            f.write(f"{entity}\n")
    
    # 统计信息
    print("\n" + "="*70)
    print("统计信息:")
    print("="*70)
    print(f"总实体数（包含重复）: {len(all_entities):,}")
    print(f"唯一实体数: {len(entity_counter):,}")
    print(f"过滤后实体数: {len(sorted_entities):,}")
    print(f"总频次: {sum(filtered_entities.values()):,}")
    print(f"平均频次: {sum(filtered_entities.values()) / len(filtered_entities):.2f}")
    
    # 词长分布
    print("\n词长分布:")
    length_dist = Counter(len(entity) for entity in filtered_entities.keys())
    for length in sorted(length_dist.keys()):
        print(f"  {length}字词: {length_dist[length]:,} 个")
    
    # Top 20 高频实体
    print("\nTop 20 高频实体:")
    for i, (entity, freq) in enumerate(sorted_entities[:20], 1):
        print(f"  {i:2d}. {entity:10s} (频次: {freq:,})")
    
    print("\n✅ 词典生成完成！")


if __name__ == "__main__":
    main()
