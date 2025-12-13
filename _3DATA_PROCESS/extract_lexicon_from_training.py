#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练数据集抽取专家词典

用途：
1. 从 BMES 标注的训练数据中提取所有实体
2. 生成专家词典文件供 NER 模型使用
3. 可选：按实体频次过滤，只保留高频实体
"""

import argparse
from collections import Counter
from pathlib import Path


def load_bmes_file(file_path):
    """加载 BMES 格式文件"""
    sentences = []
    current_tokens = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append({
                        'tokens': current_tokens,
                        'labels': current_labels
                    })
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[1]
                    current_tokens.append(token)
                    current_labels.append(label)
        
        # 处理最后一个句子
        if current_tokens:
            sentences.append({
                'tokens': current_tokens,
                'labels': current_labels
            })
    
    return sentences


def extract_entities(tokens, labels):
    """从 BMES 标签中提取实体
    
    Args:
        tokens: 字符列表
        labels: BMES 标签列表
        
    Returns:
        实体列表，每个实体是 {'text': str, 'type': str} 格式
    """
    entities = []
    current_entity = []
    current_type = None
    
    for token, label in zip(tokens, labels):
        if label == 'O':
            if current_entity:
                entities.append({
                    'text': ''.join(current_entity),
                    'type': current_type
                })
                current_entity = []
                current_type = None
        elif label.startswith('B-'):
            if current_entity:
                entities.append({
                    'text': ''.join(current_entity),
                    'type': current_type
                })
            current_entity = [token]
            current_type = label[2:]
        elif label.startswith('M-'):
            current_entity.append(token)
        elif label.startswith('E-'):
            current_entity.append(token)
            entities.append({
                'text': ''.join(current_entity),
                'type': current_type
            })
            current_entity = []
            current_type = None
        elif label.startswith('S-'):
            entities.append({
                'text': token,
                'type': label[2:]
            })
    
    if current_entity:
        entities.append({
            'text': ''.join(current_entity),
            'type': current_type
        })
    
    return entities


def extract_lexicon_from_data(data_path, min_freq=1, max_length=None, save_with_type=False):
    """从训练数据提取专家词典
    
    Args:
        data_path: 训练数据路径
        min_freq: 最小频次阈值
        max_length: 最大实体长度限制（None表示不限制）
        save_with_type: 是否保存实体类型信息
        
    Returns:
        entity_counter: 实体计数器
        entity_types: 实体类型字典
    """
    print(f"📖 加载训练数据: {data_path}")
    data = load_bmes_file(data_path)
    print(f"   加载了 {len(data)} 个句子\n")
    
    # 提取所有实体
    print("🔍 提取实体...")
    entity_counter = Counter()
    entity_types = {}  # 记录每个实体的类型（取频次最高的类型）
    entity_type_counter = {}  # 每个实体的类型计数
    
    for sent in data:
        entities = extract_entities(sent['tokens'], sent['labels'])
        for entity in entities:
            text = entity['text']
            ent_type = entity['type']
            
            # 长度过滤
            if max_length is not None and len(text) > max_length:
                continue
            
            entity_counter[text] += 1
            
            # 统计类型
            if text not in entity_type_counter:
                entity_type_counter[text] = Counter()
            entity_type_counter[text][ent_type] += 1
    
    # 确定每个实体的主要类型
    for text in entity_counter:
        entity_types[text] = entity_type_counter[text].most_common(1)[0][0]
    
    print(f"   提取了 {len(entity_counter)} 个唯一实体")
    print(f"   总实体数: {sum(entity_counter.values())} 次\n")
    
    # 频次过滤
    if min_freq > 1:
        filtered_counter = Counter({k: v for k, v in entity_counter.items() if v >= min_freq})
        print(f"🔽 频次过滤 (min_freq={min_freq}):")
        print(f"   过滤前: {len(entity_counter)} 个实体")
        print(f"   过滤后: {len(filtered_counter)} 个实体\n")
        entity_counter = filtered_counter
    
    return entity_counter, entity_types


def save_lexicon(entity_counter, entity_types, output_path, save_with_type=False):
    """保存专家词典
    
    Args:
        entity_counter: 实体计数器
        entity_types: 实体类型字典
        output_path: 输出路径
        save_with_type: 是否保存类型信息
    """
    # 按频次排序（高频在前）
    sorted_entities = sorted(entity_counter.items(), key=lambda x: x[1], reverse=True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entity_text, count in sorted_entities:
            if save_with_type:
                ent_type = entity_types.get(entity_text, 'UNK')
                f.write(f"{entity_text}\t{ent_type}\t{count}\n")
            else:
                f.write(f"{entity_text}\n")
    
    print(f"💾 保存专家词典: {output_path}")
    print(f"   词典大小: {len(sorted_entities)} 个词条")
    
    # 显示统计信息
    total_freq = sum(entity_counter.values())
    print(f"\n📊 统计信息:")
    print(f"   总实体数（含重复）: {total_freq:,}")
    print(f"   唯一实体数: {len(sorted_entities):,}")
    print(f"   平均频次: {total_freq / len(sorted_entities):.2f}")
    
    # 显示高频实体示例
    print(f"\n🔝 高频实体 (Top 20):")
    for i, (text, count) in enumerate(sorted_entities[:20], 1):
        ent_type = entity_types.get(text, 'UNK')
        print(f"   {i:2d}. {text:15s} ({ent_type:8s}) - {count:4d} 次")


def main():
    parser = argparse.ArgumentParser(description='从训练数据集抽取专家词典')
    parser.add_argument('--train_path', type=str, default='data/HZ/hz_train.bmes',
                        help='训练数据路径')
    parser.add_argument('--output_path', type=str, default='data/HZ/expert_lexicon_auto.txt',
                        help='输出词典路径')
    parser.add_argument('--min_freq', type=int, default=1,
                        help='最小频次阈值（默认1，即保留所有实体）')
    parser.add_argument('--max_length', type=int, default=None,
                        help='最大实体长度限制（默认不限制）')
    parser.add_argument('--save_with_type', action='store_true',
                        help='保存时包含实体类型和频次信息')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔧 从训练数据抽取专家词典")
    print("=" * 70)
    print()
    
    # 提取词典
    entity_counter, entity_types = extract_lexicon_from_data(
        args.train_path,
        min_freq=args.min_freq,
        max_length=args.max_length,
        save_with_type=args.save_with_type
    )
    
    # 保存词典
    save_lexicon(
        entity_counter,
        entity_types,
        args.output_path,
        save_with_type=args.save_with_type
    )
    
    print()
    print("=" * 70)
    print("✅ 专家词典抽取完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
