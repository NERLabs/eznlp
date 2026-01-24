#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 RedJujube 和 PingGuo 数据集的实体种类和数量"""

import os
from collections import defaultdict

def count_entities(file_path):
    """统计单个 BMES 文件中的实体"""
    entity_counts = defaultdict(int)
    current_entity = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                current_entity = None
                continue
            
            parts = line.split()
            if len(parts) != 2:
                continue
            
            char, tag = parts
            
            if tag.startswith('B-'):
                entity_type = tag[2:]
                current_entity = entity_type
            elif tag.startswith('S-'):
                entity_type = tag[2:]
                entity_counts[entity_type] += 1
                current_entity = None
            elif tag.startswith('E-'):
                entity_type = tag[2:]
                if current_entity == entity_type:
                    entity_counts[entity_type] += 1
                current_entity = None
            elif tag == 'O':
                current_entity = None
    
    return dict(entity_counts)

def count_dataset(data_dir, name):
    """统计整个数据集"""
    print(f"\n{'='*60}")
    print(f"数据集: {name}")
    print(f"{'='*60}")
    
    total_counts = defaultdict(lambda: {'train': 0, 'dev': 0, 'test': 0})
    
    file_mapping = {
        'train': ['train.char.bmes', 'train1.char.bmes', 'redjujube_train.bmes', 'hz_train.bmes'],
        'dev': ['dev.char.bmes', 'dev1.char.bmes', 'redjujube_dev.bmes', 'hz_dev.bmes'],
        'test': ['test.char.bmes', 'test1.char.bmes', 'redjujube_test.bmes', 'hz_test.bmes']
    }
    
    for split, filenames in file_mapping.items():
        for filename in filenames:
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                counts = count_entities(file_path)
                for entity_type, count in counts.items():
                    total_counts[entity_type][split] += count
                break
    
    print(f"\n{'实体类型':<15} {'训练集':>10} {'验证集':>10} {'测试集':>10} {'合计':>10}")
    print("-" * 60)
    
    grand_total = {'train': 0, 'dev': 0, 'test': 0}
    for entity_type in sorted(total_counts.keys()):
        counts = total_counts[entity_type]
        total = counts['train'] + counts['dev'] + counts['test']
        print(f"{entity_type:<15} {counts['train']:>10} {counts['dev']:>10} {counts['test']:>10} {total:>10}")
        for split in ['train', 'dev', 'test']:
            grand_total[split] += counts[split]
    
    print("-" * 60)
    total_all = grand_total['train'] + grand_total['dev'] + grand_total['test']
    print(f"{'总计':<15} {grand_total['train']:>10} {grand_total['dev']:>10} {grand_total['test']:>10} {total_all:>10}")
    print(f"\n实体种类数: {len(total_counts)}")
    
    return total_counts

if __name__ == "__main__":
    base_dir = "_2DATA"
    
    # 红枣数据集（bmes文件夹，hz前缀）
    count_dataset("bmes", "红枣 (RedJujube)")
    
    # 苹果数据集
    count_dataset(os.path.join(base_dir, "pingguo"), "苹果 (PingGuo)")