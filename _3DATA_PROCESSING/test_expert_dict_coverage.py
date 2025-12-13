#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试专家词典在 HZ 数据集上的覆盖率和匹配效果
"""

import os
import re
from collections import Counter, defaultdict


class SimpleLexiconMatcher:
    """简单的词典匹配器（最大正向匹配）"""
    
    def __init__(self, lexicon_file, max_len=10):
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            self.lexicon = set(line.strip() for line in f if line.strip())
        self.max_len = max_len
    
    def tokenize(self, text):
        """匹配文本中的词典词条（最大正向匹配）"""
        matches = []
        i = 0
        
        while i < len(text):
            matched = False
            # 从最长到最短尝试匹配
            for length in range(min(self.max_len, len(text) - i), 0, -1):
                word = text[i:i+length]
                if word in self.lexicon:
                    matches.append((word, i, i+length))
                    i += length
                    matched = True
                    break
            
            if not matched:
                i += 1
        
        return matches


def load_bmes_file(file_path):
    """加载 BMES 格式的数据"""
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
                if len(parts) == 2:
                    token, label = parts
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
    """从 BMES 标签中提取实体"""
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


def analyze_expert_dict_coverage(data, expert_lexicon_path):
    """分析专家词典的覆盖率"""
    # 加载专家词典
    print(f"📖 加载专家词典：{expert_lexicon_path}")
    with open(expert_lexicon_path, 'r', encoding='utf-8') as f:
        expert_dict = set(line.strip() for line in f if line.strip())
    print(f"   词典大小：{len(expert_dict)} 个词条\n")
    
    # 统计实体
    all_entities = []
    entity_counter = Counter()
    entity_type_counter = Counter()
    
    for sent in data:
        entities = extract_entities(sent['tokens'], sent['labels'])
        all_entities.extend(entities)
        for entity in entities:
            entity_counter[entity['text']] += 1
            entity_type_counter[entity['type']] += 1
    
    # 分析覆盖情况
    covered_entities = set()
    uncovered_entities = set()
    covered_count = 0
    total_count = len(all_entities)
    
    type_coverage = defaultdict(lambda: {'covered': 0, 'total': 0})
    
    for entity in all_entities:
        entity_text = entity['text']
        entity_type = entity['type']
        
        type_coverage[entity_type]['total'] += 1
        
        if entity_text in expert_dict:
            covered_entities.add(entity_text)
            covered_count += 1
            type_coverage[entity_type]['covered'] += 1
        else:
            uncovered_entities.add(entity_text)
    
    return {
        'expert_dict_size': len(expert_dict),
        'total_entities': total_count,
        'unique_entities': len(entity_counter),
        'covered_count': covered_count,
        'covered_rate': covered_count / total_count if total_count > 0 else 0,
        'covered_unique': len(covered_entities),
        'uncovered_unique': len(uncovered_entities),
        'entity_counter': entity_counter,
        'entity_type_counter': entity_type_counter,
        'type_coverage': dict(type_coverage),
        'covered_entities': covered_entities,
        'uncovered_entities': uncovered_entities
    }


def test_tokenizer_matching(data, expert_lexicon_path, num_samples=10):
    """测试词典匹配器的匹配效果"""
    print(f"\n{'='*70}")
    print(f"🔍 测试专家词典匹配效果（前 {num_samples} 个句子）")
    print(f"{'='*70}\n")
    
    # 加载词典分词器
    tokenizer = SimpleLexiconMatcher(expert_lexicon_path, max_len=10)
    
    for i, sent in enumerate(data[:num_samples]):
        text = ''.join(sent['tokens'])
        tokens = sent['tokens']
        labels = sent['labels']
        
        print(f"句子 {i+1}:")
        print(f"  原文: {text}")
        
        # 提取真实实体
        entities = extract_entities(tokens, labels)
        if entities:
            entity_strs = [f"{e['text']}({e['type']})" for e in entities]
            print(f"  实体: {', '.join(entity_strs)}")
        
        # 使用词典匹配
        matched = tokenizer.tokenize(text)
        if matched:
            print(f"  词典匹配: {', '.join([m[0] for m in matched])}")
        else:
            print(f"  词典匹配: (无)")
        print()


def main():
    # 数据路径
    train_file = "data/HZ/hz_train.bmes"
    dev_file = "data/HZ/hz_dev.bmes"
    test_file = "data/HZ/hz_test.bmes"
    expert_lexicon = "data/HZ/expert_lexicon.txt"
    
    print(f"{'='*70}")
    print(f"🧪 专家词典在 HZ 数据集上的覆盖率分析")
    print(f"{'='*70}\n")
    
    # 加载数据
    print("📥 加载数据集...")
    train_data = load_bmes_file(train_file)
    dev_data = load_bmes_file(dev_file)
    test_data = load_bmes_file(test_file)
    
    print(f"  训练集: {len(train_data)} 个句子")
    print(f"  验证集: {len(dev_data)} 个句子")
    print(f"  测试集: {len(test_data)} 个句子\n")
    
    # 分析各数据集
    for name, data in [("训练集", train_data), ("验证集", dev_data), ("测试集", test_data)]:
        print(f"{'='*70}")
        print(f"📊 {name}分析")
        print(f"{'='*70}\n")
        
        stats = analyze_expert_dict_coverage(data, expert_lexicon)
        
        print(f"📈 整体覆盖率：")
        print(f"  - 专家词典大小: {stats['expert_dict_size']:,} 个词条")
        print(f"  - 数据集实体总数: {stats['total_entities']:,} 个")
        print(f"  - 唯一实体数: {stats['unique_entities']:,} 个")
        print(f"  - 词典覆盖实体数: {stats['covered_count']:,} 个")
        print(f"  - 覆盖率: {stats['covered_rate']:.2%}")
        print(f"  - 覆盖唯一实体: {stats['covered_unique']:,} 个")
        print(f"  - 未覆盖唯一实体: {stats['uncovered_unique']:,} 个\n")
        
        print(f"📋 实体类型分布：")
        for entity_type, count in sorted(stats['entity_type_counter'].items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"  - {entity_type}: {count:,} 个")
        print()
        
        print(f"🎯 各类型覆盖率：")
        for entity_type, coverage in sorted(stats['type_coverage'].items(), 
                                           key=lambda x: (x[0] is None, x[0])):
            covered = coverage['covered']
            total = coverage['total']
            rate = covered / total if total > 0 else 0
            type_name = entity_type if entity_type is not None else 'None'
            print(f"  - {type_name}: {covered}/{total} ({rate:.2%})")
        print()
        
        # 显示高频未覆盖实体
        print(f"⚠️  高频未覆盖实体 (Top 20):")
        uncovered_freq = {text: count for text, count in stats['entity_counter'].items() 
                         if text in stats['uncovered_entities']}
        for i, (text, count) in enumerate(sorted(uncovered_freq.items(), 
                                                 key=lambda x: x[1], 
                                                 reverse=True)[:20], 1):
            print(f"  {i:2d}. {text} ({count} 次)")
        print()
    
    # 测试匹配效果
    test_tokenizer_matching(train_data, expert_lexicon, num_samples=10)
    
    print(f"{'='*70}")
    print(f"✅ 分析完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
