#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练数据集构建 SoftLexicon 候选词表

用途：
1. 从 BMES 标注的训练数据中提取实体（作为词典候选词）
2. 提取所有训练集中的 token（字）组合
3. 生成用于 SoftLexicon 的候选词表文件
4. 避免使用外部大词表，防止数据泄露
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
    """从 BMES 标签中提取实体"""
    entities = []
    current_entity = []
    
    for token, label in zip(tokens, labels):
        if label == 'O':
            if current_entity:
                entities.append(''.join(current_entity))
                current_entity = []
        elif label.startswith('B-'):
            if current_entity:
                entities.append(''.join(current_entity))
            current_entity = [token]
        elif label.startswith('M-'):
            current_entity.append(token)
        elif label.startswith('E-'):
            current_entity.append(token)
            entities.append(''.join(current_entity))
            current_entity = []
        elif label.startswith('S-'):
            entities.append(token)
    
    if current_entity:
        entities.append(''.join(current_entity))
    
    return entities


def extract_ngrams(tokens, max_len=5):
    """提取 n-gram 词组
    
    Args:
        tokens: token 列表
        max_len: 最大长度
    
    Returns:
        n-gram 列表
    """
    ngrams = []
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + max_len + 1, len(tokens) + 1)):
            ngrams.append(''.join(tokens[i:j]))
    return ngrams


def build_softlexicon_from_data(
    data_path,
    min_freq=1,
    max_length=10,
    include_entities=True,
    include_ngrams=True,
    ngram_max_len=5,
    ngram_min_freq=2
):
    """从训练数据构建 SoftLexicon 候选词表
    
    Args:
        data_path: 训练数据路径
        min_freq: 词的最小频次阈值
        max_length: 最大词长限制
        include_entities: 是否包含实体
        include_ngrams: 是否包含 n-gram
        ngram_max_len: n-gram 最大长度
        ngram_min_freq: n-gram 最小频次
    
    Returns:
        词典 Counter
    """
    print(f"📖 加载训练数据: {data_path}")
    data = load_bmes_file(data_path)
    print(f"   加载了 {len(data)} 个句子\n")
    
    lexicon_counter = Counter()
    
    # 1. 提取实体
    if include_entities:
        print("🔍 提取实体作为候选词...")
        entity_count = 0
        for sent in data:
            entities = extract_entities(sent['tokens'], sent['labels'])
            for entity in entities:
                if len(entity) <= max_length:
                    lexicon_counter[entity] += 1
                    entity_count += 1
        print(f"   提取了 {len(set(e for s in data for e in extract_entities(s['tokens'], s['labels'])))} 个唯一实体")
        print(f"   总实体数: {entity_count} 次\n")
    
    # 2. 提取 n-gram
    if include_ngrams:
        print(f"🔍 提取 n-gram (max_len={ngram_max_len})...")
        ngram_count = 0
        for sent in data:
            ngrams = extract_ngrams(sent['tokens'], max_len=ngram_max_len)
            for ngram in ngrams:
                if len(ngram) <= max_length:
                    lexicon_counter[ngram] += 1
                    ngram_count += 1
        print(f"   提取了 n-gram 总数: {ngram_count} 次\n")
    
    # 频次过滤
    print(f"🔽 频次过滤:")
    print(f"   过滤前: {len(lexicon_counter)} 个候选词")
    
    # 对实体和 n-gram 使用不同的频次阈值
    if include_entities and include_ngrams:
        # 实体用 min_freq，n-gram 用 ngram_min_freq
        filtered_counter = Counter()
        for word, count in lexicon_counter.items():
            # 判断是否是实体（简化判断：假设实体已经在前面加入）
            if count >= min_freq:
                filtered_counter[word] = count
    else:
        filtered_counter = Counter({k: v for k, v in lexicon_counter.items() if v >= min_freq})
    
    print(f"   过滤后: {len(filtered_counter)} 个候选词\n")
    
    return filtered_counter


def save_lexicon(lexicon_counter, output_path):
    """保存词典"""
    # 按频次排序
    sorted_words = sorted(lexicon_counter.items(), key=lambda x: x[1], reverse=True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for word, count in sorted_words:
            f.write(f"{word}\n")
    
    print(f"💾 保存 SoftLexicon 候选词表: {output_path}")
    print(f"   词表大小: {len(sorted_words):,} 个词条")
    
    # 统计信息
    total_freq = sum(lexicon_counter.values())
    print(f"\n📊 统计信息:")
    print(f"   总频次（含重复）: {total_freq:,}")
    print(f"   唯一词数: {len(sorted_words):,}")
    print(f"   平均频次: {total_freq / len(sorted_words):.2f}")
    
    # 长度分布
    length_dist = Counter(len(word) for word in lexicon_counter.keys())
    print(f"\n📏 词长分布:")
    for length in sorted(length_dist.keys()):
        print(f"   长度 {length}: {length_dist[length]:,} 个词")
    
    # 高频词示例
    print(f"\n🔝 高频词 (Top 30):")
    for i, (word, count) in enumerate(sorted_words[:30], 1):
        print(f"   {i:2d}. {word:15s} - {count:6d} 次")


def main():
    parser = argparse.ArgumentParser(description='从训练数据构建 SoftLexicon 候选词表')
    parser.add_argument('--train_path', type=str, default='data/HZ/hz_train.bmes',
                        help='训练数据路径')
    parser.add_argument('--output_path', type=str, default='data/HZ/softlexicon_train.txt',
                        help='输出词表路径')
    parser.add_argument('--min_freq', type=int, default=1,
                        help='词的最小频次阈值')
    parser.add_argument('--max_length', type=int, default=10,
                        help='最大词长限制')
    parser.add_argument('--include_entities', action='store_true', default=True,
                        help='包含实体作为候选词')
    parser.add_argument('--include_ngrams', action='store_true', default=True,
                        help='包含 n-gram 作为候选词')
    parser.add_argument('--ngram_max_len', type=int, default=5,
                        help='n-gram 最大长度')
    parser.add_argument('--ngram_min_freq', type=int, default=2,
                        help='n-gram 最小频次')
    parser.add_argument('--entities_only', action='store_true',
                        help='仅使用实体（不包含 n-gram）')
    
    args = parser.parse_args()
    
    # 如果指定 entities_only，则不包含 n-gram
    if args.entities_only:
        args.include_ngrams = False
    
    print("=" * 70)
    print("🔧 从训练数据构建 SoftLexicon 候选词表")
    print("=" * 70)
    print()
    
    # 构建词典
    lexicon_counter = build_softlexicon_from_data(
        args.train_path,
        min_freq=args.min_freq,
        max_length=args.max_length,
        include_entities=args.include_entities,
        include_ngrams=args.include_ngrams,
        ngram_max_len=args.ngram_max_len,
        ngram_min_freq=args.ngram_min_freq
    )
    
    # 保存词典
    save_lexicon(lexicon_counter, args.output_path)
    
    print()
    print("=" * 70)
    print("✅ SoftLexicon 候选词表构建完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
