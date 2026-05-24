#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版 SoftLexicon 词典提取脚本 - 高质量过滤策略

改进点：
1. 更高频次阈值（min_freq=5~10）
2. 长度限制（2-4字为主）
3. 实体优先权重
4. 停用词过滤
5. PMI互信息过滤（可选）

目标：从20w词缩减到5-10k高质量词
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import math


# 中文停用词列表（简化版）
STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没',
    '看', '好', '自己', '这', '那', '里', '就是', '得', '为', '能', '可以'
}


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


def extract_ngrams(tokens, min_len=2, max_len=4):
    """提取 n-gram 词组（限制长度范围）
    
    Args:
        tokens: token 列表
        min_len: 最小长度（默认2，过滤单字）
        max_len: 最大长度（默认4，避免过长）
    """
    ngrams = []
    for i in range(len(tokens)):
        for length in range(min_len, max_len + 1):
            if i + length <= len(tokens):
                ngram = ''.join(tokens[i:i+length])
                ngrams.append(ngram)
    return ngrams


def is_valid_word(word):
    """判断词是否有效
    
    过滤规则：
    1. 不是停用词
    2. 不是纯数字
    3. 不是纯标点符号
    4. 不包含标点符号（新增）
    5. 长度在合理范围内
    """
    if word in STOPWORDS:
        return False
    
    if word.isdigit():
        return False
    
    # 检查是否包含至少一个汉字
    if not any('\u4e00' <= c <= '\u9fff' for c in word):
        return False
    
    # 过滤包含标点符号的词（关键改进）
    import string
    chinese_punctuation = '，。！？；：""''（）《》、·…—'
    all_punctuation = string.punctuation + chinese_punctuation
    if any(c in all_punctuation for c in word):
        return False
    
    # 长度检查
    if len(word) < 2 or len(word) > 6:
        return False
    
    return True


def calculate_pmi(word, word_freq, char_freq, total_chars):
    """计算词的点互信息（PMI）
    
    PMI衡量词内部的凝聚度
    高PMI表示词内字符经常一起出现，是有意义的词
    """
    if len(word) < 2:
        return 0
    
    # 词频概率
    p_word = word_freq / total_chars
    
    # 各字符独立概率的乘积
    p_chars = 1.0
    for char in word:
        p_chars *= (char_freq.get(char, 1) / total_chars)
    
    if p_chars == 0:
        return 0
    
    pmi = math.log2(p_word / p_chars)
    return pmi


def build_filtered_lexicon(
    train_path,
    entity_min_freq=2,      # 实体最小频次
    ngram_min_freq=10,      # n-gram最小频次（更严格）
    ngram_min_len=2,        # n-gram最小长度
    ngram_max_len=4,        # n-gram最大长度
    entity_weight=3.0,      # 实体权重加成
    use_pmi=True,           # 是否使用PMI过滤
    pmi_threshold=2.0       # PMI阈值
):
    """构建过滤后的高质量词典
    
    策略：
    1. 实体优先：频次要求低，权重高
    2. n-gram严格：频次要求高，长度限制
    3. 质量过滤：停用词、PMI等
    """
    print(f"📖 加载训练数据: {train_path}")
    data = load_bmes_file(train_path)
    print(f"   加载了 {len(data)} 个句子\n")
    
    # 存储词频和类型
    lexicon_counter = Counter()
    word_types = {}  # word -> 'entity' or 'ngram'
    
    # 统计字符频率（用于PMI计算）
    char_freq = Counter()
    total_chars = 0
    for sent in data:
        for token in sent['tokens']:
            char_freq[token] += 1
            total_chars += 1
    
    # 1. 提取实体（优先级高）
    print("🔍 提取标注实体...")
    entity_set = set()
    for sent in data:
        entities = extract_entities(sent['tokens'], sent['labels'])
        for entity in entities:
            if is_valid_word(entity):
                lexicon_counter[entity] += 1
                word_types[entity] = 'entity'
                entity_set.add(entity)
    
    print(f"   提取了 {len(entity_set)} 个唯一实体")
    print(f"   应用实体权重: {entity_weight}x\n")
    
    # 应用实体权重
    for entity in entity_set:
        lexicon_counter[entity] = int(lexicon_counter[entity] * entity_weight)
    
    # 2. 提取 n-gram（限制长度）
    print(f"🔍 提取 n-gram ({ngram_min_len}-{ngram_max_len}字)...")
    ngram_set = set()
    for sent in data:
        ngrams = extract_ngrams(sent['tokens'], ngram_min_len, ngram_max_len)
        for ngram in ngrams:
            if ngram not in entity_set and is_valid_word(ngram):
                lexicon_counter[ngram] += 1
                if ngram not in word_types:
                    word_types[ngram] = 'ngram'
                ngram_set.add(ngram)
    
    print(f"   提取了 {len(ngram_set)} 个唯一 n-gram\n")
    
    # 3. 频次过滤
    print("🔍 应用频次过滤...")
    filtered_lexicon = {}
    
    for word, freq in lexicon_counter.items():
        # 根据词类型应用不同阈值
        if word_types.get(word) == 'entity':
            if freq >= entity_min_freq * entity_weight:  # 考虑权重后的阈值
                filtered_lexicon[word] = freq
        else:
            if freq >= ngram_min_freq:
                filtered_lexicon[word] = freq
    
    print(f"   频次过滤后: {len(filtered_lexicon)} 个词\n")
    
    # 4. PMI过滤（可选）
    if use_pmi:
        print(f"🔍 应用PMI过滤 (阈值={pmi_threshold})...")
        pmi_filtered = {}
        
        for word, freq in filtered_lexicon.items():
            # 实体跳过PMI过滤
            if word_types.get(word) == 'entity':
                pmi_filtered[word] = freq
            else:
                pmi = calculate_pmi(word, freq, char_freq, total_chars)
                if pmi >= pmi_threshold:
                    pmi_filtered[word] = freq
        
        print(f"   PMI过滤后: {len(pmi_filtered)} 个词\n")
        filtered_lexicon = pmi_filtered
    
    return filtered_lexicon, word_types


def main():
    parser = argparse.ArgumentParser(description='提取高质量过滤后的SoftLexicon词典')
    parser.add_argument('--train_path', type=str, required=True,
                        help='训练数据路径 (BMES格式)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出词典路径')
    parser.add_argument('--entity_min_freq', type=int, default=2,
                        help='实体最小频次阈值 (默认: 2)')
    parser.add_argument('--ngram_min_freq', type=int, default=10,
                        help='n-gram最小频次阈值 (默认: 10)')
    parser.add_argument('--ngram_min_len', type=int, default=2,
                        help='n-gram最小长度 (默认: 2)')
    parser.add_argument('--ngram_max_len', type=int, default=4,
                        help='n-gram最大长度 (默认: 4)')
    parser.add_argument('--entity_weight', type=float, default=3.0,
                        help='实体权重加成 (默认: 3.0)')
    parser.add_argument('--use_pmi', action='store_true',
                        help='是否使用PMI过滤')
    parser.add_argument('--pmi_threshold', type=float, default=2.0,
                        help='PMI阈值 (默认: 2.0)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 高质量 SoftLexicon 词典提取")
    print("="*70)
    print(f"\n配置:")
    print(f"  - 实体最小频次: {args.entity_min_freq}")
    print(f"  - n-gram最小频次: {args.ngram_min_freq}")
    print(f"  - n-gram长度范围: {args.ngram_min_len}-{args.ngram_max_len}")
    print(f"  - 实体权重: {args.entity_weight}x")
    print(f"  - PMI过滤: {'启用' if args.use_pmi else '禁用'}")
    if args.use_pmi:
        print(f"  - PMI阈值: {args.pmi_threshold}")
    print()
    
    # 构建词典
    lexicon, word_types = build_filtered_lexicon(
        train_path=args.train_path,
        entity_min_freq=args.entity_min_freq,
        ngram_min_freq=args.ngram_min_freq,
        ngram_min_len=args.ngram_min_len,
        ngram_max_len=args.ngram_max_len,
        entity_weight=args.entity_weight,
        use_pmi=args.use_pmi,
        pmi_threshold=args.pmi_threshold
    )
    
    # 保存词典
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 按频次排序
    sorted_lexicon = sorted(lexicon.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for word, freq in sorted_lexicon:
            f.write(f"{word}\n")
    
    print(f"✅ 词典已保存到: {output_path}")
    print(f"   总词数: {len(lexicon)}")
    
    # 统计信息
    entity_count = sum(1 for w in lexicon if word_types.get(w) == 'entity')
    ngram_count = len(lexicon) - entity_count
    
    print(f"\n📊 词典统计:")
    print(f"   实体词: {entity_count}")
    print(f"   n-gram: {ngram_count}")
    print(f"   总计: {len(lexicon)}")
    
    # 显示高频词示例
    print(f"\n🔝 Top 20 高频词:")
    for i, (word, freq) in enumerate(sorted_lexicon[:20], 1):
        word_type = word_types.get(word, 'unknown')
        print(f"   {i:2d}. {word:8s} (freq={freq:4d}, type={word_type})")
    
    print("\n" + "="*70)
    print("✅ 完成！")
    print("="*70)


if __name__ == '__main__':
    main()
