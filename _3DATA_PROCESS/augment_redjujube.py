#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube 数据增强模块 - 基于实体替换的数据增强

支持两种模式：
1. BMES 文件级别增强：读取/写入 BMES 格式文件
2. eznlp 数据级别增强：直接对 eznlp 数据结构进行增强

14 种实体类型：PAR, AGR, CUL, LOC, EQU, DRU, PER, DIS, PRO, NUT, PES, FER, WED, TAX
"""

import argparse
import copy
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# 添加项目根目录到 Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# BMES 文件处理函数
# ============================================================================

def read_bmes_file(filepath: str, sep: str = None) -> List[List[Tuple[str, str]]]:
    """读取 BMES 格式文件，返回句子列表
    
    Args:
        filepath: BMES 文件路径
        sep: 分隔符，默认自动检测（None = 尝试 tab 和空格）
    
    Returns:
        句子列表，每个句子是 [(char, tag), ...] 的列表
    """
    sentences = []
    current_sentence = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                # 空行表示句子结束
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                # 自动检测分隔符
                if sep is not None:
                    parts = line.split(sep)
                elif "\t" in line:
                    parts = line.split("\t")
                else:
                    # 从右侧分割，因为字符可能是空格
                    parts = line.rsplit(" ", 1)
                
                if len(parts) >= 2:
                    char, tag = parts[0], parts[1]
                    current_sentence.append((char, tag))
                elif len(parts) == 1:
                    # 只有字符没有标签的情况
                    current_sentence.append((parts[0], "O"))
    
    # 处理最后一个句子（如果文件不以空行结尾）
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def write_bmes_file(filepath: str, sentences: List[List[Tuple[str, str]]]):
    """将句子列表写入 BMES 格式文件
    
    Args:
        filepath: 输出文件路径
        sentences: 句子列表，每个句子是 [(char, tag), ...] 的列表
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for char, tag in sentence:
                f.write(f"{char}\t{tag}\n")
            f.write("\n")  # 句子间空行分隔


def extract_entities_from_sentence(sentence: List[Tuple[str, str]]) -> List[Tuple[str, str, int, int]]:
    """从单个句子中提取所有实体
    
    Args:
        sentence: [(char, tag), ...] 列表
    
    Returns:
        实体列表：[(entity_text, entity_type, start_idx, end_idx), ...]
        end_idx 是实体最后一个字符的索引（包含）
    """
    entities = []
    i = 0
    
    while i < len(sentence):
        char, tag = sentence[i]
        
        if tag.startswith("S-"):
            # 单字实体
            entity_type = tag[2:]
            entities.append((char, entity_type, i, i))
            i += 1
        elif tag.startswith("B-"):
            # 实体开始
            entity_type = tag[2:]
            entity_chars = [char]
            start_idx = i
            i += 1
            
            # 继续读取 M 和 E 标签
            while i < len(sentence):
                next_char, next_tag = sentence[i]
                if next_tag == f"M-{entity_type}":
                    entity_chars.append(next_char)
                    i += 1
                elif next_tag == f"E-{entity_type}":
                    entity_chars.append(next_char)
                    end_idx = i
                    entities.append(("".join(entity_chars), entity_type, start_idx, end_idx))
                    i += 1
                    break
                else:
                    # 标签序列不完整，跳过
                    break
        else:
            i += 1
    
    return entities


def extract_entities_by_type(sentences: List[List[Tuple[str, str]]]) -> Dict[str, List[str]]:
    """从句子列表中提取实体，按类型分组
    
    Args:
        sentences: 句子列表
    
    Returns:
        dict: {entity_type: [entity_text, ...]}
        例如: {'PAR': ['温度', '湿度', ...], 'DIS': ['根腐病', ...]}
    """
    entity_pool = defaultdict(set)  # 使用 set 去重
    
    for sentence in sentences:
        entities = extract_entities_from_sentence(sentence)
        for entity_text, entity_type, _, _ in entities:
            entity_pool[entity_type].add(entity_text)
    
    # 转换为列表并排序（保证可重复性）
    return {k: sorted(list(v)) for k, v in entity_pool.items()}


def generate_bmes_tags(entity_text: str, entity_type: str) -> List[str]:
    """为实体生成 BMES 标签序列
    
    Args:
        entity_text: 实体文本
        entity_type: 实体类型
    
    Returns:
        标签列表
    
    示例：
        - "疫"（1字）-> ["S-DIS"]
        - "锈病"（2字）-> ["B-DIS", "E-DIS"]
        - "根腐病"（3字）-> ["B-DIS", "M-DIS", "E-DIS"]
        - "炭疽病菌"（4字）-> ["B-DIS", "M-DIS", "M-DIS", "E-DIS"]
    """
    length = len(entity_text)
    
    if length == 1:
        return [f"S-{entity_type}"]
    elif length == 2:
        return [f"B-{entity_type}", f"E-{entity_type}"]
    else:
        tags = [f"B-{entity_type}"]
        tags.extend([f"M-{entity_type}"] * (length - 2))
        tags.append(f"E-{entity_type}")
        return tags


def replace_entity_in_sentence(
    sentence: List[Tuple[str, str]], 
    entity_pool: Dict[str, List[str]], 
    replace_prob: float = 0.3,
    rng: random.Random = None
) -> List[Tuple[str, str]]:
    """对单个句子中的实体进行随机替换
    
    Args:
        sentence: [(char, tag), ...] 列表
        entity_pool: {type: [entity_text, ...]} 实体替换池
        replace_prob: 每个实体被替换的概率
        rng: 随机数生成器
    
    Returns:
        新句子 [(char, tag), ...]
    """
    if rng is None:
        rng = random.Random()
    
    # 提取当前句子中的所有实体及其位置
    entities = extract_entities_from_sentence(sentence)
    
    if not entities:
        # 没有实体，原样返回
        return list(sentence)
    
    # 决定哪些实体要替换
    replacements = []  # [(start_idx, end_idx, new_entity_text, entity_type), ...]
    
    for entity_text, entity_type, start_idx, end_idx in entities:
        if rng.random() < replace_prob:
            # 检查是否有可替换的同类型实体
            if entity_type in entity_pool and len(entity_pool[entity_type]) > 0:
                # 获取可替换的实体（排除自身）
                candidates = [e for e in entity_pool[entity_type] if e != entity_text]
                if candidates:
                    new_entity = rng.choice(candidates)
                    replacements.append((start_idx, end_idx, new_entity, entity_type))
    
    if not replacements:
        # 没有替换发生，原样返回
        return list(sentence)
    
    # 从后向前替换（避免索引偏移问题）
    new_sentence = list(sentence)
    for start_idx, end_idx, new_entity, entity_type in reversed(replacements):
        # 生成新实体的 BMES 标签
        new_tags = generate_bmes_tags(new_entity, entity_type)
        new_chars = list(new_entity)
        
        # 替换
        new_tokens = list(zip(new_chars, new_tags))
        new_sentence = new_sentence[:start_idx] + new_tokens + new_sentence[end_idx + 1:]
    
    return new_sentence


def augment_bmes_dataset(
    sentences: List[List[Tuple[str, str]]], 
    entity_pool: Dict[str, List[str]], 
    aug_ratio: int = 2, 
    replace_prob: float = 0.3, 
    seed: int = 42
) -> List[List[Tuple[str, str]]]:
    """对整个 BMES 数据集进行增强
    
    Args:
        sentences: 原始句子列表
        entity_pool: 实体替换池
        aug_ratio: 扩增倍数（生成 aug_ratio 倍新数据）
        replace_prob: 替换概率
        seed: 随机种子
    
    Returns:
        原始句子 + 增强句子
    """
    rng = random.Random(seed)
    
    augmented_sentences = list(sentences)  # 保留原始数据
    
    for _ in range(aug_ratio):
        for sentence in sentences:
            # 对每个句子进行实体替换增强
            new_sentence = replace_entity_in_sentence(
                sentence, entity_pool, replace_prob, rng
            )
            # 只有当新句子与原句子不同时才添加
            if new_sentence != sentence:
                augmented_sentences.append(new_sentence)
    
    return augmented_sentences


# ============================================================================
# eznlp 数据格式处理函数
# ============================================================================

def extract_entities_from_eznlp_entry(entry: dict) -> List[Tuple[str, str]]:
    """从 eznlp 数据条目中提取实体
    
    Args:
        entry: eznlp 数据条目，包含 'tokens' 和 'chunks'
    
    Returns:
        实体列表：[(entity_text, entity_type), ...]
    """
    tokens = entry["tokens"]
    chunks = entry["chunks"]  # [(type, start, end), ...]
    
    # 获取字符列表
    chars = tokens.raw_text if hasattr(tokens, 'raw_text') else [t.raw_text for t in tokens.token_list]
    
    entities = []
    for entity_type, start, end in chunks:
        entity_text = "".join(chars[start:end])
        entities.append((entity_text, entity_type))
    
    return entities


def extract_entity_pool_from_eznlp(data: list) -> Dict[str, List[str]]:
    """从 eznlp 数据中提取实体池
    
    Args:
        data: eznlp 格式的数据列表
    
    Returns:
        dict: {entity_type: [entity_text, ...]}
    """
    entity_pool = defaultdict(set)
    
    for entry in data:
        entities = extract_entities_from_eznlp_entry(entry)
        for entity_text, entity_type in entities:
            entity_pool[entity_type].add(entity_text)
    
    return {k: sorted(list(v)) for k, v in entity_pool.items()}


def eznlp_entry_to_bmes(entry: dict) -> List[Tuple[str, str]]:
    """将 eznlp 数据条目转换为 BMES 格式
    
    Args:
        entry: eznlp 数据条目，包含 'tokens' 和 'chunks'
    
    Returns:
        BMES 格式的句子 [(char, tag), ...]
    """
    tokens = entry["tokens"]
    chunks = entry["chunks"]  # [(type, start, end), ...]
    
    # 获取字符列表
    chars = tokens.raw_text if hasattr(tokens, 'raw_text') else [t.raw_text for t in tokens.token_list]
    
    # 初始化所有标签为 O
    tags = ["O"] * len(chars)
    
    # 根据 chunks 设置 BMES 标签
    for entity_type, start, end in chunks:
        length = end - start
        if length == 1:
            tags[start] = f"S-{entity_type}"
        elif length == 2:
            tags[start] = f"B-{entity_type}"
            tags[end - 1] = f"E-{entity_type}"
        else:
            tags[start] = f"B-{entity_type}"
            for i in range(start + 1, end - 1):
                tags[i] = f"M-{entity_type}"
            tags[end - 1] = f"E-{entity_type}"
    
    return list(zip(chars, tags))


def bmes_to_eznlp_entry(
    sentence: List[Tuple[str, str]], 
    original_entry: dict = None
) -> dict:
    """将 BMES 格式句子转换回 eznlp 数据条目
    
    Args:
        sentence: BMES 格式的句子 [(char, tag), ...]
        original_entry: 原始 eznlp 条目（用于获取配置信息）
    
    Returns:
        eznlp 格式的数据条目
    """
    from eznlp.io import ConllIO
    from eznlp.utils import ChunksTagsTranslator
    
    chars = [c for c, t in sentence]
    tags = [t for c, t in sentence]
    
    # 使用 ChunksTagsTranslator 将标签转换为 chunks
    translator = ChunksTagsTranslator(scheme="BMES", sep="-", breaking_for_types=True)
    chunks = translator.tags2chunks(tags)
    
    # 创建 TokenSequence
    if original_entry is not None:
        original_tokens = original_entry["tokens"]
        token_sep = original_tokens.token_sep
        pad_token = original_tokens.pad_token
        none_token = original_tokens.none_token
    else:
        token_sep = ""
        pad_token = "<pad>"
        none_token = "<none>"
    
    from eznlp.token import Token, TokenSequence
    token_list = [Token(c) for c in chars]
    tokens = TokenSequence(token_list, token_sep=token_sep, pad_token=pad_token, none_token=none_token)
    
    return {
        "tokens": tokens,
        "chunks": chunks,
        "doc_idx": original_entry.get("doc_idx", "0") if original_entry else "0"
    }


def augment_eznlp_entry(
    entry: dict,
    entity_pool: Dict[str, List[str]],
    replace_prob: float = 0.3,
    rng: random.Random = None
) -> dict:
    """对单个 eznlp 数据条目进行增强
    
    Args:
        entry: eznlp 数据条目
        entity_pool: 实体替换池
        replace_prob: 替换概率
        rng: 随机数生成器
    
    Returns:
        增强后的 eznlp 数据条目
    """
    # 转换为 BMES 格式
    bmes_sentence = eznlp_entry_to_bmes(entry)
    
    # 进行实体替换
    new_bmes_sentence = replace_entity_in_sentence(bmes_sentence, entity_pool, replace_prob, rng)
    
    # 如果没有变化，返回 None
    if new_bmes_sentence == bmes_sentence:
        return None
    
    # 转换回 eznlp 格式
    new_entry = bmes_to_eznlp_entry(new_bmes_sentence, entry)
    
    return new_entry


def augment_eznlp_data(
    train_data: list,
    aug_ratio: int = 2,
    replace_prob: float = 0.3,
    seed: int = 42
) -> list:
    """对 eznlp 格式的训练数据进行增强
    
    这是供训练脚本调用的主要接口。
    
    Args:
        train_data: eznlp 格式的训练数据列表（每个元素有 'tokens' 和 'chunks'）
        aug_ratio: 扩增倍数
        replace_prob: 替换概率
        seed: 随机种子
    
    Returns:
        增强后的数据列表（原始 + 增强）
    """
    rng = random.Random(seed)
    
    # 从训练数据中提取实体池
    entity_pool = extract_entity_pool_from_eznlp(train_data)
    
    # 记录实体池统计信息
    total_entities = sum(len(v) for v in entity_pool.values())
    print(f"[数据增强] 实体池统计: {len(entity_pool)} 种类型, 共 {total_entities} 个不同实体")
    for ent_type, ent_list in sorted(entity_pool.items()):
        print(f"  - {ent_type}: {len(ent_list)} 个实体")
    
    # 保留原始数据
    augmented_data = list(train_data)
    
    # 进行多轮增强
    for round_idx in range(aug_ratio):
        augmented_count = 0
        for entry in train_data:
            new_entry = augment_eznlp_entry(entry, entity_pool, replace_prob, rng)
            if new_entry is not None:
                augmented_data.append(new_entry)
                augmented_count += 1
        print(f"[数据增强] 第 {round_idx + 1} 轮: 生成 {augmented_count} 条增强数据")
    
    return augmented_data


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RedJujube BMES 数据增强工具")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="输入 BMES 文件路径")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="输出增强后的 BMES 文件路径")
    parser.add_argument("--aug_ratio", type=int, default=2, 
                        help="扩增倍数（默认：2）")
    parser.add_argument("--replace_prob", type=float, default=0.3, 
                        help="实体替换概率（默认：0.3）")
    parser.add_argument("--seed", type=int, default=42, 
                        help="随机种子（默认：42）")
    parser.add_argument("--stats_only", action="store_true",
                        help="仅输出统计信息，不进行增强")
    
    args = parser.parse_args()
    
    print(f"读取输入文件: {args.input_file}")
    sentences = read_bmes_file(args.input_file)
    print(f"读取 {len(sentences)} 个句子")
    
    # 提取实体池
    entity_pool = extract_entities_by_type(sentences)
    
    # 输出统计信息
    print("\n实体池统计:")
    total_entities = sum(len(v) for v in entity_pool.values())
    print(f"  共 {len(entity_pool)} 种实体类型, {total_entities} 个不同实体")
    for ent_type, ent_list in sorted(entity_pool.items()):
        print(f"  - {ent_type}: {len(ent_list)} 个实体")
        # 展示前5个示例
        examples = ent_list[:5]
        print(f"    示例: {', '.join(examples)}")
    
    if args.stats_only:
        return
    
    # 进行数据增强
    print(f"\n开始数据增强:")
    print(f"  扩增倍数: {args.aug_ratio}")
    print(f"  替换概率: {args.replace_prob}")
    print(f"  随机种子: {args.seed}")
    
    augmented_sentences = augment_bmes_dataset(
        sentences, 
        entity_pool, 
        aug_ratio=args.aug_ratio, 
        replace_prob=args.replace_prob, 
        seed=args.seed
    )
    
    print(f"\n增强结果:")
    print(f"  原始句子数: {len(sentences)}")
    print(f"  增强后句子数: {len(augmented_sentences)}")
    print(f"  实际增加: {len(augmented_sentences) - len(sentences)} 个句子")
    
    # 写入输出文件
    write_bmes_file(args.output_file, augmented_sentences)
    print(f"\n增强数据已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
