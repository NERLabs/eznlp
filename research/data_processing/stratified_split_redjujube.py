#!/usr/bin/env python3
"""
RedJujube NER 数据集分层随机划分脚本

按实体类型分布做分层划分，确保 train/dev/test 中每种实体类型的比例一致。
分层策略：多标签分层划分，优先保证稀有类型的均匀分布
划分比例：8:1:1 (train:dev:test)
随机种子：42
"""

import random
import os
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set


def parse_bmes_file(filepath: str) -> List[List[Tuple[str, str]]]:
    """解析 BMES 格式文件，返回句子列表，每个句子是 (char, tag) 元组列表"""
    sentences = []
    current_sentence = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # 空行表示句子结束
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    char, tag = parts[0], parts[1]
                    current_sentence.append((char, tag))
                elif len(parts) == 1:
                    # 可能是只有标签或字符的情况
                    current_sentence.append((parts[0], 'O'))
        
        # 处理最后一个句子
        if current_sentence:
            sentences.append(current_sentence)
    
    return sentences


def write_bmes_file(filepath: str, sentences: List[List[Tuple[str, str]]]):
    """将句子列表写入 BMES 格式文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            for char, tag in sentence:
                f.write(f"{char} {tag}\n")
            # 句子之间用空行分隔，最后一个句子后也加空行
            if i < len(sentences) - 1:
                f.write("\n")


def extract_entity_types(sentence: List[Tuple[str, str]]) -> Set[str]:
    """从句子中提取所有实体类型"""
    entity_types = set()
    for char, tag in sentence:
        if tag != 'O':
            # 标签格式: B-TYPE, M-TYPE, E-TYPE, S-TYPE
            entity_type = tag.split('-')[1] if '-' in tag else None
            if entity_type:
                entity_types.add(entity_type)
    return entity_types


def count_entities(sentences: List[List[Tuple[str, str]]]) -> Counter:
    """统计实体数量（按完整实体计数，不是标签数）"""
    entity_counter = Counter()
    
    for sentence in sentences:
        current_entity_type = None
        for char, tag in sentence:
            if tag.startswith('B-') or tag.startswith('S-'):
                # 新实体开始
                entity_type = tag.split('-')[1]
                entity_counter[entity_type] += 1
                current_entity_type = entity_type
            elif tag.startswith('M-') or tag.startswith('E-'):
                # 实体延续或结束，不重复计数
                pass
            else:
                current_entity_type = None
    
    return entity_counter


def get_entity_type_rarity(sentences: List[List[Tuple[str, str]]]) -> Dict[str, int]:
    """计算每种实体类型的稀有度（出现总次数）"""
    entity_counts = count_entities(sentences)
    return dict(entity_counts)


def build_entity_matrix(sentences: List[List[Tuple[str, str]]], 
                        entity_types: List[str]) -> np.ndarray:
    """构建多标签矩阵，每行对应一个句子，每列对应一个实体类型的数量"""
    type_to_idx = {t: i for i, t in enumerate(entity_types)}
    matrix = np.zeros((len(sentences), len(entity_types)), dtype=np.float32)
    
    for i, sentence in enumerate(sentences):
        entity_counts = count_entities([sentence])
        for entity_type, count in entity_counts.items():
            if entity_type in type_to_idx:
                matrix[i, type_to_idx[entity_type]] = count
    
    return matrix


def multilabel_stratified_split(sentences: List[List[Tuple[str, str]]], 
                                  train_ratio: float = 0.8,
                                  dev_ratio: float = 0.1,
                                  test_ratio: float = 0.1,
                                  random_seed: int = 42) -> Tuple[List, List, List]:
    """
    多标签分层随机划分
    
    算法思路：
    1. 按实体类型稀有度排序
    2. 对每种实体类型，从包含该类型的句子中按比例分配到 train/dev/test
    3. 已分配的句子不再重复分配
    4. 优先处理稀有类型，确保它们均匀分布
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    n_samples = len(sentences)
    indices = list(range(n_samples))
    
    # 计算实体类型稀有度
    entity_rarity = get_entity_type_rarity(sentences)
    # 按稀有度排序（最稀有的优先处理）
    sorted_types = sorted(entity_rarity.keys(), key=lambda t: entity_rarity[t])
    
    print(f"\n实体类型稀有度（出现次数，按稀有度排序）:")
    for t in sorted_types:
        print(f"  {t}: {entity_rarity[t]}")
    
    # 为每个句子提取实体类型集合
    sentence_entity_types = [extract_entity_types(s) for s in sentences]
    
    # 初始化分配
    train_indices = set()
    dev_indices = set()
    test_indices = set()
    assigned = set()
    
    # 按稀有度顺序处理每种实体类型
    for entity_type in sorted_types:
        # 找出包含该实体类型且未分配的句子
        type_indices = [i for i in indices if entity_type in sentence_entity_types[i] and i not in assigned]
        
        if not type_indices:
            continue
        
        random.shuffle(type_indices)
        n = len(type_indices)
        
        # 计算目标数量（考虑已分配的样本）
        # 统计该类型在各集合中的当前数量
        current_train = sum(1 for i in train_indices if entity_type in sentence_entity_types[i])
        current_dev = sum(1 for i in dev_indices if entity_type in sentence_entity_types[i])
        current_test = sum(1 for i in test_indices if entity_type in sentence_entity_types[i])
        current_total = current_train + current_dev + current_test
        
        # 总目标数量
        total_for_type = sum(1 for types in sentence_entity_types if entity_type in types)
        
        # 计算还需要分配多少到各集合
        target_train = int(total_for_type * train_ratio) - current_train
        target_dev = int(total_for_type * dev_ratio) - current_dev
        target_test = total_for_type - int(total_for_type * train_ratio) - int(total_for_type * dev_ratio) - current_test
        
        # 确保非负
        target_train = max(0, target_train)
        target_dev = max(0, target_dev)
        target_test = max(0, target_test)
        
        # 按目标比例分配
        total_needed = target_train + target_dev + target_test
        if total_needed == 0:
            # 均匀分配
            n_train = int(n * train_ratio)
            n_dev = int(n * dev_ratio)
            n_test = n - n_train - n_dev
        else:
            ratio_sum = total_needed
            n_train = int(n * target_train / ratio_sum) if ratio_sum > 0 else int(n * train_ratio)
            n_dev = int(n * target_dev / ratio_sum) if ratio_sum > 0 else int(n * dev_ratio)
            n_test = n - n_train - n_dev
        
        # 确保至少有一些分配
        if n >= 3:
            n_train = max(1, n_train)
            n_dev = max(1, n_dev) if n > n_train else 0
            n_test = n - n_train - n_dev
        elif n == 2:
            n_train = 1
            n_dev = 1
            n_test = 0
        else:
            n_train = 1
            n_dev = 0
            n_test = 0
        
        # 分配
        train_indices.update(type_indices[:n_train])
        dev_indices.update(type_indices[n_train:n_train + n_dev])
        test_indices.update(type_indices[n_train + n_dev:])
        assigned.update(type_indices)
    
    # 处理没有实体的句子
    no_entity_indices = [i for i in indices if i not in assigned]
    if no_entity_indices:
        random.shuffle(no_entity_indices)
        n = len(no_entity_indices)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        
        train_indices.update(no_entity_indices[:n_train])
        dev_indices.update(no_entity_indices[n_train:n_train + n_dev])
        test_indices.update(no_entity_indices[n_train + n_dev:])
    
    # 转换为句子列表
    train_data = [sentences[i] for i in train_indices]
    dev_data = [sentences[i] for i in dev_indices]
    test_data = [sentences[i] for i in test_indices]
    
    # 最后打乱
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    
    return train_data, dev_data, test_data


def iterative_stratified_split(sentences: List[List[Tuple[str, str]]], 
                                train_ratio: float = 0.8,
                                dev_ratio: float = 0.1,
                                test_ratio: float = 0.1,
                                random_seed: int = 42) -> Tuple[List, List, List]:
    """
    多标签分层划分算法 - 确保每种实体类型在 train/dev/test 中都有样本
    
    核心策略：
    1. 对每种实体类型，先按比例划分到 train/dev/test
    2. 处理多标签冲突：如果一个句子已被分配，跳过
    3. 优先处理稀有类型，确保它们的均匀分布
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    n_samples = len(sentences)
    
    # 获取所有实体类型及其稀有度
    entity_rarity = get_entity_type_rarity(sentences)
    sorted_types = sorted(entity_rarity.keys(), key=lambda t: entity_rarity[t])
    
    print(f"\n实体类型数量: {len(sorted_types)}")
    print(f"实体类型（按稀有度排序）:")
    for t in sorted_types:
        print(f"  {t}: {entity_rarity[t]}")
    
    # 为每个句子提取实体类型集合
    sentence_entity_types = [extract_entity_types(s) for s in sentences]
    
    # 初始化分配
    assignment = [None] * n_samples  # None = 未分配, 'train', 'dev', 'test'
    
    # 按稀有度顺序处理每种实体类型
    for entity_type in sorted_types:
        # 找出包含该实体类型的所有句子索引
        type_indices = [i for i in range(n_samples) if entity_type in sentence_entity_types[i]]
        
        # 分开已分配和未分配的
        unassigned = [i for i in type_indices if assignment[i] is None]
        assigned_train = [i for i in type_indices if assignment[i] == 'train']
        assigned_dev = [i for i in type_indices if assignment[i] == 'dev']
        assigned_test = [i for i in type_indices if assignment[i] == 'test']
        
        # 计算该类型的目标分配数量
        total = len(type_indices)
        target_train = max(1, int(total * train_ratio))
        target_dev = max(1, int(total * dev_ratio))
        target_test = max(1, total - target_train - target_dev)
        
        # 确保至少 3 个样本时才分配到 3 个集合
        if total < 3:
            target_train = max(1, total - 2)
            target_dev = min(1, total - target_train)
            target_test = total - target_train - target_dev
        
        # 计算还需要分配多少
        need_train = max(0, target_train - len(assigned_train))
        need_dev = max(0, target_dev - len(assigned_dev))
        need_test = max(0, target_test - len(assigned_test))
        
        # 打乱未分配的索引
        random.shuffle(unassigned)
        
        # 按需求分配
        idx = 0
        
        # 先分配到 dev 和 test（确保它们有样本）
        for _ in range(need_dev):
            if idx < len(unassigned):
                assignment[unassigned[idx]] = 'dev'
                idx += 1
        
        for _ in range(need_test):
            if idx < len(unassigned):
                assignment[unassigned[idx]] = 'test'
                idx += 1
        
        # 剩余的分配到 train
        while idx < len(unassigned):
            assignment[unassigned[idx]] = 'train'
            idx += 1
    
    # 处理没有实体的句子
    no_entity_indices = [i for i in range(n_samples) if assignment[i] is None]
    random.shuffle(no_entity_indices)
    
    n = len(no_entity_indices)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    
    for i in range(n):
        if i < n_train:
            assignment[no_entity_indices[i]] = 'train'
        elif i < n_train + n_dev:
            assignment[no_entity_indices[i]] = 'dev'
        else:
            assignment[no_entity_indices[i]] = 'test'
    
    # 转换为句子列表
    train_data = [sentences[i] for i in range(n_samples) if assignment[i] == 'train']
    dev_data = [sentences[i] for i in range(n_samples) if assignment[i] == 'dev']
    test_data = [sentences[i] for i in range(n_samples) if assignment[i] == 'test']
    
    # 最后打乱
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    
    return train_data, dev_data, test_data


def print_statistics(name: str, sentences: List[List[Tuple[str, str]]]):
    """打印数据集统计信息"""
    entity_counts = count_entities(sentences)
    total_entities = sum(entity_counts.values())
    total_sentences = len(sentences)
    
    print(f"\n{name}:")
    print(f"  句子数: {total_sentences}")
    print(f"  实体总数: {total_entities}")
    print(f"  实体类型分布:")
    
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        ratio = count / total_entities * 100 if total_entities > 0 else 0
        print(f"    {entity_type}: {count} ({ratio:.2f}%)")
    
    return entity_counts


def compare_distributions(original_stats: Dict, new_stats: Dict, entity_types: List[str]):
    """对比原始划分和新划分的分布"""
    print("\n" + "=" * 80)
    print("原始划分 vs 新划分 对比")
    print("=" * 80)
    
    headers = ["实体类型", "原Train", "原Dev", "原Test", "新Train", "新Dev", "新Test", "分布变化"]
    col_widths = [10, 12, 12, 12, 12, 12, 12, 15]
    
    # 打印表头
    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    for entity_type in sorted(entity_types):
        orig_train = original_stats['train'].get(entity_type, 0)
        orig_dev = original_stats['dev'].get(entity_type, 0)
        orig_test = original_stats['test'].get(entity_type, 0)
        
        new_train = new_stats['train'].get(entity_type, 0)
        new_dev = new_stats['dev'].get(entity_type, 0)
        new_test = new_stats['test'].get(entity_type, 0)
        
        # 计算占比
        orig_total = orig_train + orig_dev + orig_test
        new_total = new_train + new_dev + new_test
        
        def ratio_str(count, total):
            return f"{count}({count/total*100:.1f}%)" if total > 0 else "0(0%)"
        
        orig_dev_ratio = orig_dev / orig_total * 100 if orig_total > 0 else 0
        orig_test_ratio = orig_test / orig_total * 100 if orig_total > 0 else 0
        new_dev_ratio = new_dev / new_total * 100 if new_total > 0 else 0
        new_test_ratio = new_test / new_total * 100 if new_total > 0 else 0
        
        # 判断改进
        orig_imbalance = abs(orig_dev_ratio - orig_test_ratio)
        new_imbalance = abs(new_dev_ratio - new_test_ratio)
        
        if new_imbalance < orig_imbalance:
            change = "✓ 更均衡"
        elif new_imbalance > orig_imbalance:
            change = "✗ 更不均"
        else:
            change = "- 持平"
        
        row = [
            entity_type.center(col_widths[0]),
            ratio_str(orig_train, orig_total).center(col_widths[1]),
            ratio_str(orig_dev, orig_total).center(col_widths[2]),
            ratio_str(orig_test, orig_total).center(col_widths[3]),
            ratio_str(new_train, new_total).center(col_widths[4]),
            ratio_str(new_dev, new_total).center(col_widths[5]),
            ratio_str(new_test, new_total).center(col_widths[6]),
            change.center(col_widths[7]),
        ]
        print(" | ".join(row))


def main():
    random_seed = 42
    data_dir = "/home/shiwenlong/NERlabs/eznlp/datasets/raw/RedJujube"
    
    train_file = os.path.join(data_dir, "redjujube_train.bmes.orig")
    dev_file = os.path.join(data_dir, "redjujube_dev.bmes.orig")
    test_file = os.path.join(data_dir, "redjujube_test.bmes.orig")
    
    # 检查备份文件
    for f in [train_file, dev_file, test_file]:
        if not os.path.exists(f):
            print(f"错误: 备份文件不存在 {f}")
            print("请先备份原始文件（添加 .orig 后缀）")
            return
    
    print("=" * 80)
    print("RedJujube NER 数据集分层随机划分")
    print(f"随机种子: {random_seed}")
    print(f"划分比例: train:dev:test = 8:1:1")
    print("=" * 80)
    
    # 1. 读取原始数据
    print("\n1. 读取原始数据...")
    orig_train = parse_bmes_file(train_file)
    orig_dev = parse_bmes_file(dev_file)
    orig_test = parse_bmes_file(test_file)
    
    print(f"原始 Train: {len(orig_train)} 句子")
    print(f"原始 Dev: {len(orig_dev)} 句子")
    print(f"原始 Test: {len(orig_test)} 句子")
    
    # 统计原始分布
    orig_train_stats = count_entities(orig_train)
    orig_dev_stats = count_entities(orig_dev)
    orig_test_stats = count_entities(orig_test)
    
    original_stats = {
        'train': orig_train_stats,
        'dev': orig_dev_stats,
        'test': orig_test_stats
    }
    
    print("\n原始数据分布:")
    print_statistics("原始 Train", orig_train)
    print_statistics("原始 Dev", orig_dev)
    print_statistics("原始 Test", orig_test)
    
    # 2. 合并所有数据
    print("\n" + "=" * 80)
    print("2. 合并所有数据...")
    all_data = orig_train + orig_dev + orig_test
    print(f"总句子数: {len(all_data)}")
    
    all_entity_counts = count_entities(all_data)
    all_entity_types = set(all_entity_counts.keys())
    print(f"总实体数: {sum(all_entity_counts.values())}")
    print(f"实体类型: {sorted(all_entity_types)}")
    
    # 3. 分层划分
    print("\n" + "=" * 80)
    print("3. 执行迭代分层划分 (Iterative Stratification)...")
    new_train, new_dev, new_test = iterative_stratified_split(
        all_data, 
        train_ratio=0.8, 
        dev_ratio=0.1, 
        test_ratio=0.1,
        random_seed=random_seed
    )
    
    print(f"\n新 Train: {len(new_train)} 句子")
    print(f"新 Dev: {len(new_dev)} 句子")
    print(f"新 Test: {len(new_test)} 句子")
    
    # 4. 统计新划分的分布
    print("\n" + "=" * 80)
    print("4. 新划分数据分布:")
    new_train_stats = print_statistics("新 Train", new_train)
    new_dev_stats = print_statistics("新 Dev", new_dev)
    new_test_stats = print_statistics("新 Test", new_test)
    
    new_stats = {
        'train': new_train_stats,
        'dev': new_dev_stats,
        'test': new_test_stats
    }
    
    # 5. 对比分布
    compare_distributions(original_stats, new_stats, all_entity_types)
    
    # 6. 计算分布一致性指标
    print("\n" + "=" * 80)
    print("5. 分布一致性评估")
    print("=" * 80)
    
    def calc_distribution_variance(stats: Dict, entity_types: Set[str]) -> float:
        """计算 dev 和 test 中各实体类型占比的平均方差"""
        variances = []
        for entity_type in entity_types:
            train_count = stats['train'].get(entity_type, 0)
            dev_count = stats['dev'].get(entity_type, 0)
            test_count = stats['test'].get(entity_type, 0)
            
            total = train_count + dev_count + test_count
            if total == 0:
                continue
            
            train_ratio = train_count / total
            dev_ratio = dev_count / total
            test_ratio = test_count / total
            
            # 期望比例: 0.8, 0.1, 0.1
            expected = [0.8, 0.1, 0.1]
            actual = [train_ratio, dev_ratio, test_ratio]
            
            variance = sum((a - e) ** 2 for a, e in zip(actual, expected)) / 3
            variances.append(variance)
        
        return sum(variances) / len(variances) if variances else 0
    
    orig_variance = calc_distribution_variance(original_stats, all_entity_types)
    new_variance = calc_distribution_variance(new_stats, all_entity_types)
    
    print(f"原始划分 - 平均分布方差: {orig_variance:.6f}")
    print(f"新划分   - 平均分布方差: {new_variance:.6f}")
    print(f"改进: {(1 - new_variance / orig_variance) * 100:.1f}%" if orig_variance > 0 else "N/A")
    
    # 7. 写入新文件
    print("\n" + "=" * 80)
    print("6. 写入新文件...")
    
    output_train = os.path.join(data_dir, "redjujube_train.bmes")
    output_dev = os.path.join(data_dir, "redjujube_dev.bmes")
    output_test = os.path.join(data_dir, "redjujube_test.bmes")
    
    write_bmes_file(output_train, new_train)
    write_bmes_file(output_dev, new_dev)
    write_bmes_file(output_test, new_test)
    
    print(f"已写入: {output_train}")
    print(f"已写入: {output_dev}")
    print(f"已写入: {output_test}")
    
    print("\n" + "=" * 80)
    print("分层划分完成！")
    print("原始文件已备份为 .orig 后缀")
    print("=" * 80)


if __name__ == "__main__":
    main()
