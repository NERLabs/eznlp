#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练数据集抽取带类型的专家词典 (Typed Lexicon)

用法:
python research.data_processing/extract_typed_lexicon.py \
    --input datasets/raw/RedJujube/redjujube_train.bmes \
    --output datasets/raw/RedJujube/expert_lexicon_typed.txt \
    --min_freq 2
"""

import argparse
from collections import Counter
from pathlib import Path
import sys
import os

# 将项目根目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.data_processing.extract_lexicon_from_training import load_bmes_file, extract_entities, extract_lexicon_from_data, save_lexicon

def main():
    parser = argparse.ArgumentParser(description='从训练数据集抽取带类型的专家词典')
    parser.add_argument('--input', type=str, required=True,
                        help='训练数据路径 (BMES格式)')
    parser.add_argument('--output', type=str, required=True,
                        help='输出词典路径')
    parser.add_argument('--min_freq', type=int, default=1,
                        help='最小频次阈值 (默认1)')
    parser.add_argument('--max_length', type=int, default=None,
                        help='最大实体长度限制')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔧 提取带类型的专家词典 (Typed Lexicon Extraction)")
    print(f"   输入: {args.input}")
    print(f"   输出: {args.output}")
    print(f"   最小频次: {args.min_freq}")
    print("=" * 70)
    print()
    
    # 提取词典 (强制使用 save_with_type=True 的逻辑)
    entity_counter, entity_types = extract_lexicon_from_data(
        args.input,
        min_freq=args.min_freq,
        max_length=args.max_length,
        save_with_type=True
    )
    
    # 保存词典 (强制使用 save_with_type=True)
    save_lexicon(
        entity_counter,
        entity_types,
        args.output,
        save_with_type=True
    )
    
    print()
    print("=" * 70)
    print("✅ 带类型词典提取完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
