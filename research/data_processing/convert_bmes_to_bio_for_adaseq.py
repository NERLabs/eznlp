#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMES格式转换为BIO格式
用于AdaSeq项目数据适配

BMES格式:
- B-X: 实体开始
- M-X: 实体中间
- E-X: 实体结束
- S-X: 单字实体
- O: 非实体

BIO格式:
- B-X: 实体开始
- I-X: 实体中间（包括BMES的M和E）
- O: 非实体

注意：AdaSeq期望的是BIO格式，不支持M-标签
"""

import os
import sys


def convert_bmes_to_bio(bmes_file, bio_file):
    """
    将BMES格式文件转换为BIO格式
    
    Args:
        bmes_file: BMES格式输入文件路径
        bio_file: BIO格式输出文件路径
    """
    converted_lines = 0
    entity_count = 0
    
    with open(bmes_file, 'r', encoding='utf-8') as f_in, open(bio_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                # 空行保持不变
                f_out.write('\n')
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                tag = parts[1]
                
                # BMES -> BIO 转换规则
                if tag == 'O':
                    new_tag = 'O'
                elif tag.startswith('B-'):
                    # B- 保持不变
                    new_tag = tag
                    entity_count += 1
                elif tag.startswith('M-') or tag.startswith('E-'):
                    # M- 和 E- 都转换为 I-
                    entity_type = tag[2:]
                    new_tag = 'I-' + entity_type
                elif tag.startswith('S-'):
                    # S-（单字实体）转换为 B-（因为AdaSeq不支持S-）
                    entity_type = tag[2:]
                    new_tag = 'B-' + entity_type
                    entity_count += 1
                else:
                    # 其他情况保持不变
                    new_tag = tag
                
                f_out.write(f"{token} {new_tag}\n")
                converted_lines += 1
    
    return converted_lines, entity_count


def main():
    # 红枣数据集路径
    source_dir = "/home/shiwenlong/NERlabs/eznlp/datasets/raw/RedJujube"
    target_dir = "/home/shiwenlong/NERlabs/AdaSeq/adaseq/data/redjujube_bio"
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    print("=" * 60)
    print("BMES to BIO Conversion for AdaSeq")
    print("=" * 60)
    
    files = [
        ('redjujube_train.bmes', 'train.bio'),
        ('redjujube_dev.bmes', 'dev.bio'),
        ('redjujube_test.bmes', 'test.bio'),
    ]
    
    total_lines = 0
    total_entities = 0
    
    for src_file, tgt_file in files:
        src_path = os.path.join(source_dir, src_file)
        tgt_path = os.path.join(target_dir, tgt_file)
        
        if os.path.exists(src_path):
            lines, entities = convert_bmes_to_bio(src_path, tgt_path)
            total_lines += lines
            total_entities += entities
            print(f"✅ {src_file} -> {tgt_file}")
            print(f"   Lines: {lines}, Entities converted: {entities}")
        else:
            print(f"❌ Source file not found: {src_path}")
    
    print("=" * 60)
    print(f"Total lines converted: {total_lines}")
    print(f"Total entities: {total_entities}")
    print(f"Output directory: {target_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()