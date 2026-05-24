#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIO格式 → BMES格式转换脚本

支持两个数据集:
1. WeiboNER: 特殊格式 `字N\tTAG`, 需去除数字后缀, 合并.NAM/.NOM
2. People's Daily: 标准格式 `字 TAG`
"""

import os
import re
from collections import defaultdict
from pathlib import Path


def parse_weibo_line(line):
    """解析WeiboNER行: `字N\tTAG` -> (字, TAG)"""
    if not line.strip():
        return None, None
    
    parts = line.strip().split('\t')
    if len(parts) != 2:
        return None, None
    
    char_with_num, tag = parts
    # 去掉字符后的数字后缀 (如 "科0" -> "科")
    char = re.sub(r'\d+$', '', char_with_num)
    
    # 简化实体类型: 去掉 .NAM/.NOM 后缀
    if tag != 'O':
        tag = re.sub(r'\.(NAM|NOM)$', '', tag)
    
    return char, tag


def parse_standard_line(line):
    """解析标准格式行: `字 TAG` -> (字, TAG)"""
    if not line.strip():
        return None, None
    
    parts = line.strip().split()
    if len(parts) != 2:
        return None, None
    
    return parts[0], parts[1]


def bio_to_bmes(sentences):
    """
    BIO → BMES 转换
    
    规则:
    - B 后面跟 I → B 不变，中间的 I 变 M，最后一个 I 变 E
    - B 后面跟 O 或另一个 B 或句子结束 → B 变 S (单字实体)
    """
    converted_sentences = []
    
    for sentence in sentences:
        if not sentence:
            continue
        
        chars = [item[0] for item in sentence]
        tags = [item[1] for item in sentence]
        new_tags = tags.copy()
        
        i = 0
        while i < len(tags):
            tag = tags[i]
            
            if tag.startswith('B-'):
                entity_type = tag[2:]
                # 找到实体结束位置
                j = i + 1
                while j < len(tags) and tags[j] == f'I-{entity_type}':
                    j += 1
                
                entity_len = j - i
                
                if entity_len == 1:
                    # 单字实体: B → S
                    new_tags[i] = f'S-{entity_type}'
                else:
                    # 多字实体: B, M..., E
                    new_tags[i] = f'B-{entity_type}'
                    for k in range(i + 1, j - 1):
                        new_tags[k] = f'M-{entity_type}'
                    new_tags[j - 1] = f'E-{entity_type}'
                
                i = j
            else:
                i += 1
        
        converted_sentences.append(list(zip(chars, new_tags)))
    
    return converted_sentences


def read_bio_file(filepath, parser_func):
    """读取BIO格式文件, 返回句子列表"""
    sentences = []
    current_sentence = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            char, tag = parser_func(line)
            
            if char is None:
                # 空行表示句子结束
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append((char, tag))
    
    # 处理最后一个句子
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def write_bmes_file(sentences, filepath):
    """写入BMES格式文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            for char, tag in sentence:
                f.write(f'{char} {tag}\n')
            if i < len(sentences) - 1:
                f.write('\n')


def get_entity_stats(sentences):
    """统计实体类型分布"""
    entity_counts = defaultdict(int)
    
    for sentence in sentences:
        for char, tag in sentence:
            if tag.startswith('B-') or tag.startswith('S-'):
                entity_type = tag[2:]
                entity_counts[entity_type] += 1
    
    return dict(entity_counts)


def convert_dataset(name, src_files, dst_files, dst_files_alt, parser_func):
    """转换单个数据集"""
    print(f'\n{"="*60}')
    print(f'转换数据集: {name}')
    print(f'{"="*60}')
    
    total_sentences = 0
    total_entity_stats = defaultdict(int)
    
    for split in ['train', 'dev', 'test']:
        src_file = src_files[split]
        dst_file = dst_files[split]
        dst_file_alt = dst_files_alt[split]
        
        if not os.path.exists(src_file):
            print(f'  [警告] 源文件不存在: {src_file}')
            continue
        
        # 读取并转换
        sentences = read_bio_file(src_file, parser_func)
        bmes_sentences = bio_to_bmes(sentences)
        
        # 写入两个位置
        write_bmes_file(bmes_sentences, dst_file)
        write_bmes_file(bmes_sentences, dst_file_alt)
        
        # 统计
        entity_stats = get_entity_stats(bmes_sentences)
        
        print(f'\n  [{split}]')
        print(f'    句子数: {len(bmes_sentences)}')
        print(f'    实体分布: {dict(entity_stats)}')
        print(f'    输出: {dst_file}')
        print(f'    输出: {dst_file_alt}')
        
        total_sentences += len(bmes_sentences)
        for k, v in entity_stats.items():
            total_entity_stats[k] += v
    
    print(f'\n  [总计]')
    print(f'    总句子数: {total_sentences}')
    print(f'    总实体分布: {dict(total_entity_stats)}')


def main():
    base_dir = Path('/home/shiwenlong/NERlabs/eznlp/datasets/raw')
    
    # 数据集1: WeiboNER
    weibo_src = {
        'train': '/tmp/cnlp_corpus/NER/Weibo/weiboNER_2nd_conll.train',
        'dev': '/tmp/cnlp_corpus/NER/Weibo/weiboNER_2nd_conll.dev',
        'test': '/tmp/cnlp_corpus/NER/Weibo/weiboNER_2nd_conll.test',
    }
    weibo_dst = {
        'train': str(base_dir / 'WeiboNER/train.char.bmes'),
        'dev': str(base_dir / 'WeiboNER/dev.char.bmes'),
        'test': str(base_dir / 'WeiboNER/test.char.bmes'),
    }
    weibo_dst_alt = {
        'train': str(base_dir / 'weibo_as_redjujube/redjujube_train.bmes'),
        'dev': str(base_dir / 'weibo_as_redjujube/redjujube_dev.bmes'),
        'test': str(base_dir / 'weibo_as_redjujube/redjujube_test.bmes'),
    }
    
    convert_dataset(
        name='WeiboNER (微博NER)',
        src_files=weibo_src,
        dst_files=weibo_dst,
        dst_files_alt=weibo_dst_alt,
        parser_func=parse_weibo_line
    )
    
    # 数据集2: People's Daily
    pd_src = {
        'train': "/tmp/cnlp_corpus/NER/People's Daily/example.train",
        'dev': "/tmp/cnlp_corpus/NER/People's Daily/example.dev",
        'test': "/tmp/cnlp_corpus/NER/People's Daily/example.test",
    }
    pd_dst = {
        'train': str(base_dir / 'PeopleDaily/train.char.bmes'),
        'dev': str(base_dir / 'PeopleDaily/dev.char.bmes'),
        'test': str(base_dir / 'PeopleDaily/test.char.bmes'),
    }
    pd_dst_alt = {
        'train': str(base_dir / 'peopledaily_as_redjujube/redjujube_train.bmes'),
        'dev': str(base_dir / 'peopledaily_as_redjujube/redjujube_dev.bmes'),
        'test': str(base_dir / 'peopledaily_as_redjujube/redjujube_test.bmes'),
    }
    
    convert_dataset(
        name="People's Daily (人民日报)",
        src_files=pd_src,
        dst_files=pd_dst,
        dst_files_alt=pd_dst_alt,
        parser_func=parse_standard_line
    )
    
    print(f'\n{"="*60}')
    print('转换完成!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
