#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMES格式数据转换为piqn所需的JSON格式

piqn数据格式要求：
- tokens: 字符列表
- entities: 实体span列表 [{"type": "XXX", "start": N, "end": M}]
- 每个样本为一个JSON对象，整个文件为JSON数组
"""

import json
import os
import sys


def parse_bmes_file(bmes_file_path):
    """
    解析BMES格式文件，提取句子和实体
    
    Args:
        bmes_file_path: BMES格式文件路径
        
    Returns:
        list of dict: 每个dict包含tokens和entities
    """
    sentences = []
    current_tokens = []
    current_entities = []
    entity_start = None
    entity_type = None
    
    with open(bmes_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # 空行表示句子结束
                if current_tokens:
                    sentences.append({
                        "tokens": current_tokens,
                        "entities": current_entities,
                        "relations": [],
                        "org_id": str(len(sentences)),
                        "pos": ["NO"] * len(current_tokens),
                        "ltokens": [],
                        "rtokens": []
                    })
                    current_tokens = []
                    current_entities = []
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                tag = parts[1]
                current_tokens.append(token)
                
                # 解析BMES标签
                if tag.startswith('B-'):
                    # 实体开始
                    entity_type = tag[2:]
                    entity_start = len(current_tokens) - 1
                elif tag.startswith('M-'):
                    # 实体中间，继续
                    pass
                elif tag.startswith('E-'):
                    # 实体结束
                    if entity_start is not None:
                        entity_end = len(current_tokens)  # exclusive索引
                        current_entities.append({
                            "type": entity_type,
                            "start": entity_start,
                            "end": entity_end
                        })
                        entity_start = None
                        entity_type = None
                elif tag.startswith('S-'):
                    # 单字实体
                    entity_type = tag[2:]
                    current_entities.append({
                        "type": entity_type,
                        "start": len(current_tokens) - 1,
                        "end": len(current_tokens)  # exclusive索引
                    })
                    entity_start = None
                    entity_type = None
                else:
                    # O标签或其他
                    entity_start = None
                    entity_type = None
    
    # 处理最后一个句子（如果没有空行结尾）
    if current_tokens:
        sentences.append({
            "tokens": current_tokens,
            "entities": current_entities,
            "relations": [],
            "org_id": str(len(sentences)),
            "pos": ["NO"] * len(current_tokens),
            "ltokens": [],
            "rtokens": []
        })
    
    return sentences


def convert_bmes_to_piqn_json(bmes_file_path, json_file_path):
    """
    将BMES格式文件转换为piqn所需的JSON格式
    
    Args:
        bmes_file_path: BMES格式文件路径
        json_file_path: 输出JSON文件路径
    """
    sentences = parse_bmes_file(bmes_file_path)
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成: {bmes_file_path} -> {json_file_path}")
    print(f"共 {len(sentences)} 个句子")
    
    # 统计实体数量
    total_entities = sum(len(s['entities']) for s in sentences)
    print(f"共 {total_entities} 个实体")


def generate_types_json(json_file_path, entity_types):
    """
    生成piqn所需的types.json文件
    
    Args:
        json_file_path: 输出JSON文件路径
        entity_types: 实体类型列表
    """
    types_dict = {
        "entities": {},
        "relations": {}
    }
    
    # 实体类型描述
    type_descriptions = {
        "CUL": "枣品种",
        "PRO": "枣产品",
        "PAR": "枣部位",
        "PER": "生长时期",
        "TAX": "生物分类",
        "NUT": "营养成分",
        "DIS": "病害",
        "PES": "虫害",
        "EQU": "机械装备",
        "FER": "肥料",
        "DRU": "农药",
        "AGR": "农艺",
        "WED": "杂草",
        "LOC": "地理位置"
    }
    
    for etype in entity_types:
        types_dict["entities"][etype] = {
            "verbose": type_descriptions.get(etype, etype),
            "short": etype
        }
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(types_dict, f, ensure_ascii=False, indent=2)
    
    print(f"类型文件生成完成: {json_file_path}")


if __name__ == "__main__":
    # 红枣数据集路径
    data_dir = "/home/shiwenlong/NERlabs/eznlp/_2DATA/RedJujube"
    output_dir = "/home/shiwenlong/NERlabs/piqn/_2DATA/redjujube"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 实体类型列表（根据数据集统计）
    entity_types = ["CUL", "PRO", "PAR", "PER", "TAX", "NUT", "DIS", "PES", "EQU", "FER", "DRU", "AGR", "WED", "LOC"]
    
    # 转换训练集
    convert_bmes_to_piqn_json(
        os.path.join(data_dir, "redjujube_train.bmes"),
        os.path.join(output_dir, "redjujube_train_context.json")
    )
    
    # 转换验证集
    convert_bmes_to_piqn_json(
        os.path.join(data_dir, "redjujube_dev.bmes"),
        os.path.join(output_dir, "redjujube_dev_context.json")
    )
    
    # 转换测试集
    convert_bmes_to_piqn_json(
        os.path.join(data_dir, "redjujube_test.bmes"),
        os.path.join(output_dir, "redjujube_test_context.json")
    )
    
    # 生成类型文件
    generate_types_json(
        os.path.join(output_dir, "redjujube_types.json"),
        entity_types
    )