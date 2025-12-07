#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将实体列表转换为专家词典格式
输入：data/实体列表.txt（包含 CUL、PRO、PAR 等类别的实体）
输出：data/HZ/expert_lexicon.txt（每行一个词条）
"""

import re
import os


def parse_entity_list(file_path):
    """解析实体列表文件"""
    entities = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配每一行的类别定义
    # 格式：CUL=["实体1", "实体2", ...]
    pattern = r'(\w+)=\[(.*?)\]'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for category, entity_string in matches:
        # 解析实体列表
        # 使用正则提取引号内的内容
        entity_pattern = r'"([^"]+)"'
        category_entities = re.findall(entity_pattern, entity_string)
        
        print(f"类别 {category}: 发现 {len(category_entities)} 个实体")
        entities.extend(category_entities)
    
    return entities


def save_expert_lexicon(entities, output_path):
    """保存为专家词典格式（每行一个词条）"""
    # 去重并排序
    unique_entities = sorted(set(entities))
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for entity in unique_entities:
            # 过滤掉空白或特殊字符
            entity = entity.strip()
            if entity and entity != ' ':  # 排除空白实体
                f.write(entity + '\n')
    
    print(f"\n✅ 成功保存 {len(unique_entities)} 个唯一实体到 {output_path}")


def main():
    # 输入输出路径
    input_file = "data/实体列表.txt"
    output_file = "data/HZ/expert_lexicon.txt"
    
    print(f"📖 读取实体列表：{input_file}")
    entities = parse_entity_list(input_file)
    
    print(f"\n📊 统计信息：")
    print(f"  - 总实体数（含重复）：{len(entities)}")
    print(f"  - 唯一实体数：{len(set(entities))}")
    
    print(f"\n💾 保存专家词典：{output_file}")
    save_expert_lexicon(entities, output_file)
    
    # 展示前 20 个实体
    print(f"\n🔍 前 20 个实体示例：")
    for i, entity in enumerate(sorted(set(entities))[:20], 1):
        print(f"  {i:2d}. {entity}")


if __name__ == "__main__":
    main()
