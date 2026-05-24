#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLAT Lattice 数据处理器

参考: references/external_projects/Flat-Lattice-Transformer-master/V1/add_lattice.py

核心功能:
1. 使用 Trie 树匹配句子中的词汇
2. 生成 lattice 序列（字符 + 词汇拼接）
3. 生成 pos_s, pos_e 位置信息
4. 生成 seq_len, lex_num 长度信息
"""

import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class Trie:
    """Trie 树用于快速词汇匹配"""
    
    def __init__(self):
        self.root = {}
        self.end_flag = '<END>'
    
    def insert(self, word: str):
        """插入一个词"""
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.end_flag] = True
    
    def get_lexicon(self, sentence: str) -> List[Tuple[int, int, str]]:
        """
        获取句子中所有匹配的词汇
        
        Args:
            sentence: 输入句子（字符串）
            
        Returns:
            List of (start, end, word) 元组，end 是包含的最后一个字符的位置
        """
        result = []
        for start in range(len(sentence)):
            node = self.root
            for end in range(start, len(sentence)):
                char = sentence[end]
                if char not in node:
                    break
                node = node[char]
                if self.end_flag in node:
                    # 找到一个词，记录 (start, end, word)
                    word = sentence[start:end+1]
                    if len(word) > 1:  # 只保留多字词
                        result.append((start, end, word))
        return result


class FLATDataProcessor:
    """FLAT 数据处理器
    
    将原始 NER 数据转换为 FLAT 模型需要的 Lattice 格式
    """
    
    def __init__(self, word_list: List[str], max_seq_len: int = 256):
        """
        初始化
        
        Args:
            word_list: 词表列表（用于构建 Trie 树）
            max_seq_len: 最大序列长度
        """
        self.max_seq_len = max_seq_len
        self.trie = Trie()
        
        # 构建 Trie 树
        for word in word_list:
            if len(word) > 1:  # 只添加多字词
                self.trie.insert(word)
        
        print(f"✓ 构建 Trie 树完成，词汇数量: {len(word_list)}")
        
        # 词汇表
        self.char_vocab = {'<PAD>': 0, '<UNK>': 1}
        self.word_vocab = {'<PAD>': 0, '<UNK>': 1}
        self.label_vocab = {'O': 0}
        
        # 反向词汇表
        self.idx2char = {0: '<PAD>', 1: '<UNK>'}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.idx2label = {0: 'O'}
    
    def build_vocab(self, data_list: List[Dict]):
        """
        构建词汇表
        
        Args:
            data_list: 数据列表，每个元素包含 'tokens' 和 'chunks'
        """
        # 收集所有字符
        for data in data_list:
            if hasattr(data, 'tokens'):
                tokens = data.tokens
            else:
                tokens = data.get('tokens', [])
            
            # 处理 tokens
            if hasattr(tokens, 'raw_text'):
                chars = list(tokens.raw_text)
            elif isinstance(tokens, list):
                chars = tokens
            else:
                chars = list(str(tokens))
            
            for char in chars:
                if char not in self.char_vocab:
                    idx = len(self.char_vocab)
                    self.char_vocab[char] = idx
                    self.idx2char[idx] = char
        
        # 收集所有标签
        for data in data_list:
            if hasattr(data, 'chunks'):
                chunks = data.chunks
            else:
                chunks = data.get('chunks', [])
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    label = chunk.label
                else:
                    label = chunk.get('label', 'O')
                
                for prefix in ['B-', 'I-', 'E-', 'S-', 'M-']:
                    tag = f"{prefix}{label}"
                    if tag not in self.label_vocab:
                        idx = len(self.label_vocab)
                        self.label_vocab[tag] = idx
                        self.idx2label[idx] = tag
        
        print(f"✓ 词汇表构建完成:")
        print(f"  - 字符数量: {len(self.char_vocab)}")
        print(f"  - 标签数量: {len(self.label_vocab)}")
    
    def process_sentence(self, chars: List[str], labels: Optional[List[str]] = None) -> Dict:
        """
        处理单个句子，生成 FLAT 格式数据
        
        Args:
            chars: 字符列表
            labels: 标签列表（可选）
            
        Returns:
            包含 lattice, pos_s, pos_e, seq_len, lex_num, target 的字典
        """
        seq_len = min(len(chars), self.max_seq_len)
        chars = chars[:seq_len]
        
        # 获取词汇匹配
        sentence = ''.join(chars)
        lexicons = self.trie.get_lexicon(sentence)
        
        # 过滤超出范围的词汇
        lexicons = [(s, e, w) for s, e, w in lexicons if e < seq_len]
        
        lex_num = len(lexicons)
        
        # 构建 lattice 序列：字符 + 词汇
        lattice = chars + [w for _, _, w in lexicons]
        
        # 构建位置信息
        # 字符位置：pos_s[i] = i, pos_e[i] = i
        # 词汇位置：pos_s[i] = start, pos_e[i] = end
        pos_s = list(range(seq_len)) + [s for s, _, _ in lexicons]
        pos_e = list(range(seq_len)) + [e for _, e, _ in lexicons]
        
        # 转换为索引
        lattice_ids = []
        for item in lattice:
            if len(item) == 1:  # 字符
                lattice_ids.append(self.char_vocab.get(item, 1))
            else:  # 词汇
                # 词汇用字符的索引序列表示，或者单独建词表
                lattice_ids.append(self.char_vocab.get(item[0], 1))
        
        # 处理标签
        if labels is not None:
            labels = labels[:seq_len]
            target = [self.label_vocab.get(l, 0) for l in labels]
        else:
            target = [0] * seq_len
        
        return {
            'chars': chars,
            'lattice': lattice,
            'lattice_ids': lattice_ids,
            'pos_s': pos_s,
            'pos_e': pos_e,
            'seq_len': seq_len,
            'lex_num': lex_num,
            'target': target,
            'lexicons': lexicons,
        }
    
    def process_batch(self, batch_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        处理一批数据，生成可用于模型训练的 tensor
        
        Args:
            batch_data: 批次数据列表
            
        Returns:
            包含各种 tensor 的字典
        """
        batch_size = len(batch_data)
        
        # 计算最大长度
        max_seq_len = max(d['seq_len'] for d in batch_data)
        max_lattice_len = max(d['seq_len'] + d['lex_num'] for d in batch_data)
        
        # 初始化张量
        lattice_ids = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        pos_s = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        pos_e = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        seq_len = torch.zeros(batch_size, dtype=torch.long)
        lex_num = torch.zeros(batch_size, dtype=torch.long)
        target = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        for i, d in enumerate(batch_data):
            cur_len = d['seq_len'] + d['lex_num']
            lattice_ids[i, :cur_len] = torch.tensor(d['lattice_ids'][:cur_len])
            pos_s[i, :cur_len] = torch.tensor(d['pos_s'][:cur_len])
            pos_e[i, :cur_len] = torch.tensor(d['pos_e'][:cur_len])
            seq_len[i] = d['seq_len']
            lex_num[i] = d['lex_num']
            target[i, :d['seq_len']] = torch.tensor(d['target'])
        
        return {
            'lattice': lattice_ids,
            'pos_s': pos_s,
            'pos_e': pos_e,
            'seq_len': seq_len,
            'lex_num': lex_num,
            'target': target,
        }


class FLATDataset(torch.utils.data.Dataset):
    """FLAT 数据集"""
    
    def __init__(self, data_list: List[Dict], processor: FLATDataProcessor):
        """
        初始化
        
        Args:
            data_list: 原始数据列表
            processor: FLAT 数据处理器
        """
        self.processor = processor
        self.processed_data = []
        
        for data in data_list:
            # 提取字符和标签
            if hasattr(data, 'tokens'):
                tokens = data.tokens
                if hasattr(tokens, 'raw_text'):
                    chars = list(tokens.raw_text)
                else:
                    chars = list(tokens)
            else:
                chars = data.get('tokens', [])
            
            # 提取标签（BMES格式）
            labels = self._extract_labels(data, len(chars))
            
            # 处理句子
            processed = processor.process_sentence(chars, labels)
            self.processed_data.append(processed)
    
    def _extract_labels(self, data, seq_len: int) -> List[str]:
        """从 chunks 中提取 BMES 标签"""
        labels = ['O'] * seq_len
        
        if hasattr(data, 'chunks'):
            chunks = data.chunks
        else:
            chunks = data.get('chunks', [])
        
        for chunk in chunks:
            if hasattr(chunk, 'start'):
                start, end = chunk.start, chunk.end
                label = chunk.label
            else:
                start = chunk.get('start', 0)
                end = chunk.get('end', 0)
                label = chunk.get('label', 'O')
            
            if end > seq_len:
                continue
            
            length = end - start
            if length == 1:
                labels[start] = f'S-{label}'
            elif length == 2:
                labels[start] = f'B-{label}'
                labels[start+1] = f'E-{label}'
            else:
                labels[start] = f'B-{label}'
                for i in range(start+1, end-1):
                    labels[i] = f'M-{label}'
                labels[end-1] = f'E-{label}'
        
        return labels
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
    
    def collate_fn(self, batch):
        """DataLoader 的 collate 函数"""
        return self.processor.process_batch(batch)


def load_word_list(word_file: str) -> List[str]:
    """
    从文件加载词表
    
    Args:
        word_file: 词表文件路径
        
    Returns:
        词汇列表
    """
    words = []
    with open(word_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()[0] if line.strip() else ''
            if word and len(word) > 1:
                words.append(word)
    print(f"✓ 加载词表: {len(words)} 个词")
    return words


if __name__ == '__main__':
    # 测试代码
    print("测试 FLAT 数据处理器")
    
    # 创建测试词表
    test_words = ['北京', '上海', '中国', '红枣', '新疆', '南京', '南京市']
    
    # 创建处理器
    processor = FLATDataProcessor(test_words)
    
    # 测试句子
    test_chars = list('我爱北京天安门')
    result = processor.process_sentence(test_chars, labels=['O']*len(test_chars))
    
    print(f"\n输入: {''.join(test_chars)}")
    print(f"Lattice: {result['lattice']}")
    print(f"pos_s: {result['pos_s']}")
    print(f"pos_e: {result['pos_e']}")
    print(f"seq_len: {result['seq_len']}")
    print(f"lex_num: {result['lex_num']}")
    print(f"lexicons: {result['lexicons']}")
