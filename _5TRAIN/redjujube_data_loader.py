#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube 数据加载器

功能：
- 封装数据加载和预处理逻辑
- 支持专家词典和软词典特征构建
- 提供统一的数据加载接口
"""

from eznlp.io import ConllIO
from eznlp.token import LexiconTokenizer


class RedJujubeDataLoader:
    """RedJujube 数据加载器"""
    
    def __init__(self, data_dir):
        """初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.io = ConllIO(
            text_col_id=0,
            tag_col_id=1,
            scheme="BMES",
            encoding="utf-8",
            token_sep="",
            pad_token="<pad>"
        )
    
    def load_data(self):
        """加载 RedJujube 数据集
        
        Returns:
            tuple: (train_data, dev_data, test_data)
        """
        train_data = self.io.read(f"{self.data_dir}/redjujube_train.bmes")
        dev_data = self.io.read(f"{self.data_dir}/redjujube_dev.bmes")
        test_data = self.io.read(f"{self.data_dir}/redjujube_test.bmes")
        
        return train_data, dev_data, test_data
    
    @staticmethod
    def load_lexicon(dict_path):
        """加载词典文件
        
        Args:
            dict_path: 词典文件路径
            
        Returns:
            list: 词典列表
        """
        lexicon = []
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    lexicon.append(word)
        return lexicon
    
    @staticmethod
    def add_expert_dict_features(data_partitions, expert_lexicon, max_len=10):
        """为数据添加专家词典匹配特征
        
        Args:
            data_partitions: 数据分区列表 (train_data, dev_data, test_data)
            expert_lexicon: 专家词典列表
            max_len: 最大词长
        """
        tokenizer = LexiconTokenizer(expert_lexicon, max_len=max_len)
        
        for partition in data_partitions:
            for entry in partition:
                entry["tokens"].build_expert_dict_tags(tokenizer.tokenize)
    
    @staticmethod
    def add_softlexicon_features(data_partitions, lexicon, max_len=10):
        """为数据添加 SoftLexicon 特征
        
        Args:
            data_partitions: 数据分区列表 (train_data, dev_data, test_data)
            lexicon: 词表列表（可以是 vectors.itos 或自定义词表）
            max_len: 最大词长
        """
        tokenizer = LexiconTokenizer(lexicon, max_len=max_len)
        
        for partition in data_partitions:
            for entry in partition:
                entry["tokens"].build_softwords(tokenizer.tokenize)
                entry["tokens"].build_softlexicons(tokenizer.tokenize)


class DataPreparationPipeline:
    """数据准备流水线
    
    根据模型类型自动准备所需的数据特征
    """
    
    def __init__(self, data_loader):
        """初始化数据准备流水线
        
        Args:
            data_loader: RedJujubeDataLoader 实例
        """
        self.data_loader = data_loader
    
    def prepare_for_baseline(self, data_partitions):
        """为 Baseline 模型准备数据
        
        Args:
            data_partitions: 数据分区列表
            
        Returns:
            原数据（无额外特征）
        """
        # Baseline 不需要额外特征
        return data_partitions
    
    def prepare_for_expert_dict(self, data_partitions, expert_dict_path):
        """为 ExpertDict 模型准备数据
        
        Args:
            data_partitions: 数据分区列表
            expert_dict_path: 专家词典路径
            
        Returns:
            添加了专家词典特征的数据
        """
        expert_lexicon = self.data_loader.load_lexicon(expert_dict_path)
        print(f"专家词典大小: {len(expert_lexicon)} 个词")
        
        self.data_loader.add_expert_dict_features(data_partitions, expert_lexicon)
        return data_partitions
    
    def prepare_for_softlexicon(self, data_partitions, softlex_lexicon):
        """为 SoftLexicon 模型准备数据
        
        Args:
            data_partitions: 数据分区列表
            softlex_lexicon: 软词典词表（可以是 vectors.itos 或自定义词表）
            
        Returns:
            添加了 SoftLexicon 特征的数据
        """
        print(f"软词典大小: {len(softlex_lexicon):,} 个词")
        
        self.data_loader.add_softlexicon_features(data_partitions, softlex_lexicon)
        return data_partitions
    
    def prepare_for_fusion(self, data_partitions, expert_dict_path, softlex_lexicon):
        """为融合模型准备数据
        
        Args:
            data_partitions: 数据分区列表
            expert_dict_path: 专家词典路径
            softlex_lexicon: 软词典词表
            
        Returns:
            同时添加了专家词典和 SoftLexicon 特征的数据
        """
        # 加载专家词典
        expert_lexicon = self.data_loader.load_lexicon(expert_dict_path)
        print(f"专家词典大小: {len(expert_lexicon)} 个词")
        print(f"软词典大小: {len(softlex_lexicon):,} 个词")
        
        # 添加两种特征
        self.data_loader.add_expert_dict_features(data_partitions, expert_lexicon)
        self.data_loader.add_softlexicon_features(data_partitions, softlex_lexicon)
        
        return data_partitions
    
    def prepare(self, model_type, expert_dict_path=None, softlex_lexicon=None):
        """根据模型类型自动准备数据
        
        Args:
            model_type: 模型类型
            expert_dict_path: 专家词典路径（可选）
            softlex_lexicon: 软词典词表（可选）
            
        Returns:
            tuple: (train_data, dev_data, test_data)
        """
        # 加载原始数据
        data_partitions = list(self.data_loader.load_data())
        
        # 根据模型类型准备特征
        if model_type == 'baseline':
            return self.prepare_for_baseline(data_partitions)
        
        elif model_type in ['expert_dict', 'expert_dict_auto', 'expert_dict_manual']:
            if expert_dict_path is None:
                raise ValueError(f"模型类型 {model_type} 需要提供 expert_dict_path")
            return self.prepare_for_expert_dict(data_partitions, expert_dict_path)
        
        elif model_type in ['softlexicon', 'softlexicon_trainlex']:
            if softlex_lexicon is None:
                raise ValueError(f"模型类型 {model_type} 需要提供 softlex_lexicon")
            return self.prepare_for_softlexicon(data_partitions, softlex_lexicon)
        
        elif model_type.startswith('fusion_'):
            if expert_dict_path is None or softlex_lexicon is None:
                raise ValueError(f"融合模型需要同时提供 expert_dict_path 和 softlex_lexicon")
            return self.prepare_for_fusion(data_partitions, expert_dict_path, softlex_lexicon)
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
