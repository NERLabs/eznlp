# -*- coding: utf-8 -*-
"""
HZ 数据集 NER 训练示例 - 使用专家词典特征

这个脚本展示了如何为 HZ 数据集添加专家词典特征来增强 NER 性能
"""
import torch
from eznlp.io import ConllIO
from eznlp.model import (
    BertLikeConfig,
    ExpertDictConfig,
    EncoderConfig,
    ExtractorConfig,
)
from eznlp.model.decoder import SequenceTaggingDecoderConfig
from eznlp.token import LexiconTokenizer
from eznlp.dataset import Dataset
from eznlp.training import Trainer


def load_expert_lexicon(dict_path):
    """加载专家词典
    
    Args:
        dict_path: 词典文件路径，每行一个词（或 "词\t类型" 格式）
    
    Returns:
        lexicon: 词典列表
    """
    lexicon = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            word = parts[0] if parts else line.strip()
            if word:
                lexicon.append(word)
    return lexicon


def preprocess_data_with_expert_dict(data, expert_tokenizer):
    """为数据添加专家词典匹配特征
    
    Args:
        data: 数据列表，每个元素包含 'tokens' 字段
        expert_tokenizer: LexiconTokenizer 实例
    """
    for entry in data:
        entry["tokens"].build_expert_dict_tags(expert_tokenizer.tokenize)
    return data


def main():
    # 1. 加载专家词典
    print("加载专家词典...")
    expert_lexicon = load_expert_lexicon("data/HZ/expert_lexicon.txt")
    expert_tokenizer = LexiconTokenizer(expert_lexicon, max_len=10)
    print(f"词典大小: {len(expert_lexicon)} 个词")
    
    # 2. 读取数据
    print("\n读取 HZ 数据集...")
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        tokenize_callback="char",  # 字符级
        token_sep="",
        pad_token="<pad>"
    )
    
    train_data = io.read("data/HZ/train.bmes")
    dev_data = io.read("data/HZ/dev.bmes")
    test_data = io.read("data/HZ/test.bmes")
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(dev_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
    # 3. 为数据添加专家词典匹配
    print("\n添加专家词典匹配特征...")
    train_data = preprocess_data_with_expert_dict(train_data, expert_tokenizer)
    dev_data = preprocess_data_with_expert_dict(dev_data, expert_tokenizer)
    test_data = preprocess_data_with_expert_dict(test_data, expert_tokenizer)
    
    # 4. 构建模型配置
    print("\n构建模型配置...")
    
    # BERT 配置
    bert_config = BertLikeConfig(
        bert_arch="hfl/chinese-macbert-base",  # 中文 MacBERT
        bert_max_length=512,
        bert_drop_rate=0.2,
        freeze=False,
        mix_layers="top"
    )
    
    # 专家词典特征配置
    expert_dict_config = ExpertDictConfig(
        emb_dim=50,              # 词典特征嵌入维度
        agg_mode="wtd_mean_pooling"  # 加权平均池化
    )
    
    # 编码器配置
    encoder_config = EncoderConfig(
        arch="LSTM",
        hid_dim=256,
        num_layers=1,
        in_drop_rates=(0.5, 0.0, 0.0)
    )
    
    # 解码器配置
    decoder_config = SequenceTaggingDecoderConfig(
        scheme="BMES",
        in_drop_rates=(0.5,)
    )
    
    # 整体模型配置
    config = ExtractorConfig(
        bert_like=bert_config,
        nested_ohots={
            "expert_dict": expert_dict_config  # 🔴 关键：添加专家词典特征
        },
        encoder=encoder_config,
        decoder=decoder_config
    )
    
    # 5. 构建数据集和词表
    print("\n构建数据集和词表...")
    train_set = Dataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims(dev_data)
    
    # 为 ExpertDictConfig 构建词频统计
    if hasattr(config.nested_ohots["expert_dict"], "build_freqs"):
        config.nested_ohots["expert_dict"].build_freqs(train_data, dev_data)
    
    dev_set = Dataset(dev_data, config, training=False)
    test_set = Dataset(test_data, config, training=False)
    
    print(train_set.summary)
    
    # 6. 实例化模型
    print("\n实例化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.instantiate().to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 7. 训练配置
    print("\n配置训练器...")
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        collate_fn=train_set.collate
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        batch_size=16,
        shuffle=False,
        collate_fn=dev_set.collate
    )
    
    # 你可以在这里添加 Trainer 和训练循环
    # trainer = Trainer(model, device, ...)
    # trainer.train(train_loader, dev_loader, ...)
    
    print("\n✅ 配置完成！模型已准备好训练。")
    print("\n💡 提示：")
    print("  - 专家词典特征维度: 50")
    print("  - 总特征维度: BERT(768) + ExpertDict(50*4) = 968")
    print("  - 你可以通过调整 ExpertDictConfig 的参数来优化性能")


if __name__ == "__main__":
    main()
