# -*- coding: utf-8 -*-
import torch
from eznlp.dataset import Dataset
from eznlp.training import Trainer, evaluate_entity_recognition
from eznlp.io import ConllIO

def load_msra_data():
    """加载MSRA数据集"""
    conll_io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="",
    )
    
    # 加载数据
    train_data = conll_io.read("data/MSRA/hz_train.bmes")
    dev_data = conll_io.read("data/MSRA/hz_dev.bmes")
    test_data = conll_io.read("data/MSRA/hz_test.bmes")
    
    return train_data, dev_data, test_data

def evaluate_model():
    """使用训练好的模型评估MSRA数据集"""
    print("使用CPU设备进行评估...")
    device = torch.device("cpu")
    
    # 加载数据
    print("加载MSRA数据集...")
    train_data, dev_data, test_data = load_msra_data()
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(dev_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 加载模型配置和模型
    print("加载训练好的模型...")
    config = torch.load(
        "cache/MSRA-ER/20250822-195241-646229/text-softlexicon-MacBERT_base-LSTM-BIOES-CRF-config.pth", 
        map_location='cpu',
        weights_only=False
    )
    model = torch.load(
        "cache/MSRA-ER/20250822-195241-646229/text-softlexicon-MacBERT_base-LSTM-BIOES-CRF.pth", 
        map_location='cpu',
        weights_only=False
    )
    
    # 禁用SoftLexicon以避免索引错误
    print("禁用SoftLexicon以避免索引错误...")
    if hasattr(config, 'nested_ohots'):
        config.nested_ohots = None
    if hasattr(model, 'nested_ohots'):
        model.nested_ohots = None
    
    # 将模型移动到CPU设备
    model = model.to(device)
    
    # 构建数据集（不使用SoftLexicon）
    print("构建数据集...")
    train_set = Dataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims(dev_data)
    
    # 构建测试集
    test_set = Dataset(test_data, train_set.config, training=False)
    
    # 创建训练器（使用CPU）
    trainer = Trainer(model, device=device)
    
    # 在测试集上评估
    print("在测试集上评估模型...")
    print("=" * 50)
    evaluate_entity_recognition(
        trainer,
        test_set,
        batch_size=16,
        save_preds=False
    )

if __name__ == "__main__":
    evaluate_model()
