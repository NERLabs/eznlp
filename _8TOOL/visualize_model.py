#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成模型结构可视化图
"""
import os
import sys
import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchviz import make_dot
from eznlp.io import ConllIO
from eznlp.dataset import Dataset
from eznlp.model import (
    OneHotConfig,
    EncoderConfig,
    BertLikeConfig,
)
from eznlp.model.model import ExtractorConfig
from eznlp.model.decoder import BoundarySelectionDecoderConfig


def visualize_model(data_dir: str, output_path: str, bert_path: str):
    """从头构建模型并生成计算图"""
    
    # 加载数据
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="<pad>",
    )
    train_data = io.read(os.path.join(data_dir, "redjujube_train.bmes"))[:10]
    
    print(f"加载 BERT: {bert_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
    bert_model = transformers.AutoModel.from_pretrained(bert_path, local_files_only=True)
    
    # 构建配置（与训练脚本一致）
    config = ExtractorConfig(
        ohots={"text": OneHotConfig(field="text", min_freq=1)},
        bert_like=BertLikeConfig(
            tokenizer=tokenizer,
            bert_like=bert_model,
            freeze=False,
            mix_layers="top",
            bert_max_length=512,
            truncation=True,
        ),
        intermediate2=EncoderConfig(arch="FFN"),
        decoder=BoundarySelectionDecoderConfig(
            sb_epsilon=0.1,
            sb_size=2,
        ),
    )
    
    # 构建词表和维度
    print("构建词表...")
    config.build_vocabs_and_dims(train_data)
    
    # 构建数据集
    dataset = Dataset(train_data, config, training=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate
    )
    
    # 实例化模型
    print("实例化模型...")
    model = config.instantiate()
    model.eval()
    
    # 获取示例 batch
    batch = next(iter(loader))
    
    # 前向传播（需要启用梯度才能生成计算图）
    print("前向传播...")
    losses, states = model(batch, return_states=True)
    
    # 生成计算图
    print("生成计算图...")
    dot = make_dot(losses, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    
    # 保存
    dot.render(output_path, format="pdf", cleanup=True)
    print(f"✅ 模型图已保存: {output_path}.pdf")
    
    # 同时打印模型结构
    print("\n" + "=" * 70)
    print("模型结构:")
    print("=" * 70)
    print(model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="可视化模型结构")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output", type=str, default="model_graph", help="输出文件名")
    parser.add_argument("--bert", type=str, default="assets/transformers/hfl/chinese-macbert-base",
                        help="BERT 模型路径")
    args = parser.parse_args()
    
    visualize_model(args.data_dir, args.output, args.bert)