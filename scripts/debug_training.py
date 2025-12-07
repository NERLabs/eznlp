#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试训练问题"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import transformers
from eznlp.io import ConllIO
from eznlp.dataset import Dataset
from eznlp.model.model import ExtractorConfig
from eznlp.model.bert_like import BertLikeConfig
from eznlp.model.encoder import EncoderConfig
from eznlp.model.decoder import SequenceTaggingDecoderConfig

# 1. 加载数据
print("="*70)
print("1. 加载数据")
print("="*70)
io = ConllIO(
    text_col_id=0,
    tag_col_id=1,
    scheme="BMES",
    encoding="utf-8",
    token_sep="",
    pad_token="<pad>"
)

train_data = io.read("data/HZ/hz_train.bmes")
print(f"训练集: {len(train_data)} 条")
print(f"\n第1个样本:")
print(f"  Tokens[:10]: {train_data[0]['tokens'].raw_text[:10]}")
print(f"  Chunks: {train_data[0]['chunks'][:5]}")

# 2. 构建模型配置
print(f"\n{'='*70}")
print("2. 构建模型配置")
print("="*70)

# 加载 BERT
bert_model = transformers.AutoModel.from_pretrained("hfl/chinese-macbert-base")
tokenizer = transformers.AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

bert_config = BertLikeConfig(
    tokenizer=tokenizer,
    bert_like=bert_model,
    freeze=False,
    mix_layers="top"
)

encoder_config = EncoderConfig(
    arch="LSTM",
    hid_dim=256,
    num_layers=1,
    in_drop_rates=(0.5, 0.0, 0.0)
)

decoder_config = SequenceTaggingDecoderConfig(
    scheme="BMES",
    use_crf=True,
    in_drop_rates=(0.5,)
)

config = ExtractorConfig(
    bert_like=bert_config,
    encoder=encoder_config,
    decoder=decoder_config
)

print("模型配置创建成功")

# 3. 构建数据集
print(f"\n{'='*70}")
print("3. 构建数据集")
print("="*70)

train_set = Dataset(train_data[:100], config, training=True)  # 只用100条测试
train_set.build_vocabs_and_dims()
print(train_set.summary)

# 打印标签词表
print(f"\n标签词表 (idx2tag):")
if hasattr(config.decoder, 'idx2tag'):
    print(f"  总共 {len(config.decoder.idx2tag)} 个标签")
    print(f"  idx2tag: {config.decoder.idx2tag}")
    print(f"  tag2idx: {config.decoder.tag2idx}")
else:
    print(f"  ❌ decoder 没有 idx2tag 属性！")

# 4. 创建数据加载器
print(f"\n{'='*70}")
print("4. 测试数据批次")
print("="*70)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=False,
    collate_fn=train_set.collate
)

# 获取第一个 batch
batch = next(iter(train_loader))
print(f"Batch 类型: {type(batch)}")
print(f"Batch 属性: {[k for k in dir(batch) if not k.startswith('_')]}")
print(f"Batch.seq_lens: {batch.seq_lens}")
print(f"Batch.tags_objs: {len(batch.tags_objs)} 个")
print(f"\n第一个 tags_obj:")
if hasattr(batch.tags_objs[0], 'chunks'):
    print(f"  chunks: {batch.tags_objs[0].chunks[:3]}")
if hasattr(batch.tags_objs[0], 'tag_ids'):
    print(f"  tag_ids shape: {batch.tags_objs[0].tag_ids.shape}")
    print(f"  tag_ids[:20]: {batch.tags_objs[0].tag_ids[:20]}")
else:
    print(f"  ❌ 没有 tag_ids 字段！")

# 5. 实例化模型并测试前向传播
print(f"\n{'='*70}")
print("5. 测试模型前向传播")
print("="*70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

model = config.instantiate().to(device)
batch = batch.to(device)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 测试前向传播
model.eval()
with torch.no_grad():
    try:
        losses, states = model(batch, return_states=True)
        print(f"\n✅ 前向传播成功!")
        print(f"Losses shape: {losses.shape}")
        print(f"Losses[:4]: {losses[:4]}")
        print(f"Loss mean: {losses.mean().item():.6f}")
        
        if losses.mean().item() < 0.001:
            print(f"\n⚠️  警告: Loss 接近 0，可能有问题！")
            
    except Exception as e:
        print(f"\n❌ 前向传播失败!")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

# 6. 测试训练一步
print(f"\n{'='*70}")
print("6. 测试训练一步")
print("="*70)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

try:
    losses, states = model(batch, return_states=True)
    loss = losses.mean()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    
    if loss.item() < 0.001:
        print(f"\n⚠️  Loss 太小了！检查:")
        print(f"  - batch.tags_objs 数量: {len(batch.tags_objs)}")
        if hasattr(batch.tags_objs[0], 'tag_ids'):
            print(f"  - 第1个样本 tag_ids 长度: {len(batch.tags_objs[0].tag_ids)}")
            print(f"  - 第1个样本 tag_ids[:20]: {batch.tags_objs[0].tag_ids[:20]}")
        else:
            print(f"  - ❌ tag_ids 不存在！")
    
    loss.backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    if has_grad:
        print(f"✅ 梯度计算成功")
    else:
        print(f"❌ 没有梯度！")
    
    optimizer.step()
    print(f"✅ 优化器更新成功")
    
except Exception as e:
    print(f"❌ 训练步骤失败!")
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
print("调试完成")
print("="*70)
