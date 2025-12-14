# FLAT 模型构建器使用指南

## 📋 概述

本文档介绍如何使用 FLAT (Flat-Lattice Transformer) 模型构建器在 eznlp 框架中进行中文 NER 任务。

FLAT 模型是由 Li et al. (ACL 2020) 提出的基于 Lattice 结构的 Transformer 模型，核心创新包括：
- **Lattice 结构**: 将字符和词汇扁平化表示，避免传统 Lattice LSTM 的复杂性
- **四位置编码**: SS/SE/ES/EE 四种相对位置的融合，精确建模字符与词汇的位置关系
- **全连接注意力**: 字-字、字-词、词-词的全局交互

## 🏗️ 模型架构

### 核心组件

```
输入层
  ├── 字符嵌入 (Char Embedding)
  ├── 词汇嵌入 (Word Embedding)  
  └── Bigram 嵌入 (可选)
        ↓
四位置融合编码
  ├── SS: Start-Start 位置
  ├── SE: Start-End 位置
  ├── ES: End-Start 位置
  └── EE: End-End 位置
        ↓
Lattice Self-Attention × N 层
  ├── 多头注意力机制
  ├── 层归一化 (LayerNorm)
  ├── 位置感知前馈网络 (FFN)
  └── 残差连接 (Residual)
        ↓
CRF 解码层
        ↓
输出 (NER 标签序列)
```

### 与原始 FLAT 论文的对应关系

| 组件 | 论文描述 | 实现模块 |
|------|---------|---------|
| Lattice 输入 | 字符 + 匹配词汇 | 数据预处理 |
| 四位置编码 | Relative Position | `FourPositionFusion` |
| Lattice Attention | Self-Attention with PE | `LatticeSelfAttention` |
| Transformer Layer | Multi-head + FFN | `TransformerEncoderLayer` |
| 层处理序列 | Pre/Post Process | `LayerProcess` |

## 🚀 快速开始

### 1. 基本使用

```python
import argparse
from _4MODELS.models.flat_model_builder import FLATModelFactory

# 创建参数对象
args = argparse.Namespace(
    # 模型结构参数
    hidden_size=512,          # 隐藏层维度
    num_layers=4,             # Transformer 层数
    num_heads=8,              # 多头注意力头数
    ff_size=2048,             # 前馈网络维度
    max_seq_len=512,          # 最大序列长度
    
    # Dropout 参数
    dropout=0.15,             # 主 Dropout
    embed_dropout=0.5,        # 嵌入层 Dropout
    attn_dropout=0.0,         # 注意力 Dropout
    
    # 位置编码参数
    four_pos_fusion='ff',     # 四位置融合方式: 'ff'/'attn'/'gate'
    learnable_position=False, # 位置编码是否可学习
    
    # 层处理参数
    layer_preprocess='n',     # 层前处理: 'n' (LayerNorm)
    layer_postprocess='dan',  # 层后处理: 'dan' (Dropout->Add->LayerNorm)
)

# 创建 FLAT Baseline 模型
model_config = FLATModelFactory.create_model_config('flat_baseline', args)
```

### 2. 使用配置文件

```bash
# 使用提供的配置文件启动训练
python _5TRAIN/train_flat_ner.py \
    --config _1CONFIG/redjujube/flat_redjujube_config.json \
    --gpu 0
```

### 3. 三种模型变体

#### (1) FLAT Baseline
```python
# 不使用 BERT，仅使用字符和词汇嵌入
model_config = FLATModelFactory.create_model_config('flat_baseline', args)
```

#### (2) FLAT + BERT
```python
# 在 FLAT 基础上加入 BERT 预训练特征
args.bert_arch = 'hfl/chinese-macbert-base'
args.freeze_bert = False
model_config = FLATModelFactory.create_model_config('flat_bert', args)
```

#### (3) FLAT + SoftLexicon
```python
# 在 FLAT 基础上加入软词典特征
from eznlp import Vectors
vectors = Vectors.from_file('path/to/word_vectors.txt')
model_config = FLATModelFactory.create_model_config('flat_lexicon', args, vectors=vectors)
```

## ⚙️ 配置参数详解

### 模型结构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 512 | 隐藏层维度，必须能被 num_heads 整除 |
| `num_layers` | 4 | Transformer 编码层数 |
| `num_heads` | 8 | 多头注意力头数 |
| `ff_size` | 2048 | 前馈网络隐藏层维度，通常为 hidden_size 的 4 倍 |
| `max_seq_len` | 512 | 最大序列长度 |

### Dropout 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dropout` | 0.15 | 主 Dropout 率，应用于层处理和前馈网络 |
| `embed_dropout` | 0.5 | 嵌入层 Dropout 率 |
| `attn_dropout` | 0.0 | 注意力权重 Dropout 率 |

### 位置编码参数

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `four_pos_fusion` | `'ff'`, `'attn'`, `'gate'` | 四位置融合方式 |
| `learnable_position` | `True`, `False` | 位置编码是否可学习 |

**融合方式说明**:
- `'ff'`: 前馈网络融合（推荐，速度快）
- `'attn'`: 注意力融合（效果好，但慢）
- `'gate'`: 门控融合（平衡方案）

### 层处理参数

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `layer_preprocess` | `'n'` | 层前处理序列 |
| `layer_postprocess` | `'dan'` | 层后处理序列 |

**处理序列符号**:
- `'a'`: Add (残差连接)
- `'d'`: Dropout
- `'n'`: LayerNorm

**常用组合**:
- 前处理: `'n'` (仅 LayerNorm)
- 后处理: `'dan'` (Dropout → Add → LayerNorm)

## 📊 训练配置示例

完整的训练配置文件位于 `_1CONFIG/redjujube/flat_redjujube_config.json`

```json
{
  "model_config": {
    "hidden_size": 512,
    "num_layers": 4,
    "num_heads": 8,
    "ff_size": 2048,
    "dropout": 0.15,
    "four_pos_fusion": "ff",
    "learnable_position": false
  },
  
  "training_config": {
    "num_epochs": 100,
    "batch_size": 10,
    "learning_rate": 0.0006,
    "optimizer": "sgd",
    "momentum": 0.9,
    "grad_clip": 5.0,
    "early_stop_patience": 25
  },
  
  "embedding_config": {
    "char_emb_dim": 50,
    "word_emb_dim": 50,
    "use_bigram": true,
    "pretrained_word_path": "_2DATA/embeddings/gigaword_chn.all.a2b.uni.ite50.vec"
  }
}
```

## 🔧 高级使用

### 自定义 FLAT 编码器

```python
from _4MODELS.models.flat_model_builder import FLATEncoder

# 创建自定义编码器
encoder = FLATEncoder(
    hidden_size=768,          # 更大的隐藏层
    num_layers=6,             # 更深的网络
    num_heads=12,             # 更多注意力头
    ff_size=3072,
    dropout=0.1,
    four_pos_fusion='attn',   # 使用注意力融合
    learnable_position=True,  # 可学习位置编码
)

# 前向传播
output = encoder(
    embedded=embedded_input,  # [batch, seq_len+lex_num, hidden_size]
    pos_s=start_positions,    # [batch, seq_len+lex_num]
    pos_e=end_positions,      # [batch, seq_len+lex_num]
    seq_len=char_lengths,     # [batch]
    lex_num=word_counts,      # [batch]
    mask=attention_mask       # [batch, seq_len+lex_num]
)
```

### 集成到现有 Pipeline

```python
# 在训练脚本中使用
from _4MODELS.models import FLATModelFactory

def build_model(args):
    # 根据参数创建模型
    model_config = FLATModelFactory.create_model_config(
        model_type=args.model_type,
        args=args,
        vectors=load_vectors() if args.use_lexicon else None
    )
    
    # 实例化模型
    model = initialize_model(model_config)
    return model
```

## 📈 性能优化建议

### 1. 内存优化
- 降低 `batch_size` (默认 10)
- 减少 `max_seq_len` (默认 256)
- 使用 `four_pos_fusion='ff'` (最快)

### 2. 精度优化
- 增加 `num_layers` (4 → 6)
- 使用 `four_pos_fusion='attn'` (最准确)
- 设置 `learnable_position=True`

### 3. 速度优化
- 使用 `four_pos_fusion='ff'`
- 降低 `num_heads` (8 → 4)
- 减少 `ff_size` (2048 → 1024)

## 🔍 与其他模型的对比

| 模型 | 架构 | 词汇交互 | 复杂度 | 适用场景 |
|------|------|----------|--------|----------|
| **FLAT** | Transformer | 全连接注意力 | O((n+m)²) | 高精度 NER |
| BiLSTM-CRF | RNN | 无 | O(n) | 简单快速 |
| SoftLexicon | 特征拼接 | 加权平均 | O(n) | 轻量级词汇增强 |
| Lattice LSTM | LSTM | 动态路径 | O(n·m) | 内存受限场景 |

## 📚 参考文献

- Li et al. (2020). "FLAT: Chinese NER Using Flat-Lattice Transformer". ACL 2020.
- Zhang & Yang (2018). "Chinese NER Using Lattice LSTM". ACL 2018.

## 🐛 常见问题

### Q1: `hidden_size` 必须能被 `num_heads` 整除

```python
# ❌ 错误
hidden_size = 500
num_heads = 8  # 500 / 8 = 62.5 (不能整除)

# ✅ 正确
hidden_size = 512
num_heads = 8  # 512 / 8 = 64
```

### Q2: 显存不足

```python
# 减小批次大小
batch_size = 10 → 5

# 减小序列长度
max_seq_len = 512 → 256

# 减小模型规模
hidden_size = 512 → 256
num_layers = 4 → 2
```

### Q3: 训练速度慢

```python
# 使用更快的融合方式
four_pos_fusion = 'attn' → 'ff'

# 减少注意力头数
num_heads = 12 → 8

# 减少前馈网络维度
ff_size = 3072 → 2048
```

## 📝 TODO

- [ ] 添加混合精度训练支持 (AMP)
- [ ] 支持分布式训练 (DDP)
- [ ] 添加模型蒸馏接口
- [ ] 优化 Lattice 构建效率
- [ ] 支持动态序列长度

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**最后更新**: 2025-12-14
**维护者**: eznlp 项目组
