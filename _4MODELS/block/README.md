# 模型通用组件模块

本目录包含从 FLAT/LatticeLSTM 等项目中提取并重新实现的核心组件，符合 eznlp 框架规范。

## 📂 目录结构

```
block/
├── encoder_builder.py       # 编码器配置构建器
├── decoder_builder.py       # 解码器配置构建器  
├── embedding_builder.py     # 嵌入层配置构建器
├── position_encoding.py     # 位置编码模块 ⭐
├── lattice_modules.py       # Lattice 结构模块 ⭐
├── lattice_attention.py     # Lattice 注意力机制 ⭐⭐ 新增核心
└── __init__.py
```

## 🎯 新增组件

### 1. 位置编码模块 (`position_encoding.py`)

从 **FLAT (Flat-Lattice-Transformer)** 项目提取的核心位置编码组件。

#### 1.1 SinusoidalPositionEncoding
标准 Transformer 正弦位置编码

```python
from _4MODELS.block import SinusoidalPositionEncoding

# 创建位置编码
pos_enc = SinusoidalPositionEncoding(max_len=512, d_model=768)

# 使用
positions = torch.arange(0, seq_len).unsqueeze(0)  # [1, seq_len]
pos_embeddings = pos_enc(positions)  # [1, seq_len, 768]
```

#### 1.2 RelativePositionEncoding
相对位置编码，支持 Transformer 中的相对位置建模

```python
from _4MODELS.block import RelativePositionEncoding

# 创建相对位置编码
rel_pos_enc = RelativePositionEncoding(
    max_len=256, 
    d_model=768,
    learnable=True,      # 可学习
    symmetric=True       # 对称初始化 (-max_len ~ max_len)
)

# 使用
rel_positions = pos_i.unsqueeze(-1) - pos_j.unsqueeze(-2)  # [batch, seq, seq]
rel_pos_embeddings = rel_pos_enc(rel_positions)  # [batch, seq, seq, 768]
```

#### 1.3 FourPositionFusion ⭐
**FLAT 核心组件**：四位置融合，用于 Lattice 结构

```python
from _4MODELS.block import FourPositionFusion

# 创建四位置融合模块
four_pos = FourPositionFusion(
    hidden_size=768,
    max_len=512,
    fusion_mode='ff',    # 'ff', 'attn', 'gate', 'concat'
    learnable=True,
    shared=True          # 四个位置编码共享参数
)

# 使用（Lattice 结构需要起始和结束位置）
pos_s = ...  # [batch, seq_len] 开始位置
pos_e = ...  # [batch, seq_len] 结束位置
fused_pos = four_pos(pos_s, pos_e)  # [batch, seq_len, seq_len, 768]
```

**四种相对位置**：
- **SS** (Start-Start): 开始位置到开始位置
- **SE** (Start-End): 开始位置到结束位置
- **ES** (End-Start): 结束位置到开始位置
- **EE** (End-End): 结束位置到结束位置

**融合模式**：
- `ff`: 前馈网络融合（默认，性能最好）
- `attn`: 注意力加权融合
- `gate`: 门控融合
- `concat`: 直接拼接

### 2. Lattice 模块 (`lattice_modules.py`)

从 **LatticeLSTM** 项目提取的核心组件。

#### 2.1 MultiInputLSTMCell ⭐
**LatticeLSTM 核心**：支持多输入的 LSTM Cell

```python
from _4MODELS.block import MultiInputLSTMCell

# 创建多输入 LSTM Cell
cell = MultiInputLSTMCell(input_size=300, hidden_size=256)

# 使用
input_ = char_embedding  # [1, 300] 字符嵌入
c_input = [word_c1, word_c2]  # List of [1, 256] 词汇 cell 状态
hx = (h_0, c_0)  # 前一时刻状态

h_1, c_1 = cell(input_, c_input, hx)
```

**特性**：
- 同时处理字符序列和匹配词汇
- 使用 Alpha 门控融合词汇信息
- 正交初始化保证训练稳定性

#### 2.2 LatticeLSTMEncoder
Lattice LSTM 编码器（接口框架）

```python
from _4MODELS.block import LatticeLSTMEncoder

encoder = LatticeLSTMEncoder(
    input_size=300,
    hidden_size=256,
    num_layers=2,
    bidirectional=True,
    dropout=0.5
)
```

**注意**：完整实现需要根据具体的 lattice 结构调整。

### 3. Lattice 注意力机制 (`lattice_attention.py`) ⭐⭐

**FLAT 核心创新**：从项目中提取的完整注意力实现。

#### 3.1 MultiHeadAttentionWithRelativePosition ⭐⭐
**Transformer-XL 风格的相对位置注意力**

```python
from _4MODELS.block import MultiHeadAttentionWithRelativePosition

# 创建带相对位置的注意力
attn = MultiHeadAttentionWithRelativePosition(
    hidden_size=768,
    num_heads=12,
    dropout=0.1,
    scaled=True,
    use_projection=True
)

# 使用
output = attn(
    query=hidden_states,
    key=hidden_states,
    value=hidden_states,
    rel_pos_embedding=rel_pos,  # [batch, seq, seq, hidden]
    mask=attention_mask
)
```

**核心创新**：四分量注意力计算
```
Attention = Softmax((A + B + C + D) / √d) V

A = Q @ K^T           # 内容-内容
B = Q @ R^T           # 内容-位置
C = u @ K^T           # 全局内容偏置
D = v @ R^T           # 全局位置偏置
```

其中 u 和 v 是可学习的全局参数，R 是相对位置编码。

#### 3.2 LatticeSelfAttention ⭐⭐
**完整的 Lattice 自注意力模块**

```python
from _4MODELS.block import LatticeSelfAttention

# 创建 Lattice 自注意力
lattice_attn = LatticeSelfAttention(
    hidden_size=768,
    num_heads=12,
    max_len=512,
    dropout=0.1,
    four_pos_fusion_mode='ff'  # 'ff', 'attn', 'gate'
)

# 使用（自动处理四位置融合）
output = lattice_attn(
    hidden_states=hidden_states,
    pos_s=start_positions,  # [batch, seq_len]
    pos_e=end_positions,    # [batch, seq_len]
    mask=attention_mask
)
```

**特性**：
- 自动集成四位置融合
- 端到端的 Lattice 注意力
- 即插即用，无需手动处理位置

#### 3.3 TransformerEncoderLayer ⭐
**FLAT 风格的 Transformer 层**

```python
from _4MODELS.block import TransformerEncoderLayer

# 创建编码器层
encoder_layer = TransformerEncoderLayer(
    hidden_size=768,
    num_heads=12,
    max_len=512,
    ff_size=3072,
    dropout=0.1,
    activation='relu'
)

# 使用
output = encoder_layer(
    hidden_states=hidden_states,
    pos_s=start_positions,
    pos_e=end_positions,
    mask=attention_mask
)
```

**架构**：
```
Input
  |
  +-> LatticeSelfAttention -> LayerNorm -> Residual
       |
       +-> FeedForward -> LayerNorm -> Residual -> Output
```

## 🔧 与现有组件的关系

### 组合使用示例

```python
from _4MODELS.block import (
    EncoderBuilder,
    DecoderBuilder,
    EmbeddingBuilder,
    FourPositionFusion,
    MultiInputLSTMCell
)

# 构建基础组件
embedding_builder = EmbeddingBuilder()
encoder_builder = EncoderBuilder()

# 添加 Lattice 特性
four_pos_fusion = FourPositionFusion(hidden_size=768, max_len=512)

# 在模型中使用
class MyLatticeModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = embedding_builder.build_from_args(args)
        self.encoder = encoder_builder.build_from_args(args)
        self.four_pos = four_pos_fusion
        # ...
```

## 📊 组件来源映射

| eznlp 组件 | 原始项目 | 原始文件 | 说明 |
|-----------|---------|---------|------|
| `SinusoidalPositionEncoding` | FLAT | `models.py::get_embedding` | 标准位置编码 |
| `RelativePositionEncoding` | FLAT | `models.py::get_embedding` | 相对位置编码 |
| `FourPositionFusion` | FLAT | `modules.py::Four_Pos_Fusion_Embedding` | 四位置融合 |
| `MultiInputLSTMCell` | LatticeLSTM | `latticelstm.py::MultiInputLSTMCell` | 多输入LSTM |
| `MultiHeadAttentionWithRelativePosition` | FLAT | `modules.py::MultiHead_Attention_Lattice_rel` | **核心注意力** ⭐⭐ |
| `LatticeSelfAttention` | FLAT | `modules.py` 组合实现 | **完整Lattice注意力** ⭐⭐ |
| `TransformerEncoderLayer` | FLAT | `modules.py::Transformer_Encoder_Layer` | **FLAT编码层** ⭐ |
| `AdaptiveDropout` | FLAT | `utils.py::MyDropout` | 自适应Dropout |
| `LayerProcess` | FLAT | `modules.py::Layer_Process` | 层处理序列 |
| `PositionwiseFeedForward` | FLAT | `modules.py::Positionwise_FeedForward` | 位置前馈网络 |
| `AbsolutePositionEmbedding` | FLAT | `modules.py::Absolute_Position_Embedding` | 绝对位置编码 |
| `StartEndPositionEmbedding` | FLAT | `models.py::Absolute_SE_Position_Embedding` | Start-End位置编码 |

## 🚀 使用建议

### 1. 实现 FLAT 风格模型
使用 `FourPositionFusion` + `RelativePositionEncoding`

### 2. 实现 LatticeLSTM 风格模型  
使用 `MultiInputLSTMCell` + 自定义 lattice 处理

### 3. 混合架构
组合多种组件创新模型

## ⚠️ 注意事项

1. **依赖兼容性**：所有组件基于 PyTorch，与 eznlp 框架兼容
2. **接口一致性**：遵循 eznlp 的设计模式和命名规范
3. **性能优化**：相比原始实现，优化了内存使用和计算效率
4. **可扩展性**：预留接口便于后续改进

---

## 🔧 辅助工具组件 (`lattice_utils.py`) ⭐ 新增

### 4.1 AdaptiveDropout

FLAT 项目中的自定义 Dropout，支持更灵活的训练控制。

```python
from _4MODELS.block import AdaptiveDropout

# 创建
 dropout = AdaptiveDropout(p=0.1)

# 使用
x = torch.randn(2, 10, 768)
out = dropout(x)  # 训练时应用 Dropout
```

**特点**：
- 仅在 `training` 模式且 `p > 0.001` 时生效
- 使用随机 mask 而非直接 dropout

### 4.2 LayerProcess

灵活的层处理序列，支持 Add/Dropout/LayerNorm 的任意组合。

```python
from _4MODELS.block import LayerProcess

# 创建层后处理：Dropout -> Add -> LayerNorm
post_process = LayerProcess(
    process_sequence='dan',  # 'd'=Dropout, 'a'=Add, 'n'=LayerNorm
    hidden_size=768,
    dropout=0.1
)

# 使用
x = torch.randn(2, 10, 768)
residual = torch.randn(2, 10, 768)
out = post_process(x, residual)  # 执行 d->a->n
```

**常用序列**：
- `'dan'`: Dropout → Add → LayerNorm（后处理）
- `'n'`: 仅 LayerNorm（前处理）
- `'ad'`: Add → Dropout

### 4.3 PositionwiseFeedForward

位置感知前馈网络，支持多层配置和灵活的激活函数。

```python
from _4MODELS.block import PositionwiseFeedForward

# 创建（标准 Transformer FFN）
ffn = PositionwiseFeedForward(
    layer_sizes=[768, 2048, 768],  # input -> hidden -> output
    dropout=0.1,
    activation='relu'  # 'relu', 'gelu', 'leaky'
)

# 使用
x = torch.randn(2, 10, 768)
out = ffn(x)  # [2, 10, 768]
```

**特点**：
- 支持多层配置（不限于 3 层）
- 每层独立的 Dropout（`dropout` 和 `dropout_2`）
- 多种激活函数选择

### 4.4 AbsolutePositionEmbedding

标准的正弦绝对位置编码。

```python
from _4MODELS.block import AbsolutePositionEmbedding

# 创建
pos_emb = AbsolutePositionEmbedding(
    hidden_size=768,
    max_len=512,
    learnable=False,      # 是否可学习
    pos_norm=False,       # 是否归一化
    fusion_mode='add'     # 'add' 或 'concat'
)

# 使用
x = torch.randn(2, 10, 768)
out = pos_emb(x)  # [2, 10, 768]，已添加位置编码
```

**融合模式**：
- `'add'`: 直接相加
- `'concat'`: 拼接后通过线性层投影

### 4.5 StartEndPositionEmbedding

Lattice 结构的 Start-End 双位置编码。

```python
from _4MODELS.block import StartEndPositionEmbedding

# 创建
se_pos_emb = StartEndPositionEmbedding(
    hidden_size=768,
    max_len=512,
    learnable=False,
    fusion_mode='add'  # 或 'concat', 'nonlinear_add', etc.
)

# 使用
x = torch.randn(2, 10, 768)
pos_s = torch.randint(0, 512, (2, 10))  # 开始位置
pos_e = torch.randint(0, 512, (2, 10))  # 结束位置
out = se_pos_emb(x, pos_s, pos_e)  # [2, 10, 768]
```

**融合模式**：
- `'add'`: 直接相加
- `'concat'`: 拼接后线性变换
- `'nonlinear_add'`: 非线性变换后相加
- `'nonlinear_concat'`: 非线性变换后拼接

**完整示例：组合 Lattice 组件**

```python
import torch
import torch.nn as nn
from _4MODELS.block import (
    LatticeSelfAttention,
    PositionwiseFeedForward,
    LayerProcess,
    StartEndPositionEmbedding
)

class LatticeTransformerLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, max_len=512):
        super().__init__()
        
        # 位置编码
        self.pos_emb = StartEndPositionEmbedding(
            hidden_size=hidden_size,
            max_len=max_len,
            fusion_mode='add'
        )
        
        # Lattice 自注意力
        self.self_attn = LatticeSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_len=max_len
        )
        
        # 前馈网络
        self.ffn = PositionwiseFeedForward(
            layer_sizes=[hidden_size, hidden_size * 4, hidden_size],
            dropout=0.1
        )
        
        # 层处理
        self.post_attn = LayerProcess('dan', hidden_size, dropout=0.1)
        self.post_ffn = LayerProcess('dan', hidden_size, dropout=0.1)
    
    def forward(self, x, pos_s, pos_e, mask=None):
        # 添加位置编码
        x = self.pos_emb(x, pos_s, pos_e)
        
        # 自注意力 + 残差
        attn_out = self.self_attn(x, pos_s, pos_e, mask=mask)
        x = self.post_attn(attn_out, residual=x)
        
        # 前馈网络 + 残差
        ffn_out = self.ffn(x)
        x = self.post_ffn(ffn_out, residual=x)
        
        return x

# 使用
layer = LatticeTransformerLayer()
x = torch.randn(2, 10, 768)
pos_s = torch.arange(0, 10).unsqueeze(0).expand(2, -1)
pos_e = pos_s + 1
out = layer(x, pos_s, pos_e)
print(out.shape)  # [2, 10, 768]
```

---

## 📚 参考文献

- **FLAT**: Li et al. "FLAT: Chinese NER Using Flat-Lattice Transformer" (ACL 2020)
- **LatticeLSTM**: Zhang and Yang. "Chinese NER Using Lattice LSTM" (ACL 2018)
