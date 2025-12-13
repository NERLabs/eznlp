# HZ 数据集 NER 训练指南

本指南介绍如何使用专家词典特征在 HZ 数据集上训练 NER 模型，并进行对比实验。

## 📋 文件说明

### 核心脚本

1. **`train_hz_ner_baseline_vs_expert_dict.py`** - 主训练脚本
   - 支持 Baseline 和 +ExpertDict 两种配置
   - 自动对比实验结果
   - 保存最佳模型和评估结果

2. **`run_hz_comparison.sh`** - 快速启动脚本
   - 一键运行对比实验
   - 预设推荐参数

3. **`test_hz_training.py`** - 快速测试脚本
   - 用于验证代码正确性
   - 仅训练 2 个 epoch

### 数据文件

- **`data/HZ/hz_train.bmes`** - 训练集（5363 句）
- **`data/HZ/hz_dev.bmes`** - 验证集（670 句）
- **`data/HZ/hz_test.bmes`** - 测试集（672 句）
- **`data/HZ/expert_lexicon.txt`** - 专家词典（2887 词条）

---

## 🚀 快速开始

### 方法 1: 使用启动脚本（推荐）

```bash
# 在项目根目录运行
bash scripts/run_hz_comparison.sh
```

### 方法 2: 直接运行 Python 脚本

```bash
python scripts/train_hz_ner_baseline_vs_expert_dict.py \
    --run_both \
    --data_dir data/HZ \
    --expert_dict_path data/HZ/expert_lexicon.txt \
    --num_epochs 30 \
    --batch_size 16
```

### 方法 3: 快速测试（2 epoch）

```bash
python scripts/test_hz_training.py
```

---

## ⚙️ 参数说明

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `data/HZ` | 数据目录 |
| `--expert_dict_path` | `data/HZ/expert_lexicon.txt` | 专家词典路径 |
| `--save_dir` | `cache/hz_ner_comparison` | 保存目录 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bert_arch` | `hfl/chinese-macbert-base` | BERT 模型 |
| `--bert_drop_rate` | `0.2` | BERT Dropout 率 |
| `--hid_dim` | `256` | LSTM 隐藏层维度 |
| `--num_layers` | `1` | LSTM 层数 |
| `--dropout` | `0.5` | Dropout 率 |
| `--expert_dict_dim` | `50` | 专家词典特征维度 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_epochs` | `30` | 训练轮数 |
| `--batch_size` | `16` | 批次大小 |
| `--lr` | `2e-3` | 学习率 |
| `--finetune_lr` | `2e-5` | BERT 微调学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--grad_clip` | `5.0` | 梯度裁剪 |
| `--use_amp` | `False` | 混合精度训练 |

### 实验参数

| 参数 | 说明 |
|------|------|
| `--run_baseline` | 仅运行 Baseline |
| `--run_expert_dict` | 仅运行 +ExpertDict |
| `--run_both` | 运行两个实验（推荐） |
| `--seed` | 随机种子（默认 42） |

---

## 📊 实验配置

### Baseline 配置

```
MacBERT (hfl/chinese-macbert-base)
  ↓
BiLSTM (dim=256, layers=1)
  ↓
CRF (BMES scheme)
```

**特征维度**: `768` (BERT)

### +ExpertDict 配置

```
MacBERT (hfl/chinese-macbert-base)  +  ExpertDict (dim=50×4)
  ↓                                     ↓
  └─────────────→ Concat ←──────────────┘
                   ↓
            BiLSTM (dim=256, layers=1)
                   ↓
            CRF (BMES scheme)
```

**特征维度**: `768 + 200 = 968` (BERT + ExpertDict)

---

## 📈 输出文件

运行后会在 `cache/hz_ner_comparison/` 目录下生成以下文件：

```
cache/hz_ner_comparison/
├── baseline_20231207-123456/
│   ├── training.log          # 训练日志
│   ├── best_model.pt         # 最佳模型权重
│   └── results.json          # 测试结果
│
├── expert_dict_20231207-123456/
│   ├── training.log
│   ├── best_model.pt
│   └── results.json
│
└── comparison_20231207-123456.json  # 对比结果
```

### 结果文件格式

**results.json**:
```json
{
  "model_type": "Baseline",
  "test_loss": 0.1234,
  "test_metrics": [0.8567],
  "total_params": 102345678,
  "trainable_params": 102345678,
  "args": {...}
}
```

**comparison_20231207-123456.json**:
```json
{
  "baseline": {...},
  "expert_dict": {...}
}
```

---

## 📝 使用示例

### 示例 1: 运行完整对比实验

```bash
python scripts/train_hz_ner_baseline_vs_expert_dict.py \
    --run_both \
    --num_epochs 30 \
    --batch_size 16 \
    --seed 42
```

### 示例 2: 仅运行 Baseline

```bash
python scripts/train_hz_ner_baseline_vs_expert_dict.py \
    --run_baseline \
    --num_epochs 30
```

### 示例 3: 仅运行 +ExpertDict

```bash
python scripts/train_hz_ner_baseline_vs_expert_dict.py \
    --run_expert_dict \
    --num_epochs 30
```

### 示例 4: 调整超参数

```bash
python scripts/train_hz_ner_baseline_vs_expert_dict.py \
    --run_both \
    --hid_dim 512 \
    --num_layers 2 \
    --expert_dict_dim 100 \
    --batch_size 32 \
    --lr 5e-3
```

### 示例 5: 使用混合精度训练（加速）

```bash
python scripts/train_hz_ner_baseline_vs_expert_dict.py \
    --run_both \
    --use_amp \
    --batch_size 32
```

---

## 🔍 预期结果

根据专家词典覆盖率测试（87%+ 覆盖率），预期：

- **Baseline F1**: ~82-85%
- **+ExpertDict F1**: ~85-88% ✨
- **预期提升**: +2-3% F1

实际结果可能因超参数、随机种子等因素有所差异。

---

## 🛠️ 常见问题

### Q1: CUDA Out of Memory

**解决方法**:
```bash
# 减小批次大小
--batch_size 8

# 或使用梯度累积
--batch_size 8 --num_grad_acc_steps 2
```

### Q2: BERT 模型下载失败

**解决方法**:
1. 手动下载 `hfl/chinese-macbert-base`
2. 放到本地目录（如 `models/chinese-macbert-base`）
3. 修改 `--bert_arch` 参数：
```bash
--bert_arch models/chinese-macbert-base
```

### Q3: 如何加载已训练的模型？

**Python 代码**:
```python
import torch
from eznlp.model import ExtractorConfig

# 加载配置和模型
config = ExtractorConfig(...)  # 使用相同配置
model = config.instantiate()
model.load_state_dict(torch.load('cache/.../best_model.pt'))
model.eval()
```

### Q4: 如何修改实体类型？

当前 HZ 数据集包含 14 种实体类型（PAR, PER, GEO, CUL, AGR, PRO, NUT, DIS, EQU, PES, FER, DRU, TAX, None）。
模型会自动从训练数据中学习实体类型，无需手动配置。

---

## 📚 相关文档

- [专家词典覆盖率分析报告](file:///home/shiwenlong/NERlabs/eznlp/expert_dict_analysis_report.txt)
- [专家词典文件](file:///home/shiwenlong/NERlabs/eznlp/data/HZ/expert_lexicon.txt)
- [eznlp 项目文档](file:///home/shiwenlong/NERlabs/eznlp/README.md)

---

## 🎯 进阶优化建议

1. **调整专家词典特征维度**
   - 尝试 `--expert_dict_dim 100` 或 `200`

2. **增加 LSTM 层数**
   - 尝试 `--num_layers 2` 或 `3`

3. **调整学习率**
   - 尝试 `--lr 5e-3` 或 `--finetune_lr 5e-5`

4. **使用更大的 BERT 模型**
   - 尝试 `--bert_arch hfl/chinese-macbert-large`

5. **补充专家词典**
   - 从数据集中提取高频未覆盖实体
   - 添加外部地名词典（解决 GEO 覆盖率低的问题）

---

**祝实验顺利！如有问题，请查看日志文件或提交 Issue。**
