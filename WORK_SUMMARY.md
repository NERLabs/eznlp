# Soft+Expert 联合模型实现工作总结

**项目**: eznlp - 命名实体识别融合特征研究  
**任务**: 实现SoftLexicon与ExpertDict联合融合模型  
**数据集**: RedJujube (医疗NER，14类实体)  
**完成时间**: 2025-12-13  

---

## 📋 任务完成情况

### ✅ 已完成任务

#### 1. 设计文档创建
- 📄 [`soft-expert-ensemble-model.md`](file:///.qoder/quests/soft-expert-ensemble-model.md)
- 包含4种融合方案的完整设计
- 详细的架构图、配置规格、参数说明

#### 2. 代码实现

##### 核心融合模块
- 📄 [`eznlp/model/model/fusion_extractor.py`](file:///home/shiwenlong/NERlabs/eznlp/eznlp/model/model/fusion_extractor.py) (420行)
  - `FusionExtractorConfig`: 融合配置类
  - `FusionLayer`: 通用融合层基类
  - `ConcatFusion`: 直接拼接融合
  - `WeightedFusion`: 可学习权重加权融合  
  - `GatedFusion`: 门控机制融合
  - `AttentionFusion`: 多头注意力融合

##### 训练脚本扩展
- 📄 [`scripts/train_redjujube_ner_comparison.py`](file:///home/shiwenlong/NERlabs/eznlp/scripts/train_redjujube_ner_comparison.py) (650行)
  - 新增4个融合方案配置函数
  - 新增命令行参数支持
  - 完整的数据预处理流程

##### 监控与分析工具
- 📄 [`scripts/monitor_training.py`](file:///home/shiwenlong/NERlabs/eznlp/scripts/monitor_training.py) (420行)
  - **通用深度学习训练监控脚本**
  - GPU实时监控、进程跟踪、日志分析
  - 可复用于所有未来训练任务
  
- 📄 [`scripts/collect_fusion_results.py`](file:///home/shiwenlong/NERlabs/eznlp/scripts/collect_fusion_results.py) (284行)
  - 自动收集融合实验结果
  - 生成对比报告和分析建议

#### 3. 文档系统

- 📄 [`MONITOR_USAGE.md`](file:///home/shiwenlong/NERlabs/eznlp/scripts/MONITOR_USAGE.md): 监控脚本使用指南
- 📄 [`TRAINING_STATUS.md`](file:///home/shiwenlong/NERlabs/eznlp/TRAINING_STATUS.md): 实时训练状态报告
- 📄 [`Fusion_Comparison_Report.md`](file:///home/shiwenlong/NERlabs/eznlp/experiments/hz_lexicon/results/Fusion_Comparison_Report.md): 融合方案对比报告

---

## 🔧 技术实现亮点

### 1. 模块化融合层设计

采用策略模式，支持4种融合方式：

```python
class FusionLayer(torch.nn.Module):
    """融合层基类"""
    
class ConcatFusion(FusionLayer):
    """方案A: 直接拼接 - 0参数"""
    
class WeightedFusion(FusionLayer):
    """方案B: 加权求和 - ~10参数"""
    
class GatedFusion(FusionLayer):
    """方案C: 门控机制 - ~500参数"""
    
class AttentionFusion(FusionLayer):
    """方案D: 多头注意力 - ~1500参数"""
```

### 2. 解决的关键技术问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| BertLikeEmbedder维度访问错误 | 配置路径错误 | 修正为 `bert_like.bert_like.config.hidden_size` |
| NestedOneHotEmbedder维度获取失败 | 从实例而非配置获取 | 改为从 `config.nested_ohots[].out_dim` 获取 |
| 特征维度不匹配 | 合并嵌入导致维度变化 | 重写 `_get_full_hidden`，分别处理特征 |
| intermediate2默认值冲突 | ExtractorConfig有默认encoder | 显式传入 `intermediate2=None` |
| Decoder维度不匹配 | 父类按拼接计算维度 | 重写 `build_vocabs_and_dims` 方法 |

### 3. 通用监控系统

创建了可复用的训练监控框架：

**核心功能**:
- 🖥️ GPU监控（利用率、显存、温度）
- 🔄 进程自动检测与跟踪
- 📋 智能日志解析（Epoch/Step/Loss/Metrics）
- ⚠️ 异常检测与状态识别
- 🎨 彩色终端输出

**适用场景**: PyTorch、TensorFlow、JAX等所有深度学习框架

---

## 📊 实验进展

### 当前状态

| 方案 | 状态 | 测试F1 | 训练进度 | 说明 |
|------|------|--------|---------|------|
| 方案A (Concat) | ✅ 完成 | 96.87% | 30/30 | 低于预期（目标97.04%） |
| 方案B (Weighted) | 🔄 训练中 | - | 3/30 | 预计50分钟完成 |
| 方案C (Gated) | ⏸️ 待启动 | - | 0/30 | 等待GPU资源 |
| 方案D (Attention) | 🔄 训练中 | - | 15/30 | 预计20分钟完成 |

### 性能对比（已完成方案）

| 方法 | 测试F1 | 提升 | 参数量 |
|------|--------|------|--------|
| Baseline | 95.51% | - | 103.1M |
| ExpertDict | 96.99% | +1.48% | 103.3M |
| **Concat (方案A)** | **96.87%** | **+1.36%** | **113.3M** |

**关键发现**: 方案A未超越单独ExpertDict，说明简单拼接不是最优策略

---

## 🛠️ 可用工具与脚本

### 监控工具

```bash
# 实时监控所有训练任务
python scripts/monitor_training.py

# 快速查看一次
python scripts/monitor_training.py --once

# 自定义刷新间隔
python scripts/monitor_training.py --interval 60 --log-lines 5
```

### 结果收集

```bash
# 收集所有融合方案结果并生成报告
python scripts/collect_fusion_results.py
```

### 训练启动

```bash
# 方案A - 直接拼接
python scripts/train_redjujube_ner_comparison.py \
  --data_dir data/RedJujube \
  --run_softlexicon_expert_concat \
  --save_dir cache/redjujube_ner_comparison \
  ...

# 方案B - 加权求和
python scripts/train_redjujube_ner_comparison.py \
  --run_softlexicon_expert_weighted ...

# 方案C - 门控机制
python scripts/train_redjujube_ner_comparison.py \
  --run_softlexicon_expert_gated ...

# 方案D - 注意力融合
python scripts/train_redjujube_ner_comparison.py \
  --run_softlexicon_expert_attention ...
```

---

## 📈 下一步工作

### 短期（完成训练后）

1. ✅ 等待方案B/D训练完成
2. 🔜 启动方案C训练
3. 🔜 收集所有4个方案的结果
4. 🔜 生成完整对比报告

### 中期（实验分析）

1. 消融实验
   - 分析SoftLex和Expert各自贡献
   - 对比联合效果vs单独效果

2. 错误分析
   - 查看不同方案的预测差异
   - 识别各方案擅长的实体类型

3. 性能调优
   - 如果未达97.04%目标，调整超参数
   - 尝试不同的融合参数配置

### 长期（文档与发布）

1. 完善实验报告
2. 更新项目文档
3. 总结最佳实践
4. 归档实验结果

---

## 💡 经验总结

### 成功经验

1. **模块化设计**: 融合层采用策略模式，易于扩展新方案
2. **充分调试**: 通过逐步修复维度问题，确保了模型正确性
3. **通用工具**: 监控脚本可复用，提高未来工作效率
4. **文档完善**: 详细记录了设计、实现、使用方法

### 待改进点

1. **方案A性能**: 直接拼接未达预期，可能需要：
   - 增加Dropout正则化
   - 调整学习率
   - 尝试其他特征归一化方法

2. **GPU资源**: 显存使用接近上限，限制了并行训练
   - 考虑减小batch_size
   - 使用梯度累积

3. **训练时间**: 每个方案约需50分钟
   - 可考虑使用混合精度训练加速
   - 优化数据加载流程

---

## 📚 参考文件

### 核心代码
- [`fusion_extractor.py`](file:///home/shiwenlong/NERlabs/eznlp/eznlp/model/model/fusion_extractor.py): 融合层实现
- [`train_redjujube_ner_comparison.py`](file:///home/shiwenlong/NERlabs/eznlp/scripts/train_redjujube_ner_comparison.py): 训练脚本

### 工具脚本
- [`monitor_training.py`](file:///home/shiwenlong/NERlabs/eznlp/scripts/monitor_training.py): 通用监控工具
- [`collect_fusion_results.py`](file:///home/shiwenlong/NERlabs/eznlp/scripts/collect_fusion_results.py): 结果收集工具

### 文档
- [`soft-expert-ensemble-model.md`](file:///.qoder/quests/soft-expert-ensemble-model.md): 设计文档
- [`MONITOR_USAGE.md`](file:///home/shiwenlong/NERlabs/eznlp/scripts/MONITOR_USAGE.md): 监控使用指南
- [`TRAINING_STATUS.md`](file:///home/shiwenlong/NERlabs/eznlp/TRAINING_STATUS.md): 训练状态报告

### 实验结果
- `cache/redjujube_softlexicon_expert/`: 方案A结果
- `cache/redjujube_ner_comparison/`: 方案B/C/D结果
- `experiments/hz_lexicon/results/Fusion_Comparison_Report.md`: 对比报告

---

**项目状态**: 🔄 训练进行中  
**预计完成**: 1-2小时内完成所有4个方案训练  
**下一个里程碑**: 收集结果并生成完整对比分析报告  

**创建时间**: 2025-12-13 18:22  
**最后更新**: 2025-12-13 18:22  
