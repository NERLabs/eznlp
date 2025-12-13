# NFLAT改进方案实施指南

## 🚀 快速开始（5分钟上手）

### 第一步：选择方案

```bash
# 查看所有可用方案
cd /home/shiwenlong/NERlabs/eznlp/experiments/hz_lexicon/analysis
python nflat_improvements.py
```

**推荐路线**：
```
今晚    → 方案1 (Gated-Improved)
明天    → 方案4 (Hierarchical-Inter) 
后天    → 超参数调优 + 消融实验
```

---

## 📦 方案1: 改进门控融合（最快见效）

### 实现步骤

#### 1. 修改extractor.py

```bash
cd /home/shiwenlong/NERlabs/eznlp/_6MODEL
cp extractor.py extractor.py.backup  # 备份
```

在`extractor.py`中添加（大约第200行，FusionExtractor类内）：

```python
# ============ 新增代码开始 ============
class ImprovedGatedFusion(nn.Module):
    """改进的门控融合（从nflat_improvements.py复制）"""
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.expert_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.soft_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, expert_feat, soft_feat):
        expert_encoded = self.expert_encoder(expert_feat)
        soft_encoded = self.soft_encoder(soft_feat)
        combined = torch.cat([expert_encoded, soft_encoded], dim=-1)
        gate = self.gate_net(combined)
        fused = gate * expert_encoded + (1 - gate) * soft_encoded
        return self.norm(fused)
# ============ 新增代码结束 ============

class FusionExtractor(nn.Module):
    def __init__(self, bert_like, nested_ohots, intermediate2, decoder,
                 fusion_strategy='concat', fusion_params=None):
        super().__init__()
        # ... 原有代码 ...
        
        # ============ 修改这里 ============
        if fusion_strategy == 'gated':
            # 原始Gated
            gate_hidden_dim = fusion_params.get('gate_hidden_dim', 768) if fusion_params else 768
            self.gate = nn.Sequential(
                nn.Linear(total_nested_dim * 2, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, total_nested_dim),
                nn.Sigmoid()
            )
        elif fusion_strategy == 'gated_improved':  # 新增！
            self.improved_fusion = ImprovedGatedFusion(
                hidden_dim=total_nested_dim,
                dropout=fusion_params.get('dropout', 0.3) if fusion_params else 0.3
            )
        # ============ 修改结束 ============
    
    def forward(self, tokens, **nested_inputs):
        bert_output = self.bert_extractor(tokens)
        nested_features = {}
        for name, extractor in self.nested_extractors.items():
            nested_features[name] = extractor(**nested_inputs)
        
        # ... 处理 nested_features ...
        
        # ============ 修改这里 ============
        if self.fusion_strategy == 'gated':
            # 原始Gated
            combined = torch.cat([nested_features['expert_dict'], 
                                  nested_features['softlexicon']], dim=-1)
            gate = self.gate(combined)
            fused = gate * nested_features['expert_dict'] + \
                    (1 - gate) * nested_features['softlexicon']
            output = torch.cat([bert_output, fused], dim=-1)
        
        elif self.fusion_strategy == 'gated_improved':  # 新增！
            fused = self.improved_fusion(
                expert_feat=nested_features['expert_dict'],
                soft_feat=nested_features['softlexicon']
            )
            output = torch.cat([bert_output, fused], dim=-1)
        # ============ 修改结束 ============
        
        return self.decoder(output)
```

#### 2. 创建训练脚本

```bash
cd /home/shiwenlong/NERlabs/eznlp/_1CONFIG/redjujube
cp run_fusion_gated.sh run_fusion_gated_improved.sh
```

修改`run_fusion_gated_improved.sh`：

```bash
#!/bin/bash

# ===== 改进门控融合实验 =====
# 使用: ExpertDict(自动) + SoftLex-v2 + Gated-Improved
# 目标: 突破97%

EXPERIMENT_NAME="redjujube_fusion_gated_improved"
OUTPUT_DIR="../../_4OUTPUT/redjujube/${EXPERIMENT_NAME}"

python train_redjujube_ner_comparison.py \
    --experiment_name "${EXPERIMENT_NAME}" \
    --model_type "soft_expert_gated_improved" \    # 新增模型类型
    --expert_dict_type "auto" \
    --softlexicon_path "../../_2DATA/RedJujube/softlexicon_filtered_v2.txt" \    # 使用v2！
    --bert_model_path "hfl/chinese-macbert-base" \
    --num_epochs 30 \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --dropout 0.3 \            # 改进版使用更高dropout
    --output_dir "${OUTPUT_DIR}" \
    --use_crf \
    --save_model
```

#### 3. 修改训练脚本支持新模型

在`train_redjujube_ner_comparison.py`中添加（约第800行）：

```python
def build_softlexicon_expert_gated_improved_config(args, vectors):
    """构建改进版Gated融合配置"""
    # ... 复制gated配置，修改fusion_strategy ...
    config = FusionExtractorConfig(
        bert_like=bert_config,
        nested_ohots={
            "softlexicon": softlexicon_config,
            "expert_dict": expert_dict_config
        },
        intermediate2=None,
        decoder=decoder_config,
        fusion_strategy="gated_improved",  # 这里改！
        fusion_params={"dropout": args.dropout}
    )
    return config

# 在main函数中添加
if args.model_type == 'soft_expert_gated_improved':
    config = build_softlexicon_expert_gated_improved_config(args, vectors)
```

#### 4. 启动训练

```bash
cd /home/shiwenlong/NERlabs/eznlp/_1CONFIG/redjujube
bash run_fusion_gated_improved.sh
```

### 预期结果

**训练日志示例**：
```
Epoch 1/30: Train Loss=0.0823, Dev F1=94.23%
Epoch 5/30: Train Loss=0.0312, Dev F1=96.18%
Epoch 10/30: Train Loss=0.0156, Dev F1=96.78%
Epoch 15/30: Train Loss=0.0089, Dev F1=97.02%  ← 期待这里！
Epoch 20/30: Train Loss=0.0067, Dev F1=97.15%
Best Dev F1: 97.15% at Epoch 20
Test F1: 97.08%  ← 最终目标
```

---

## 📦 方案4: 层次化Inter-Attention（潜力最大）

### 实现步骤

#### 1. 添加Inter-Attention层

在`extractor.py`中添加：

```python
# ============ 新增代码 ============
import math

class InterAttentionFusion(nn.Module):
    """简化版Inter-Attention（从nflat_improvements.py复制）"""
    def __init__(self, hidden_dim, n_head=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_head == 0
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.per_head_dim = hidden_dim // n_head
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.per_head_dim)
    
    def forward(self, query, key_value, mask=None):
        B, L1, _ = query.shape
        L2 = key_value.shape[1]
        
        Q = self.q_proj(query).view(B, L1, self.n_head, self.per_head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, L2, self.n_head, self.per_head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, L2, self.n_head, self.per_head_dim).transpose(1, 2)
        
        attn_score = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        if mask is not None:
            attn_score = attn_score.masked_fill(~mask.unsqueeze(1), -1e9)
        
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)
        
        output = torch.matmul(attn_weight, V)
        output = output.transpose(1, 2).contiguous().view(B, L1, self.hidden_dim)
        return self.out_proj(output)


class HierarchicalInterFusion(nn.Module):
    """层次化Inter-Attention融合"""
    def __init__(self, hidden_dim, n_head=4, dropout=0.1):
        super().__init__()
        self.expert_attn = InterAttentionFusion(hidden_dim, n_head, dropout)
        self.soft_attn = InterAttentionFusion(hidden_dim, n_head, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, bert_feat, expert_feat, soft_feat):
        # 层次1: ExpertDict增强
        residual = bert_feat
        enhanced1 = self.expert_attn(query=bert_feat, key_value=expert_feat)
        enhanced1 = self.norm1(enhanced1 + residual)
        
        # 层次2: SoftLex增强
        residual = enhanced1
        enhanced2 = self.soft_attn(query=enhanced1, key_value=soft_feat)
        enhanced2 = self.norm2(enhanced2 + residual)
        
        # FFN增强
        residual = enhanced2
        final_feat = self.ffn(enhanced2)
        return self.norm3(final_feat + residual)
# ============ 新增代码结束 ============
```

#### 2. 修改FusionExtractor

```python
class FusionExtractor(nn.Module):
    def __init__(self, fusion_strategy='hierarchical_inter', ...):
        # ... 原有代码 ...
        
        if fusion_strategy == 'hierarchical_inter':
            self.hierarchical_fusion = HierarchicalInterFusion(
                hidden_dim=768,  # BERT输出维度
                n_head=4,
                dropout=0.1
            )
    
    def forward(self, tokens, **nested_inputs):
        bert_output = self.bert_extractor(tokens)
        nested_features = {...}
        
        if self.fusion_strategy == 'hierarchical_inter':
            # 注意：这里不拼接，直接用融合后的特征
            fused = self.hierarchical_fusion(
                bert_feat=bert_output,
                expert_feat=nested_features['expert_dict'],
                soft_feat=nested_features['softlexicon']
            )
            # 重要：decoder输入维度不变（768）
            return self.decoder(fused)
```

#### 3. 创建训练脚本

```bash
cd /home/shiwenlong/NERlabs/eznlp/_1CONFIG/redjujube
cat > run_fusion_hierarchical_inter.sh << 'EOF'
#!/bin/bash

EXPERIMENT_NAME="redjujube_fusion_hierarchical_inter"
OUTPUT_DIR="../../_4OUTPUT/redjujube/${EXPERIMENT_NAME}"

python train_redjujube_ner_comparison.py \
    --experiment_name "${EXPERIMENT_NAME}" \
    --model_type "soft_expert_hierarchical_inter" \
    --expert_dict_type "auto" \
    --softlexicon_path "../../_2DATA/RedJujube/softlexicon_filtered_v2.txt" \
    --bert_model_path "hfl/chinese-macbert-base" \
    --num_epochs 30 \
    --batch_size 16 \
    --learning_rate 2e-5 \    # 注意：Inter-Attention需要更小的lr
    --warmup_ratio 0.1 \
    --dropout 0.1 \            # 注意：更低的dropout
    --output_dir "${OUTPUT_DIR}" \
    --use_crf \
    --save_model

echo "✅ 层次化Inter-Attention融合训练完成"
EOF

chmod +x run_fusion_hierarchical_inter.sh
```

---

## ⚙️ 超参数调优指南

### 基于NFLAT的推荐配置

| 超参数 | 原始Gated | Gated-Improved | Hierarchical-Inter | 说明 |
|--------|-----------|----------------|-------------------|------|
| learning_rate | 3e-5 | 3e-5 | **2e-5** | Inter-Attention需要更小lr |
| batch_size | 16 | 16 | 16 | 保持不变 |
| dropout | 0.5 | **0.3** | **0.1** | 改进版使用更精细的dropout |
| warmup_ratio | 0.1 | 0.1 | 0.1 | 保持不变 |
| num_epochs | 30 | 30 | 30 | 保持不变 |
| n_head | - | - | **4** | Inter-Attention头数 |
| hidden_dim | 768 | 768 | 768 | BERT维度 |

### 调优技巧

1. **学习率选择**：
   - Gated系列：3e-5（BERT标准）
   - Attention系列：2e-5（更稳定）
   - 如果不收敛：降到1e-5

2. **Dropout调整**：
   - 过拟合（Train F1 >> Dev F1）：增加dropout到0.5
   - 欠拟合（Train F1 < 96%）：减少dropout到0.1
   - 正常情况：0.3

3. **Batch Size影响**：
   - 16：标准，适合大多数情况
   - 32：更稳定，但收敛慢
   - 8：更快，但可能不稳定

---

## 🔍 故障排查

### 问题1: 训练不收敛（Dev F1 < 95%）

**可能原因**：
- 学习率过高
- Dropout过大
- 融合策略代码有bug

**解决方法**：
```bash
# 1. 降低学习率
--learning_rate 1e-5

# 2. 减少dropout
--dropout 0.1

# 3. 检查代码
python -c "from _6MODEL.extractor import ImprovedGatedFusion; print('✅ 导入成功')"
```

### 问题2: 过拟合（Train 98%, Dev 95%）

**解决方法**：
```bash
# 增加dropout
--dropout 0.5

# 增加L2正则
--weight_decay 0.01

# 提前停止
--early_stopping_patience 5
```

### 问题3: 内存溢出

**解决方法**：
```bash
# 减少batch_size
--batch_size 8

# 或使用梯度累积
--gradient_accumulation_steps 2
```

### 问题4: 性能不如预期（< 97%）

**检查清单**：
- [ ] 确认使用SoftLex-v2（去标点版本）
- [ ] 确认ExpertDict是自动提取版本
- [ ] 检查融合代码是否正确集成
- [ ] 查看训练日志，是否有异常
- [ ] 尝试不同的随机种子

---

## 📊 实验对比模板

创建文件`experiments/hz_lexicon/results/Improvement_Comparison.md`：

```markdown
# 改进方案对比实验

## 实验配置

- 数据集: RedJujube (Train:5372, Dev:671, Test:672)
- 基础模型: MacBERT-base
- 词典: ExpertDict(自动) + SoftLexicon-v2

## 结果对比

| 方案 | Dev F1 | Test F1 | 参数量 | 训练时间 | 备注 |
|------|--------|---------|--------|----------|------|
| Baseline (BERT) | 95.87% | 95.64% | 103M | 1h | - |
| + ExpertDict | 96.78% | 96.99% | 103M | 1h | ✅ 当前最好 |
| + SoftLex-v1 | 96.21% | 96.12% | 103M | 1.2h | 含标点 |
| + SoftLex-v2 | ?% | ?% | 103M | 1.2h | 🔄 训练中 |
| Gated (原始) | 96.58% | 96.46% | 113M | 1.5h | 融合失败 |
| **Gated-Improved** | **?%** | **?%** | 116M | 1.6h | 🎯 今晚测试 |
| Hierarchical-Inter | ?% | ?% | 125M | 2h | 🎯 明天测试 |

## 消融实验

### 改进门控的影响

| 组件 | Dev F1 | Δ |
|------|--------|---|
| Baseline Gated | 96.46% | - |
| + 独立编码器 | ?% | ?% |
| + 更深门控网络 | ?% | ?% |
| + LayerNorm | ?% | ?% |
| **完整版** | **?%** | **?%** |

### 位置编码的影响

| 配置 | Dev F1 | Δ |
|------|--------|---|
| 无位置编码 | ?% | - |
| + ExpertDict位置 | ?% | ?% |
| + SoftLex位置 | ?% | ?% |
| + 双重位置 | ?% | ?% |
```

---

## 🎯 成功标准

### 阶段性目标

**短期目标（今晚-明天）**：
- ✅ 成功集成Gated-Improved代码
- ✅ 训练收敛（Dev F1稳定）
- 🎯 **Test F1 ≥ 97.0%**

**中期目标（本周）**：
- 🎯 Test F1 ≥ 97.2%
- ✅ 完成消融实验
- ✅ 确定最优配置

**长期目标（下周）**：
- 🎯 Test F1 ≥ 97.5%（如果可能）
- ✅ 撰写实验报告
- ✅ 准备论文/部署

### 验收标准

**代码质量**：
- [ ] 所有改进代码已集成到extractor.py
- [ ] 训练脚本可独立运行
- [ ] 有完整的实验日志

**性能指标**：
- [ ] 至少1个方案 > 96.99%（超越ExpertDict单独）
- [ ] 最好方案 ≥ 97.0%
- [ ] 训练稳定，可复现

**文档完整**：
- [ ] 有详细的实验对比表
- [ ] 有消融实验分析
- [ ] 有故障排查记录

---

## 📞 需要帮助？

### 常见问题

**Q1: 代码集成后训练报错？**
A: 检查`import`语句，确保`torch`, `torch.nn.functional`都导入了

**Q2: 性能不如预期？**
A: 先确认SoftLex-v2单独的性能，如果v2也不行，说明词典质量还需优化

**Q3: 要不要完整移植NFLAT？**
A: 建议先把这3个方案都试完，如果还不行再考虑

### 联系方式

- 实验日志路径: `_4OUTPUT/redjujube/*/train.log`
- 结果文件: `experiments/hz_lexicon/results/`
- 代码仓库: `/home/shiwenlong/NERlabs/eznlp/`

---

**祝实验顺利！期待看到97%+的结果！🎉**
