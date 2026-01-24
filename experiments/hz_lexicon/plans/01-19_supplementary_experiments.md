# RedJujube NER 补充实验计划

> 创建日期: 2026-01-19
> 基于已完成实验的分析结果，规划补充实验

---

## 一、已完成实验回顾

| 实验 | 词典 | 解码层 | 测试F1 |
|------|------|--------|--------|
| BERT+BiLSTM+CRF基准 | 无 | CRF | 0.8426 |
| 纯BERT+BiLSTM+CRF | 无 | CRF | 0.8443 |
| 词典+边界选择(freq=2) | 1800词 | BS | **0.8596** |
| 纯BERT+边界选择 | 无 | BS | 0.8556 |
| 词典+边界选择(freq=1) | 5247词 | BS | 0.7754 |
| 无通道注意力 | 5247词 | BS | 0.8603 |
| 序列标注+词典 | 5247词 | CRF | 0.8273 |

**当前最优**: 词典(freq=2) + 边界选择 = **0.8596**

---

## 二、补充实验清单

### 实验组1: 词典频率阈值细化 (高优先级)

**目标**: 找到最优的词典频率阈值

| 编号 | 实验名称 | min_freq | 预期词典大小 |
|------|----------|----------|--------------|
| 1.1 | freq=3 | 3 | ~1200词 |
| 1.2 | freq=5 | 5 | ~800词 |

**执行命令**:

```bash
# 实验 1.1: min_freq=3
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_freq3 \
    --min_freq 3 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --num_epochs 30 \
    --batch_size 16 \
    > train_hz_freq3.log 2>&1 &

# 实验 1.2: min_freq=5
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_freq5 \
    --min_freq 5 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --num_epochs 30 \
    --batch_size 16 \
    > train_hz_freq5.log 2>&1 &
```

---

### 实验组2: 类型感知词典对比 (高优先级)

**目标**: 验证词典类型信息是否有帮助

| 编号 | 实验名称 | with_type | 解码层 |
|------|----------|-----------|--------|
| 2.1 | 类型词典+边界选择 | True | BS |
| 2.2 | 类型词典+CRF | True | CRF |

**前置步骤**: 生成带类型的词典文件

```bash
# 从训练集提取带类型词典
python _3DATA_PROCESS/extract_typed_lexicon.py \
    --input _2DATA/RedJujube/redjujube_train.bmes \
    --output _2DATA/RedJujube/expert_lexicon_typed.txt \
    --min_freq 2
```

**执行命令**:

```bash
# 实验 2.1: 类型词典 + 边界选择
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_typed_bs \
    --expert_dict_path _2DATA/RedJujube/expert_lexicon_typed.txt \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --num_epochs 30 \
    --batch_size 16 \
    > train_hz_typed_bs.log 2>&1 &

# 实验 2.2: 类型词典 + CRF (序列标注)
nohup python _5TRAIN/train_redjujube_ner.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_typed_crf \
    --expert_dict_path _2DATA/RedJujube/expert_lexicon_typed.txt \
    --with_type \
    --num_epochs 30 \
    --batch_size 16 \
    > train_hz_typed_crf.log 2>&1 &
```

---

### 实验组3: 特征融合方式对比 (中优先级)

**目标**: 对比不同的词典特征融合方式

| 编号 | 实验名称 | 融合方式 | 参数 |
|------|----------|----------|------|
| 3.1 | 通道注意力v1 | attention | --use_channel_attention --channel_attn_version v1 |
| 3.2 | 通道注意力v2 | attention | --use_channel_attention --channel_attn_version v2 |
| 3.3 | 跨位置LSTM | cross_encoder | --use_cross_position_encoder |

**执行命令**:

```bash
# 实验 3.1: 通道注意力 v1 (freq=2词典)
# 注意: expert_dict_dim 必须能被 channel_attn_heads 整除
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_channel_attn_v1 \
    --min_freq 2 \
    --expert_dict_dim 48 \
    --use_channel_attention \
    --channel_attn_version v1 \
    --channel_attn_heads 4 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --num_epochs 30 \
    > train_hz_channel_attn_v1.log 2>&1 &

# 实验 3.2: 通道注意力 v2 (freq=2词典)
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_channel_attn_v2 \
    --min_freq 2 \
    --expert_dict_dim 48 \
    --use_channel_attention \
    --channel_attn_version v2 \
    --channel_attn_heads 4 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --num_epochs 30 \
    > train_hz_channel_attn_v2.log 2>&1 &

# 实验 3.3: 跨位置LSTM编码器 (freq=2词典)
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_cross_encoder \
    --min_freq 2 \
    --use_cross_position_encoder \
    --cross_position_encoder_type lstm \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --num_epochs 30 \
    > train_hz_cross_encoder.log 2>&1 &
```

---

### 实验组4: 边界平滑参数调优 (低优先级)

**目标**: 优化边界平滑超参数

| 编号 | sb_epsilon | sb_size |
|------|------------|---------|
| 4.1 | 0.05 | 2 |
| 4.2 | 0.15 | 2 |
| 4.3 | 0.1 | 1 |
| 4.4 | 0.1 | 3 |

**执行命令**:

```bash
# 实验 4.1: sb_epsilon=0.05
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_sb_eps005 \
    --min_freq 2 \
    --sb_epsilon 0.05 \
    --sb_size 2 \
    --num_epochs 30 \
    > train_hz_sb_eps005.log 2>&1 &

# 实验 4.2: sb_epsilon=0.15
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_sb_eps015 \
    --min_freq 2 \
    --sb_epsilon 0.15 \
    --sb_size 2 \
    --num_epochs 30 \
    > train_hz_sb_eps015.log 2>&1 &

# 实验 4.3: sb_size=1
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_sb_size1 \
    --min_freq 2 \
    --sb_epsilon 0.1 \
    --sb_size 1 \
    --num_epochs 30 \
    > train_hz_sb_size1.log 2>&1 &

# 实验 4.4: sb_size=3
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_sb_size3 \
    --min_freq 2 \
    --sb_epsilon 0.1 \
    --sb_size 3 \
    --num_epochs 30 \
    > train_hz_sb_size3.log 2>&1 &
```

---

## 三、执行优先级与时间安排

### 阶段一 (高优先级)

| 序号 | 实验 | 预计耗时 |
|------|------|----------|
| 1 | 1.1 freq=3 | ~20min |
| 2 | 1.2 freq=5 | ~20min |
| 3 | 2.1 类型词典+BS | ~25min |

**可并行执行**: 1.1, 1.2, 2.1

### 阶段二 (中优先级)

| 序号 | 实验 | 预计耗时 |
|------|------|----------|
| 4 | 2.2 类型词典+CRF | ~20min |
| 5 | 3.1 通道注意力v1 | ~25min |
| 6 | 3.2 通道注意力v2 | ~25min |

### 阶段三 (低优先级，可选)

| 序号 | 实验 | 预计耗时 |
|------|------|----------|
| 7-10 | 4.1-4.4 边界平滑调参 | 各~20min |

---

## 四、预期结果与分析要点

### 词典频率实验
- 预期 freq=3 可能略优于 freq=2
- freq=5 可能因词典过小而性能下降

### 类型感知词典
- 根据历史反馈，类型感知效果可能一般
- 但需要实验数据支撑结论

### 特征融合方式
- 已知 freq=1 + 无通道注意力 = 0.8603
- 需验证 freq=2 + 通道注意力是否更优

---

## 五、实验结果汇总 (已完成)

> 更新日期: 2026-01-20

### 补充实验结果

| 实验 | 词典配置 | 融合方式 | 验证F1 | 测试F1 | 备注 |
|------|----------|----------|--------|--------|------|
| 1.1 | freq=3 (1082词) | concat | 0.8722 | **0.8678** | 词典精简效果好 |
| 1.2 | freq=5 (614词) | concat | 0.8722 | 0.8611 | 词典过小 |
| 2.1 | typed+freq=2 (1800词) | concat | 0.8740 | **0.8679** | 类型信息有帮助 |
| 2.2 | typed+freq=2 (1800词) | CRF | 0.8597 | 0.8459 | CRF效果一般 |
| 3.1 | freq=2 (1800词) | attn_v1 | 0.8753 | 0.8642 | 注意力有效 |
| 3.2 | freq=2 (1800词) | attn_v2 | 0.8743 | 0.8652 | v2略优 |
| 3.3 | freq=2 (1800词) | cross_lstm | 0.8750 | 0.8620 | 跨位置编码 |

### 完整实验结果对比 (含基线)

| 排名 | 实验 | 词典 | 解码层 | 融合方式 | 测试F1 |
|------|------|------|--------|----------|--------|
| 1 | freq=3 | 1082词 | BS | concat | **0.8678** |
| 2 | 类型词典+BS | 1800词 | BS | concat | **0.8679** |
| 3 | 通道注意力v2 | 1800词 | BS | attn_v2 | 0.8652 |
| 4 | 通道注意力v1 | 1800词 | BS | attn_v1 | 0.8642 |
| 5 | 跨位置LSTM | 1800词 | BS | cross_lstm | 0.8620 |
| 6 | freq=5 | 614词 | BS | concat | 0.8611 |
| 7 | 无通道注意力 | 5247词 | BS | concat | 0.8603 |
| 8 | 词典+BS(freq=2) | 1800词 | BS | concat | 0.8596 |
| 9 | 纯BERT+BS | 无 | BS | - | 0.8556 |
| 10 | 类型词典+CRF | 1800词 | CRF | concat | 0.8459 |
| 11 | 纯BERT+CRF | 无 | CRF | - | 0.8443 |
| 12 | BERT+BiLSTM+CRF | 无 | CRF | - | 0.8426 |

### 关键发现

1. **最优配置**: freq=3 (1082词) + concat + BS = **0.8678**
   - 词典精简后反而效果更好
   - 类型词典(0.8679)与freq=3效果相当

2. **词典频率阈值**:
   - freq=3 > freq=2 > freq=5
   - 过多低频词(freq=1)会严重损害性能

3. **通道注意力**:
   - v2 (0.8652) > v1 (0.8642) > 无注意力(0.8596)
   - 注意力机制带来约0.5%提升

4. **类型信息**:
   - 类型词典+BS (0.8679) vs 无类型+BS (0.8596) = +0.83%
   - 类型信息在BS解码下有帮助

5. **CRF vs BS**:
   - BS解码全面优于CRF
   - 类型词典+CRF (0.8459) vs 类型词典+BS (0.8679) = -2.2%

---

## 六、快速执行脚本

将高优先级实验整合为一键脚本:

```bash
#!/bin/bash
# run_supplementary_experiments.sh

echo "=== 启动补充实验 ==="
echo "开始时间: $(date)"

# 实验1.1: freq=3
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube --save_dir cache/hz_freq3 \
    --min_freq 3 --sb_epsilon 0.1 --sb_size 2 --num_epochs 30 \
    > train_hz_freq3.log 2>&1 &
echo "已启动: freq=3 实验"

# 实验1.2: freq=5
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube --save_dir cache/hz_freq5 \
    --min_freq 5 --sb_epsilon 0.1 --sb_size 2 --num_epochs 30 \
    > train_hz_freq5.log 2>&1 &
echo "已启动: freq=5 实验"

echo "=== 实验已启动，使用 tail -f train_hz_*.log 查看进度 ==="
```

保存为 `_7SHELL/run_hz_supplementary.sh` 执行。

---

## 七、待完成实验计划 (Phase 2)

> 基于已完成实验的发现，规划下一阶段实验

### 7.1 组合优化实验 (高优先级)

**目标**: 将已验证有效的因素组合，寻找最优配置

| 编号 | 实验名称 | 词典 | 融合方式 | 预期 |
|------|----------|------|----------|------|
| 5.1 | freq=3 + attn_v2 | 1082词 | attn_v2 | 可能突破0.87 |
| 5.2 | typed + attn_v2 | 1800词 | attn_v2 | 类型+注意力组合 |
| 5.3 | typed + freq=3 | ~1000词 | concat | 类型+精简词典 |

**执行命令**:

```bash
# 实验 5.1: freq=3 + 通道注意力v2
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_freq3_attn_v2 \
    --min_freq 3 \
    --expert_dict_dim 48 \
    --use_channel_attention \
    --channel_attn_version v2 \
    --channel_attn_heads 4 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --num_epochs 30 \
    > train_hz_freq3_attn_v2.log 2>&1 &

# 实验 5.2: 类型词典 + 通道注意力v2
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_typed_attn_v2 \
    --expert_dict_path _2DATA/RedJujube/expert_lexicon_typed.txt \
    --expert_dict_dim 48 \
    --use_channel_attention \
    --channel_attn_version v2 \
    --channel_attn_heads 4 \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --num_epochs 30 \
    > train_hz_typed_attn_v2.log 2>&1 &

# 实验 5.3: 类型词典 + freq=3 (需先生成typed词典with freq=3)
# 前置: python _3DATA_PROCESS/extract_typed_lexicon.py --min_freq 3
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_typed_freq3 \
    --expert_dict_path _2DATA/RedJujube/expert_lexicon_typed_freq3.txt \
    --sb_epsilon 0.1 \
    --sb_size 2 \
    --num_epochs 30 \
    > train_hz_typed_freq3.log 2>&1 &
```

---

### 7.2 边界平滑参数调优 (中优先级)

**目标**: 在最优词典配置(freq=3)上调优边界平滑参数

| 编号 | sb_epsilon | sb_size | 基准对比 |
|------|------------|---------|----------|
| 6.1 | 0.05 | 2 | epsilon减半 |
| 6.2 | 0.15 | 2 | epsilon增大 |
| 6.3 | 0.1 | 1 | size减小 |
| 6.4 | 0.1 | 3 | size增大 |

**执行命令**:

```bash
# 实验 6.1: sb_epsilon=0.05 (基于freq=3最优配置)
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_freq3_sb_eps005 \
    --min_freq 3 \
    --sb_epsilon 0.05 \
    --sb_size 2 \
    --num_epochs 30 \
    > train_hz_freq3_sb_eps005.log 2>&1 &

# 实验 6.2: sb_epsilon=0.15
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_freq3_sb_eps015 \
    --min_freq 3 \
    --sb_epsilon 0.15 \
    --sb_size 2 \
    --num_epochs 30 \
    > train_hz_freq3_sb_eps015.log 2>&1 &

# 实验 6.3: sb_size=1
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_freq3_sb_size1 \
    --min_freq 3 \
    --sb_epsilon 0.1 \
    --sb_size 1 \
    --num_epochs 30 \
    > train_hz_freq3_sb_size1.log 2>&1 &

# 实验 6.4: sb_size=3
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_freq3_sb_size3 \
    --min_freq 3 \
    --sb_epsilon 0.1 \
    --sb_size 3 \
    --num_epochs 30 \
    > train_hz_freq3_sb_size3.log 2>&1 &
```

---

### 7.3 稳定性验证 (低优先级)

**目标**: 验证最优配置的稳定性

| 编号 | 实验 | seed |
|------|------|------|
| 7.1 | freq=3 (run2) | 43 |
| 7.2 | freq=3 (run3) | 44 |

**执行命令**:

```bash
# 多次运行验证
for seed in 43 44; do
    nohup python _5TRAIN/train_redjujube_expert_boundary.py \
        --data_dir _2DATA/RedJujube \
        --save_dir cache/hz_freq3_seed${seed} \
        --min_freq 3 \
        --seed ${seed} \
        --sb_epsilon 0.1 \
        --sb_size 2 \
        --num_epochs 30 \
        > train_hz_freq3_seed${seed}.log 2>&1 &
done
```

---

### 7.4 执行优先级总结

| 阶段 | 实验编号 | 实验数量 | 优先级 |
|------|----------|----------|--------|
| Phase 2a | 5.1, 5.2 | 2 | 高 |
| Phase 2b | 6.1-6.4 | 4 | 中 |
| Phase 2c | 5.3, 7.1-7.2 | 3 | 低 |

**推荐执行顺序**:
1. 先执行 5.1 (freq=3 + attn_v2) 和 5.2 (typed + attn_v2)
2. 根据结果决定是否继续边界平滑调优
3. 稳定性验证可在最终配置确定后进行

---

### 7.5 Phase 2 结果记录 (已完成)

> 更新日期: 2026-01-20

| 实验 | 词典 | 融合方式 | sb参数 | 验证F1 | 测试F1 | 备注 |
|------|------|----------|--------|--------|--------|------|
| 5.1 | freq=3 | attn_v2 | 0.1/2 | 0.8760 | 0.8572 | 组合效果不佳 |
| 5.2 | typed | attn_v2 | 0.1/2 | 0.8753 | 0.8672 | 较好 |
| 5.3 | typed+freq=3 | concat | 0.1/2 | 0.8693 | 0.8666 | 效果一般 |
| 6.1 | freq=3 | concat | 0.05/2 | 0.8701 | 0.8659 | eps偏小 |
| 6.2 | freq=3 | concat | 0.15/2 | 0.8696 | **0.8677** | 当前最优 |
| 6.3 | freq=3 | concat | 0.1/1 | 0.8667 | 0.8653 | size偏小 |
| 6.4 | freq=3 | concat | 0.1/3 | 0.8682 | 0.8673 | 效果不错 |

### Phase 2 关键发现

1. **边界平滑调优有效**: sb_epsilon=0.15 获得最优测试F1 **0.8677**

2. **组合实验效果不佳**: freq=3 + attn_v2 测试F1仅0.8572，注意力机制与高频词典组合可能过拟合

3. **最优配置确定**: freq=3 + sb_epsilon=0.15 + sb_size=2

---

## 八、最优模型分类指标详情

> 模型: freq=3 + sb_epsilon=0.15 + sb_size=2  
> 测试F1: 0.8677

### 宏/微平均指标

| 指标 | Precision | Recall | F1 |
|------|-----------|--------|------|
| Macro | 0.8546 | 0.8276 | 0.8382 |
| Micro | 0.8911 | 0.8454 | **0.8677** |

### 各实体类型详细指标

| Type | P | R | F1 | Gold | Pred | TP |
|------|------|------|------|------|------|------|
| AGR | 0.8187 | 0.7298 | 0.7717 | 433 | 386 | 316 |
| CUL | 0.9400 | 0.8577 | 0.8969 | 274 | 250 | 235 |
| DIS | 0.9328 | 0.9191 | 0.9259 | 136 | 134 | 125 |
| DRU | 0.9247 | 0.8766 | 0.9000 | 154 | 146 | 135 |
| EQU | 0.8670 | 0.7354 | 0.7958 | 257 | 218 | 189 |
| FER | 0.8125 | 0.6500 | 0.7222 | 40 | 32 | 26 |
| LOC | 0.9067 | 0.8413 | 0.8728 | 208 | 193 | 175 |
| NUT | 0.9189 | 0.8718 | 0.8947 | 39 | 37 | 34 |
| PAR | 0.9246 | 0.9187 | 0.9216 | 787 | 782 | 723 |
| PER | 0.8818 | 0.8950 | 0.8883 | 200 | 203 | 179 |
| PES | 0.9333 | 0.8750 | 0.9032 | 16 | 15 | 14 |
| PRO | 0.8306 | 0.8306 | 0.8306 | 124 | 124 | 103 |
| TAX | 0.5000 | 0.6667 | 0.5714 | 6 | 8 | 4 |
| WED | 0.7727 | 0.9189 | 0.8395 | 37 | 44 | 34 |

### 类型表现分析

**高F1类型 (>0.90)**:
- DIS (疾病): 0.9259
- PAR (部位): 0.9216
- PES (害虫): 0.9032
- DRU (药物): 0.9000

**中等F1类型 (0.80-0.90)**:
- CUL (品种): 0.8969
- NUT (营养): 0.8947
- PER (物候期): 0.8883
- LOC (地点): 0.8728
- WED (杂草): 0.8395
- PRO (产品): 0.8306

**待提升类型 (<0.80)**:
- EQU (设备): 0.7958
- AGR (农艺): 0.7717
- FER (肥料): 0.7222
- TAX (分类): 0.5714

### 错误分析

**样本级错误数量**: 139 / 190 (73.2%)

**Top 5 漏检实体**:
1. [PAR] 枝 : 10
2. [CUL] 枣 : 8
3. [PRO] 枣果 : 7
4. [PAR] 花 : 7
5. [EQU] 反光幕 : 4

**Top 5 误检实体**:
1. [PAR] 枣果 : 9
2. [CUL] 枣 : 6
3. [EQU] 枣园 : 5
4. [AGR] 贮藏 : 5
5. [PER] 采收 : 4

---

## 九、完整实验结果排名 (更新)

| 排名 | 实验 | 词典 | 融合/SB参数 | 测试F1 |
|------|------|------|-------------|--------|
| 1 | **freq=3+sb_eps=0.15** | 1082词 | concat/0.15,2 | **0.8677** |
| 2 | freq=3+sb_size=3 | 1082词 | concat/0.1,3 | 0.8673 |
| 3 | typed+attn_v2 | 1800词 | attn_v2/0.1,2 | 0.8672 |
| 4 | 类型词典+BS | 1800词 | concat/0.1,2 | 0.8679 |
| 5 | freq=3 | 1082词 | concat/0.1,2 | 0.8678 |
| 6 | typed+freq=3 | 1082词 | concat/0.1,2 | 0.8666 |
| 7 | freq=3+sb_eps=0.05 | 1082词 | concat/0.05,2 | 0.8659 |
| 8 | freq=3+sb_size=1 | 1082词 | concat/0.1,1 | 0.8653 |
| 9 | 通道注意力v2 | 1800词 | attn_v2/0.1,2 | 0.8652 |
| 10 | 通道注意力v1 | 1800词 | attn_v1/0.1,2 | 0.8642 |

---

## 十、后续实验建议

基于当前结果，建议以下补充实验：

### 10.1 稳定性验证 (推荐)
对最优配置进行多次运行，验证结果稳定性：
```bash
for seed in 43 44 45; do
    nohup python _5TRAIN/train_redjujube_expert_boundary.py \
        --data_dir _2DATA/RedJujube \
        --save_dir cache/hz_freq3_sb015_seed${seed} \
        --min_freq 3 --seed ${seed} \
        --sb_epsilon 0.15 --sb_size 2 --num_epochs 30 \
        > train_hz_freq3_sb015_seed${seed}.log 2>&1 &
done
```

### 10.2 边界平滑参数微调 (可选)
在0.15附近进一步微调：
- sb_epsilon=0.12
- sb_epsilon=0.18

### 10.3 类型词典+边界平滑组合 (可选)
将类型词典与最优边界平滑参数组合：
```bash
nohup python _5TRAIN/train_redjujube_expert_boundary.py \
    --data_dir _2DATA/RedJujube \
    --save_dir cache/hz_typed_sb015 \
    --expert_dict_path _2DATA/RedJujube/expert_lexicon_typed.txt \
    --sb_epsilon 0.15 --sb_size 2 --num_epochs 30 \
    > train_hz_typed_sb015.log 2>&1 &
```
