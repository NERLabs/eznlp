# Soft+Expert 联合模型训练状态报告

**更新时间**: 2025-12-13 18:20

## 📊 当前训练进度

### ✅ 已完成
- **方案A (Concat) - 首次运行**: 
  - 状态: ✅ 完成
  - 测试集 F1: **96.87%**
  - 训练时长: ~47分钟
  - 结果文件: `cache/redjujube_softlexicon_expert/softlexicon_expert_concat_20251213-172348/`
  - **结论**: 性能低于预期（目标>97.04%），可能需要调参

### 🔄 正在训练中

1. **方案D (Attention)**
   - 状态: 🔄 训练中
   - 当前进度: Epoch 15/30 (50%)
   - 最新指标: Dev F1 96.38%
   - 运行时长: 25分钟
   - PID: 1033338
   - 日志: `cache/redjujube_ner_comparison/softlexicon_expert_attention_20251213-175441/`

2. **方案B (Weighted)**
   - 状态: 🔄 训练中
   - 当前进度: Epoch 3/30 (10%)
   - 最新指标: Dev F1 93.91%
   - 运行时长: 5分钟
   - PID: 1041563
   - 日志: `cache/redjujube_softlexicon_expert_weighted/softlexicon_expert_weighted_20251213-181422/`

3. **方案A (Concat) - 重新训练**
   - 状态: 🔄 训练中
   - 当前进度: Epoch 3/30 (10%)
   - 最新指标: Dev F1 89.25%
   - 运行时长: 5分钟
   - PID: 1041555
   - 日志: `cache/redjujube_softlexicon_expert_concat/softlexicon_expert_concat_20251213-181422/`

### ⏸️ 待启动
- **方案C (Gated)**: 等待GPU资源释放后启动

## 🖥️ 资源使用情况

- **GPU 型号**: NVIDIA GeForce RTX 4090
- **GPU 利用率**: 98%
- **显存使用**: 21,369 / 24,564 MiB (87.0%)
- **温度**: 66°C
- **运行进程数**: 4个训练任务

## 📝 下一步行动

### 短期（1-2小时）
1. ✅ 监控当前3个训练任务完成情况
2. ⏳ 等待方案D完成（预计剩余15-20分钟）
3. ⏳ 等待方案B完成（预计剩余40-50分钟）
4. ⏳ 等待方案A重训练完成（预计剩余40-50分钟）
5. 🔜 启动方案C (Gated) 训练

### 中期（2-4小时）
1. 等待所有4个方案完成训练
2. 收集和对比实验结果
3. 分析性能差异原因
4. 生成实验对比报告

### 长期（后续工作）
1. 消融实验：分析SoftLex和Expert各自贡献
2. 实体类型详细分析
3. 超参数调优（如果性能未达标）
4. 更新项目文档

## 🛠️ 可用工具

### 监控脚本
```bash
# 实时监控（持续刷新，30秒间隔）
python scripts/monitor_training.py

# 快速查看一次
python scripts/monitor_training.py --once

# 自定义刷新间隔
python scripts/monitor_training.py --interval 60 --log-lines 5
```

### 查看特定日志
```bash
# 方案D
tail -f cache/redjujube_ner_comparison/softlexicon_expert_attention_20251213-175441/training.log

# 方案B
tail -f cache/redjujube_softlexicon_expert_weighted/softlexicon_expert_weighted_20251213-181422/training.log

# 方案A重训练
tail -f cache/redjujube_softlexicon_expert_concat/softlexicon_expert_concat_20251213-181422/training.log
```

### 检查GPU状态
```bash
nvidia-smi
watch -n 5 nvidia-smi
```

## 📌 注意事项

1. **GPU显存限制**: 当前使用87%，接近上限，方案C需等待资源释放
2. **方案A性能**: 首次训练结果96.87%低于预期，重训练可能需要调整超参数
3. **训练时长**: 每个方案预计需要45-60分钟完成30个epoch
4. **数据集**: 使用RedJujube数据集（HZ的更新版本）

## 🎯 成功标准

- 至少一个方案达到测试集 F1 > 97.04%（超越单独ExpertDict的96.99%）
- 验证SoftLex + Expert联合特征的互补性
- 识别最优融合策略

## 📊 预期结果对比

| 方案 | 融合方式 | 新增参数 | 预期F1 | 实际F1 | 状态 |
|------|---------|---------|--------|--------|------|
| Baseline | - | 0 | 95.51% | - | 参考 |
| ExpertDict | - | 0 | 96.99% | - | 参考 |
| 方案A (Concat) | 直接拼接 | 0 | >97.0% | 96.87% | ⚠️ 低于预期 |
| 方案B (Weighted) | 加权求和 | ~10 | >97.0% | 训练中 | 🔄 |
| 方案C (Gated) | 门控机制 | ~500 | >97.0% | 待启动 | ⏸️ |
| 方案D (Attention) | 注意力融合 | ~1500 | >97.0% | 训练中 | 🔄 |

---

**最后更新**: 2025-12-13 18:20:00
**监控脚本**: `scripts/monitor_training.py`
**训练日志目录**: `cache/`
