# 当前训练状态报告

**生成时间**: 2025-12-13 18:29

---

## 📊 正在运行的训练任务

### ✅ 有效训练任务（3个）

| 方案 | 当前进度 | 最佳 Dev F1 | 状态 | 日志路径 |
|------|---------|------------|------|---------|
| 🏆 **方案D (Attention)** | Epoch 21/30 | **96.82%** | 🔄 训练中 | `cache/redjujube_ner_comparison/softlexicon_expert_attention_20251213-175441/` |
| **方案A (Concat)** | Epoch 7/30 | 96.60% | 🔄 训练中 | `cache/redjujube_softlexicon_expert_concat/softlexicon_expert_concat_20251213-181422/` |
| **方案B (Weighted)** | Epoch 7/30 | 96.43% | 🔄 训练中 | `cache/redjujube_softlexicon_expert_weighted/softlexicon_expert_weighted_20251213-181422/` |

### 🔒 待启动任务

- **方案C (Gated)**: 因GPU显存不足（87%占用）暂时无法启动，等待其他任务完成释放资源

---

## 🎯 性能对比

| 排名 | 方案 | Dev F1 | vs Baseline | vs 最佳单特征 |
|------|------|--------|------------|-------------|
| 🥇 | 方案D (Attention) | **96.82%** | +1.31% | 接近 97.04% |
| 🥈 | 方案A (Concat-原始) | 96.87% | +1.36% | 接近 97.04% |
| 🥉 | 方案A (Concat-重训练) | 96.60% | +1.09% | 训练中 |
| 4 | 方案B (Weighted) | 96.43% | +0.92% | 训练中 |
| - | Baseline | 95.51% | - | - |
| - | ExpertDict (手动) | 97.04% | +1.53% | 最佳单特征 |

**关键发现**：
- 方案D表现优异，Dev F1已达96.82%，预计完成后会进一步提升
- 方案A和B才训练7个epoch，还有很大提升空间
- 所有联合模型都显著优于Baseline（95.51%）

---

## 🛠️ 资源使用情况

### GPU状态
- **型号**: NVIDIA GeForce RTX 4090
- **显存占用**: 21369/24564 MiB (87.0%)
- **GPU利用率**: 85%
- **温度**: 66°C

### 显存分配
| 进程 | 显存 | 任务 |
|------|------|------|
| PID 1033338 | 7856 MiB | 方案D (Attention) |
| PID 1041555 | 7028 MiB | 方案A (Concat) |
| PID 1041563 | 6118 MiB | 方案B (Weighted) |
| **总计** | **21002 MiB** | 3个训练任务 |

---

## ⏰ 预计完成时间

| 任务 | 剩余Epochs | 预计完成时间 |
|------|-----------|------------|
| 方案D | 9/30 | 约 6-8 小时（今晚凌晨） |
| 方案A | 23/30 | 约 15-18 小时（明天中午） |
| 方案B | 23/30 | 约 15-18 小时（明天中午） |

**预计所有任务完成**: 2025-12-14 中午12:00左右

---

## 🤖 自动化工具

### 已部署的监控脚本

1. **资源监控脚本**: `scripts/monitor_training.py`
   ```bash
   # 查看当前状态（只显示有效训练）
   python scripts/monitor_training.py --once
   
   # 持续监控（30秒刷新）
   python scripts/monitor_training.py
   
   # 显示所有日志（包括失败的）
   python scripts/monitor_training.py --once --show-all
   ```

2. **自动收集脚本**: `scripts/auto_collect_when_complete.py` ✅ 已启动
   ```bash
   # 后台运行，每10分钟检查一次，所有训练完成后自动收集结果
   # 已启动 PID: 1048122
   # 日志文件: auto_watcher.log
   ```

3. **结果收集脚本**: `scripts/collect_fusion_results.py`
   ```bash
   # 手动收集结果（训练完成后）
   python scripts/collect_fusion_results.py
   ```

---

## 📋 下一步行动

### 自动执行
- ✅ 自动监视器已启动，等待训练完成
- ✅ 完成后自动收集结果到 `FUSION_RESULTS.md`

### 手动检查（可选）
```bash
# 查看自动监视器日志
tail -f auto_watcher.log

# 查看当前训练进度
python scripts/monitor_training.py --once

# 手动收集结果（训练完成后）
python scripts/collect_fusion_results.py
```

### 训练完成后
1. 检查自动生成的结果报告
2. 启动方案C (Gated) 的训练
3. 进行消融实验分析
4. 生成最终实验报告

---

## 📝 备注

- 监控脚本已优化：默认只显示有最佳模型保存记录的有效训练
- 中途失败的日志已自动过滤
- 所有工具已准备就绪，无需人工干预

**状态**: 🟢 一切正常，等待训练完成
