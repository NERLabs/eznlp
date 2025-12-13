# 训练监控工具使用指南

本目录包含完整的训练监控和结果收集工具集。

---

## 🛠️ 可用工具

### 1. 通用训练监控脚本 `monitor_training.py`

**功能**: 实时监控GPU、进程和训练日志

**基础用法**:
```bash
# 查看当前状态（只显示有效训练）
python scripts/monitor_training.py --once

# 持续监控（30秒刷新一次）
python scripts/monitor_training.py

# 自定义刷新间隔（60秒）
python scripts/monitor_training.py --interval 60
```

**高级参数**:
```bash
# 显示所有日志（包括失败的任务）
python scripts/monitor_training.py --once --show-all

# 指定监控特定PID
python scripts/monitor_training.py --pids 12345 67890

# 指定日志目录
python scripts/monitor_training.py --log-dir cache/my_experiments

# 调整日志显示行数
python scripts/monitor_training.py --log-lines 10
```

**特性**:
- ✅ 自动过滤失败的训练任务（默认只显示有最佳模型保存的）
- ✅ 彩色输出，易于阅读
- ✅ 显示GPU使用情况、进程信息和训练进度
- ✅ 支持多种框架（PyTorch、TensorFlow等）

---

### 2. 自动结果收集脚本 `auto_collect_when_complete.py`

**功能**: 自动监控训练完成状态，完成后自动收集结果

**用法**:
```bash
# 后台运行，每10分钟检查一次
nohup python scripts/auto_collect_when_complete.py --interval 600 > auto_watcher.log 2>&1 &

# 只检查一次（不循环）
python scripts/auto_collect_when_complete.py --once

# 自定义检查间隔（5分钟）
python scripts/auto_collect_when_complete.py --interval 300
```

**工作流程**:
1. 每隔指定时间检查训练状态
2. 当所有任务完成时自动调用 `collect_fusion_results.py`
3. 生成结果对比报告

**当前状态**:
- ✅ 已启动 (PID: 1048123)
- ⏰ 检查间隔: 600秒 (10分钟)
- 📄 日志文件: `auto_watcher.log`

---

### 3. 结果收集脚本 `collect_fusion_results.py`

**功能**: 收集所有融合方案的实验结果并生成对比报告

**用法**:
```bash
# 自动查找并收集结果
python scripts/collect_fusion_results.py

# 指定基础目录
python scripts/collect_fusion_results.py --base-dir cache

# 指定输出文件
python scripts/collect_fusion_results.py --output FUSION_RESULTS.md
```

**输出**:
- 生成Markdown格式的对比报告
- 包含性能对比表、参数量统计、优化建议

---

### 4. 快速状态检查脚本 `quick_status.sh`

**功能**: 快速查看GPU和训练进度

**用法**:
```bash
bash scripts/quick_status.sh
```

**输出示例**:
```
================================
📊 快速状态检查
================================

🖥️  GPU状态:
0, NVIDIA GeForce RTX 4090, 45, 21369, 24564, 68

🔄 训练进程:
  运行中: 4 个

📋 训练进度:

🏆 方案D (Attention):
[2025-12-13 18:31:24] Epoch: 23 | Step: 3700

📌 方案A (Concat):
[2025-12-13 18:31:48] Epoch: 9 | Dev F1: 96.65%

🤖 自动监视器:
  ✅ 运行中 (PID: 1048123)
================================
```

---

## 📊 当前监控的任务

### 正在训练（3个）

| 方案 | 日志路径 | 状态 |
|------|---------|------|
| 方案D (Attention) | `cache/redjujube_ner_comparison/softlexicon_expert_attention_20251213-175441/` | 🔄 Epoch 23/30 |
| 方案A (Concat) | `cache/redjujube_softlexicon_expert_concat/softlexicon_expert_concat_20251213-181422/` | 🔄 Epoch 9/30 |
| 方案B (Weighted) | `cache/redjujube_softlexicon_expert_weighted/softlexicon_expert_weighted_20251213-181422/` | 🔄 Epoch 9/30 |

### 等待启动

- **方案C (Gated)**: 显存不足，待其他任务完成后启动

---

## 🔍 故障排查

### 问题1: 监控脚本显示太多失败的日志

**解决方案**: 默认已过滤，只显示有效训练。如需查看所有日志：
```bash
python scripts/monitor_training.py --once --show-all
```

### 问题2: 自动监视器没有运行

**检查**:
```bash
ps aux | grep auto_collect_when_complete
```

**重新启动**:
```bash
nohup python scripts/auto_collect_when_complete.py --interval 600 > auto_watcher.log 2>&1 &
```

### 问题3: 如何查看自动监视器日志

```bash
tail -f auto_watcher.log
```

---

## 💡 最佳实践

1. **日常监控**: 使用 `quick_status.sh` 快速查看进度
2. **详细检查**: 使用 `monitor_training.py --once` 查看详细信息
3. **后台监控**: 启动 `auto_collect_when_complete.py` 自动收集结果
4. **结果分析**: 训练完成后查看自动生成的 `FUSION_RESULTS.md`

---

## 📝 注意事项

- ✅ 所有脚本都支持 `--help` 参数查看详细帮助
- ✅ 监控脚本不会影响训练性能（轻量级）
- ✅ 自动监视器会在所有任务完成后自动退出
- ✅ 建议定期检查 `auto_watcher.log` 确保监视器正常运行

---

**更新时间**: 2025-12-13 18:32
**维护者**: eznlp 项目组
