# 通用训练监控脚本使用指南

## 功能特性

`monitor_training.py` 是一个通用的深度学习训练任务监控脚本，适用于所有PyTorch/TensorFlow训练任务。

### 核心功能

1. **实时GPU监控**
   - GPU利用率、显存使用、温度
   - 彩色状态指示（绿色/黄色/红色）
   - 多GPU支持

2. **进程状态跟踪**
   - 自动检测训练进程
   - 显示运行时间、GPU显存占用
   - 支持指定PID监控

3. **日志智能分析**
   - 自动解析训练进度（Epoch、Step、Loss、Metrics）
   - 显示最近日志行（格式：@training.log 行号范围）
   - 检测训练状态（运行中/已完成/错误）

4. **异常检测**
   - 自动识别错误日志
   - 高亮显示警告信息

## 使用方法

### 基础用法

```bash
# 1. 监控所有训练任务（持续监控，每30秒刷新）
python scripts/monitor_training.py

# 2. 只运行一次，不持续监控
python scripts/monitor_training.py --once

# 3. 自定义刷新间隔（每10秒）
python scripts/monitor_training.py --interval 10

# 4. 指定日志目录
python scripts/monitor_training.py --log-dir cache

# 5. 监控指定进程
python scripts/monitor_training.py --pids 1018542 1033338

# 6. 显示更多日志行
python scripts/monitor_training.py --log-lines 10
```

### 高级用法

```bash
# 组合参数：监控指定进程，每15秒刷新，显示8行日志
python scripts/monitor_training.py \
  --pids 12345 67890 \
  --interval 15 \
  --log-lines 8 \
  --log-dir experiments/cache

# 后台持续监控并保存输出
nohup python scripts/monitor_training.py --interval 60 > monitor.log 2>&1 &
```

## 输出示例

```
======================================================================
📊 训练任务监控 - 2025-12-13 18:11:43
======================================================================

======================================================================
🖥️  GPU 状态
======================================================================

GPU 0: NVIDIA GeForce RTX 4090
  利用率:  23%  显存: 15251/24564 MiB (62.1%)  温度: 62°C

======================================================================
🔄 运行中的进程 (2)
======================================================================

PID 1018542 | 运行时间: 47:58 | GPU显存: 7028 MiB
  命令: python scripts/train_redjujube_ner_comparison.py ...

PID 1033338 | 运行时间: 17:06 | GPU显存: 7856 MiB
  命令: python scripts/train_redjujube_ner_comparison.py ...

======================================================================
📋 训练日志 (2)
======================================================================

🟢 cache/redjujube_ner_comparison/training.log
  Epoch 11 | Step 1800 | Loss 3.042 | Train 99.23%
  @training.log 116-120
    [2025-12-13 18:11:14] Train Loss: 3.364 | Train Metrics: 99.13%
    [2025-12-13 18:11:39] Epoch: 11 | Step: 1800
    [2025-12-13 18:11:39] Train Loss: 3.042 | Train Metrics: 99.23%

🟢 cache/redjujube_softlexicon_expert/training.log
  Epoch 29 | Step 9600 | Loss 0.540 | Train 99.95% | Dev 97.07%
  @training.log 479-483
    [2025-12-13 18:11:29] Epoch: 29 | Step: 9600
    [2025-12-13 18:11:29] Train Loss: 0.540 | Train Metrics: 99.95%
    [2025-12-13 18:11:38] Dev Loss: 7.132 | Dev Metrics: 97.07%
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--pids` | int列表 | 自动检测 | 指定要监控的进程PID |
| `--log-dir` | str | cache | 训练日志目录 |
| `--interval` | int | 30 | 刷新间隔（秒） |
| `--log-lines` | int | 5 | 每个日志显示的行数 |
| `--once` | flag | False | 只运行一次，不持续监控 |

## 状态图标说明

- 🟢 绿色圆点：训练正在进行
- ✅ 绿色对勾：训练已完成
- ❌ 红色叉号：训练出现错误
- ⚠️ 黄色警告：日志为空或异常

## 适用场景

1. **多任务并行训练**：同时监控多个训练任务的进度
2. **长时间训练**：定期检查训练状态，避免频繁查看日志
3. **服务器共享**：查看GPU资源使用情况
4. **调试阶段**：快速定位错误日志
5. **实验对比**：并排查看多个实验的训练进度

## 注意事项

1. 默认监控 `cache` 目录下的所有 `training.log` 文件
2. 日志文件需包含标准的训练输出格式（Epoch、Step、Loss等）
3. 需要安装 `nvidia-smi`（用于GPU监控）
4. 使用 `Ctrl+C` 停止持续监控

## 集成到训练脚本

可以在训练脚本中添加监控功能：

```python
import subprocess
import threading

def start_monitor():
    """后台启动监控脚本"""
    subprocess.Popen([
        'python', 'scripts/monitor_training.py',
        '--log-dir', 'cache',
        '--interval', '60'
    ])

# 在训练开始时调用
# start_monitor()
```

## 未来扩展

- [ ] 支持TensorBoard集成
- [ ] 添加邮件/钉钉通知
- [ ] 支持远程监控（HTTP API）
- [ ] 添加性能分析功能
- [ ] 支持训练曲线可视化
