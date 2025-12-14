# 实验自动化流水线 - 快速入门

## 5分钟上手

### 方式1: 使用Taskfile（推荐）

```bash
cd /home/shiwenlong/NERlabs/eznlp

# 运行Baseline实验
task pipeline:baseline

# 运行SoftLexicon实验
task pipeline:softlexicon

# 运行ExpertDict实验
task pipeline:expert-dict

# 批量运行所有实验
task pipeline:all
```

### 方式2: 直接使用脚本

```bash
cd /home/shiwenlong/NERlabs/eznlp

# 使用预设配置
bash _7PIPELINE/run_pipeline.sh --name "我的实验" --baseline

# 使用自定义配置
bash _7PIPELINE/run_pipeline.sh \
  --name "自定义实验" \
  --config my_config.json
```

### 方式3: 直接调用Python

```bash
python _7PIPELINE/experiment_pipeline.py \
  --name "测试实验" \
  --config _7PIPELINE/config_templates/baseline_example.json
```

## 完整示例

### 示例1: 单个实验（含邮件通知）

```bash
bash _7PIPELINE/run_pipeline.sh \
  --name "RedJujube_Baseline_20251214" \
  --baseline \
  --email-sender "your@163.com" \
  --email-receiver "target@example.com" \
  --email-password "your_smtp_auth_code"
```

### 示例2: 跳过测试运行（加速）

```bash
bash _7PIPELINE/run_pipeline.sh \
  --name "快速实验" \
  --baseline \
  --skip-test
```

### 示例3: 批量运行多个实验

编辑 `_7PIPELINE/run_all_pipelines.sh` 中的邮件配置，然后：

```bash
bash _7PIPELINE/run_all_pipelines.sh
```

## 流水线阶段说明

每个实验会自动执行以下7个阶段：

| 阶段 | 功能 | 耗时估计 |
|-----|------|---------|
| 1. 参数修改 | 应用参数覆盖，保存配置 | < 1秒 |
| 2. 测试运行 | 1个epoch验证配置正确性 | 2-5分钟 |
| 3. 后台正式运行 | 启动完整训练任务 | 主要时长 |
| 4. 资源监控 | 监控CPU/内存/GPU使用 | 与训练同步 |
| 5. 结果对比 | 收集同数据集所有结果 | 1-2秒 |
| 6. 生成报告 | 生成Markdown格式报告 | < 1秒 |
| 7. 邮件通知 | 发送完成通知邮件 | 2-3秒 |

## 查看结果

### 实时查看流水线日志

```bash
# 查找最新的流水线运行
ls -lt pipeline_runs/

# 查看流水线日志
tail -f pipeline_runs/实验名称_*/pipeline.log
```

### 实时查看训练输出

```bash
tail -f pipeline_runs/实验名称_*/training_output.log
```

### 查看实验报告

```bash
cat pipeline_runs/实验名称_*/experiment_report.md
```

### 查看结果对比

```bash
cat pipeline_runs/实验名称_*/comparison_results.json
```

## 常见问题

### Q: 如何修改实验参数？

**方法1**: 修改配置模板
```bash
# 复制模板
cp _7PIPELINE/config_templates/baseline_example.json my_config.json

# 编辑参数
vim my_config.json

# 使用自定义配置
bash _7PIPELINE/run_pipeline.sh --name "实验" --config my_config.json
```

**方法2**: 程序化覆盖
```python
# 在代码中使用param_overrides
pipeline = ExperimentPipeline(
    exp_name="实验",
    exp_config=base_config,
    param_overrides={'num_epochs': 50, 'lr': 0.001}
)
```

### Q: 如何跳过某些阶段？

修改配置：
```bash
# 跳过测试运行
bash _7PIPELINE/run_pipeline.sh --name "实验" --config config.json --skip-test
```

或在代码中自定义阶段列表（见README.md高级用法）。

### Q: 如何配置邮件通知？

163邮箱示例：
```bash
bash _7PIPELINE/run_pipeline.sh \
  --name "实验" \
  --baseline \
  --email-sender "your@163.com" \
  --email-receiver "target@example.com" \
  --email-password "SMTP授权码" \
  --smtp-server "smtp.163.com" \
  --smtp-port 465
```

**重要**: 密码是SMTP授权码，不是邮箱登录密码！

### Q: 如何监控多个并行实验？

```bash
# 启动批量实验
bash _7PIPELINE/run_all_pipelines.sh

# 监控所有实验
watch -n 10 'ps aux | grep experiment_pipeline'

# 查看GPU使用
watch -n 1 nvidia-smi
```

### Q: 实验失败了怎么办？

1. 查看流水线日志找到失败阶段
```bash
cat pipeline_runs/实验名称_*/pipeline.log
```

2. 查看训练输出日志
```bash
cat pipeline_runs/实验名称_*/training_output.log
```

3. 查看实验报告中的错误信息
```bash
cat pipeline_runs/实验名称_*/experiment_report.md
```

## 下一步

- 阅读 [完整文档](README.md)
- 查看 [配置示例](config_templates/)
- 了解 [高级用法](README.md#高级用法)
- 集成到 [CI/CD流程](README.md#与taskfile集成)

## 技巧与最佳实践

### 1. 快速验证配置

首次使用新配置时，建议：
```bash
# 不跳过测试，验证配置正确性
bash _7PIPELINE/run_pipeline.sh --name "测试" --config new_config.json
```

### 2. 长时间实验

对于需要运行很久的实验：
```bash
# 延长监控间隔减少开销
bash _7PIPELINE/run_pipeline.sh \
  --name "长实验" \
  --config config.json \
  --monitor-interval 1800  # 每30分钟检查
```

### 3. 资源受限环境

避免同时运行太多实验：
```bash
# 顺序运行而非并行
for config in baseline softlexicon expert_dict; do
    bash _7PIPELINE/run_pipeline.sh --name "Exp_$config" --$config
done
```

### 4. 调试模式

快速定位问题：
```bash
# 使用少量epoch测试
# 在config中设置 "num_epochs": 2

# 查看实时输出而不后台运行
python _7PIPELINE/experiment_pipeline.py \
  --name "Debug" \
  --config debug_config.json
```

## 示例工作流

完整的研究工作流示例：

```bash
# 1. 准备数据和词典
python _3DATA_PROCESS/extract_lexicon_from_training.py

# 2. 运行baseline获取基准
task pipeline:baseline

# 3. 等待完成后运行其他方法
task pipeline:softlexicon
task pipeline:expert-dict

# 4. 查看对比报告
cat pipeline_runs/*/experiment_report.md

# 5. 根据结果调整参数，重新运行
bash _7PIPELINE/run_pipeline.sh \
  --name "Optimized" \
  --config optimized_config.json
```

---

**快速链接**:
- [详细文档](README.md)
- [Taskfile任务](../Taskfile.yml)
- [配置模板](config_templates/)
- [故障排查](README.md#故障排查)
