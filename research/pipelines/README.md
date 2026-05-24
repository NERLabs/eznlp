# 实验自动化流水线系统

完整的实验生命周期管理，从参数配置到结果报告的全自动化流程。

## 📋 功能特性

实验自动化流水线包含以下7个阶段：

1. **参数修改** - 应用参数覆盖，保存实验配置
2. **测试运行** - 1个epoch快速验证配置正确性
3. **后台正式运行** - 启动完整训练并后台运行
4. **资源监控** - 实时监控CPU、内存、GPU使用情况
5. **结果对比** - 自动收集同数据集所有实验结果并对比
6. **生成报告** - 生成详细的Markdown格式实验报告
7. **邮件通知** - 完成后自动发送邮件通知

## 🚀 快速开始

### 基础用法

```bash
cd /home/shiwenlong/NERlabs/eznlp

# 使用配置文件运行流水线
python research/pipelines/experiment_pipeline.py \
  --name "我的实验" \
  --config research/pipelines/config_templates/baseline_example.json
```

### 完整用法（含邮件通知）

```bash
python research/pipelines/experiment_pipeline.py \
  --name "RedJujube_Baseline" \
  --config research/pipelines/config_templates/baseline_example.json \
  --email-sender "your_email@163.com" \
  --email-receiver "target@example.com" \
  --email-password "your_smtp_password" \
  --smtp-server "smtp.163.com" \
  --smtp-port 465
```

### 跳过测试运行

如果配置已验证，可跳过测试阶段：

```bash
python research/pipelines/experiment_pipeline.py \
  --name "快速实验" \
  --config my_config.json \
  --skip-test
```

### 调整监控间隔

```bash
python research/pipelines/experiment_pipeline.py \
  --name "实验" \
  --config config.json \
  --monitor-interval 600  # 每10分钟检查一次
```

## 📁 配置文件格式

### Baseline实验配置

```json
{
  "experiment_name": "RedJujube_Baseline",
  "dataset": "RedJujube",
  "script": "python research/configs/redjujube/train_redjujube_ner_comparison.py",
  "data_dir": "datasets/raw/RedJujube",
  "save_dir": "cache/pipeline_test/baseline",
  "bert_arch": "hfl/chinese-macbert-base",
  "hid_dim": 256,
  "num_layers": 1,
  "dropout": 0.5,
  "num_epochs": 30,
  "batch_size": 16,
  "lr": 0.002,
  "finetune_lr": 0.00002,
  "run_baseline": true
}
```

### SoftLexicon实验配置

```json
{
  "experiment_name": "RedJujube_SoftLexicon",
  "dataset": "RedJujube",
  "script": "python research/configs/redjujube/train_redjujube_ner_comparison.py",
  "data_dir": "datasets/raw/RedJujube",
  "softlex_train_path": "datasets/raw/RedJujube/softlexicon_train.txt",
  "save_dir": "cache/pipeline_test/softlexicon",
  "run_softlexicon": true,
  ...
}
```

### ExpertDict实验配置

```json
{
  "experiment_name": "RedJujube_ExpertDict",
  "dataset": "RedJujube",
  "script": "python research/configs/redjujube/train_redjujube_ner_comparison.py",
  "data_dir": "datasets/raw/RedJujube",
  "expert_dict_auto_path": "datasets/raw/RedJujube/expert_lexicon_auto.txt",
  "save_dir": "cache/pipeline_test/expert_dict",
  "expert_dict_dim": 50,
  "run_expert_dict_auto": true,
  ...
}
```

更多配置示例见 [`config_templates/`](config_templates/) 目录。

## 📊 流水线执行流程

```
┌─────────────────────┐
│  1. 参数修改         │ 应用参数覆盖，保存配置
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  2. 测试运行         │ 1 epoch快速验证（可跳过）
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  3. 后台正式运行      │ 启动完整训练任务
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  4. 资源监控         │ 监控CPU/内存/GPU/进度
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  5. 结果对比         │ 收集同数据集所有结果
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  6. 生成报告         │ 生成Markdown报告
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  7. 邮件通知         │ 发送完成通知（可选）
└─────────────────────┘
```

## 📂 输出文件结构

每次运行会在 `pipeline_runs/` 下创建独立的工作目录：

```
pipeline_runs/
└── 实验名称_20251214_143022/
    ├── pipeline.log              # 流水线执行日志
    ├── pipeline_config.json      # 实验配置（含覆盖）
    ├── test_run/                 # 测试运行输出
    ├── training_output.log       # 训练进程输出
    ├── comparison_results.json   # 结果对比数据
    └── experiment_report.md      # 实验报告
```

## 📧 邮件配置

### QQ邮箱配置

1. 登录QQ邮箱网页版
2. 设置 → 账户 → POP3/IMAP/SMTP服务
3. 开启"POP3/SMTP服务"并生成授权码（16位）
4. 使用授权码作为密码（不是邮箱登录密码）

```bash
--email-sender "your@qq.com"
--email-password "abcd1234efgh5678"  # 授权码
--smtp-server "smtp.qq.com"
--smtp-port 465
```

### 163邮箱配置

```bash
--email-sender "your@163.com"
--email-password "your_auth_code"
--smtp-server "smtp.163.com"
--smtp-port 465
```

### Outlook邮箱配置

```bash
--email-sender "your@outlook.com"
--email-password "your_password"
--smtp-server "smtp-mail.outlook.com"
--smtp-port 587
```

## 🔧 高级用法

### 批量运行多个实验

创建批量运行脚本 `run_multiple.sh`：

```bash
#!/bin/bash

# Baseline
python research/pipelines/experiment_pipeline.py \
  --name "RedJujube_Baseline" \
  --config research/pipelines/config_templates/baseline_example.json &

# SoftLexicon
python research/pipelines/experiment_pipeline.py \
  --name "RedJujube_SoftLexicon" \
  --config research/pipelines/config_templates/softlexicon_example.json &

# ExpertDict
python research/pipelines/experiment_pipeline.py \
  --name "RedJujube_ExpertDict" \
  --config research/pipelines/config_templates/expert_dict_example.json &

wait
echo "所有实验完成！"
```

### 与Taskfile集成

在 `Taskfile.yml` 中添加流水线任务：

```yaml
tasks:
  pipeline:baseline:
    desc: 运行Baseline流水线
    cmds:
      - python research/pipelines/experiment_pipeline.py
          --name "RedJujube_Baseline"
          --config research/pipelines/config_templates/baseline_example.json

  pipeline:all:
    desc: 运行所有流水线实验
    cmds:
      - bash research/pipelines/run_all_pipelines.sh
```

### 自定义阶段跳过

修改 `experiment_pipeline.py` 中的 `__init__` 方法：

```python
# 跳过某些阶段
self.stages = [
    ParameterModificationStage(self),
    # TestRunStage(self),  # 跳过测试
    BackgroundTrainingStage(self),
    ResourceMonitoringStage(self),
    ResultComparisonStage(self),
    ReportGenerationStage(self),
    # EmailNotificationStage(self)  # 跳过邮件
]
```

## 📊 监控与日志

### 实时查看流水线日志

```bash
tail -f pipeline_runs/实验名称_*/pipeline.log
```

### 实时查看训练输出

```bash
tail -f pipeline_runs/实验名称_*/training_output.log
```

### 查看GPU使用情况

```bash
watch -n 1 nvidia-smi
```

### 检查进程状态

```bash
ps aux | grep experiment_pipeline
```

## ⚠️ 注意事项

1. **资源管理**
   - 确保有足够的GPU显存运行实验
   - 监控间隔不要设置太短（建议≥300秒）
   - 批量运行时注意资源冲突

2. **配置文件**
   - 确保所有路径正确（数据集、词典等）
   - 布尔参数使用 `true`/`false`（小写）
   - 必须包含 `script` 字段指定训练脚本

3. **邮件通知**
   - SMTP授权码不是邮箱登录密码
   - 首次发送可能需要验证
   - 建议使用专门的监控邮箱

4. **测试运行**
   - 首次使用建议开启测试（不跳过）
   - 测试阶段会运行1个epoch验证配置
   - 配置验证后可使用 `--skip-test` 加速

## 🛠️ 故障排查

### Q1: 流水线启动失败

**检查项**:
- 配置文件格式是否正确（JSON）
- 训练脚本路径是否存在
- 数据集路径是否正确

### Q2: 测试运行超时

**解决方案**:
- 检查数据加载是否正常
- 确认GPU可用性
- 使用 `--skip-test` 跳过测试阶段

### Q3: 邮件发送失败

**检查项**:
- SMTP授权码是否正确
- 服务器和端口是否匹配
- 网络连接是否正常

### Q4: 资源监控无数据

**原因**:
- 训练进程PID未正确获取
- 日志文件路径不匹配

**解决方案**:
- 检查 `training_output.log`
- 确认 `save_dir` 配置正确

## 📝 示例场景

### 场景1: 单个实验完整流程

```bash
python research/pipelines/experiment_pipeline.py \
  --name "RedJujube_Baseline_20251214" \
  --config research/pipelines/config_templates/baseline_example.json \
  --email-sender "monitor@163.com" \
  --email-receiver "researcher@example.com" \
  --email-password "smtp_auth_code"
```

### 场景2: 快速验证（跳过测试和邮件）

```bash
python research/pipelines/experiment_pipeline.py \
  --name "Quick_Test" \
  --config my_config.json \
  --skip-test
```

### 场景3: 长时间实验（延长监控间隔）

```bash
python research/pipelines/experiment_pipeline.py \
  --name "Long_Running_Exp" \
  --config config.json \
  --monitor-interval 1800  # 每30分钟检查
```

## 🔗 相关文档

- [Taskfile任务编排](../Taskfile.yml)
- [监控系统文档](../tools/monitoring/MONITORING_TOOLS.md)
- [邮件通知系统](../_4MONITORING/README.md)
- [结果收集脚本](../evaluation/format_redjujube_results.py)

## 📅 更新日志

- **2025-12-14**: 初始版本
  - 7阶段自动化流水线
  - 完整生命周期管理
  - 邮件通知集成
  - 资源监控功能

---

**维护**: eznlp项目组
**最后更新**: 2025-12-14
