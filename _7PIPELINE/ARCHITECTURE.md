# 实验自动化流水线 - 系统架构

## 系统概览

实验自动化流水线是一个完整的实验生命周期管理系统，整合了参数管理、训练执行、资源监控、结果分析和通知报告等功能。

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    实验自动化流水线系统                       │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
   │  配置层  │         │  执行层  │        │  监控层  │
   └────┬────┘         └────┬────┘        └────┬────┘
        │                   │                   │
   ┌────▼────────┐     ┌────▼────────┐    ┌────▼────────┐
   │ 参数管理     │     │ 训练调度     │    │ 资源监控     │
   │ 配置验证     │     │ 进程管理     │    │ 进度追踪     │
   └─────────────┘     └─────────────┘    └─────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
   │  分析层  │         │  报告层  │        │  通知层  │
   └────┬────┘         └────┬────┘        └────┬────┘
        │                   │                   │
   ┌────▼────────┐     ┌────▼────────┐    ┌────▼────────┐
   │ 结果对比     │     │ 报告生成     │    │ 邮件通知     │
   │ 性能分析     │     │ 文档输出     │    │ 状态反馈     │
   └─────────────┘     └─────────────┘    └─────────────┘
```

## 核心组件

### 1. PipelineStage（流水线阶段基类）

所有阶段的抽象基类，提供统一的生命周期管理。

**关键方法**:
- `execute()`: 阶段执行逻辑（需子类实现）
- `on_start()`: 阶段开始钩子
- `on_success()`: 阶段成功钩子
- `on_failure()`: 阶段失败钩子
- `log()`: 统一日志记录

**状态管理**:
- PENDING: 等待执行
- RUNNING: 正在执行
- SUCCESS: 执行成功
- FAILED: 执行失败
- SKIPPED: 已跳过

### 2. ExperimentPipeline（流水线编排器）

流水线的核心控制器，负责阶段编排和执行。

**主要职责**:
- 初始化各个阶段
- 顺序执行阶段
- 错误处理和恢复
- 全局状态管理
- 日志聚合

### 3. 七大执行阶段

#### 阶段1: ParameterModificationStage
- **功能**: 参数覆盖和配置保存
- **输入**: 基础配置 + 参数覆盖
- **输出**: `pipeline_config.json`
- **耗时**: < 1秒

#### 阶段2: TestRunStage
- **功能**: 快速验证配置正确性
- **执行**: 1个epoch的训练
- **目的**: 提前发现配置错误
- **耗时**: 2-5分钟
- **可跳过**: 是

#### 阶段3: BackgroundTrainingStage
- **功能**: 启动完整训练任务
- **执行**: 后台subprocess
- **监控**: 保存PID用于后续监控
- **输出**: `training_output.log`
- **耗时**: 根据num_epochs

#### 阶段4: ResourceMonitoringStage
- **功能**: 实时监控资源使用
- **监控项**: 
  - CPU使用率（psutil）
  - 内存使用（psutil）
  - GPU显存（nvidia-smi）
  - 训练进度（日志解析）
- **检查间隔**: 可配置（默认300秒）
- **退出条件**: 训练进程结束

#### 阶段5: ResultComparisonStage
- **功能**: 收集并对比实验结果
- **收集范围**: 同数据集所有实验
- **对比维度**: F1、参数量、模型类型
- **输出**: `comparison_results.json`
- **排序**: 按F1降序

#### 阶段6: ReportGenerationStage
- **功能**: 生成Markdown实验报告
- **内容**: 
  - 流水线执行概况
  - 实验配置详情
  - 结果对比表格
  - 性能排名
- **输出**: `experiment_report.md`

#### 阶段7: EmailNotificationStage
- **功能**: 发送邮件通知
- **支持**: SMTP/SMTP_SSL
- **格式**: HTML富文本
- **内容**: 
  - 最佳F1分数
  - 各阶段状态
  - 报告文件路径
- **可选**: 是

## 数据流

```
配置文件(JSON)
    │
    ├──> 参数修改阶段 ──> pipeline_config.json
    │
    └──> 测试运行阶段 ──> 验证配置有效性
              │
              └──> 后台训练阶段 ──> training_output.log
                        │                    │
                        │                    ▼
                        │           资源监控阶段
                        │           (实时监控)
                        │                    │
                        └────────────────────┘
                                    │
                                    ▼
                            结果对比阶段 ──> comparison_results.json
                                    │
                                    ▼
                            报告生成阶段 ──> experiment_report.md
                                    │
                                    ▼
                            邮件通知阶段 ──> 发送邮件
```

## 目录结构

```
_7PIPELINE/
├── experiment_pipeline.py       # 核心流水线实现
├── run_pipeline.sh             # 快速启动脚本
├── run_all_pipelines.sh        # 批量运行脚本
├── README.md                   # 完整文档
├── QUICKSTART.md               # 快速入门
├── ARCHITECTURE.md             # 本文档
└── config_templates/           # 配置模板目录
    ├── baseline_example.json
    ├── softlexicon_example.json
    └── expert_dict_example.json
```

## 工作流程详解

### 1. 初始化阶段

```python
pipeline = ExperimentPipeline(
    exp_name="实验名称",
    exp_config=config_dict,
    work_dir="pipeline_runs/...",
    email_config=email_settings,
    skip_test=False,
    monitor_interval=300
)
```

**创建内容**:
- 工作目录
- 日志文件
- 阶段实例列表
- 状态变量

### 2. 执行阶段

```python
for stage in self.stages:
    success = stage.execute()
    if not success and stage.status == "FAILED":
        break  # 失败则终止流水线
```

**执行逻辑**:
- 顺序执行各阶段
- 记录时间和状态
- 失败时立即终止
- 跳过的阶段继续执行

### 3. 状态反馈

每个阶段都会：
- 打印进度到控制台
- 写入日志到文件
- 更新阶段状态
- 记录时间戳

### 4. 资源管理

**进程管理**:
```python
# 后台启动训练
process = subprocess.Popen(cmd, stdout=log_file)
pid = process.pid

# 监控进程
while psutil.pid_exists(pid):
    monitor_resources(pid)
```

**资源清理**:
- 训练进程正常结束
- 临时文件保留（用于调试）
- 工作目录完整保存

## 扩展性设计

### 添加新阶段

```python
class CustomStage(PipelineStage):
    def __init__(self, pipeline):
        super().__init__("自定义阶段", pipeline)
    
    def execute(self) -> bool:
        self.on_start()
        try:
            # 执行自定义逻辑
            self.on_success()
            return True
        except Exception as e:
            self.on_failure(str(e))
            return False

# 在ExperimentPipeline.__init__中添加
self.stages.append(CustomStage(self))
```

### 自定义报告格式

子类化 `ReportGenerationStage` 并覆盖 `_generate_report()` 方法。

### 支持新的邮件服务

修改 `EmailNotificationStage._send_email()` 添加新的SMTP配置。

## 性能优化

### 1. 测试阶段优化

- 可选跳过测试运行
- 测试时使用最小epoch数
- 使用小batch_size加速

### 2. 监控开销控制

- 可配置检查间隔
- 轻量级资源查询
- 异步日志写入

### 3. 并行执行

当前版本顺序执行阶段，未来可优化：
- 监控与训练并行
- 结果分析在后台进行
- 报告生成异步化

## 容错机制

### 阶段失败处理

```python
if not success and stage.status == "FAILED":
    # 记录失败信息
    # 保留已完成阶段的输出
    # 生成部分报告
    # 终止后续阶段
```

### 训练进程崩溃

监控阶段会检测：
- 进程是否存在
- 日志是否更新
- 资源使用是否正常

### 网络中断（邮件）

邮件发送失败不影响流水线：
- 阶段标记为SKIPPED
- 保留本地报告
- 继续完成流水线

## 日志系统

### 日志级别

- **INFO**: 阶段开始/完成
- **WARNING**: 配置覆盖、资源异常
- **ERROR**: 阶段失败、异常信息

### 日志位置

1. **控制台输出**: 实时反馈
2. **pipeline.log**: 流水线日志
3. **training_output.log**: 训练输出
4. **阶段特定日志**: 各阶段可选

### 日志格式

```
[2025-12-14 14:30:22] [阶段名称] 日志消息
```

## 与现有系统集成

### 1. Taskfile集成

```yaml
pipeline:baseline:
  desc: 运行Baseline流水线
  cmds:
    - bash _7PIPELINE/run_pipeline.sh --baseline
```

### 2. 监控系统集成

复用现有监控代码：
- `_8TOOL/monitoring/`
- `_4MONITORING/`
- GPU资源查询

### 3. 结果收集集成

使用现有脚本：
- `_6EVALUATE/format_redjujube_results.py`
- `_6EVALUATE/collect_comparison_results.py`

### 4. 邮件系统集成

复用邮件发送逻辑：
- `_4MONITORING/send_experiment_result.py`
- HTML邮件模板

## 安全考虑

### 1. 配置验证

- 参数类型检查
- 路径存在性验证
- 权限检查

### 2. 进程隔离

- 独立工作目录
- 子进程管理
- 资源限制

### 3. 敏感信息

- 邮件密码通过参数传递
- 不记录到日志
- 建议使用环境变量

## 未来规划

### 短期（v1.1）

- [ ] 支持断点续跑
- [ ] 添加Web UI
- [ ] 实验队列管理
- [ ] 更丰富的监控指标

### 中期（v2.0）

- [ ] 分布式训练支持
- [ ] 超参数自动调优
- [ ] 实验对比可视化
- [ ] 云平台集成

### 长期（v3.0）

- [ ] 多机协同
- [ ] 自动化论文生成
- [ ] 实验知识图谱
- [ ] AI辅助调参

---

**维护**: eznlp项目组  
**最后更新**: 2025-12-14
