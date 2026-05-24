# 论文端与实验端分支方案

本文档用于协调当前红枣栽培 NER 论文写作端和服务器实验端，目标是让实验端可以独立执行补实验，同时保证论文主分支保持稳定、干净、可交付。

## 1. 分支职责

| 分支 | 使用端 | 职责 | 允许内容 | 不建议内容 |
|---|---|---|---|---|
| `master` | 论文端 | 稳定论文稿、已确认结果、交付包 | 最终论文、结果登记表、可追溯的最终指标、少量必要配置 | 临时日志、大 checkpoint、失败实验、中间缓存 |
| `experiment/current-rjnd-baselines` | 实验端 | 当前 RJND 口径补实验 | 实验脚本、配置、运行日志摘要、候选结果、最终结果 | 不要直接改投稿最终稿，除非只是登记结果 |

核心原则：实验端负责“产出可验证结果”，论文端负责“筛选并写入论文”。

`master` 保留的是论文端的稳定资产，不是所有实验过程：

```text
1. 最终论文源文件和交付包
2. 已确认可进入论文的结果登记表
3. 支撑论文表格的最小结果文件
4. 可复现实验的最小配置和命令记录
5. 审稿、投稿、复现说明等文档
```

`master` 不保留实验端的过程垃圾：

```text
1. 训练 checkpoint 和模型权重
2. 大体积 cache
3. 重复或失败实验的完整输出目录
4. 临时 debug 日志
5. 未确认口径的候选高分结果
```

换句话说，`master` 是“论文事实库 + 交付包”，实验分支是“实验工作区”。

## 2. 论文端准备命令

论文端在本机维护 `master`：

```bash
cd /home/lenovo/projects/eznlp
git checkout master
git pull --ff-only
git status --short --branch
```

如果需要创建或刷新实验分支：

```bash
git checkout master
git pull --ff-only
git checkout -B experiment/current-rjnd-baselines
git push -u origin experiment/current-rjnd-baselines
git checkout master
```

如果远端已经有实验分支，只需要确保服务器端拉取它，不要重复创建。

## 3. 实验端执行命令

服务器实验端首次使用：

```bash
cd /path/to/eznlp
git fetch origin
git checkout experiment/current-rjnd-baselines
git pull --ff-only
```

如果服务器上还没有仓库：

```bash
cd /path/to/projects
git clone git@github.com:NERLabs/eznlp.git
cd eznlp
git checkout experiment/current-rjnd-baselines
```

实验端不合并整条 `master`。每次接收论文端新需求时，只从 `origin/master` 同步需求文件：

```bash
git fetch origin
git checkout experiment/current-rjnd-baselines
git checkout origin/master -- \
  docs/paper/current_rjnd_experiment_requirements.md \
  docs/paper/needed_experiment_results.md \
  docs/paper/branch_workflow_for_paper_and_experiments.md
git add docs/paper/current_rjnd_experiment_requirements.md \
  docs/paper/needed_experiment_results.md \
  docs/paper/branch_workflow_for_paper_and_experiments.md
git commit -m "docs: sync paper experiment requirements"
git push origin experiment/current-rjnd-baselines
```

如果同步后没有差异，`git commit` 会提示 nothing to commit，可直接继续实验。

实验端优先读取任务清单：

```bash
sed -n '1,220p' docs/paper/current_rjnd_experiment_requirements.md
```

实验完成后，先只提交可追溯结果和必要配置：

```bash
git status --short
git add docs/paper/current_rjnd_experiment_requirements.md
git add docs/paper/needed_experiment_results.md
git add docs/paper/paper_result_registry.md
git add experiments/EXP-010-optimization/results_needed_YYYYMMDD/
git commit -m "experiments: add current RJND baseline results"
git push origin experiment/current-rjnd-baselines
```

如果某些结果目录很大，先不要直接 `git add`，只提交：

```text
1. metrics/results json
2. config/yaml/sh
3. training command
4. evaluation command
5. small log summary
```

不要提交：

```text
checkpoint
*.pt
*.pth
*.ckpt
cache
完整大日志
临时 debug 输出
```

## 4. 实验结果返回格式

实验端每个模型至少返回以下字段，便于论文端直接登记：

| 字段 | 要求 |
|---|---|
| model_name | 模型名，如 `Boundary Smoothing` |
| seed | 当前主表使用 `42` |
| dataset_split | 必须确认是当前 RJND/RedJujube 论文划分 |
| precision | test Precision/% |
| recall | test Recall/% |
| f1 | test F1/% |
| result_path | `results.json`、`metrics.json` 或日志路径 |
| config_path | 参数文件、shell 脚本或命令记录 |
| eval_script | 评估脚本路径 |
| bert_backbone | 使用的 BERT/MacBERT 模型 |
| lexicon | 是否使用词典、词典来源、词典规模 |
| note | 口径差异、失败原因或环境限制 |

## 5. 论文端合并方式

不要默认执行整分支合并：

```bash
git merge experiment/current-rjnd-baselines
```

推荐论文端从实验分支挑选文件：

```bash
cd /home/lenovo/projects/eznlp
git checkout master
git pull --ff-only
git fetch origin

git checkout origin/experiment/current-rjnd-baselines -- \
  docs/paper/paper_result_registry.md \
  docs/paper/needed_experiment_results.md \
  experiments/EXP-010-optimization/results_needed_YYYYMMDD/
```

检查没有带入大文件或临时文件：

```bash
git status --short
git diff --stat
git diff --check
```

确认结果可用后提交到 `master`：

```bash
git add docs/paper/paper_result_registry.md
git add docs/paper/needed_experiment_results.md
git add experiments/EXP-010-optimization/results_needed_YYYYMMDD/
git commit -m "docs: register current RJND baseline results"
git push origin master
```

如果实验分支上的某个提交非常干净，只包含要进入论文端的结果，也可以使用：

```bash
git checkout master
git pull --ff-only
git cherry-pick <commit_hash>
git push origin master
```

优先使用“按文件挑选”，因为实验分支通常会包含临时过程文件。

## 6. 论文更新流程

论文端确认实验结果后，再更新最终稿：

```bash
cd /home/lenovo/projects/eznlp/docs/paper
python3 tools/validate_numeric_consistency.py 农业机械学报_红枣NER_投稿稿.md
```

如果更新了 Markdown，需要同步交付包、重新生成 HTML/DOCX/PDF，并再次检查：

```bash
cd /home/lenovo/projects/eznlp
diff -u docs/paper/农业机械学报_红枣NER_投稿稿.md docs/paper/submission_package/农业机械学报_红枣NER_投稿稿.md
git diff --check
git status --short
```

PDF 生成依赖 LibreOffice，PDF 质量检查依赖 Poppler。若 `soffice`、`pdfinfo`、`pdftotext` 或 `pdfimages` 不存在，需要先安装：

```bash
sudo apt-get update
sudo apt-get install -y libreoffice-writer libreoffice-java-common poppler-utils
```

## 7. 冲突处理

如果实验端推送失败，先同步远端实验分支：

```bash
git checkout experiment/current-rjnd-baselines
git pull --rebase origin experiment/current-rjnd-baselines
git push origin experiment/current-rjnd-baselines
```

如果论文端挑文件后发现冲突或误带大文件：

```bash
git status --short
git restore --staged <path>
git restore <path>
```

只恢复误带文件，不要使用：

```bash
git reset --hard
```

除非已经确认没有需要保留的本地改动。

## 8. 推荐提交命名

实验端：

```text
experiments: add current RJND boundary smoothing result
experiments: add current RJND mrc baseline result
experiments: record failed nflat reproduction attempt
```

论文端：

```text
docs: register current RJND baseline results
docs: update paper comparison table
docs: regenerate final paper package
```

## 9. 当前优先实验分支

当前建议使用：

```text
experiment/current-rjnd-baselines
```

该分支服务于 `docs/paper/current_rjnd_experiment_requirements.md` 中列出的当前 RJND 补实验任务。
