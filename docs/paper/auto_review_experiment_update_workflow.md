# 自动审稿、实验设计与论文更新流程

本文档定义当前红枣栽培 NER 论文的自动化协作流程。论文端以 `master` 为稳定主线，实验端以 `experiment/current-rjnd-baselines` 为补实验主线。

## 1. 本地技能

已安装到本机 Codex skills：

| skill | 用途 |
|---|---|
| `academic-writing` | 学术写作、研究方法、引用和同行评审准备 |
| `paper-writing` | 论文结构、段落压缩、引言/讨论/结论修订 |
| `academic-paper-reviewer` | 多视角模拟审稿和复审 |
| `academic-paper` | 学术论文写作流水线 |
| `academic-pipeline` | research -> write -> review -> revise -> finalize 总控 |
| `deep-research` | 文献检索、证据综合和事实核查 |
| `academic-paper-strategist` | 论文策略设计和研究空白判断 |
| `academic-paper-composer` | 根据提纲写完整稿件 |
| `agricultural-ner-paper-workflow` | 本项目专用：红枣 NER 论文审稿、实验需求、结果登记和最终包更新 |

新安装的 skills 需要重启 Codex 后自动出现在可用 skills 列表中。

## 2. 自动审稿输入

默认审稿对象：

```text
docs/paper/final_current/农业机械学报_红枣NER_最终查看稿.md
```

证据文件：

```text
docs/paper/paper_result_registry.md
docs/paper/current_rjnd_experiment_requirements.md
docs/paper/final_current/论文完善审查_2026-05-24.md
docs/paper/submission_package/evidence/
```

本地参考论文来源：

```text
/mnt/d/Lenovo/Documents/世界树/世界树/
/mnt/e/Users/Lenovo/Zotero/storage/
```

## 3. 自动审稿输出

自动审稿报告写入：

```text
docs/paper/review/auto_review_YYYY-MM-DD.md
```

报告必须包含：

```text
1. 编辑初筛意见
2. 方法审稿意见
3. 农业领域审稿意见
4. 实验充分性审稿意见
5. 可复现性和投稿格式意见
6. 必须修改项
7. 建议修改项
8. 不建议修改项
9. 是否需要新增实验
```

## 4. 自动实验设计

如果审稿报告指出实验不足，应同步更新：

```text
docs/paper/current_rjnd_experiment_requirements.md
docs/paper/needed_experiment_results.md
```

每条实验任务必须给出：

```text
model_name
priority
seed
dataset_split
required_metrics
result_path_required
config_path_required
eval_script_required
backbone
lexicon
data_process
paper_claim_supported
acceptance_rule
```

当前正文主表实验统一要求：

```text
当前 RJND/RedJujube 数据划分
seed=42
test split
实体级严格 P/R/F1
同一评估脚本
```

不能把旧版红枣结果、HZ 数据、公开数据集结果或 dev F1 直接放入当前表 3。

## 5. 自动结果登记

实验端返回结果后，论文端只登记可追溯结果。

必须更新：

```text
docs/paper/paper_result_registry.md
docs/paper/submission_package/evidence/paper_result_registry.md
```

如果结果改变正文结论，再更新：

```text
docs/paper/农业机械学报_红枣NER_投稿稿.md
docs/paper/submission_package/农业机械学报_红枣NER_投稿稿.md
```

## 6. 自动论文更新

论文正文更新后执行：

```bash
cd /home/lenovo/projects/eznlp/docs/paper

uv run --no-project --with markdown \
  python tools/md_to_html.py \
  农业机械学报_红枣NER_投稿稿.md \
  农业机械学报_红枣NER_投稿稿.html

uv run --no-project --with python-docx \
  python tools/md_to_docx_with_images.py \
  --journal-style \
  农业机械学报_红枣NER_投稿稿.md \
  农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx

python3 tools/validate_numeric_consistency.py \
  农业机械学报_红枣NER_投稿稿.md
```

同步交付包和最终查看目录：

```text
docs/paper/submission_package/
docs/paper/final_current/
```

## 7. 自动验证门禁

提交前必须通过：

```bash
cd /home/lenovo/projects/eznlp

git diff --check

python3 docs/paper/tools/validate_numeric_consistency.py \
  docs/paper/农业机械学报_红枣NER_投稿稿.md

diff -u \
  docs/paper/农业机械学报_红枣NER_投稿稿.md \
  docs/paper/submission_package/农业机械学报_红枣NER_投稿稿.md

test -d .venv && echo .venv-exists || echo no-venv

git status --short --branch
```

DOCX 数值检查使用 Python `zipfile`，不要依赖 `unzip`。

## 8. PDF 规则

只有重新从最新 DOCX 渲染并验证通过的 PDF 才能放入最终查看目录。

当前若缺少 LibreOffice 或 PDF 检查工具，先安装：

```bash
sudo apt-get update
sudo apt-get install -y libreoffice-writer libreoffice-java-common poppler-utils
```

旧 PDF 不得作为最新论文判断依据。

## 9. Git 规则

论文端：

```text
master
```

实验端：

```text
experiment/current-rjnd-baselines
```

实验端不要直接改最终投稿稿。论文端不要整分支合并实验端；优先按文件挑选 verified result、config、registry 和必要论文更新。
