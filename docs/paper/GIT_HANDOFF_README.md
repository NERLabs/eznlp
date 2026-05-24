# 论文 Git 交接说明

用途：让本地机器通过 `git pull` 获取论文写作所需材料；服务器继续只负责跑实验。

## 本次纳入 Git 的材料

1. 投稿稿与交接包：`docs/paper/农业机械学报_红枣NER_投稿稿.md`、期刊格式 DOCX、PDF、投稿 zip。
2. 图件：`docs/paper/figures/`、`docs/paper/figures_png/`。
3. 论文证据与验证报告：`docs/paper/*validation*.md`、`full_submission_check_report_2026-05-23.md`、`paper_result_registry.md`、多轮审稿记录和投稿就绪审计。
4. 实验充分性材料：`experiment_sufficiency_analysis_2026-05-23.md`、`experiment_sufficiency_groups_2026-05-23.csv`、`experiment_sufficiency_seeds_2026-05-23.csv`。
5. 轻量实验结果：`experiments/EXP-010-optimization/results_newdata/**/results.json`、公开数据集 `results_public/**/results.json`、必要 `auto_lexicon.txt`。
6. 词典策略分析结果：`experiments/EXP-011-lexicon_strategy/analysis/`。
7. RJND 当前写作所需数据：`datasets/raw/RedJujube/redjujube_train.bmes`、`redjujube_dev.bmes`、`redjujube_test.bmes` 和专家词典/SoftLexicon 文本。

## 未纳入 Git 的材料

1. 模型权重：`*.pth`、`*.pt`。
2. TensorBoard 事件文件。
3. 大规模公开数据原始集和实验缓存。
4. 临时 Python 缓存：`__pycache__`、`*.pyc`。

## 本地使用方式

在本地机器执行：

```bash
git pull origin master
```

论文主要入口：

```text
docs/paper/农业机械学报_红枣NER_投稿交接包_2026-05-23.zip
docs/paper/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx
docs/paper/rendered/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf
docs/paper/experiment_sufficiency_analysis_2026-05-23.md
```

服务器继续跑新实验时，只需把新的 `results.json`、分析 CSV、论文图表或稿件修订提交到 Git；不要提交权重文件。
