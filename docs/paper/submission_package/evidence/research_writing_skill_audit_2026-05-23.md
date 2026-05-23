# Research Writing Skill Audit

审计日期：2026-05-23

任务：继续执行 `docs/paper/plans/农业机械学报投稿论文_goal计划.md`，并按用户要求参考项目中其他 Markdown 论文，重点处理图中文字乱码和公式排版问题。

## 使用的写作流程依据

已安装并读取 `/home/shiwenlong/.codex/skills/research-writing-skill` 中的规则：

1. `skills/using-research-writing/SKILL.md`：确认本任务属于论文写作与投稿准备任务，应记录能力使用审计。
2. `skills/writing-core/SKILL.md`：采用中文期刊论文自然表达、证据可追溯、避免编造文献和完成后验证的规则。

本轮未完整执行 `brainstorming-research` 的原因：当前任务不是新开题或从零写作，而是在既有 goal 计划和既有投稿稿基础上继续修订；用户明确要求继续推进并解决具体版式问题。

## 参考的同刊 Markdown 论文

本轮定位并读取了 `_2DATA/papers/` 下 3 篇《农业机械学报》同类 NER 论文的 Markdown 转写结果：

1. `李春春 等 - 2025 - 基于BERT-BiLSTM-CRF的茶树病虫害命名实体识别方法/auto/李春春 等 - 2025 - 基于BERT-BiLSTM-CRF的茶树病虫害命名实体识别方法.md`
2. `路阳 等 - 2026 - 基于曼哈顿注意力机制的水稻病虫害命名实体识别/auto/路阳 等 - 2026 - 基于曼哈顿注意力机制的水稻病虫害命名实体识别.md`
3. `蒲攀 等 - 2024 - 融合动态词典特征和CBAM的苹果病虫害命名实体识别方法/auto/蒲攀 等 - 2024 - 融合动态词典特征和CBAM的苹果病虫害命名实体识别方法.md`

已形成对照文档：`docs/paper/农业机械学报同类NER论文写法对照.md`。

## 本轮完成的修订

1. 图中文字乱码处理：将 5 张 SVG 图的字体栈从优先 `Microsoft YaHei` 调整为本机可用的 `Noto Sans CJK SC`，并重新导出 2 倍分辨率 PNG。
2. 图像质量处理：重新生成 `figures_png/*.png`，当前 PDF 检查显示 5 张正文图均不低于 250 ppi。
3. 图题位置处理：主稿中 5 张图调整为图片在前、图题在图下；正文仍保留图号先导引用。
4. 行内公式处理：`tools/md_to_docx_with_images.py` 已清理 Word/PDF 中正文 `$...$` Markdown 数学标记。
5. 展示公式处理：期刊格式 DOCX 中 10 个展示公式保持居中，编号靠右，编号连续。
6. 图表题注处理：`tools/md_to_docx_with_images.py` 已将真正的图题、表题和英文题名居中输出，同时避免把“图 1 为…”“表 1 给出…”等正文导语误判为题注。
7. 编号列表处理：`tools/md_to_docx_with_images.py` 已将贡献条目和结论条目按编号分段输出，避免 Word/PDF 中被压缩成同一段。
8. 投稿包同步：已将最新主稿、DOCX、PDF、PNG、工具脚本和新增对照审计材料同步到 `submission_package/`。
9. 交接包收口：已重新验证 `submission_package/` 草稿模式、同步最新全量检查报告，并重建 `农业机械学报_红枣NER_投稿交接包_2026-05-23.zip`。
10. 用户 DOCX 融合：已抽取 `融合专家词典与边界选择的红枣栽培命名实体识别方法.docx`，吸收其中语料来源、文本抽取、人工校对和半自动标注流程写法；未采用其中与当前证据冲突的 `EDBS`、通道注意力主模型和旧主结果数值。
11. 本地论文学习：已补充参考吴钊等苹果栽培 NER 论文，并将茶树、水稻、苹果病虫害、苹果栽培等论文的写法归纳到 `农业机械学报同类NER论文写法对照.md`。
12. 多轮审稿：按 `peer-review` 技能和外部 `academic-paper-reviewer` 量表完成 Round 2 审稿、Round 2 修改和 Round 3 复审；审稿记录见 `peer_review_round2.md`、`revision_log_round2.md`、`peer_review_round3.md` 和 `revision_log_round3.md`。
13. 实验充分性补强：汇总 50 组已完成实验，生成 `experiment_sufficiency_analysis_2026-05-23.md`、`experiment_sufficiency_groups_2026-05-23.csv` 和 `experiment_sufficiency_seeds_2026-05-23.csv`；据此在主稿 2.1 节补充有原始三种子结果支撑的配对 t 检验。

## 验证命令

已执行并通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 docs/paper/tools/validate_figures_tables.py docs/paper/农业机械学报_红枣NER_投稿稿.md --report docs/paper/figure_table_validation_2026-05-23.md
PYTHONDONTWRITEBYTECODE=1 python3 docs/paper/tools/validate_equations_symbols.py docs/paper/农业机械学报_红枣NER_投稿稿.md docs/paper/农业机械学报_红枣NER_官方核对附录.md --docx docs/paper/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx
PYTHONDONTWRITEBYTECODE=1 python3 docs/paper/tools/validate_rendered_pdf.py docs/paper/rendered/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf --report docs/paper/rendered_pdf_validation_2026-05-23.md
PYTHONDONTWRITEBYTECODE=1 python3 docs/paper/tools/run_all_submission_checks.py --report docs/paper/full_submission_check_report_2026-05-23.md
PYTHONDONTWRITEBYTECODE=1 python3 docs/paper/tools/validate_submission_package.py docs/paper/submission_package --report docs/paper/submission_package/evidence/package_validation_draft_2026-05-23.md
unzip -t docs/paper/农业机械学报_红枣NER_投稿交接包_2026-05-23.zip
```

本轮收口时再次执行 `run_all_submission_checks.py`、`validate_submission_package.py` 和 `unzip -t`，结果均通过；唯一保留警告为投稿首页作者信息占位，属于真实作者信息未提供导致的预期草稿状态。

Round 3 复审后将再次执行上述全量验证，并以最新报告为准。

## 当前剩余风险

1. DOCX 仍为投稿格式初稿，作者、单位、基金、中图分类号、作者简介和通信作者信息仍需作者补齐。
2. 公式已去除 Markdown 标记并实现居中编号，但仍是可编辑文本形式，不是 Word 原生公式对象；正式投稿前建议在 Word/WPS 中将 10 个公式替换为公式对象或按编辑部要求处理。
3. 编号列表分段后 PDF 由 13 页变为 14 页，贡献条目附近存在自然分页；正式投稿前可在 Word/WPS 中微调段前距、图件高度或分页位置。
4. Word/WPS 终检仍需人工完成，包括三线表跨页、参考文献悬挂缩进、作者单位首页布局和图表分页。

## 结论

本轮修订使稿件在图中文字、图像分辨率、图题位置、表题格式、行内公式标记、公式编号和编号列表可读性方面更接近《农业机械学报》同类 NER 论文版式。自动检查已通过，但真实投稿终稿仍依赖作者信息和 Word/WPS 人工终检。
