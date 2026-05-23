# 投稿计划执行完成度审计

审计对象：`plans/农业机械学报投稿论文_goal计划.md`

审计日期：2026-05-23

## 已完成项

| 计划要求 | 当前证据 | 状态 |
|---|---|---|
| 建立结果注册表，统一最终主结果 | `paper_result_registry.md` | 已完成 |
| 统一模型名称和术语 | 投稿稿初版统一使用 EDBP；注册表说明 Boundary Prediction 与 Boundary Selection 的关系 | 已完成 |
| 生成投稿主文稿 Markdown 版 | `农业机械学报_红枣NER_投稿稿.md` | 已完成初版 |
| 生成可转 Word 的中间稿 | `农业机械学报_红枣NER_投稿稿.html`、`农业机械学报_红枣NER_投稿稿.docx`、`农业机械学报_红枣NER_投稿稿_带图.docx`、`农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx`、`农业机械学报_红枣NER_投稿稿_lo.odt`、`农业机械学报_红枣NER_投稿稿_lo.pdf`、`农业机械学报_红枣NER_Word转排说明.md` | 已完成期刊格式 Word 初稿 |
| 生成投稿包清单 | `农业机械学报_投稿包_manifest.md` | 已完成 |
| 生成图表清单 | `农业机械学报_红枣NER_图表清单.md` | 已完成 |
| 生成自审意见 | `internal_review_round1.md` | 已完成 |
| 生成修改记录 | `revision_log_round1.md` | 已完成 |
| 生成投稿前检查清单 | `投稿前检查清单.md` | 已完成 |
| 生成 Word 转排说明 | `农业机械学报_红枣NER_Word转排说明.md` | 已完成 |
| 生成方法图 | `figures/fig1_edbp_architecture.svg`、`figures/fig2_bmes_dictionary_encoding.svg`、`figures/fig3_boundary_prediction_decoder.svg`；PNG 见 `figures_png/` | 已完成初版 |
| 生成类别分析和错误示例图 | `figures/fig4_entity_f1_by_category.svg`、`figures/fig5_boundary_error_cases.svg`；PNG 见 `figures_png/`；图 5 证据见 `figure5_error_evidence.md` | 已完成初版 |
| 按官方编改核对表审计 | `official_checklist_audit_2026-05-23.md` | 已完成初版 |
| 生成官方核对附录 | `农业机械学报_红枣NER_官方核对附录.md` | 已完成 |
| 生成 PDF 渲染预览 | `rendered/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf`、`rendered/render_audit_2026-05-23.md` | 已完成 |
| 生成投稿交接包 | `submission_package/`、`农业机械学报_红枣NER_投稿交接包_2026-05-23.zip` | 已完成并通过压缩包测试 |
| 自动验证投稿交接包 | `tools/validate_submission_package.py`、`submission_package/evidence/package_validation_draft_2026-05-23.md` | 草稿模式通过；终稿模式需替换作者信息后执行 |
| 终稿构建流程 | `tools/build_final_submission_package.py` | 已用测试作者信息跑通；真实终稿需作者填写 `submission_info.json` |
| 投稿信息 JSON 严格校验 | `tools/validate_submission_info.py`、`submission_info_template_validation_2026-05-23.md` | 模板结构校验通过；终稿构建前会拦截占位信息 |
| 全量投稿检查总控 | `tools/run_all_submission_checks.py`、`full_submission_check_report_2026-05-23.md` | 已通过；汇总引用、文献格式、数值、图表、公式、主稿质量、投稿信息模板、投稿包和 zip 完整性 |
| 复现实验说明 | `reproducibility_statement.md`、`reproducibility_validation_2026-05-23.md` | 已完成；覆盖 RJND/公开数据集路径、训练脚本、结果目录和采用口径 |
| PDF 渲染质量门禁 | `rendered_pdf_validation_2026-05-23.md` | 已完成；PDF 为 A4、14 页、含关键文本、5 张正文图且图像不低于 250 ppi |
| DOCX 版式门禁 | `docx_layout_validation_2026-05-23.md` | 已完成；题注居中、正文导语未误居中、行内公式 Markdown 标记已清理、编号列表独立成段 |
| 用户 DOCX 与本地论文写法吸收 | `user_draft_融合专家词典与边界选择的红枣栽培NER_抽取稿.md`、`农业机械学报同类NER论文写法对照.md`、`plans/evidence-map.md`、`plans/chapter-blueprints/intro-materials-blueprint.md` | 已完成本轮吸收；语料来源、PDF 抽取/OCR 校对、半自动标注流程和引言问题链已并入主稿 |
| 源码一致性核对 | `source_consistency_check.md` | 已完成初核 |
| 官方论文写作模板 PDF | `official_docs/论文写作模板_2019.pdf` | 已下载并用于内容项核对 |
| 官方 2026 编改专项核对表 | `official_docs/农业机械学报论文编改专项核对表_2026.doc` | 已下载；终稿 Word/WPS 人工核对时优先参考 |
| 参考文献不少于 25 条 | 投稿稿当前列出 28 条；已替换 1、24-25 的风险条目，并补充 1-25、27-28 的 DOI/URL；核验记录见 `reference_audit_round1.md` | 数量满足，格式仍需终核 |
| 正文引用顺序和覆盖 | `reference_citation_validation_2026-05-23.md` 显示文末 [1]-[28] 连续、正文引用覆盖所有参考文献且首次引用顺序递增 | 已完成 |
| 参考文献基本著录格式 | `reference_format_validation_2026-05-23.md` 显示 28 条文献均有类型标识、年份和 DOI/URL，含近年文献 18 条、外文文献 19 条 | 已完成 |
| 核心数值一致性 | `numeric_consistency_validation_2026-05-23.md` 显示摘要、英文摘要、结果表、泛化实验和结论核心数值一致 | 已完成 |
| 图表编号和文件一致性 | `figure_table_validation_2026-05-23.md` 显示图 1-5、表 1-8 连续，均有双语题名、正文先导引用和可用图件 | 已完成 |
| 公式和符号一致性 | `equation_symbol_validation_2026-05-23.md` 显示主稿 10 个展示公式、核心公式、符号附录和 DOCX 公式编号通过 | 已完成 |
| 主稿质量门禁 | `manuscript_quality_validation_2026-05-23.md` 显示题名长度、关键词、章节结构、参考文献数量、术语统一和非投稿内容检查通过 | 已完成 |
| 多轮审稿与复审 | `peer_review_round2.md`、`revision_log_round2.md`、`peer_review_round3.md`、`revision_log_round3.md` 显示正文科学叙述风险已收束，剩余问题集中于投稿信息和 Word/WPS 人工终检 | 已完成正文复审 |
| 实验充分性补强 | `experiment_sufficiency_analysis_2026-05-23.md`、`experiment_sufficiency_groups_2026-05-23.csv`、`experiment_sufficiency_seeds_2026-05-23.csv` 汇总 50 组已完成实验，并定位主表三种子原始结果与配对 t 检验 | 已完成补强 |

## 未完成项

| 计划要求 | 缺口 | 下一步 |
|---|---|---|
| 图 4 不同实体类别识别效果 | 已绘制 SVG 并导出 PNG，但仍需按期刊最终版式微调 | 投稿前统一图字体和线宽 |
| 图 5 典型边界错误示例 | 已补充预测文件逐条证据并导出 PNG；仍需决定是否用正式主实验预测文件替换 `EXP-009` 辅助预测样例 | 若投稿强调严格主实验一致性，应从主实验 `predictions_test.pt` 中再抽取同类错例 |
| 参考文献格式完全满足 GB/T 7714 | 1-28 已补 DOI/URL 或稳定来源，正文引用已覆盖全部文献，基本著录门禁已通过；仍需 Word 转排时检查标点空格细节 | Word/WPS 终检时复核标点、空格和悬挂缩进 |
| 方法实现与代码完全一致 | `source_consistency_check.md` 已完成初核；当前主稿不声称通道注意力、BMES span feature、enhanced size embedding 或 LogN scaling 为主贡献 | 若后续换用新实验结果，需重新核对方法章节 |
| 投稿模板适配 | 已生成 HTML、基础 `.docx`、带图 `.docx`、期刊格式 `.docx`、ODT、PDF、渲染核验和 Word 转排说明；已取得官方写作模板 PDF，但未取得 Word 模板 | 后续若编辑部提供 Word 模板，再按 Word 模板转排 |
| 投稿首页信息 | 作者、单位、基金、中图分类号、作者简介和通信作者信息缺失 | 由作者填写 `农业机械学报_红枣NER_投稿信息补全表.md` |
| 投稿信息替换流程 | 已提供 `submission_info.example.json`、`tools/validate_submission_info.py`、`tools/fill_docx_front_matter.py` 和 `tools/build_final_submission_package.py`，并用测试作者信息跑通无占位终稿验证 | 作者填写 JSON 后运行总控脚本生成无占位终稿 |
| 中图分类号建议 | 已生成 `农业机械学报_红枣NER_中图分类号建议.md`，建议优先 `TP391.1`、备选 `TP391` | 终稿前由作者确认 |
| 投稿前终检 | 自动化引用顺序、核心数值、图表编号、公式符号、参考文献格式、主稿质量门禁、复现路径、PDF 渲染质量和全量检查已完成；仍缺 Word/WPS 人工显示终检 | 补齐作者信息后在 Word/WPS 中逐页检查 |

## 当前结论

计划已从“期刊格式 Word 初稿阶段”推进到“投稿交接包阶段”。目前不应标记整个 goal 完成，因为作者/单位/基金等投稿首页信息缺失，Word 终稿显示检查、作者单位和参考文献悬挂缩进尚未完成。下一轮优先补齐投稿信息并按官方编改核对表完成 Word/WPS 终检。

## 2026-05-23 同步验证记录

本轮已将最新主稿、期刊格式 DOCX、PDF、5 张 PNG 图、验证脚本、全量检查报告、DOCX 版式报告、同刊论文写法对照和能力使用审计同步至 `submission_package/`，并重新生成 `农业机械学报_红枣NER_投稿交接包_2026-05-23.zip`。

已复核的命令：

```bash
python3 docs/paper/tools/run_all_submission_checks.py --report docs/paper/full_submission_check_report_2026-05-23.md
python3 docs/paper/tools/validate_submission_package.py docs/paper/submission_package --report docs/paper/submission_package/evidence/package_validation_draft_2026-05-23.md
unzip -t docs/paper/农业机械学报_红枣NER_投稿交接包_2026-05-23.zip
```

验证结果：全量检查通过；投稿包草稿模式通过；压缩包完整性通过。保留的 `WARN` 为预期状态，即 DOCX 首页仍含作者、单位、基金等待作者补齐的占位内容。

## 2026-05-23 Round 3 复审记录

已按同行评审技能和外部审稿量表完成 Round 3 复审。复审结论为：正文层面可进入投稿前终检阶段，但不能标记为最终可投终稿。原因是作者、单位、基金、作者简介、通信作者和中图分类号仍未替换为真实信息，且 DOCX 公式和分页仍需 Word/WPS 人工显示终检。
