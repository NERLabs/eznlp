# 《农业机械学报》投稿包清单

生成日期：2026-05-23

## 1. 主稿文件

| 文件 | 用途 | 状态 |
|---|---|---|
| `农业机械学报_红枣NER_投稿稿.md` | 投稿主文稿 Markdown 版 | 已完成可转排初稿 |
| `农业机械学报_红枣NER_投稿稿.html` | HTML 转排中间稿，可用于浏览器预览或 Word/WPS 打开后二次排版 | 已生成 |
| `农业机械学报_红枣NER_投稿稿.docx` | Word 初稿，由标准库脚本生成，保留正文、表格和图件占位 | 已生成，未套官方模板 |
| `农业机械学报_红枣NER_投稿稿_带图.docx` | Word 初稿，由 `python-docx` 生成，正文含 8 张表并嵌入 5 张 PNG 图 | 已生成，未套官方模板 |
| `农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx` | Word 初稿，含英文题名和投稿信息占位、公式编号、三线表边框、5 张嵌入图 | 已生成，优先用于终稿转排 |
| `rendered/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf` | 由期刊格式 Word 初稿渲染得到的 PDF 预览 | 已生成并核验 |
| `农业机械学报_红枣NER_投稿稿_lo.odt` | LibreOffice 从 HTML 生成的 ODT 中间稿 | 已生成 |
| `农业机械学报_红枣NER_投稿稿_lo.pdf` | LibreOffice 从 HTML 生成的 PDF 预览稿 | 已生成 |
| `农业机械学报_红枣NER_Word转排说明.md` | Word 转排流程和当前工具限制说明 | 已完成 |
| `农业机械学报_红枣NER_投稿信息补全表.md` | 作者、单位、基金、中图分类号等投稿系统信息补全表 | 待作者填写 |
| `submission_info.example.json` | 作者信息脚本化替换模板 | 已完成 |
| `tools/validate_submission_info.py` | 投稿首页作者、单位、基金、中图分类号等 JSON 字段严格校验脚本 | 已完成 |
| `tools/fill_docx_front_matter.py` | 替换期刊格式 DOCX 首页占位的脚本 | 已完成 |
| `tools/validate_submission_package.py` | 投稿交接包自动验证脚本，支持草稿/终稿两种模式 | 已完成 |
| `tools/build_final_submission_package.py` | 作者信息补齐后，一键构建终稿交接包的总控脚本 | 已完成并用测试信息跑通 |
| `tools/validate_references.py` | 正文引用顺序与文末参考文献连续性验证脚本 | 已完成 |
| `tools/validate_reference_format.py` | 参考文献类型标识、年份、DOI/URL、近年文献和外文文献门禁脚本 | 已完成 |
| `tools/validate_numeric_consistency.py` | 摘要、结果表和结论核心数值一致性验证脚本 | 已完成 |
| `tools/validate_figures_tables.py` | 正文图表编号、双语题名、先导引用和图件文件一致性验证脚本 | 已完成 |
| `tools/validate_equations_symbols.py` | 公式数量、核心公式、符号附录和 DOCX 公式编号一致性验证脚本 | 已完成 |
| `tools/validate_docx_layout.py` | DOCX 中题注居中、正文导语、行内公式标记和编号列表版式验证脚本 | 已完成 |
| `tools/validate_manuscript_quality.py` | 题名长度、关键词、章节结构、术语混用和非投稿内容终检脚本 | 已完成 |
| `tools/md_to_html.py` | Markdown 投稿稿生成 HTML/LibreOffice 转排中间稿脚本 | 已完成 |
| `tools/run_all_submission_checks.py` | 一键运行引用、参考文献、数值、图表、公式、主稿质量、投稿信息和整包验证 | 已完成 |
| `tools/validate_reproducibility_paths.py` | 复现实验说明关键数据、代码和结果路径验证脚本 | 已完成 |
| `tools/validate_rendered_pdf.py` | PDF 页数、A4 页面、关键文本、图像数量和分辨率验证脚本 | 已完成 |

## 2. 图件

| 图号 | 文件 | 状态 |
|---|---|---|
| 图 1 | `figures/fig1_edbp_architecture.svg` | 已生成 SVG |
| 图 2 | `figures/fig2_bmes_dictionary_encoding.svg` | 已生成 SVG |
| 图 3 | `figures/fig3_boundary_prediction_decoder.svg` | 已生成 SVG |
| 图 4 | `figures/fig4_entity_f1_by_category.svg` | 已生成 SVG |
| 图 5 | `figures/fig5_boundary_error_cases.svg` | 已生成 SVG |

当前进展：5 张 SVG 已同步导出为 `figures_png/*.png`，并嵌入 `农业机械学报_红枣NER_投稿稿_带图.docx`。正式提交前仍需按期刊终稿版式检查图片清晰度、字体和线宽。

## 3. 证据和审计文件

| 文件 | 用途 | 状态 |
|---|---|---|
| `paper_result_registry.md` | 实验结果证据表和数值采用原则 | 已完成 |
| `figure5_error_evidence.md` | 图 5 错误案例预测文件证据 | 已完成 |
| `reference_audit_round1.md` | 参考文献核验记录 | 已完成初核 |
| `source_consistency_check.md` | 方法描述与源码一致性核对 | 已完成初核 |
| `internal_review_round1.md` | 内部审稿意见 | 已完成 |
| `revision_log_round1.md` | 逐条修改记录 | 已完成 |
| `submission_readiness_audit.md` | 投稿计划执行完成度审计 | 已更新 |
| `投稿前检查清单.md` | 投稿前终检清单 | 已完成 |
| `official_guidelines_index.md` | 官网投稿依据和已下载官方附件索引 | 已完成 |
| `official_docs/论文写作模板_2019.pdf` | 官网论文写作模板 PDF | 已下载 |
| `official_docs/论文编改专项核对表_2018-8-20.pdf` | 官网编改专项核对表附件 | 已下载 |
| `official_checklist_audit_2026-05-23.md` | 按官方编改专项核对表逐项审计 | 已完成 |
| `农业机械学报_红枣NER_官方核对附录.md` | 变量表和结论性数据汇总表 | 已完成 |
| `rendered/render_audit_2026-05-23.md` | 期刊格式 Word 初稿渲染核验记录 | 已完成 |
| `submission_package/evidence/package_validation_draft_2026-05-23.md` | 投稿交接包草稿模式自动验证报告 | 已完成 |
| `农业机械学报_红枣NER_中图分类号建议.md` | 中图分类号候选建议与依据 | 已完成，待作者确认 |
| `reference_citation_validation_2026-05-23.md` | 正文引用顺序与参考文献覆盖验证报告 | 已完成 |
| `reference_format_validation_2026-05-23.md` | 参考文献基本著录格式验证报告 | 已完成 |
| `numeric_consistency_validation_2026-05-23.md` | 核心数值一致性验证报告 | 已完成 |
| `figure_table_validation_2026-05-23.md` | 图 1-5、表 1-8 编号与文件一致性验证报告 | 已完成 |
| `equation_symbol_validation_2026-05-23.md` | 公式和符号附录一致性验证报告 | 已完成 |
| `docx_layout_validation_2026-05-23.md` | DOCX 题注、行内公式标记和编号列表版式验证报告 | 已完成 |
| `manuscript_quality_validation_2026-05-23.md` | 投稿主稿质量门禁验证报告 | 已完成 |
| `submission_info_template_validation_2026-05-23.md` | 投稿信息 JSON 模板结构验证报告 | 已完成 |
| `full_submission_check_report_2026-05-23.md` | 全量投稿检查汇总报告 | 已完成 |
| `reproducibility_statement.md` | 数据、代码入口、结果来源和重跑注意事项说明 | 已完成 |
| `reproducibility_validation_2026-05-23.md` | 复现实验关键路径验证报告 | 已完成 |
| `rendered_pdf_validation_2026-05-23.md` | 期刊格式 PDF 预览质量验证报告 | 已完成 |

## 4. 投稿交接包

| 文件/目录 | 用途 | 状态 |
|---|---|---|
| `submission_package/` | 投稿交接目录，集中放置优先使用稿件、PDF 预览、图件、证据材料和官方附件 | 已生成 |
| `农业机械学报_红枣NER_投稿交接包_2026-05-23.zip` | `submission_package/` 的压缩包 | 已生成并通过 `unzip -t` 测试 |

## 5. 当前仍未完成的投稿动作

1. 获取《农业机械学报》官方 Word 模板并套用；当前已取得官网论文写作模板 PDF 和编改核对表 PDF，但未直接取得 Word 模板文件，且投稿须知说明对格式版式无要求。
2. 补齐作者、单位、基金、中图分类号、作者简介和通信作者信息。
3. 在 Word/WPS 中检查三线表显示、公式编号、图题表题、参考文献悬挂缩进、页眉页脚和作者单位格式。

## 6. 建议提交前顺序

1. 优先打开 `submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx`。
2. 根据 `submission_package/农业机械学报_红枣NER_投稿信息补全表.md` 替换作者、单位、基金、中图分类号等占位。
3. 对照 `submission_package/official_docs/论文编改专项核对表_2018-8-20.pdf` 和 `submission_package/evidence/official_checklist_audit_2026-05-23.md` 完成终检。
4. 若 Word/WPS 中图件显示异常，使用 `submission_package/figures_png/` 重新插入图 1-5。
5. 用 `submission_readiness_audit.md` 更新最终状态。
