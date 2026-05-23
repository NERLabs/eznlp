# 修改记录 Round 1

对应计划：`plans/农业机械学报投稿论文_goal计划.md`

## 已完成

1. 新建结果注册表：`paper_result_registry.md`。
2. 明确主结果采用 88.28±0.22，单次 88.54 仅用于分类分析。
3. 统一模型名称建议：EDBP，中文为“专家词典与边界预测模型”。
4. 新建图表清单：`农业机械学报_红枣NER_图表清单.md`。
5. 新建内部审稿意见：`internal_review_round1.md`。
6. 开始生成投稿稿初版：`农业机械学报_红枣NER_投稿稿.md`。
7. 生成 3 张方法图 SVG：
   - `figures/fig1_edbp_architecture.svg`
   - `figures/fig2_bmes_dictionary_encoding.svg`
   - `figures/fig3_boundary_prediction_decoder.svg`
8. 将投稿稿中的图 1、图 2、图 3 替换为 SVG 图像引用。
9. 生成并嵌入图 4、图 5：
   - `figures/fig4_entity_f1_by_category.svg`
   - `figures/fig5_boundary_error_cases.svg`
10. 根据源码核对，补充说明主模型未将通道注意力作为核心组件，边界预测实现包含 span 长度嵌入、边界平滑和 fl_gamma 触发的 Focal Loss。
11. 替换投稿稿中信息不完整的参考文献 26-28，补入农业机械学报茶树病虫害 NER、GatedMan 作物病虫害 NER、作物病虫害多模态 NER 文献。
12. 新增 `figure5_error_evidence.md`，把图 5 更新为有金标和预测文件支撑的肥料类漏检/边界合并案例，并移除未在当前测试错例中定位到的“人粪尿”实证展示。
13. 替换未能稳定核验的农业知识图谱综述参考文献 24-25，改为期刊官网可核验的农业工程学报/智慧农业综述文献。
14. 对参考文献 1、7-23、27-28 进行集中核验，补充 DOI/URL，并修正第 20 条会议来源。
15. 新增 `投稿前检查清单.md` 和 `农业机械学报_红枣NER_Word转排说明.md`，补齐计划要求的投稿前检查与 Word 转排准备材料。
16. 在实验参数表中补充边界平滑系数和邻域大小，并更新源码一致性核对结论。
17. 生成 `农业机械学报_红枣NER_投稿稿.html` 作为可浏览和可转排中间稿。
18. 编写 `tools/md_to_simple_docx.py`，生成基础 Word 初稿 `农业机械学报_红枣NER_投稿稿.docx`；该版本保留正文、表格和图件占位，但尚未套用《农业机械学报》官方模板。
19. 新增 `农业机械学报_投稿包_manifest.md` 和 `submission_readiness_audit.md`，明确投稿包文件、当前完成度和剩余模板适配工作。
20. 安装并使用 `cairosvg` 将 5 张 SVG 图导出为 `figures_png/*.png`。
21. 编写 `tools/md_to_docx_with_images.py`，生成含 5 张嵌入图件和 8 张表的 Word 初稿 `农业机械学报_红枣NER_投稿稿_带图.docx`。
22. 使用 LibreOffice 生成 `农业机械学报_红枣NER_投稿稿_lo.odt` 和 `农业机械学报_红枣NER_投稿稿_lo.pdf`，作为 Word/WPS 转排中间稿和预览稿。
23. 新增 `official_checklist_audit_2026-05-23.md`，按官网编改专项核对表逐项审计主稿。
24. 新增 `农业机械学报_红枣NER_投稿信息补全表.md`，列出作者、单位、基金、中图分类号和通信作者等需作者提供的信息。
25. 扩展 `tools/md_to_docx_with_images.py`，新增 `--journal-style` 选项，自动生成英文题名、投稿信息占位、公式编号和三线表边框。
26. 生成 `农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx`，作为当前优先转排入口。
27. 新增 `农业机械学报_红枣NER_官方核对附录.md`，整理文中主要符号表和结论性数据汇总表。
28. 使用 LibreOffice 将期刊格式 Word 初稿渲染为 `rendered/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf`，并新增 `rendered/render_audit_2026-05-23.md` 记录页数、图像和文本核验。
29. 建立 `submission_package/` 投稿交接目录并生成 `农业机械学报_红枣NER_投稿交接包_2026-05-23.zip`，压缩包已通过完整性测试。
30. 复核参考文献 [26]，修正作者名为“张华扬”，补 DOI `10.6041/j.issn.1000-1298.2025.11.050`，并重建期刊格式 Word/PDF 预览。
31. 新增 `submission_info.example.json` 和 `tools/fill_docx_front_matter.py`，支持作者信息补齐后的首页占位自动替换。
32. 新增 `tools/validate_submission_package.py`，对交接包文件完整性、DOCX 图表数量、PDF 文本、参考文献数量和压缩包完整性进行自动验证；草稿模式已通过。
33. 新增 `tools/build_final_submission_package.py`，用测试作者信息跑通“替换占位 -> 渲染 PDF -> 终稿验证 -> 生成终稿压缩包”流程。
34. 新增 `农业机械学报_红枣NER_中图分类号建议.md`，建议中图分类号优先 `TP391.1`、备选 `TP391`，并同步到投稿信息补全表和 JSON 示例。
35. 在引言中补齐正文参考文献引用编号，覆盖 [1]-[28]，并新增 `tools/validate_references.py` 与 `reference_citation_validation_2026-05-23.md` 验证引用顺序和覆盖。
36. 新增 `tools/validate_numeric_consistency.py` 与 `numeric_consistency_validation_2026-05-23.md`，验证摘要、结果表、泛化实验和结论中的核心数值一致性。
37. 新增 `tools/validate_figures_tables.py` 与 `figure_table_validation_2026-05-23.md`，验证图 1-5、表 1-8 的编号、双语题名、正文先导引用和图件文件一致性，并接入投稿包自动验证。
38. 根据原计划补入 EXP-011 词典构建策略对比表，将公开数据集泛化实验顺延为表 8，并重建 HTML、DOCX、ODT 和 PDF 中间稿。
39. 下载官网 `论文写作模板_2019.pdf`，并新增 `tools/validate_manuscript_quality.py` 与 `manuscript_quality_validation_2026-05-23.md`，对题名长度、关键词、章节结构、术语混用和非投稿内容进行自动终检。
40. 新增 `tools/validate_reference_format.py` 与 `reference_format_validation_2026-05-23.md`，验证文献类型标识、年份、DOI/URL、近年文献和外文文献数量，并接入投稿包自动验证。
41. 新增 `tools/validate_submission_info.py` 与 `submission_info_template_validation_2026-05-23.md`，对作者、单位、基金、中图分类号等投稿信息 JSON 做严格校验，并接入填充脚本、终稿构建脚本和投稿包验证。
42. 补齐官方核对附录中的词典层和边界层符号，并新增 `tools/validate_equations_symbols.py` 与 `equation_symbol_validation_2026-05-23.md`，验证公式数量、核心公式、符号附录和 DOCX 公式编号。
43. 新增 `tools/run_all_submission_checks.py` 与 `full_submission_check_report_2026-05-23.md`，一键汇总引用、参考文献、数值、图表、公式、主稿质量、投稿信息模板、投稿包和 zip 完整性检查。
44. 新增 `reproducibility_statement.md`、`tools/validate_reproducibility_paths.py` 与 `reproducibility_validation_2026-05-23.md`，整理 RJND/公开数据集、训练脚本、结果目录和结果采用口径，并接入全量检查。
45. 新增 `tools/validate_rendered_pdf.py` 与 `rendered_pdf_validation_2026-05-23.md`，验证 PDF 预览的页数、A4 页面、关键文本、正文图像数量和分辨率，并接入投稿包验证。

## 核心修改

| 问题 | 修改前 | 修改后 |
|---|---|---|
| 模型名混乱 | EDBS/EDBP 混用 | 统一使用 EDBP |
| 主结果冲突 | 摘要写 88.54，正文写 88.28±0.22 | 摘要主结果采用 88.28±0.22 |
| 投稿定位偏 NLP | 强调模型模块 | 强调红枣栽培知识结构化与农业信息技术 |
| Mermaid 图直接入稿 | 图为代码块 | 建立正式图表清单，后续绘图 |
| 新旧数据混用风险 | 2025/2026 结果可能混写 | 注册表中分离旧稿主结果和新数据结果 |
| 方法图缺失 | 仅有“待绘制”占位 | 先生成 3 张 SVG 方法图 |
| 类别分析缺少可视化 | 仅有表 6 | 新增图 4 类别 F1 柱状图 |
| 错误案例缺少图示 | 只有文字说明 | 新增图 5 低频实体错误示例图，并补充预测文件证据索引 |
| Word 转排材料缺失 | 只有 Markdown 稿件 | 生成 HTML 中间稿、基础 DOCX 和 Word 转排说明 |
| 图件未嵌入 Word | 基础 DOCX 中只有图件占位 | 生成 PNG 图件并建立带图 DOCX |
| 官方编改核对不足 | 只有普通投稿前清单 | 按官网 PDF 核对表生成逐项审计 |
| 三线表和公式编号需手工处理 | 带图 DOCX 仍为网格表，公式无编号 | 生成期刊格式 DOCX，自动加入三线表边框和公式编号 |
| 交接材料分散 | 投稿稿、图件、证据和官方附件分散在 `docs/paper/` | 建立 `submission_package/` 并生成压缩交接包 |
| 作者信息替换易出错 | Word 首页占位需手动替换 | 提供 JSON 模板和替换脚本 |
| 终检依赖人工记忆 | 投稿包文件、图件、PDF、参考文献需逐项手工查 | 提供投稿包自动验证脚本和验证报告 |
| 填完作者信息后步骤分散 | 需手动替换、导出 PDF、验证、打包 | 提供终稿构建总控脚本 |
| 正文无文献引用编号 | 文末有参考文献但正文未按序引用 | 在正文补齐 [1]-[28] 引用并加入自动验证 |
| 核心数值需人工比对 | 摘要、结果表、结论可能出现数值不一致 | 加入数值一致性自动验证 |
| 图表终检依赖人工比对 | 图号、表号、图件文件和正文先导引用需逐项核对 | 加入图表一致性自动验证 |
| 主稿质量门禁分散 | 题名、关键词、章节和非投稿内容依赖清单人工勾选 | 加入主稿质量自动验证 |
| 参考文献格式仍靠人工抽查 | 只验证引用覆盖和数量，未验证类型标识与 DOI/URL | 加入参考文献格式自动门禁 |
| 投稿信息占位可能误入终稿 | 只检查 JSON 键存在，不拦截占位内容 | 加入投稿信息严格校验并在终稿构建前执行 |
| 公式符号终检依赖人工 | 只有渲染核验记录，缺少公式和符号覆盖检查 | 加入公式符号自动门禁 |
| 多个验证脚本容易漏跑 | 单项门禁分散在多个脚本 | 加入全量投稿检查总控脚本 |
| 复现实验路径分散 | 数据、脚本和结果来源分散在注册表、Taskfile 和实验目录 | 新增复现实验说明和路径校验 |
| PDF 预览质量只靠人工记录 | 渲染审计不是可执行门禁 | 加入 PDF 渲染质量自动验证 |

## 待完成

1. 继续确认参考文献 26 是否已有 DOI，并在 Word 转排前统一 GB/T 7714 标点与大小写。
2. 作者补齐投稿首页信息：作者、单位、基金、中图分类号、作者简介和通信作者。
3. 在 Word/WPS 中检查期刊格式 DOCX 的三线表、公式编号、参考文献悬挂缩进和页眉页脚。
4. 若投稿需要完全对应主实验单次运行，从主实验 `predictions_test.pt` 中抽取真实错误案例替换当前 `EXP-009` 辅助预测样例。
