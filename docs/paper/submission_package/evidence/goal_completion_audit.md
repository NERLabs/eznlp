# Goal 交付项完成度审计

审计日期：2026-05-23

审计对象：`plans/农业机械学报投稿论文_goal计划.md`

## 1. 最终交付物核对

| 交付物 | 目标要求 | 当前文件 | 完成度 | 说明 |
|---|---|---|---|---|
| 投稿主文稿 Markdown 版 | 形成完整中文论文 | `农业机械学报_红枣NER_投稿稿.md` | 已完成初稿 | 已包含中英文摘要、引言、材料与方法、结果与分析、讨论、结论、参考文献 |
| 可转 Word 的规范稿件结构 | 可进入 Word/WPS 转排 | `农业机械学报_红枣NER_投稿稿.html`、`农业机械学报_红枣NER_投稿稿.docx`、`农业机械学报_红枣NER_投稿稿_带图.docx`、`农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx`、`农业机械学报_红枣NER_投稿稿_lo.odt`、`农业机械学报_红枣NER_投稿稿_lo.pdf`、`农业机械学报_红枣NER_Word转排说明.md` | 基本完成 | 已生成期刊格式 DOCX、带图 DOCX、ODT 和 PDF；尚未套官方模板，作者信息需补齐 |
| 论文图表与图题表题清单 | 图表齐全且可追溯 | `农业机械学报_红枣NER_图表清单.md`、`figures/`、`figures_png/` | 已完成初版 | 5 张 SVG 图、5 张 PNG 图、8 张表已在主稿中引用；正式投稿前需检查图件清晰度 |
| 实验结果证据表和可追溯结果索引 | 明确所有采用结果来源 | `paper_result_registry.md`、`figure5_error_evidence.md` | 已完成初核 | 主结果统一为 88.28%±0.22%；代表性分类分析使用 88.54 |
| 自审意见与逐条修改记录 | 模拟审稿并记录修改 | `internal_review_round1.md`、`revision_log_round1.md` | 已完成 Round 1 | 已覆盖投稿定位、数值一致性、方法一致性、文献风险和图件问题 |
| 投稿前检查清单 | 提供终检依据 | `投稿前检查清单.md`、`submission_readiness_audit.md`、`official_checklist_audit_2026-05-23.md`、`农业机械学报_红枣NER_官方核对附录.md` | 已完成初版 | 已按官方编改核对表做逐项审计，并补变量表和结论性数据汇总表；终检仍依赖作者信息和 Word 显示检查 |
| 官方投稿依据索引 | 联网核对投稿要求 | `official_guidelines_index.md`、`official_docs/论文编改专项核对表_2018-8-20.pdf`、`official_docs/农业机械学报论文编改专项核对表_2026.doc` | 已完成初核 | 官网投稿须知说明参考文献至少 25 条、英文摘要参照 EI 要求，且“对论文格式、版式无要求”；未直接取得 Word 论文写作模板文件 |
| 主稿质量自动终检 | 题名、关键词、章节、术语和非投稿内容可验证 | `manuscript_quality_validation_2026-05-23.md` | 已完成 | 题名 24 个汉字、关键词 6 个、章节完整、无 Mermaid/代码块/EDBS/TODO |
| 参考文献格式自动终检 | 文献数量、类型标识、年份、DOI/URL、近年与外文文献可验证 | `reference_format_validation_2026-05-23.md` | 已完成 | 28 条文献均有类型标识、年份和 DOI/URL，近年文献 18 条、外文文献 19 条 |
| 公式和符号自动终检 | 展示公式、核心公式、符号附录和 DOCX 公式编号可验证 | `equation_symbol_validation_2026-05-23.md` | 已完成 | 主稿 10 个展示公式，符号附录覆盖 28 个必需符号，DOCX 公式编号连续 |
| 全量投稿检查 | 所有自动门禁和交接包完整性可一次性验证 | `full_submission_check_report_2026-05-23.md` | 已完成 | 引用、文献格式、数值、图表、公式、主稿质量、投稿信息模板、投稿包和 zip 完整性均通过 |
| 复现实验说明 | 数据、代码、模型与结果来源可追溯 | `reproducibility_statement.md`、`reproducibility_validation_2026-05-23.md` | 已完成 | RJND/公开数据集、训练入口、结果目录和采用口径均已记录并校验 |
| PDF 渲染质量自动终检 | PDF 页数、页面规格、关键文本和图像质量可验证 | `rendered_pdf_validation_2026-05-23.md` | 已完成 | PDF 为 A4、14 页、含关键文本、5 张正文图且图像不低于 250 ppi |
| 投稿信息补全表 | 补齐作者相关投稿字段 | `农业机械学报_红枣NER_投稿信息补全表.md` | 已建立，待作者填写 | 作者、单位、基金、中图分类号等不能由项目文件推断 |
| 投稿信息 JSON 校验 | 防止占位信息进入终稿 | `tools/validate_submission_info.py`、`submission_info_template_validation_2026-05-23.md` | 已完成 | 严格模式会拦截“待作者补充/To be completed”，终稿构建已用测试信息跑通 |
| 投稿交接包 | 集中交付可用稿件和证据材料 | `submission_package/`、`农业机械学报_红枣NER_投稿交接包_2026-05-23.zip` | 已完成 | 压缩包已通过 `unzip -t` 完整性测试 |

## 2. 计划阶段核对

| 阶段 | 计划目标 | 当前证据 | 状态 |
|---|---|---|---|
| Phase 1 证据归档与数值统一 | 建立唯一可信结果来源 | `paper_result_registry.md`，主稿摘要/表 3/结论均采用 88.28%±0.22% | 已完成 |
| Phase 2 期刊适配改写 | 改为农业信息技术论文 | 主稿围绕红枣栽培知识结构化、知识图谱、智能检索和问答应用组织 | 已完成初稿 |
| Phase 3 方法章节规范化 | 方法描述与源码一致 | `source_consistency_check.md`；主稿未把通道注意力、BMES span feature、enhanced size embedding 或 LogN scaling 写成主贡献 | 已完成初核 |
| Phase 4 图表重构 | 不使用 Mermaid，生成正式图表 | `figures/fig1` 至 `fig5`，`figures_png/fig1` 至 `fig5`，主稿 8 张表 | 已完成初版 |
| Phase 5 内部审稿 | 输出审稿意见和修改记录 | `internal_review_round1.md`、`revision_log_round1.md` | 已完成 Round 1 |
| Phase 6 参考文献补齐 | 不少于 25 条并尽量核验 | 主稿 28 条；`reference_audit_round1.md` 记录 DOI/URL 核验 | 数量完成，格式待终核 |
| Phase 7 投稿前终检 | 形成可提交版本 | `投稿前检查清单.md`、`submission_readiness_audit.md`、`official_checklist_audit_2026-05-23.md` | 未完全完成 |

## 3. 当前不能判定为最终投稿完成的原因

1. 官网检索到下载中心、论文写作模板 PDF 和编改核对表，但未直接取得《农业机械学报》官方 Word 模板；官网投稿须知同时说明对格式、版式无要求。
2. 作者、单位、基金、中图分类号、作者简介和通信作者信息缺失，不能由当前项目自动补齐。
3. 期刊格式 `.docx` 已嵌入图片、补入英文题名和投稿信息占位、生成公式编号和三线表边框，并已渲染 PDF；仍需在 Word/WPS 中确认最终显示效果。
4. Word/WPS 中的表格跨页、图题表题、页眉页脚、作者单位和参考文献悬挂缩进尚未人工终检。

## 4. 下一步最短路径

1. 作者填写 `农业机械学报_红枣NER_投稿信息补全表.md`。
2. 优先用 `submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx` 进入 Word/WPS。
3. 按官方编改核对表替换作者单位和基金占位，并检查三线表、公式编号、图表位置。
4. 对照 `投稿前检查清单.md` 与 `official_checklist_audit_2026-05-23.md` 做终检，并更新 `submission_readiness_audit.md`。
