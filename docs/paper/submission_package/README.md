# 《农业机械学报》投稿交接包

生成日期：2026-05-23

## 1. 优先使用文件

1. `农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx`
   - 当前最接近投稿形态的 Word 初稿。
   - 已包含英文题名、作者/单位/基金/中图分类号占位、公式编号、三线表边框、5 张嵌入图和 8 张表。
2. `农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf`
   - 由 LibreOffice 渲染的预览稿，用于快速检查页数、图表和正文显示。
3. `农业机械学报_红枣NER_投稿信息补全表.md`
   - 作者、单位、基金、中图分类号、作者简介和通信作者信息补全表。
4. `submission_info.example.json`
   - 作者信息脚本化替换模板。

## 2. 图件

`figures_png/` 中包含 5 张 PNG 图件：

1. `fig1_edbp_architecture.png`
2. `fig2_bmes_dictionary_encoding.png`
3. `fig3_boundary_prediction_decoder.png`
4. `fig4_entity_f1_by_category.png`
5. `fig5_boundary_error_cases.png`

这些图件已嵌入 Word 初稿。若 Word/WPS 中显示异常，可重新插入本目录 PNG。

## 3. 证据材料

`evidence/` 目录包含结果注册表、图表清单、参考文献核验、复现实验说明、官方核对表审计、图 5 错例证据和官方核对附录。它们不一定随稿提交，但用于投稿前自查和答复审稿。

其中 `农业机械学报_红枣NER_中图分类号建议.md` 给出中图分类号建议：优先 `TP391.1`，备选 `TP391`，终稿前仍需作者确认。

## 4. 官方附件

`official_docs/论文写作模板_2019.pdf` 和 `official_docs/论文编改专项核对表_2018-8-20.pdf` 为《农业机械学报》官网附件，终稿前需逐项核对。

## 5. 提交前必须替换的占位

1. 作者中文名和英文名。
2. 作者单位中文全称、英文全称、城市和邮编。
3. 中图分类号。
4. 基金项目名称和编号。
5. 第一作者简介。
6. 通信作者简介。
7. 投稿联系人信息。

如需脚本化替换 Word 首页占位，可填写 `submission_info.example.json`，另存为 `submission_info.json`，然后使用 `tools/fill_docx_front_matter.py`。

也可使用终稿构建总控脚本，一次性完成占位替换、PDF 渲染、终稿验证和压缩包生成：

```bash
python3 tools/validate_submission_info.py submission_info.json

python3 tools/build_final_submission_package.py \
  --info submission_info.json \
  --output-dir ../final_submission_package \
  --zip ../农业机械学报_红枣NER_终稿交接包_2026-05-23.zip
```

替换后可运行投稿包验证：

```bash
python3 tools/validate_submission_package.py . --final
```

当前未填写作者信息时，可运行草稿模式：

```bash
python3 tools/validate_submission_package.py .
```

也可运行全量检查：

```bash
python3 tools/run_all_submission_checks.py
```

## 6. 已验证

1. Word 初稿 zip 结构通过。
2. Word 初稿内含 5 张嵌入图和 8 张表。
3. Word 初稿内检出英文题名、作者占位、中图分类号占位、公式编号和主结果 88.28%±0.22%。
4. PDF 预览为 A4，共 14 页。
5. PDF 中图像约 300 dpi。
6. 参考文献 [26] 已修正作者名并补 DOI `10.6041/j.issn.1000-1298.2025.11.050`。
7. 草稿模式投稿包验证通过，唯一警告为作者信息占位未替换。
8. 终稿构建总控脚本已用测试作者信息跑通，终稿模式验证可通过。
9. 正文引用编号已补齐并验证，正文引用覆盖文末 [1]-[28]，首次引用顺序递增。
10. 摘要、结果表、泛化实验和结论中的核心数值已通过一致性验证。
11. 正文图 1-5、表 1-8 的编号、双语题名、正文先导引用和图件文件已通过一致性验证。
12. 词典构建策略表已补入主稿，数据来自 EXP-011 词典策略分析结果。
13. 主稿质量门禁已通过，覆盖题名长度、关键词数量、章节结构、参考文献数量、术语统一和非投稿内容检查。
14. 参考文献格式门禁已通过，覆盖类型标识、年份、DOI/URL、近年文献和外文文献数量。
15. 投稿信息 JSON 严格校验已接入填充脚本和终稿构建流程；用测试作者信息跑通终稿模式验证。
16. 公式和符号门禁已通过，覆盖 10 个展示公式、28 个符号和 DOCX 连续公式编号。
17. 全量投稿检查已通过，报告见 `evidence/full_submission_check_report_2026-05-23.md`。
18. 复现实验说明和关键路径校验已完成，报告见 `evidence/reproducibility_validation_2026-05-23.md`。
19. PDF 渲染质量门禁已通过，报告见 `evidence/rendered_pdf_validation_2026-05-23.md`。
