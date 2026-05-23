# 《农业机械学报》Word 转排说明

对应主稿：`农业机械学报_红枣NER_投稿稿.md`

当前状态：Markdown 投稿稿已具备完整论文结构。已生成基础 `.docx`、带图 `.docx`、期刊格式 `.docx`、ODT 和 PDF 中间稿。期刊格式 `.docx` 已嵌入 5 张 PNG 图件，补入英文题名和投稿信息占位，并生成公式编号和三线表边框，但仍未套用官方模板。

已补充官方依据索引：`official_guidelines_index.md`。其中 `official_docs/论文编改专项核对表_2018-8-20.pdf` 已下载到本地，可用于后续 Word 终稿核对。

工具检查结果：

- 未检测到 `pandoc`。
- 已安装并使用 `python-docx` 生成带图 Word 初稿。
- 未检测到 `rsvg-convert`、`inkscape`、ImageMagick `convert`。
- 已安装并使用 `cairosvg` 将 SVG 导出为 PNG。

## 1. 转排前文件

| 用途 | 文件 |
|---|---|
| 主文稿 | `docs/paper/农业机械学报_红枣NER_投稿稿.md` |
| HTML 转排中间稿 | `docs/paper/农业机械学报_红枣NER_投稿稿.html` |
| Word 初稿 | `docs/paper/农业机械学报_红枣NER_投稿稿.docx` |
| 带图 Word 初稿 | `docs/paper/农业机械学报_红枣NER_投稿稿_带图.docx` |
| 期刊格式 Word 初稿 | `docs/paper/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx` |
| ODT 中间稿 | `docs/paper/农业机械学报_红枣NER_投稿稿_lo.odt` |
| PDF 预览稿 | `docs/paper/农业机械学报_红枣NER_投稿稿_lo.pdf` |
| 图表清单 | `docs/paper/农业机械学报_红枣NER_图表清单.md` |
| 结果注册表 | `docs/paper/paper_result_registry.md` |
| 图 5 证据 | `docs/paper/figure5_error_evidence.md` |
| 参考文献核验 | `docs/paper/reference_audit_round1.md` |
| 自审意见 | `docs/paper/internal_review_round1.md` |
| 修改记录 | `docs/paper/revision_log_round1.md` |

## 2. Word 结构要求

建议按以下顺序转排：

1. 题名
2. 中文摘要、关键词
3. 英文摘要、关键词
4. `0 引言`
5. `1 材料与方法`
6. `2 结果与分析`
7. `3 讨论`
8. `4 结论`
9. 参考文献

## 3. 图表处理

当前 5 张图均为 SVG，并已导出 PNG：

| 图号 | 文件 |
|---|---|
| 图 1 | `figures/fig1_edbp_architecture.svg`；`figures_png/fig1_edbp_architecture.png` |
| 图 2 | `figures/fig2_bmes_dictionary_encoding.svg`；`figures_png/fig2_bmes_dictionary_encoding.png` |
| 图 3 | `figures/fig3_boundary_prediction_decoder.svg`；`figures_png/fig3_boundary_prediction_decoder.png` |
| 图 4 | `figures/fig4_entity_f1_by_category.svg`；`figures_png/fig4_entity_f1_by_category.png` |
| 图 5 | `figures/fig5_boundary_error_cases.svg`；`figures_png/fig5_boundary_error_cases.png` |

Word 转排时建议：

1. 若模板支持 SVG，直接插入 SVG。
2. 若模板不支持 SVG，使用已导出的 PNG 图件。
3. 图题保留中英文双语，中文在前，英文在后。
4. 图中字体、线宽和字号在终版前统一。

## 4. 表格处理

主稿内表格均为 Markdown 表格。转 Word 后需检查：

1. 表题中英文双语是否完整。
2. 表头是否居中，数值保留位数是否统一。
3. 表 3、表 4、表 8 中 `±` 前后格式是否一致。
4. 表 5 需保留说明：该表为代表性运行，统计口径不同于表 3 的三种子均值。

## 5. 自动转排命令

若后续安装 `pandoc`，可在仓库根目录执行：

```bash
pandoc docs/paper/农业机械学报_红枣NER_投稿稿.md \
  -o docs/paper/农业机械学报_红枣NER_投稿稿.docx \
  --resource-path=docs/paper
```

当前已生成 HTML 中间稿：

```text
docs/paper/农业机械学报_红枣NER_投稿稿.html
```

该文件可用浏览器预览，也可尝试用 Word/WPS 打开后另存为 `.docx`，再套用官方模板。

当前也已生成基础 Word 初稿：

```text
docs/paper/农业机械学报_红枣NER_投稿稿.docx
```

该文件由 `docs/paper/tools/md_to_simple_docx.py` 使用 Python 标准库生成，保留 8 个表格和 5 个图件占位。正式投稿前需要将图件占位替换为真实图片，并按官方模板调整版式。

当前已生成带图 Word 初稿：

```text
docs/paper/农业机械学报_红枣NER_投稿稿_带图.docx
```

该文件由 `docs/paper/tools/md_to_docx_with_images.py` 使用 `python-docx` 生成，包含 8 个表格和 5 张嵌入 PNG 图。优先用该文件进入 Word/WPS 终稿转排。

当前已生成期刊格式 Word 初稿：

```text
docs/paper/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx
```

该文件使用同一脚本的 `--journal-style` 选项生成，包含英文题名、作者/单位/基金/中图分类号占位、公式编号和三线表边框。若作者信息尚未补齐，优先打开此文件替换占位内容。

若获得《农业机械学报》官方 Word 模板，可使用：

```bash
pandoc docs/paper/农业机械学报_红枣NER_投稿稿.md \
  -o docs/paper/农业机械学报_红枣NER_投稿稿.docx \
  --resource-path=docs/paper \
  --reference-doc=/path/to/农业机械学报模板.docx
```

## 6. 当前未完成

1. 未获取官方 Word 模板；官网投稿须知说明对格式版式无要求，但要求参照写作模板完善内容。
2. 期刊格式 `.docx` 已生成，但未套官方模板，三线表和公式编号仍需在 Word/WPS 中确认显示效果。
3. 已生成 PNG 图件；如期刊要求 TIFF，仍需额外导出。
4. 作者、单位、基金、中图分类号、作者简介和通信作者信息仍需作者补齐。
5. 转排后仍需人工检查图表跨页、表格宽度和参考文献悬挂缩进。
