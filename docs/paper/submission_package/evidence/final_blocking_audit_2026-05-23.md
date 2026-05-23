# Final Blocking Audit

审计日期：2026-05-23

目标：执行 `docs/paper/plans/农业机械学报投稿论文_goal计划.md`，形成面向《农业机械学报》的红枣栽培命名实体识别投稿稿和交接包。

## 已完成并可自动验证的部分

1. 投稿主稿已完成：`docs/paper/农业机械学报_红枣NER_投稿稿.md`。
2. 期刊格式 Word 初稿已生成：`docs/paper/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx`。
3. PDF 渲染预览已生成：`docs/paper/rendered/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf`。
4. 5 张正文图、8 张正文表已完成并通过编号、题名、引用和文件存在性检查。
5. 28 条参考文献已完成正文引用覆盖、首次引用顺序、基本格式和 DOI/URL 检查。
6. 主结果统一为 `88.28%±0.22%`，代表性单次运行 `89.51 / 87.58 / 88.54` 已在文中区分。
7. 公开数据集泛化结果、消融结果、词典策略结果和类别结果已纳入主稿。
8. 投稿交接包已生成：`docs/paper/submission_package/`。
9. 投稿交接压缩包已生成：`docs/paper/农业机械学报_红枣NER_投稿交接包_2026-05-23.zip`。
10. 全量自动检查已通过：`docs/paper/full_submission_check_report_2026-05-23.md`。
11. 投稿包自动检查已通过草稿模式：`docs/paper/submission_package/evidence/package_validation_draft_2026-05-23.md`。
12. PDF 自动质量检查已通过：A4 近似页面、13 页、关键文本存在、5 张正文图、图像不低于 250 ppi。

## 当前不能继续自动完成的部分

以下信息不能从项目文件或公开信息中可靠推断，必须由作者确认：

1. 中文作者姓名，建议不超过 6 人。
2. 英文作者姓名及拼写。
3. 中文单位、英文单位、城市、邮编。
4. 基金项目名称和编号。
5. 中图分类号，当前建议优先 `TP391.1`，需作者确认。
6. 作者简介：姓名、出生年、性别、职称/学位、研究方向、邮箱。
7. 通信作者简介：姓名、职称/学位、研究方向、邮箱。
8. 是否需要补充致谢、利益冲突或数据开放声明。

## 后续最短执行路径

1. 按 `docs/paper/submission_info.example.json` 填写真实信息，保存为 `docs/paper/submission_info.json`。
2. 运行：

```bash
python3 docs/paper/tools/validate_submission_info.py docs/paper/submission_info.json
python3 docs/paper/tools/build_final_submission_package.py --info docs/paper/submission_info.json
```

3. 在 Word/WPS 中打开生成的终稿，逐页人工检查：
   - 作者、单位、基金和作者简介是否完整；
   - 图题在图下、表题在表上；
   - 三线表、跨页表和公式编号显示是否正常；
   - 参考文献悬挂缩进、标点和换行是否符合投稿要求；
   - 页眉页脚、页码、英文摘要和关键词是否显示正常。

## 结论

当前工作已经达到“投稿交接包草稿完成并通过自动门禁”的状态。由于真实作者投稿信息和 Word/WPS 终稿显示检查不具备自动推断条件，本 goal 不能被判定为完全完成；在作者补齐 `submission_info.json` 并完成人工显示终检前，继续自动修改不会消除核心阻塞。
