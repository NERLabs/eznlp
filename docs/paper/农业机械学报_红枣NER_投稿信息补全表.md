# 投稿信息补全表

用途：补齐《农业机械学报》投稿系统和论文写作模板中需要作者确认的信息。当前主稿正文已完成，但以下信息不能由代码或实验结果推断，需要作者提供。

## 1. 作者与单位

| 项目 | 待填写内容 | 备注 |
|---|---|---|
| 作者中文名 |  | 建议不超过 6 位作者 |
| 作者英文名 |  | 英文题名下作者姓字母大写，如 ZHANG San |
| 作者排序 |  | 与投稿系统一致 |
| 作者单位中文全称 |  | 至少到二级单位，使用规范全称 |
| 作者单位英文全称 |  | 与单位官方英文名称一致 |
| 城市、邮编 |  | 中文和英文单位均需对应 |

## 2. 论文首页信息

| 项目 | 待填写内容 | 备注 |
|---|---|---|
| 英文题名 | Named Entity Recognition Method for Red Jujube Cultivation Based on Expert Dictionary and Boundary Prediction | 可作为初稿英文题名，终稿可再润色 |
| 中图分类号 | 建议 `TP391.1`，备选 `TP391` | 不超过 2 个，需按《中国图书馆分类法》正式表确认；依据见 `农业机械学报_红枣NER_中图分类号建议.md` |
| 文献标识码 | A | 通常由模板给出，投稿前确认 |
| 文章编号 | 编辑部填写 | 不需要作者填写 |
| 收稿日期 | 投稿后填写 | 初投稿可留空 |
| 修回日期 | 修改稿填写 | 初投稿可留空 |

## 3. 基金与作者简介

| 项目 | 待填写内容 | 备注 |
|---|---|---|
| 基金项目 |  | 填写项目名称和编号；如无基金需确认投稿要求 |
| 第一作者简介 |  | 姓名、出生年、性别、学历/职称、研究领域、邮箱 |
| 通信作者简介 |  | 姓名、职称/学历、研究领域、邮箱 |
| 投稿联系人 |  | 与投稿系统联系人保持一致 |

## 4. 终稿前必须确认

1. 作者排序、单位、基金和通信作者信息与投稿系统完全一致。
2. 英文作者名和英文单位采用官方写法。
3. 中图分类号由作者按正式分类表确认。
4. 若使用基金项目，编号必须与立项文件一致。

## 5. 可选脚本化替换

若需要自动替换 `农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx` 中的首页占位，可复制并填写：

```text
docs/paper/submission_info.example.json
```

填写后执行：

```bash
python3 docs/paper/tools/validate_submission_info.py \
  docs/paper/submission_info.json

python3 docs/paper/tools/fill_docx_front_matter.py \
  docs/paper/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx \
  docs/paper/submission_info.json \
  docs/paper/农业机械学报_红枣NER_投稿稿_终稿待检.docx
```

生成的 `.docx` 仍需在 Word/WPS 中检查显示效果。

如需一次性生成终稿交接包，可执行：

```bash
python3 docs/paper/tools/build_final_submission_package.py \
  --info docs/paper/submission_info.json
```

该流程会先严格校验投稿信息 JSON，若仍含“待作者补充”“待作者确认”或 `To be completed`，会直接停止。
