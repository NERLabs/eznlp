# Skill 审查后论文修订记录

日期：2026-05-24

使用的本地 skill：

1. `agricultural-ner-paper-workflow`
2. `academic-paper-reviewer`
3. `academic-writing`

## 修订目标

根据自动审稿报告中可立即处理、且不依赖新增实验结果的意见，对当前论文进行文本层面的稳健化修订。

## 已完成修订

| 位置 | 修订内容 | 目的 |
|---|---|---|
| 引言 | 将红枣栽培任务与单一病虫害或单一环节 NER 区分开，突出生产全链条、多类别和低频复合术语 | 强化研究空白和红枣数据集独特性 |
| 引言 | 在三类困难中加入并列实体、复合术语边界偏移，并明确需要领域术语先验、位置区分和边界整体判断 | 让方法动机与 EDBP 结构更直接对应 |
| 1.1 语料构建 | 补充统一实体类别定义、边界判定规则、逐句复核和易分歧样本人工修正说明 | 增强数据标注质量说明，不虚构一致性数值 |
| 2.1 主对比 | 将“基线模型”限定为“已完成同口径基线模型” | 避免在强基线补实验完成前过度泛化 |
| 2.1 主对比 | 将“取得更高 F1”限定为“在已完成对比模型中取得更高 F1” | 保持结论与当前证据边界一致 |

## 未处理但保留的审稿意见

1. Boundary Smoothing、BERT-MRC、BERT-MRC+DSC、BERT+SoftLexicon 和 RA_NER/AdaSeq BERT-CRF 当前 RJND 口径补实验仍需服务器端执行。
2. 作者、单位、基金、作者简介、通信作者和中图分类号仍需作者提供。
3. 最新 PDF 仍需安装 LibreOffice 后重新生成。
4. Word/WPS 逐页终检仍需人工完成。

## 生成物

已重新生成：

```text
docs/paper/农业机械学报_红枣NER_投稿稿.html
docs/paper/农业机械学报_红枣NER_投稿稿_lo.html
docs/paper/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx
docs/paper/submission_package/农业机械学报_红枣NER_投稿稿.html
docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx
docs/paper/final_current/农业机械学报_红枣NER_最终查看稿.md
docs/paper/final_current/农业机械学报_红枣NER_最终查看稿.html
docs/paper/final_current/农业机械学报_红枣NER_最终查看稿_期刊格式初稿.docx
```
