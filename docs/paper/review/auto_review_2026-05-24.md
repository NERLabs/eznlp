# 自动审稿报告

审稿日期：2026-05-24

审稿对象：`docs/paper/final_current/农业机械学报_红枣NER_最终查看稿.md`

审稿依据：

1. 当前主稿与 `paper_result_registry.md`。
2. 当前实验需求清单 `current_rjnd_experiment_requirements.md`。
3. 本地同类论文资料：苹果栽培 NER、茶树病虫害 NER、苹果病虫害动态词典 NER、农业实体标注准则、FLAT。
4. 学术写作技能与 IMRaD 科学论文结构。

## 1. 编辑初筛意见

当前稿件具备农业信息技术论文的基本结构，摘要、引言、材料与方法、结果分析、讨论、结论和参考文献完整。主题聚焦红枣栽培命名实体识别，应用目标指向农业知识图谱、智能检索和问答服务，符合农业工程类期刊的应用导向。

暂不建议直接投稿，主要原因是投稿首页信息缺失、最新 PDF 尚未生成、主对比表仍缺当前 RJND 口径下的 Boundary Smoothing、BERT-MRC、BERT-MRC+DSC 和 BERT+SoftLexicon 等强基线。

## 2. 方法审稿意见

EDBP 的“训练集自动专家词典 + BMES 四通道编码 + 边界预测 + Focal Loss”技术链条清楚，方法章节已有公式和结构图。当前稿件没有再声称通道注意力为主模型组件，这一点与结果登记表保持一致。

主要风险在于边界预测与已有 span-based NER、Boundary Smoothing、BERT-MRC 类方法之间的差异还不够由实验充分支撑。若服务器补实验显示相关强基线明显高于 EDBP，摘要和结论需要调整为“可解释、领域词典友好、有竞争力”，而不是隐含“最优”。

## 3. 农业领域审稿意见

数据集构建部分已覆盖红枣生产全链条实体类型，包括品种、部位、时期、病虫害、药剂、肥料、产品/工艺等，较单一病虫害语料更分散。该点应继续作为红枣数据集区别于茶树病虫害、苹果病虫害 NER 的核心叙述。

不足是标注质量说明仍偏弱。若有人工复核比例、双人标注、一致性统计或标注冲突处理记录，应补入 1.1 节；若没有，则不要编造指标，保留局限性说明。

## 4. 实验充分性审稿意见

当前表 3 已比旧稿更完整，包含 BiLSTM-CRF、BERT-wwm-ext+BiLSTM+CRF、MacBERT-base+BiLSTM+CRF、SoftLexicon、FLAT、FLAT+BERT 和 EDBP。

仍需补强：

| 优先级 | 模型 | 原因 | 当前处理 |
|---:|---|---|---|
| 1 | Boundary Smoothing | 与边界预测主题最接近，是直接强基线 | 必跑 |
| 2 | BERT+SoftLexicon | 与词典增强主题最接近 | 必跑 |
| 3 | BERT-MRC | 经典 MRC-NER 强基线 | 必跑 |
| 4 | BERT-MRC+DSC | Dice/DSC 与类别不平衡损失相关 | 必跑 |
| 5 | RA_NER / AdaSeq BERT-CRF | 强 BERT-CRF 或检索增强基线 | 必跑 |
| 6 | W2NER | 新型统一建模方法，增强现代性 | 可选 |

所有结果必须使用当前 RJND/RedJujube 数据划分、seed=42、test split、同一实体级严格 P/R/F1 评估。旧版红枣结果不允许进入当前表 3。

## 5. 可复现性和投稿格式意见

可复现性材料已有基础，包括结果注册表、实验需求清单、分支工作流和 final_current 查看目录。

投稿格式仍有三项硬缺口：

1. 作者、单位、基金、作者简介、通信作者和中图分类号需要作者补齐。
2. LibreOffice 未安装，最新 DOCX 还没有重渲染 PDF。
3. Word/WPS 逐页终检尚未完成，尤其是公式对象、图表分页和参考文献悬挂缩进。

## 6. 必须修改项

1. 补齐投稿首页真实信息。
2. 安装 LibreOffice 后用最新 DOCX 重新生成 PDF。
3. 等实验端返回优先级 1-5 的当前 RJND 强基线结果后，重写表 3、摘要和结论中的对比结论。
4. 若新增实验进入正文，必须同步更新 `paper_result_registry.md` 和 submission package evidence 镜像。

## 7. 建议修改项

1. 在引言中更集中地突出“红枣栽培全链条实体体系”和“低频复合术语”的研究空白。
2. 在数据集构建中补充更具体的人工复核流程。
3. 在讨论中预设强基线可能高于 EDBP 时的解释框架：可解释词典先验、边界建模、专业农业文本适用性。

## 8. 不建议修改项

1. 不建议把旧版红枣高分结果直接写入当前正文。
2. 不建议为了追求高分混用 HZ、旧 RJND、公开数据集或 dev F1。
3. 不建议在无证据情况下新增通道注意力、动态图词典或端到端知识图谱验证等主贡献。

## 9. 是否需要新增实验

需要。当前最小必要补实验为：

```text
Boundary Smoothing
BERT+SoftLexicon / SoftLexicon+BERT
BERT-MRC
BERT-MRC+DSC
RA_NER / AdaSeq BERT-CRF
```

这些任务已同步到 `docs/paper/current_rjnd_experiment_requirements.md`，服务器实验端应优先读取该文件。
