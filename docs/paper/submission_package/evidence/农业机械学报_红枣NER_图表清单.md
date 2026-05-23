# 《农业机械学报》投稿稿图表清单

本文档对应 `农业机械学报_红枣NER_投稿稿.md`，用于后续制图、排版和投稿前核查。

## 图

| 编号 | 中文图题 | 英文图题 | 内容 | 状态 | 备注 |
|---|---|---|---|---|---|
| 图 1 | EDBP 模型总体结构 | Overall architecture of EDBP | 输入文本、MacBERT 字符表示、专家词典 BMES 四通道特征、特征拼接、边界预测解码器、Focal Loss | 已生成 SVG | `figures/fig1_edbp_architecture.svg` |
| 图 2 | 专家词典 BMES 四通道编码过程 | BMES four-channel encoding process of expert dictionary features | 词典匹配到 B/M/E/S 四通道 multi-hot，再到嵌入向量 | 已生成 SVG | `figures/fig2_bmes_dictionary_encoding.svg` |
| 图 3 | 边界预测解码器结构 | Structure of the boundary prediction decoder | 起始/结束位置 FFN、双仿射打分、span 分类矩阵 | 已生成 SVG | `figures/fig3_boundary_prediction_decoder.svg` |
| 图 4 | 不同实体类别识别效果 | Recognition performance of different entity categories | 14 类实体 F1 柱状图 | 已生成 SVG | `figures/fig4_entity_f1_by_category.svg` |
| 图 5 | 典型边界错误示例 | Typical boundary error cases | 低频复合实体、边界偏移和漏检案例；证据见 `figure5_error_evidence.md` | 已生成 SVG | `figures/fig5_boundary_error_cases.svg` |

## 表

| 编号 | 中文表题 | 英文表题 | 内容 | 状态 | 数据来源 |
|---|---|---|---|---|---|
| 表 1 | RJND 实体类别及数据集分布 | Entity categories and dataset distribution of RJND | 14 类实体定义、示例、训练/验证/测试实体数 | 已有，需排版 | 现有主稿表 1 |
| 表 2 | 实验参数设置 | Experimental parameter settings | 预训练模型、学习率、batch size、epoch、词典维度、span 参数等 | 待整理 | 结果 JSON `args` 字段 |
| 表 3 | 主模型与基线模型对比 | Comparison between EDBP and baseline models | BiLSTM-CRF、BERT-wwm、MacBERT、EDBP | 已有，需统一数值 | `paper_result_registry.md` |
| 表 4 | 消融实验结果 | Ablation results | 词典、边界预测、Focal Loss 组合 | 已有，需排版 | 主稿表 2 |
| 表 5 | 词典构建策略对比 | Comparison of lexicon construction strategies | 最小词频阈值、词典规模、长短实体覆盖率和匹配质量 | 已补入主稿 | `EXP-011-lexicon_strategy` |
| 表 6 | 解码器和损失函数对比 | Comparison of decoders and loss functions | CRF、BP、BP+Focal | 已有，需加表注 | 主稿表 3 |
| 表 7 | 不同实体类别识别效果 | Recognition results for different entity categories | 14 类实体 P/R/F1 | 已有，需排版 | 主稿表 5 |
| 表 8 | 公开数据集泛化实验 | Generalization results on public datasets | MSRA、WeiboNER、ResumeNER、Boson、CLUENER | 已有，需压缩 | 主稿表 8 |
| 补充表 | 投稿前结果溯源表 | Traceability table of reported results | 每个主数值的来源文件和统计口径 | 可作为补充材料 | `paper_result_registry.md` |

## 制图原则

1. 最终投稿图不使用 Mermaid 源码。
2. 图中术语统一为 EDBP、Expert dictionary、Boundary prediction、Focal Loss。
3. 中文图中文字尽量少，重点用模块名和箭头表达。
4. 输出格式优先 SVG/PDF，投稿前再转 300 dpi PNG/TIFF。
5. 所有图表须在正文中引用，不能只出现在清单中。
