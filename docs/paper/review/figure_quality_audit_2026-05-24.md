# 论文图片质量校验记录

日期：2026-05-24

状态：已完成图片质量审查，并已完成图 1-5 第一轮重画；当前记录作为后续终稿图件验收依据。

校验对象：

- `docs/paper/figures/fig1_edbp_architecture.svg`
- `docs/paper/figures/fig2_bmes_dictionary_encoding.svg`
- `docs/paper/figures/fig3_boundary_prediction_decoder.svg`
- `docs/paper/figures/fig4_entity_f1_by_category.svg`
- `docs/paper/figures/fig5_boundary_error_cases.svg`
- `docs/paper/figures_png/`
- `docs/paper/submission_package/figures_png/`
- `docs/paper/final_current/figures_png/`
- `docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx`
- `docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf`

## 1. 技术校验结论

| 项目 | 结论 |
|---|---|
| 图号连续性 | 通过，正文图 1-5 连续，且均在正文中先引用后给图题 |
| 源图存在性 | 通过，5 张 SVG 源图均存在 |
| PNG 镜像一致性 | 通过，`docs/paper/figures_png`、`submission_package/figures_png`、`final_current/figures_png` 三处 PNG 哈希一致 |
| DOCX 嵌图 | 通过，DOCX 内含 5 张 PNG，哈希与 `docs/paper/figures_png` 一致 |
| PDF 嵌图 | 通过，PDF 内含 5 张栅格图，均为 300 ppi |
| 尺寸 | 通过，PNG 尺寸为 2560 px 宽，满足打印清晰度 |

当前问题不是“清晰度不够”或“文件错乱”，而是视觉表达偏原始，图形仍像流程草图或占位图。

本轮校验使用以下命令作为证据来源：

```bash
python3 docs/paper/tools/validate_figures_tables.py docs/paper/农业机械学报_红枣NER_投稿稿.md
python3 docs/paper/tools/validate_rendered_pdf.py docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf
pdfimages -list docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf
sha256sum docs/paper/figures_png/*.png docs/paper/submission_package/figures_png/*.png docs/paper/final_current/figures_png/*.png
file docs/paper/figures_png/*.png docs/paper/submission_package/figures_png/*.png docs/paper/final_current/figures_png/*.png
```

关键证据摘要：

| 证据 | 结果 |
|---|---|
| PDF 内嵌图数量 | 5 张 |
| PDF 内嵌图分辨率 | 5 张均为 300 ppi |
| DOCX 内嵌图数量 | 5 张 |
| DOCX 内嵌图一致性 | 5 张图与 `docs/paper/figures_png` 哈希一致 |
| PNG 规格 | RGB、8-bit、非交错 PNG |
| PNG 尺寸 | 图 1/2/3/5 为 2560 x 1440；图 4 为 2560 x 1520 |
| 镜像目录一致性 | `figures_png`、`submission_package/figures_png`、`final_current/figures_png` 三处一致 |

## 2. 单图质量判断

| 图号 | 当前作用 | 技术状态 | 视觉问题 | 是否必须重画 | 重画优先级 |
|---|---|---|---|---|---:|
| 图 1 | EDBP 总体结构 | 清晰度合格 | 已改为分区式模型架构图，突出输入、双通道编码、融合、边界表示、解码与训练目标 | 已重画 | 1 |
| 图 2 | BMES 词典编码 | 清晰度合格 | 已加入红枣栽培短语样例、字符级 BMES 标注和四通道 multi-hot 映射 | 已重画 | 2 |
| 图 3 | 边界预测解码器 | 清晰度合格 | 已放大 span 分类矩阵，并标注 start、end、有效 span、无效区域和预测实体片段 | 已重画 | 3 |
| 图 4 | 各类别 F1 | 清晰度合格 | 已补充类别缩写解释，并突出 FER、TAX 两个低 F1 类别 | 已重画 | 4 |
| 图 5 | 错误案例 | 清晰度合格 | 已改为金标/预测 token-span 对照图，区分漏检和边界合并 | 已重画 | 2 |

## 3. 建议重画方向

重画时保持学术期刊风格，不使用过度装饰。所有图统一采用白底、低饱和配色、深灰文字和一致线宽。图中文字优先使用中文，英文仅保留图题或必要缩写。图内不要加入正文之外的新实验结论。

### 3.1 图 1

建议改成分区式模型架构图：

1. 左侧为输入与字符序列。
2. 中部上下双通道：MacBERT 上下文表示、专家词典 BMES 位置特征。
3. 中间用融合层统一收束。
4. 右侧拆成 start/end 表示、biaffine span scorer、实体类别输出。
5. 底部单独放 Focal Loss，明确它是训练目标，不是推理模块。

验收标准：

1. 图内能一眼区分编码、词典特征、融合、解码和损失 5 个区域。
2. `MacBERT`、`BMES`、`Biaffine span scorer`、`Focal Loss` 四个核心术语必须出现。
3. 箭头方向从左到右为主，不出现交叉箭头。
4. 图内文字不少于 12 pt 等效字号，PDF 中缩放后仍可读。

### 3.2 图 2

建议加入一个真实短语样例，例如“盛花期喷施硼肥”或论文中的红枣栽培实体片段，按字符展示：

```text
盛 花 期 喷 施 硼 肥
B M E O O B E
```

再展示四通道向量如何由 B/M/E/S 映射到 embedding。这样比当前纯流程图更能解释方法。

验收标准：

1. 必须有一个红枣栽培领域短语样例。
2. 必须展示字符、实体片段、BMES 标签和四通道向量之间的对应关系。
3. 必须标明词典来源为训练集自动抽取实体词典，避免被误解为人工外部词典。
4. 不能把 O 标签画成与 B/M/E/S 同等的词典通道；O 只能作为无词典匹配状态。

### 3.3 图 3

建议把 span 矩阵放大，并在矩阵旁标注：

1. 横轴为 end position。
2. 纵轴为 start position。
3. 上三角为有效 span。
4. 颜色块代表候选实体类别。
5. 加一个小公式框说明 `score(i,j,c)`。

验收标准：

1. span 矩阵占图宽度不少于 25%。
2. `start`、`end`、`span`、`entity type` 或对应中文概念必须清楚出现。
3. 上三角有效区域和无效区域必须有视觉区分。
4. 不新增与正文公式不一致的数学符号。

### 3.4 图 4

建议保留柱状图，但增强论文可读性：

1. 将类别缩写与中文类别名放进图例或脚注。
2. 用颜色突出 FER、TAX 等低 F1 类别。
3. 若有类别样本量，可加第二轴或在柱下标注样本数，解释低频类别困难。

验收标准：

1. 图中或图注必须解释 14 个类别缩写。
2. FER 和 TAX 等低 F1 类别必须在视觉上突出。
3. 如果没有可靠样本量，不要在图中添加样本量推断。
4. 坐标轴、数值和图例在 PDF 中不能重叠。

### 3.5 图 5

建议改成 token/span 对照图：

1. 每个案例一行展示原句片段。
2. 金标实体用绿色下划线或框。
3. 预测实体用蓝色框。
4. 漏检用红色虚线框。
5. 合并错误用跨 token 长框表示。

这样能比现在的文字说明更直观地呈现边界错误。

验收标准：

1. 每个案例必须同时展示金标和预测。
2. 漏检、边界合并和低频复合实体至少覆盖 2 类错误。
3. 证据来源继续指向 `redjujube_test.json` 和预测文件，不写入无法追溯的新案例。
4. 图内文字控制在短句或短标签，避免把正文分析搬进图里。

## 4. 推荐处理顺序

1. 先重画图 1，因为它决定方法章节第一印象。
2. 再重画图 2 和图 5，因为它们能体现本文的“词典先验”和“边界错误分析”特色。
3. 图 3 可在图 1 风格确定后同步美化。
4. 图 4 可最后处理，技术上已经可用，但需要补类别含义和低频类别强调。

## 5. 重画后的同步范围

每次更新图片后，至少同步以下位置：

| 类型 | 路径 |
|---|---|
| SVG 源图 | `docs/paper/figures/fig*.svg` |
| PNG 主图 | `docs/paper/figures_png/fig*.png` |
| 投稿包 PNG | `docs/paper/submission_package/figures_png/fig*.png` |
| 最终查看 PNG | `docs/paper/final_current/figures_png/fig*.png` |
| DOCX | `docs/paper/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx`、`docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.docx`、`docs/paper/final_current/农业机械学报_红枣NER_最终查看稿_期刊格式初稿.docx` |
| PDF | `docs/paper/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf`、`docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf`、`docs/paper/final_current/农业机械学报_红枣NER_最终查看稿_期刊格式初稿.pdf` |

重画后必须重新运行：

```bash
python3 docs/paper/tools/validate_figures_tables.py docs/paper/农业机械学报_红枣NER_投稿稿.md
python3 docs/paper/tools/validate_submission_package.py docs/paper/submission_package
python3 docs/paper/tools/validate_rendered_pdf.py docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf
pdfimages -list docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf
```

验收门槛：

1. 图表编号校验通过。
2. 投稿包草稿校验通过。
3. PDF 仍含 5 张正文图片。
4. PDF 内嵌图不低于 300 ppi；若 LibreOffice 压缩导致低于 250 ppi，不得进入最终包。
5. DOCX 内嵌图与 `docs/paper/figures_png` 保持哈希一致。

## 6. 重画执行记录

本轮已完成 5 张图的第一轮重画：

| 图号 | 完成内容 |
|---|---|
| 图 1 | 改为分区式 EDBP 架构图，显式呈现 `MacBERT`、`BMES`、`Biaffine span scorer` 和 `Focal Loss` |
| 图 2 | 加入“盛花期喷施硼肥”样例，展示字符、BMES 标签、四通道 multi-hot 和词典特征嵌入流程 |
| 图 3 | 放大 span 分类矩阵，标注 start/end 位置、有效候选区域、无效区域和实体类别色块 |
| 图 4 | 保留数值不变，补充 14 类实体缩写解释，并突出 FER、TAX 低频类别 |
| 图 5 | 改为 token/span 对照图，用绿色、蓝色、红色虚线分别表示金标、预测和漏检 |

已同步范围：

1. `docs/paper/figures/fig*.svg`
2. `docs/paper/figures_png/fig*.png`
3. `docs/paper/submission_package/figures_png/fig*.png`
4. `docs/paper/final_current/figures/fig*.svg`
5. `docs/paper/final_current/figures_png/fig*.png`
6. 主稿、投稿包和最终查看目录中的 DOCX/PDF。

已验证：

```bash
python3 docs/paper/tools/validate_figures_tables.py docs/paper/农业机械学报_红枣NER_投稿稿.md
python3 docs/paper/tools/validate_rendered_pdf.py docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf
python3 docs/paper/tools/validate_submission_package.py docs/paper/submission_package
pdfimages -list docs/paper/submission_package/农业机械学报_红枣NER_投稿稿_期刊格式初稿.pdf
```

验证结论：

| 项目 | 结果 |
|---|---|
| 图表编号 | 通过 |
| 投稿包草稿校验 | 通过 |
| DOCX 嵌图 | 5 张，且与 `docs/paper/figures_png` 哈希一致 |
| PDF 嵌图 | 5 张 |
| PDF 图像分辨率 | 5 张均为 300 ppi |

## 7. 当前可投稿风险判断

重画后的图片不会导致清晰度、嵌图或编号校验失败，且已明显改善模型结构、BMES 编码和错误案例图的论文观感。

正式投稿前仍建议在 Word/WPS 中人工检查分页后的图内文字是否足够清楚，尤其是图 2 的字符级标签和图 4 的类别缩写说明。若版面空间不足，可优先缩小图 4 的右侧说明框，而不要压缩图 1、图 2 和图 5。
