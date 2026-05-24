# 源码一致性核对记录

本文档核对 `农业机械学报_红枣NER_投稿稿.md` 中方法描述与当前代码实现是否一致。

## 1. 专家词典特征层

源码位置：

- `eznlp/model/nested_embedder.py`
- `research/training/train_redjujube_expert_boundary.py`

核对结果：

1. `ExpertDictConfig` 默认字段为 `expert_dict`，默认 `num_channels=4`，对应 BMES 四通道。
2. 默认嵌入维度 `emb_dim=50`，与投稿稿中“每通道 50 维”一致。
3. 默认聚合方式为 `wtd_mean_pooling`，投稿稿中未展开聚合细节，后续若精修方法章节可补充。
4. `use_channel_attention` 默认值为 `False`；训练脚本中仅在传入 `--use_channel_attention` 时启用。因此投稿稿不将通道注意力作为主模型组件是合理的。
5. 代码保留 `ExpertDictWithChannelAttention` 与跨位置编码扩展接口，当前投稿稿已说明这些扩展不计入主结果。

结论：投稿稿对专家词典特征层的描述基本一致。

## 2. 边界预测解码器

源码位置：

- `eznlp/model/decoder/boundary_selection.py`
- `research/training/train_redjujube_expert_boundary.py`

核对结果：

1. `BoundarySelectionDecoderConfig` 使用两个 reduction encoder，分别实例化为 `reduction_start` 和 `reduction_end`，与投稿稿“起始 FFN/结束 FFN”一致。
2. 解码器参数包含 `size_emb_dim`，默认 25，与投稿稿实验参数表一致。
3. 打分由双线性项 `U`、线性项 `W` 和偏置 `b` 组成，与投稿稿双仿射公式一致。
4. 代码支持 `sb_epsilon` 和 `sb_size` 边界平滑；当 `fl_gamma > 0` 且使用平滑标签时，会实例化 `SoftLabelFocalLoss`。
5. 代码还支持 enhanced size embedding、LogN scaling、max span width、BMES span feature 等扩展项，但主结果配置中默认关闭或不作为主模块描述。

结论：投稿稿对边界预测解码器主干描述一致。已在实验参数表中补充 `sb_epsilon=0.1` 与 `sb_size=2`，并将其作为训练目标平滑策略描述。

## 3. 当前需要注意的点

1. 主稿使用“边界预测”作为论文术语，源码类名为 `BoundarySelectionDecoder`。这不是矛盾，但需要在正文中保持解释：边界预测模块采用边界选择式 span 分类解码实现。
2. 若后续采用带通道注意力的新实验结果，则必须重写方法章节并补充通道注意力消融。
3. 若后续采用 BMES span feature、enhanced size embedding 或 LogN scaling 扩展结果，则需要新增公式和消融。当前主稿不将这些扩展作为主方法贡献。
