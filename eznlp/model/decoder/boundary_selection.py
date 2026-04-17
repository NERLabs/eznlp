# -*- coding: utf-8 -*-
import logging
import math
from collections import Counter
from typing import List

import numpy
import torch

from ...metrics import precision_recall_f1_report
from ...nn.init import reinit_embedding_
from ...nn.modules import CombinedDropout, SoftLabelCrossEntropyLoss
from ...utils.chunk import detect_overlapping_level, filter_clashed_by_priority
from ...wrapper import Batch
from ..encoder import EncoderConfig
from .base import DecoderBase, DecoderMixinBase, SingleDecoderConfigBase
from .boundaries import MAX_SIZE_ID_COV_RATE, Boundaries, _spans_from_upper_triangular

logger = logging.getLogger(__name__)


class BoundariesDecoderMixin(DecoderMixinBase):
    """The standard `Mixin` for span-based entity recognition."""

    @property
    def idx2label(self):
        return self._idx2label

    @idx2label.setter
    def idx2label(self, idx2label: List[str]):
        self._idx2label = idx2label
        self.label2idx = (
            {l: i for i, l in enumerate(idx2label)} if idx2label is not None else None
        )

    @property
    def voc_dim(self):
        return len(self.label2idx)

    @property
    def none_idx(self):
        return self.label2idx[self.none_label]

    def exemplify(self, data_entry: dict, training: bool = True):
        return {"boundaries_obj": Boundaries(data_entry, self, training=training)}

    def batchify(self, batch_examples: List[dict]):
        return {"boundaries_objs": [ex["boundaries_obj"] for ex in batch_examples]}

    def retrieve(self, batch: Batch):
        return [boundaries_obj.chunks for boundaries_obj in batch.boundaries_objs]

    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        """Micro-F1 for entity recognition.

        References
        ----------
        https://www.clips.uantwerpen.be/conll2000/chunking/output.html
        """
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores["micro"]["f1"]

    def _filter(self, chunks: List[tuple], confidences: List[float], boundaries_obj):
        if hasattr(boundaries_obj, "sub2ori_idx"):
            is_valid = [
                isinstance(boundaries_obj.sub2ori_idx[start], int)
                and isinstance(boundaries_obj.sub2ori_idx[end], int)
                for label, start, end in chunks
            ]
            confidences = [conf for conf, is_v in zip(confidences, is_valid) if is_v]
            chunks = [ck for ck, is_v in zip(chunks, is_valid) if is_v]

        if hasattr(boundaries_obj, "tok2sent_idx"):
            is_valid = [
                boundaries_obj.tok2sent_idx[start]
                == boundaries_obj.tok2sent_idx[end - 1]
                for label, start, end in chunks
            ]
            confidences = [conf for conf, is_v in zip(confidences, is_valid) if is_v]
            chunks = [ck for ck, is_v in zip(chunks, is_valid) if is_v]

        if self.chunk_priority.lower().startswith("len"):
            # Sort chunks by lengths: long -> short
            chunks = sorted(chunks, key=lambda ck: ck[2] - ck[1], reverse=True)
        else:
            # Sort chunks by confidences: high -> low
            chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
        chunks = filter_clashed_by_priority(chunks, allow_level=self.overlapping_level)
        return chunks


class BoundarySelectionDecoderConfig(SingleDecoderConfigBase, BoundariesDecoderMixin):
    def __init__(self, **kwargs):
        self.reduction = kwargs.pop(
            "reduction",
            EncoderConfig(
                arch="FFN",
                hid_dim=150,
                num_layers=1,
                in_drop_rates=(0.0, 0.0, 0.0),
                hid_drop_rate=0.0,
            ),
        )

        self.max_len = kwargs.pop("max_len", None)
        self.size_emb_dim = kwargs.pop("size_emb_dim", 25)
        self.in_drop_rates = kwargs.pop("in_drop_rates", (0.4, 0.0, 0.0))
        self.hid_drop_rates = kwargs.pop("hid_drop_rates", (0.2, 0.0, 0.0))

        self.neg_sampling_rate = kwargs.pop("neg_sampling_rate", 1.0)
        self.neg_sampling_power_decay = kwargs.pop(
            "neg_sampling_power_decay", 0.0
        )  # decay = 0.5, 1.0
        self.neg_sampling_surr_rate = kwargs.pop("neg_sampling_surr_rate", 0.0)
        self.neg_sampling_surr_size = kwargs.pop("neg_sampling_surr_size", 5)
        self.nested_sampling_rate = kwargs.pop("nested_sampling_rate", 1.0)

        self.none_label = kwargs.pop("none_label", "<none>")
        self.idx2label = kwargs.pop("idx2label", None)
        self.overlapping_level = kwargs.pop("overlapping_level", None)
        self.chunk_priority = kwargs.pop("chunk_priority", "confidence")

        # Boundary smoothing epsilon
        self.sb_epsilon = kwargs.pop("sb_epsilon", 0.0)
        self.sb_size = kwargs.pop("sb_size", 1)
        self.sb_size_map = kwargs.pop("sb_size_map", None)  # dict: {label_str: int} 或 None
        self.sb_adj_factor = kwargs.pop("sb_adj_factor", 1.0)
        
        # Long entity optimization techniques
        self.enhanced_size_emb = kwargs.pop("enhanced_size_emb", False)
        self.use_lognscaling = kwargs.pop("use_lognscaling", False)
        self.lognscaling_base = kwargs.pop("lognscaling_base", 512)
        self.max_span_width = kwargs.pop("max_span_width", None)
        
        # NOTE: fl_gamma is handled by parent class SingleDecoderConfigBase
        # Do not pop it here, let parent class set self.fl_gamma
        
        super().__init__(**kwargs)

    @property
    def name(self):
        return self._name_sep.join([self.reduction.arch, self.criterion])

    @property
    def valid(self):
        """Check if the config is valid.
        
        Override base class to allow optional None attributes:
        - sb_size_map: Optional type-specific size mapping
        - idx2label: Built during build_vocab
        - overlapping_level: Built during build_vocab  
        - max_len: Built during build_vocab
        """
        optional_none_attrs = {'sb_size_map', 'idx2label', 'overlapping_level', 'max_len', 'enhanced_size_emb', 'use_lognscaling', 'lognscaling_base', 'max_span_width'}
        for name, attr in self.__dict__.items():
            if name in optional_none_attrs:
                continue
            if attr is None:
                return False
            elif hasattr(attr, 'valid') and not attr.valid:
                return False
        return True

    def __repr__(self):
        repr_attr_dict = {
            key: getattr(self, key)
            for key in ["in_dim", "in_drop_rates", "hid_drop_rates", "criterion"]
        }
        return self._repr_non_config_attrs(repr_attr_dict)

    @property
    def in_dim(self):
        return self.reduction.in_dim

    @in_dim.setter
    def in_dim(self, dim: int):
        self.reduction.in_dim = dim

    @property
    def criterion(self):
        if self.sb_epsilon > 0:
            crit_name = f"SB({self.sb_epsilon:.2f}, {self.sb_size})"
            return f"B{crit_name}" if self.multilabel else crit_name
        else:
            return super().criterion

    def instantiate_criterion(self, **kwargs):
        if self.multilabel:
            # `BCEWithLogitsLoss` allows the target to be any continuous value in [0, 1]
            return torch.nn.BCEWithLogitsLoss(**kwargs)
        elif self.criterion.lower().startswith(("sb", "sl")):
            # For boundary/label smoothing, the `Boundaries` object has been accordingly changed;
            # hence, do not use `SmoothLabelCrossEntropyLoss`
            if self.fl_gamma > 0:
                # Use Soft Label Focal Loss when fl_gamma is positive
                from ...nn.modules.loss import SoftLabelFocalLoss
                return SoftLabelFocalLoss(gamma=self.fl_gamma, **kwargs)
            else:
                return SoftLabelCrossEntropyLoss(**kwargs)
        else:
            return super().instantiate_criterion(**kwargs)

    def build_vocab(self, *partitions):
        counter = Counter(
            label
            for data in partitions
            for entry in data
            for label, start, end in entry["chunks"]
        )
        self.idx2label = [self.none_label] + list(counter.keys())

        self.overlapping_level = max(
            detect_overlapping_level(entry["chunks"])
            for data in partitions
            for entry in data
        )
        logger.info(f"Overlapping level: {self.overlapping_level}")

        span_sizes = [
            end - start
            for data in partitions
            for entry in data
            for label, start, end in entry["chunks"]
        ]
        self.max_size_id = (
            math.ceil(numpy.quantile(span_sizes, MAX_SIZE_ID_COV_RATE)) - 1
        )

        self.max_len = max(
            len(data_entry["tokens"]) for data in partitions for data_entry in data
        )

    def instantiate(self):
        return BoundarySelectionDecoder(self)


class BoundarySelectionDecoder(DecoderBase, BoundariesDecoderMixin):
    """边界选择解码器
    
    兼容性说明：旧模型可能没有 self.config 属性，
    _get_config_attr 方法提供安全的默认值访问。
    """
    
    def _get_config_attr(self, attr_name, default=None):
        """安全获取 config 属性，兼容旧模型
        
        Args:
            attr_name: 属性名
            default: 默认值
            
        Returns:
            属性值或默认值
        """
        if hasattr(self, 'config') and self.config is not None:
            return getattr(self.config, attr_name, default)
        return default
    
    def __init__(self, config: BoundarySelectionDecoderConfig):
        super().__init__()
        self.multilabel = config.multilabel
        self.conf_thresh = config.conf_thresh
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.overlapping_level = config.overlapping_level
        self.chunk_priority = config.chunk_priority

        self.reduction_start = config.reduction.instantiate()
        self.reduction_end = config.reduction.instantiate()

        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(
                config.max_size_id + 1, config.size_emb_dim
            )
            reinit_embedding_(self.size_embedding)
        
        # 技术1: 参数化 Size Embedding (残差 MLP 增强)
        if config.enhanced_size_emb and config.size_emb_dim > 0:
            self.size_mlp = torch.nn.Sequential(
                torch.nn.Linear(config.size_emb_dim, config.size_emb_dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(config.size_emb_dim * 2, config.size_emb_dim)
            )
            # 零初始化最后一层，确保初始行为等同于无 MLP
            torch.nn.init.zeros_(self.size_mlp[-1].weight)
            torch.nn.init.zeros_(self.size_mlp[-1].bias)
        else:
            self.size_mlp = None
        
        # 保存 config 引用以便在 compute_scores 中使用
        self.config = config

        self.register_buffer(
            "_span_size_ids",
            torch.arange(config.max_len) - torch.arange(config.max_len).unsqueeze(-1),
        )
        # Create `_span_non_mask` before changing values of `_span_size_ids`
        self.register_buffer("_span_non_mask", self._span_size_ids >= 0)
        self._span_size_ids.masked_fill_(self._span_size_ids < 0, 0)
        self._span_size_ids.masked_fill_(
            self._span_size_ids > config.max_size_id, config.max_size_id
        )

        self.in_dropout = CombinedDropout(*config.in_drop_rates)
        self.hid_dropout = CombinedDropout(*config.hid_drop_rates)

        self.U = torch.nn.Parameter(
            torch.empty(
                config.voc_dim, config.reduction.out_dim, config.reduction.out_dim
            )
        )
        self.W = torch.nn.Parameter(
            torch.empty(
                config.voc_dim, config.reduction.out_dim * 2 + config.size_emb_dim
            )
        )
        self.b = torch.nn.Parameter(torch.empty(config.voc_dim))
        torch.nn.init.orthogonal_(self.U.data)
        torch.nn.init.orthogonal_(self.W.data)
        torch.nn.init.zeros_(self.b.data)

        self.criterion = config.instantiate_criterion(reduction="sum")

    def _get_span_size_ids(self, seq_len: int):
        return self._span_size_ids[:seq_len, :seq_len]

    def _get_span_non_mask(self, seq_len: int):
        return self._span_non_mask[:seq_len, :seq_len]

    def compute_scores(
        self,
        batch: Batch,
        full_hidden: torch.Tensor,
        expert_bmes_channels: torch.Tensor = None,
    ):
        """
        计算跨度打分

        Args:
            full_hidden: (batch, step, hid_dim)
            expert_bmes_channels: (batch, step, 4, emb_dim), 可选
        """
        reduced_start = self.reduction_start(self.in_dropout(full_hidden), batch.mask)
        reduced_end = self.reduction_end(self.in_dropout(full_hidden), batch.mask)

        # 技术2: LogN-Scaling - 对长序列稳定打分
        seq_len = full_hidden.size(1)
        use_lognscaling = self._get_config_attr('use_lognscaling', False)
        lognscaling_base = self._get_config_attr('lognscaling_base', 512)
        if use_lognscaling and seq_len > lognscaling_base:
            log_scaling = math.log(seq_len) / math.log(lognscaling_base)
            U_scaled = self.U / (log_scaling ** 0.5)
        else:
            U_scaled = self.U
        
        # reduced_start: (batch, start_step, red_dim) -> (batch, 1, start_step, red_dim)
        # reduced_end: (batch, end_step, red_dim) -> (batch, 1, red_dim, end_step)
        # scores1: (batch, 1, start_step, red_dim) * (voc_dim, red_dim, red_dim) * (batch, 1, red_dim, end_step) -> (batch, voc_dim, start_step, end_step)
        scores1 = (
            self.hid_dropout(reduced_start)
            .unsqueeze(1)
            .matmul(U_scaled)
            .matmul(self.hid_dropout(reduced_end).permute(0, 2, 1).unsqueeze(1))
        )

        # reduced_cat: (batch, start_step, end_step, red_dim*2)
        reduced_cat = torch.cat(
            [
                self.hid_dropout(reduced_start)
                .unsqueeze(2)
                .expand(-1, -1, reduced_end.size(1), -1),
                self.hid_dropout(reduced_end)
                .unsqueeze(1)
                .expand(-1, reduced_start.size(1), -1, -1),
            ],
            dim=-1,
        )

        if hasattr(self, "size_embedding"):
            # size_embedded: (start_step, end_step, emb_dim)
            size_embedded = self.size_embedding(
                self._get_span_size_ids(full_hidden.size(1))
            )
            # 技术1: 参数化 Size Embedding - 残差 MLP 增强
            if self.size_mlp is not None:
                size_embedded = size_embedded + self.size_mlp(size_embedded)
            # reduced_cat: (batch, start_step, end_step, red_dim*2 + emb_dim)
            reduced_cat = torch.cat(
                [
                    reduced_cat,
                    self.hid_dropout(size_embedded)
                    .unsqueeze(0)
                    .expand(full_hidden.size(0), -1, -1, -1),
                ],
                dim=-1,
            )

        # 额外的 BMES 通道 span 特征打分
        # expert_bmes_channels: (batch, step, 4, emb_dim)
        if expert_bmes_channels is not None:
            batch_size, seq_len, num_channels, ch_dim = expert_bmes_channels.size()
            assert num_channels == 4, "BMES通道数量必须为4(B/M/E/S)"

            # B/M/E/S 通道拆分: (batch, seq_len, ch_dim)
            b_all = expert_bmes_channels[:, :, 0, :]
            m_all = expert_bmes_channels[:, :, 1, :]
            e_all = expert_bmes_channels[:, :, 2, :]
            s_all = expert_bmes_channels[:, :, 3, :]

            # 对应矩阵坐标 (i,j) ↔ span (start=i, end=j+1)
            # B 通道特征: 取起点 i
            # b_feat: (batch, start_step, end_step, ch_dim)
            b_feat = b_all.unsqueeze(2).expand(-1, seq_len, seq_len, -1)
            # E 通道特征: 取终点 j
            # e_feat: (batch, start_step, end_step, ch_dim)
            e_feat = e_all.unsqueeze(1).expand(-1, seq_len, seq_len, -1)

            # ===== 计算 M/S 通道在 span 内的平均表示 =====
            # 目标: 对每个 span(i,j) (对应 token 区间 [i, j])，
            #       取 M/S 通道在中间区域 (i+1 .. j-1) 的平均。

            # 前缀和: (batch, seq_len+1, ch_dim)
            m_prefix = torch.cat(
                [m_all.new_zeros(batch_size, 1, ch_dim), m_all.cumsum(dim=1)],
                dim=1,
            )
            s_prefix = torch.cat(
                [s_all.new_zeros(batch_size, 1, ch_dim), s_all.cumsum(dim=1)],
                dim=1,
            )

            device = full_hidden.device
            idx = torch.arange(seq_len, device=device)

            # start_idx: (1, seq_len, 1), end_idx: (1, 1, seq_len)
            start_idx = idx.view(1, seq_len, 1)
            end_idx = idx.view(1, 1, seq_len) + 1  # 终点是左闭右开上界

            # 中间区域起点: start+1，最大不超过 seq_len
            start_plus1 = (start_idx + 1).clamp(max=seq_len)  # (1, seq_len, 1)

            # 一维索引，用于从前缀和里取值: (batch, seq_len)
            start_plus1_1d = start_plus1.view(1, seq_len).expand(batch_size, -1)
            end_idx_1d = end_idx.view(1, seq_len).expand(batch_size, -1)

            gather_idx_start = start_plus1_1d.unsqueeze(-1).expand(-1, -1, ch_dim)
            gather_idx_end = end_idx_1d.unsqueeze(-1).expand(-1, -1, ch_dim)

            # m_start_1d / m_end_1d: (batch, seq_len, ch_dim)
            m_start_1d = torch.gather(m_prefix, 1, gather_idx_start)
            m_end_1d = torch.gather(m_prefix, 1, gather_idx_end)
            s_start_1d = torch.gather(s_prefix, 1, gather_idx_start)
            s_end_1d = torch.gather(s_prefix, 1, gather_idx_end)

            # 扩展成 span 维度:
            #   m_start: 对应每个起点 i，复制到所有 end j
            #   m_end:   对应每个终点 j，复制到所有 start i
            m_start = m_start_1d.unsqueeze(2).expand(-1, seq_len, seq_len, -1)
            m_end = m_end_1d.unsqueeze(1).expand(-1, seq_len, seq_len, -1)
            s_start = s_start_1d.unsqueeze(2).expand(-1, seq_len, seq_len, -1)
            s_end = s_end_1d.unsqueeze(1).expand(-1, seq_len, seq_len, -1)

            m_sum = m_end - m_start
            s_sum = s_end - s_start
            ms_sum = 0.5 * (m_sum + s_sum)

            # 中间 token 个数: end - (start+1)，形状 (1, seq_len, seq_len)
            interior_len_2d = (end_idx - start_plus1).clamp(min=1)
            # 扩展到 (batch, start_step, end_step, 1)
            interior_len = interior_len_2d.expand(batch_size, -1, -1).unsqueeze(-1)

            # ms_avg: (batch, start_step, end_step, ch_dim)
            ms_avg = ms_sum / interior_len
            # ===== M/S 平均计算结束 =====

            # 拼成原始BMES span特征: [B(i); M/S_avg(i..j); E(j)]
            # 维度: (batch, start_step, end_step, 3*ch_dim)
            span_bmes_raw = torch.cat([b_feat, ms_avg, e_feat], dim=-1)

            # 懒初始化: 将 3*ch_dim 压到 ch_dim 维, 再映射到每个标签的加分
            if not hasattr(self, "bmes_span_proj"):
                self.bmes_span_proj = torch.nn.Linear(3 * ch_dim, ch_dim)
                self.bmes_span_proj.to(full_hidden.device)
                torch.nn.init.orthogonal_(self.bmes_span_proj.weight.data)
                torch.nn.init.zeros_(self.bmes_span_proj.bias.data)
                # V_bmes: (voc_dim, ch_dim)
                self.V_bmes = torch.nn.Parameter(
                    torch.empty(self.voc_dim, ch_dim, device=full_hidden.device)
                )
                torch.nn.init.orthogonal_(self.V_bmes.data)

            # span_bmes_feat: (batch, start_step, end_step, ch_dim)
            span_bmes_feat = self.bmes_span_proj(span_bmes_raw)
            span_bmes_feat = self.hid_dropout(span_bmes_feat)

            # scores_bmes: (batch, start_step, end_step, voc_dim)
            scores_bmes = self.V_bmes.matmul(span_bmes_feat.unsqueeze(-1)).squeeze(-1)

        else:
            scores_bmes = 0

        # scores2: (voc_dim, red_dim*2 + emb_dim) * (batch, start_step, end_step, red_dim*2 + emb_dim, 1) -> (batch, start_step, end_step, voc_dim, 1)
        scores2 = self.W.matmul(reduced_cat.unsqueeze(-1))

        # scores: (batch, start_step, end_step, voc_dim)
        scores = scores1.permute(0, 2, 3, 1) + scores2.squeeze(-1) + scores_bmes
        scores = scores + self.b
        
        # 技术3: Span 长度限制
        max_span_width = self._get_config_attr('max_span_width', None)
        if max_span_width is not None:
            # 创建 span 长度掩码
            # i_indices: (seq_len, 1), j_indices: (1, seq_len)
            i_indices = torch.arange(seq_len, device=scores.device).unsqueeze(1)
            j_indices = torch.arange(seq_len, device=scores.device).unsqueeze(0)
            span_lengths = j_indices - i_indices + 1  # (seq_len, seq_len)
            invalid_mask = (span_lengths > max_span_width)  # True = 需要屏蔽
            # scores shape: (batch, seq_len, seq_len, voc_dim)
            # 对超长 span 的所有标签得分设为 -inf
            scores = scores.masked_fill(invalid_mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
        
        return scores

    def forward(
        self,
        batch: Batch,
        full_hidden: torch.Tensor,
        expert_bmes_channels: torch.Tensor = None,
        expert_bmes_attn_weights: torch.Tensor = None,  # 新增, 当前不在decoder内部使用
    ):
        batch_scores = self.compute_scores(
            batch, full_hidden, expert_bmes_channels=expert_bmes_channels
        )

        losses = []
        for curr_scores, boundaries_obj, curr_len in zip(
            batch_scores, batch.boundaries_objs, batch.seq_lens.cpu().tolist()
        ):
            curr_non_mask = getattr(
                boundaries_obj, "non_mask", self._get_span_non_mask(curr_len)
            )

            loss = self.criterion(
                curr_scores[:curr_len, :curr_len][curr_non_mask],
                boundaries_obj.label_ids[curr_non_mask],
            )
            losses.append(loss)
        return torch.stack(losses)

    def decode(
        self,
        batch: Batch,
        full_hidden: torch.Tensor,
        expert_bmes_channels: torch.Tensor = None,
        expert_bmes_attn_weights: torch.Tensor = None,  # 同样接受该参数以兼容 states
    ):
        batch_scores = self.compute_scores(
            batch, full_hidden, expert_bmes_channels=expert_bmes_channels
        )

        batch_chunks = []
        for curr_scores, boundaries_obj, curr_len in zip(
            batch_scores, batch.boundaries_objs, batch.seq_lens.cpu().tolist()
        ):
            curr_non_mask = self._get_span_non_mask(curr_len)
            

            if not self.multilabel:
                confidences, label_ids = (
                    curr_scores[:curr_len, :curr_len][curr_non_mask]
                    .softmax(dim=-1)
                    .max(dim=-1)
                )
                labels = [self.idx2label[i] for i in label_ids.cpu().tolist()]
                chunks = [
                    (label, start, end)
                    for label, (start, end) in zip(
                        labels, _spans_from_upper_triangular(curr_len)
                    )
                    if label != self.none_label
                ]
                confidences = [
                    conf
                    for label, conf in zip(labels, confidences.cpu().tolist())
                    if label != self.none_label
                ]
            else:
                all_confidences = curr_scores[:curr_len, :curr_len][
                    curr_non_mask
                ].sigmoid()
                # Zero-out all spans according to <none> labels
                all_confidences[
                    all_confidences[:, self.none_idx] > (1 - self.conf_thresh)
                ] = 0
                # Zero-out <none> labels for all spans
                all_confidences[:, self.none_idx] = 0
                all_spans = list(_spans_from_upper_triangular(curr_len))
                assert all_confidences.size(0) == len(all_spans)

                all_confidences_list = all_confidences.cpu().tolist()
                pos_entries = (
                    torch.nonzero(all_confidences > self.conf_thresh).cpu().tolist()
                )
                # In the early training stage, the chunk-decoder may produce too many predicted chunks
                MAX_NUM_CHUNKS = 500
                if len(pos_entries) > MAX_NUM_CHUNKS:
                    pos_entries = pos_entries[:MAX_NUM_CHUNKS]

                chunks = [
                    (self.idx2label[i], *all_spans[sidx]) for sidx, i in pos_entries
                ]
                confidences = [all_confidences_list[sidx][i] for sidx, i in pos_entries]

            assert len(confidences) == len(chunks)
            chunks = self._filter(chunks, confidences, boundaries_obj)
            batch_chunks.append(chunks)
        return batch_chunks