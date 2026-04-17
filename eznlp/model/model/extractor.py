# -*- coding: utf-8 -*-
from typing import List, Union

import torch

from ...config import ConfigDict
from ...wrapper import Batch
from ..decoder import (
    BoundarySelectionDecoderConfig,
    JointExtractionDecoderConfig,
    SequenceTaggingDecoderConfig,
    SingleDecoderConfigBase,
    SpanAttrClassificationDecoderConfig,
    SpanClassificationDecoderConfig,
    SpanRelClassificationDecoderConfig,
)
from ..embedder import OneHotConfig
from ..encoder import EncoderConfig
from ..nested_embedder import SoftLexiconConfig
from .base import ModelBase, ModelConfigBase


class ExtractorConfig(ModelConfigBase):
    """Configurations of an extractor.

    extractor
      ├─decoder(*)
      └─intermediate2
          ├─intermediate1
          │   ├─ohots
          │   ├─mhots
          │   └─nested_ohots
          ├─elmo
          ├─bert_like
          ├─flair_fw
          └─flair_bw
    """

    _embedder_names = ["ohots", "mhots", "nested_ohots"]
    _encoder_names = ["intermediate1", "intermediate2"]
    _pretrained_names = ["elmo", "bert_like", "flair_fw", "flair_bw"]
    _all_names = (
        _embedder_names
        + ["intermediate1"]
        + _pretrained_names
        + ["intermediate2"]
        + ["decoder"]
    )

    def __init__(
        self,
        decoder: Union[
            SingleDecoderConfigBase, JointExtractionDecoderConfig, str
        ] = "sequence_tagging",
        **kwargs,
    ):
        self.ohots = kwargs.pop(
            "ohots", ConfigDict({"text": OneHotConfig(field="text")})
        )
        self.mhots = kwargs.pop("mhots", None)
        self.nested_ohots = kwargs.pop("nested_ohots", None)
        self.intermediate1 = kwargs.pop("intermediate1", None)

        self.elmo = kwargs.pop("elmo", None)
        self.bert_like = kwargs.pop("bert_like", None)
        self.flair_fw = kwargs.pop("flair_fw", None)
        self.flair_bw = kwargs.pop("flair_bw", None)
        self.intermediate2 = kwargs.pop("intermediate2", EncoderConfig(arch="LSTM"))

        if isinstance(decoder, (SingleDecoderConfigBase, JointExtractionDecoderConfig)):
            self.decoder = decoder
        elif isinstance(decoder, str):
            if decoder.lower().startswith("sequence_tagging"):
                self.decoder = SequenceTaggingDecoderConfig()
            elif decoder.lower().startswith("span_classification"):
                self.decoder = SpanClassificationDecoderConfig()
            elif decoder.lower().startswith("span_attr"):
                self.decoder = SpanAttrClassificationDecoderConfig()
            elif decoder.lower().startswith("span_rel"):
                self.decoder = SpanRelClassificationDecoderConfig()
            elif decoder.lower().startswith("boundary"):
                self.decoder = BoundarySelectionDecoderConfig()
            elif decoder.lower().startswith("joint_extraction"):
                self.decoder = JointExtractionDecoderConfig(
                    ck_decoder="span_classification", rel_decoder="span_rel"
                )
            else:
                raise ValueError(f"Invalid `decoder`: {decoder}")

        super().__init__(**kwargs)

    @property
    def valid(self):
        return super().valid and (
            self.bert_like is None or self.bert_like.from_tokenized
        )

    @property
    def full_emb_dim(self):
        emb_dim = 0
        if self.ohots is not None:
            emb_dim += sum(c.out_dim for c in self.ohots.values())
        if self.mhots is not None:
            emb_dim += sum(c.out_dim for c in self.mhots.values())
        if self.nested_ohots is not None:
            emb_dim += sum(c.out_dim for c in self.nested_ohots.values())
        return emb_dim

    @property
    def full_hid_dim(self):
        if self.intermediate1 is not None:
            full_hid_dim = self.intermediate1.out_dim
        else:
            full_hid_dim = self.full_emb_dim
        full_hid_dim += sum(
            getattr(self, name).out_dim
            for name in self._pretrained_names
            if getattr(self, name) is not None
        )
        return full_hid_dim

    def build_vocabs_and_dims(self, *partitions):
        if self.ohots is not None:
            for c in self.ohots.values():
                c.build_vocab(*partitions)

        if self.mhots is not None:
            for c in self.mhots.values():
                c.build_dim(partitions[0][0]["tokens"])

        if self.nested_ohots is not None:
            for c in self.nested_ohots.values():
                c.build_vocab(*partitions)
                if isinstance(c, SoftLexiconConfig):
                    # Skip the last split (assumed to be test set)
                    c.build_freqs(*partitions[:-1])

        if self.intermediate1 is not None:
            self.intermediate1.in_dim = self.full_emb_dim

        if self.intermediate2 is not None:
            self.intermediate2.in_dim = self.full_hid_dim
            self.decoder.in_dim = self.intermediate2.out_dim
        else:
            self.decoder.in_dim = self.full_hid_dim

        self.decoder.build_vocab(*partitions)

    def exemplify(self, data_entry: dict, training: bool = True):
        example = {}

        if self.ohots is not None:
            example["ohots"] = {
                f: c.exemplify(data_entry["tokens"]) for f, c in self.ohots.items()
            }

        if self.mhots is not None:
            example["mhots"] = {
                f: c.exemplify(data_entry["tokens"]) for f, c in self.mhots.items()
            }

        if self.nested_ohots is not None:
            example["nested_ohots"] = {
                f: c.exemplify(data_entry["tokens"])
                for f, c in self.nested_ohots.items()
            }

        for name in self._pretrained_names:
            if getattr(self, name) is not None:
                example[name] = getattr(self, name).exemplify(data_entry["tokens"])

        example.update(self.decoder.exemplify(data_entry, training=training))
        return example

    def batchify(self, batch_examples: List[dict]):
        batch = {}

        if self.ohots is not None:
            batch["ohots"] = {
                f: c.batchify([ex["ohots"][f] for ex in batch_examples])
                for f, c in self.ohots.items()
            }

        if self.mhots is not None:
            batch["mhots"] = {
                f: c.batchify([ex["mhots"][f] for ex in batch_examples])
                for f, c in self.mhots.items()
            }

        if self.nested_ohots is not None:
            batch["nested_ohots"] = {
                f: c.batchify([ex["nested_ohots"][f] for ex in batch_examples])
                for f, c in self.nested_ohots.items()
            }

        for name in self._pretrained_names:
            if getattr(self, name) is not None:
                batch[name] = getattr(self, name).batchify(
                    [ex[name] for ex in batch_examples]
                )

        batch.update(self.decoder.batchify(batch_examples))
        return batch

    def instantiate(self):
        # Only check validity at the most outside level
        assert self.valid
        return Extractor(self)


class Extractor(ModelBase):
    def __init__(self, config: ExtractorConfig):
        super().__init__(config)

    def _get_full_embedded(self, batch: Batch):
        embedded = []

        if hasattr(self, "ohots") and len(self.ohots) > 0:
            ohots_embedded = [self.ohots[f](batch.ohots[f]) for f in self.ohots]
            embedded.extend(ohots_embedded)

        if hasattr(self, "mhots") and len(self.mhots) > 0:
            mhots_embedded = [self.mhots[f](batch.mhots[f]) for f in self.mhots]
            embedded.extend(mhots_embedded)

        if hasattr(self, "nested_ohots") and len(self.nested_ohots) > 0:
            # 改成逐个特征处理, 以便在ExpertDict上挂载BMES通道特征
            for f in self.nested_ohots:
                out = self.nested_ohots[f](**batch.nested_ohots[f], seq_lens=batch.seq_lens)
                embedded.append(out)

                # 如果是 ExpertDictWithChannelAttention, 则读取其通道隐藏状态和注意力权重
                if (
                    f == "expert_dict"
                    and hasattr(self.nested_ohots[f], "last_channel_hidden")
                ):
                    # 形状: (batch, step, 4, emb_dim)
                    self._expert_bmes_channels = self.nested_ohots[f].last_channel_hidden

                    if hasattr(self.nested_ohots[f], "last_channel_attn_weights"):
                        # 形状: (batch, step, num_heads, 4, 4)
                        self._expert_bmes_attn_weights = self.nested_ohots[
                            f
                        ].last_channel_attn_weights

        # 如果没有任何嵌入特征，返回空张量
        if len(embedded) == 0:
            batch_size, seq_len = batch.tokens['input_ids'].shape[:2]
            return None  # 返回None，由调用方处理
        
        return torch.cat(embedded, dim=-1)

    def _get_full_hidden(self, batch: Batch):
        """
        获取全部隐藏层特征

        默认行为：拼接
          [intermediate1(ohots/mhots/nested_ohots), elmo, bert_like, flair_fw, flair_bw]

        特殊增强：当且仅当存在 BERT + ExpertDict（且 ExpertDict 作为唯一嵌套特征）时，
        对这两路特征做 concat + Linear 融合，再与其他特征（如果有）一起拼接。
        """
        embedder_hidden = None
        pretrained_hiddens = {}
        parts = []

        # 1) 嵌入侧 (ohots / mhots / nested_ohots)
        # 检查是否有任何嵌入特征，且不为空
        has_any_embedder = any(
            hasattr(self, name) and 
            (not isinstance(getattr(self, name), torch.nn.ModuleDict) or len(getattr(self, name)) > 0)
            for name in ExtractorConfig._embedder_names
        )
        if has_any_embedder:
            embedded = self._get_full_embedded(batch)
            if embedded is not None:
                if hasattr(self, "intermediate1"):
                    embedder_hidden = self.intermediate1(embedded, batch.mask)
                else:
                    embedder_hidden = embedded
            else:
                embedder_hidden = None

        # 2) 预训练侧 (elmo / bert_like / flair_fw / flair_bw)
        for name in ExtractorConfig._pretrained_names:
            if hasattr(self, name):
                pretrained_hiddens[name] = getattr(self, name)(**getattr(batch, name))

        # 3) 判断是否可以做“BERT + ExpertDict(BMES) concat+proj 融合”
        use_bert_expert_fusion = (
            hasattr(self, "bert_like")
            and hasattr(self, "nested_ohots")
            # 只在 ExpertDict 是唯一 nested_ohots 时启用，避免影响其他组合
            and isinstance(self.nested_ohots, dict)
            and len(self.nested_ohots) == 1
            and "expert_dict" in self.nested_ohots
            and embedder_hidden is not None
            and "bert_like" in pretrained_hiddens
        )

        if use_bert_expert_fusion:
            bert_hidden = pretrained_hiddens["bert_like"]

            # 维度检查：B、S 维度应一致
            assert (
                bert_hidden.size(0) == embedder_hidden.size(0)
                and bert_hidden.size(1) == embedder_hidden.size(1)
            ), "BERT 与 ExpertDict 的 batch/seq 维度不一致"

            fused_input = torch.cat([bert_hidden, embedder_hidden], dim=-1)
            in_dim = fused_input.size(-1)

            # 懒初始化：第一次调用时创建投影层，输出维度保持不变（in_dim）
            if not hasattr(self, "bert_expert_proj"):
                self.bert_expert_proj = torch.nn.Linear(in_dim, in_dim)

            fused_hidden = self.bert_expert_proj(fused_input)
            parts.append(fused_hidden)

            # 如果还有其他预训练特征（如 elmo/flair），继续拼接
            for name, h in pretrained_hiddens.items():
                if name == "bert_like":
                    continue
                parts.append(h)
        else:
            # 保持原有拼接逻辑：embedder_hidden + 所有预训练特征
            if embedder_hidden is not None:
                parts.append(embedder_hidden)
            parts.extend(pretrained_hiddens.values())

        # 4) 拼接所有部分
        full_hidden = torch.cat(parts, dim=-1)

        # 5) 通过 intermediate2
        if hasattr(self, "intermediate2"):
            return self.intermediate2(full_hidden, batch.mask)
        else:
            return full_hidden

    def pretrained_parameters(self):
        params = []

        if hasattr(self, "elmo"):
            params.extend(self.elmo.elmo._elmo_lstm.parameters())

        if hasattr(self, "bert_like"):
            params.extend(self.bert_like.bert_like.parameters())

        if hasattr(self, "flair_fw"):
            params.extend(self.flair_fw.flair_lm.parameters())
        if hasattr(self, "flair_bw"):
            params.extend(self.flair_bw.flair_lm.parameters())

        return params

    def forward2states(self, batch: Batch):
        """
        返回给解码器的状态:
        - full_hidden: 原有编码器输出
        - expert_bmes_channels: (可选) ExpertDict BMES通道隐藏状态, 形状 (batch, step, 4, emb_dim)
        - expert_bmes_attn_weights: (可选) BMES通道注意力权重, 形状 (batch, step, num_heads, 4, 4)
        """
        full_hidden = self._get_full_hidden(batch)
        states = {"full_hidden": full_hidden}
        if hasattr(self, "_expert_bmes_channels"):
            states["expert_bmes_channels"] = self._expert_bmes_channels
        if hasattr(self, "_expert_bmes_attn_weights"):
            states["expert_bmes_attn_weights"] = self._expert_bmes_attn_weights
        return states
