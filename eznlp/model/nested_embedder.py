# -*- coding: utf-8 -*-
from collections import Counter
from typing import List, Optional

import torch

from ..nn.functional import seq_lens2mask
from ..nn.modules import SequencePooling
from ..token import TokenSequence
from ..vocab import Vocab
from .embedder import OneHotConfig, OneHotEmbedder
from .encoder import EncoderConfig


class NestedOneHotConfig(OneHotConfig):
    """Config of an embedder, designed for features with the structure: `step * channel * inner_step`.
    At each position exists one (or multiple) sequence.

    If multiple sequences exist, they share a common vocabulary and corresponding embedding layer.
    """

    def __init__(self, **kwargs):
        self.tokens_key = kwargs.pop("tokens_key", "tokens")
        self.num_channels = kwargs.pop("num_channels", 1)
        self.squeeze = kwargs.pop("squeeze", True)
        if self.squeeze:
            assert self.num_channels == 1

        self.encoder: EncoderConfig = kwargs.pop("encoder", None)
        self.agg_mode = kwargs.pop("agg_mode", "mean_pooling")

        super().__init__(**kwargs)
        if self.encoder is not None:
            self.encoder.in_dim = self.emb_dim

    @property
    def valid(self):
        if self.encoder is not None and not self.encoder.valid:
            return False
        return all(
            attr is not None
            for name, attr in self.__dict__.items()
            if name not in ("vectors", "encoder")
        )

    @property
    def out_dim(self):
        if self.encoder is not None:
            return self.encoder.out_dim * self.num_channels
        else:
            return self.emb_dim * self.num_channels

    def _inner_sequences(self, tokens: TokenSequence):
        for tok_field in getattr(tokens, self.field):
            if self.squeeze:
                yield tok_field
            else:
                yield from tok_field

    def build_vocab(self, *partitions):
        counter = Counter()
        for data in partitions:
            for data_entry in data:
                for inner_seq in self._inner_sequences(data_entry[self.tokens_key]):
                    counter.update(inner_seq)
                    if self.max_len is None or len(inner_seq) > self.max_len:
                        self.max_len = len(inner_seq)

        self.vocab = Vocab(
            counter, min_freq=self.min_freq, specials=self.specials, specials_first=True
        )

    def exemplify(self, tokens: TokenSequence):
        inner_ids_list = []
        for inner_seq in self._inner_sequences(tokens):
            inner_ids_list.append(torch.tensor([self.vocab[x] for x in inner_seq]))

        # inner_ids: (step*num_channels, inner_step)
        return {"inner_ids": inner_ids_list}

    def batchify(self, batch_ex: List[dict]):
        batch_inner_ids = [
            inner_ids for ex in batch_ex for inner_ids in ex["inner_ids"]
        ]
        inner_seq_lens = torch.tensor(
            [inner_ids.size(0) for inner_ids in batch_inner_ids]
        )
        inner_mask = seq_lens2mask(inner_seq_lens)
        batch_inner_ids = torch.nn.utils.rnn.pad_sequence(
            batch_inner_ids, batch_first=True, padding_value=self.pad_idx
        )
        # inner_ids: (batch*step*num_channels, inner_step)
        return {"inner_ids": batch_inner_ids, "inner_mask": inner_mask}

    def instantiate(self):
        return NestedOneHotEmbedder(self)


class NestedOneHotEmbedder(OneHotEmbedder):
    def __init__(self, config: NestedOneHotConfig):
        super().__init__(config)
        self.num_channels = config.num_channels
        if config.encoder is not None:
            self.encoder = config.encoder.instantiate()

        self.agg_mode = config.agg_mode
        if self.agg_mode.lower().endswith(
            "_pooling"
        ) or self.agg_mode.lower().startswith("rnn_last"):
            self.aggregating = SequencePooling(
                mode=self.agg_mode.replace("_pooling", "")
            )

    def _restore_outer_shapes(self, x: torch.Tensor, seq_lens: torch.LongTensor):
        offsets = [0] + (seq_lens * self.num_channels).cumsum(dim=0).cpu().tolist()
        # x: (batch, step*num_channels, emb_dim/hid_dim)
        x = torch.nn.utils.rnn.pad_sequence(
            [x[s:e] for s, e in zip(offsets[:-1], offsets[1:])],
            batch_first=True,
            padding_value=0.0,
        )
        # x: (batch, step, num_channels * emb_dim/hid_dim)
        return x.view(x.size(0), -1, self.num_channels * x.size(-1))

    def forward(
        self,
        inner_ids: torch.LongTensor,
        inner_mask: torch.BoolTensor,
        seq_lens: torch.LongTensor,
        inner_weight: torch.FloatTensor = None,
    ):
        assert (seq_lens * self.num_channels).sum().item() == inner_ids.size(0)

        # embedded: (batch*step*num_channels, inner_step, emb_dim)
        embedded = self.embedding(inner_ids)

        # TODO: positional embedding?

        # encoding -> aggregating
        # hidden: (batch*step*num_channels, inner_step, hid_dim)
        # agg_hidden: (batch*step*num_channels, hid_dim)
        if hasattr(self, "encoder"):
            hidden = self.encoder(embedded, inner_mask)
            agg_hidden = self.aggregating(hidden, inner_mask, weight=inner_weight)
        else:
            agg_hidden = self.aggregating(embedded, inner_mask, weight=inner_weight)

        # Restore outer shapes (token-level steps)
        return self._restore_outer_shapes(agg_hidden, seq_lens)


class SoftLexiconConfig(NestedOneHotConfig):
    """Config of a soft lexicon embedder.

    References
    ----------
    Ma et al. (2020). Simplify the usage of lexicon in Chinese NER. ACL 2020.
    """

    def __init__(self, **kwargs):
        kwargs["field"] = kwargs.pop("field", "softlexicon")
        kwargs["num_channels"] = kwargs.pop("num_channels", 4)
        kwargs["squeeze"] = kwargs.pop("squeeze", False)

        kwargs["emb_dim"] = kwargs.pop("emb_dim", 50)
        kwargs["agg_mode"] = kwargs.pop("agg_mode", "wtd_mean_pooling")
        super().__init__(**kwargs)

    def __repr__(self):
        repr_attr_dict = {
            key: getattr(self, key) for key in self.__dict__.keys() if key != "freqs"
        }
        return self._repr_non_config_attrs(repr_attr_dict)

    def build_freqs(self, *partitions):
        """Ma et al. (2020): The statistical data set is constructed from a combination
        of *training* and *developing* data of the task.
        In addition, note that the frequency of `w` does not increase if `w` is
        covered by another sub-sequence that matches the lexicon
        """
        counter = Counter()
        for data in partitions:
            for data_entry in data:
                for inner_seq in self._inner_sequences(data_entry[self.tokens_key]):
                    counter.update(inner_seq)

        # NOTE: Set the minimum frequecy as 1, to avoid OOV tokens being ignored
        self.freqs = {tok: 1 for tok in self.vocab.itos}
        self.freqs.update(counter)
        self.freqs["<pad>"] = 0

    def exemplify(self, tokens: TokenSequence):
        example = super().exemplify(tokens)

        inner_freqs_list = []
        for inner_seq in self._inner_sequences(tokens):
            inner_freqs_list.append(torch.tensor([self.freqs.get(x, 0) for x in inner_seq]))

        example["inner_freqs"] = inner_freqs_list
        return example

    def batchify(self, batch_ex: List[dict]):
        batch = super().batchify(batch_ex)

        batch_inner_freqs = [
            inner_freqs for ex in batch_ex for inner_freqs in ex["inner_freqs"]
        ]
        batch_inner_freqs = torch.nn.utils.rnn.pad_sequence(
            batch_inner_freqs, batch_first=True, padding_value=0
        )
        batch["inner_weight"] = batch_inner_freqs
        return batch


class CharConfig(NestedOneHotConfig):
    """Config of a character embedder."""

    def __init__(self, **kwargs):
        kwargs["field"] = kwargs.pop("field", "raw_text")
        kwargs["num_channels"] = kwargs.pop("num_channels", 1)
        kwargs["squeeze"] = kwargs.pop("squeeze", True)

        kwargs["emb_dim"] = kwargs.pop("emb_dim", 16)
        kwargs["encoder"] = kwargs.pop(
            "encoder",
            EncoderConfig(
                arch="LSTM", hid_dim=128, num_layers=1, in_drop_rates=(0.5, 0.0, 0.0)
            ),
        )
        if kwargs["encoder"].arch.lower() in ("lstm", "gru"):
            kwargs["agg_mode"] = kwargs.pop("agg_mode", "rnn_last")
        elif kwargs["encoder"].arch.lower() in ("conv", "gehring"):
            kwargs["agg_mode"] = kwargs.pop("agg_mode", "max_pooling")
        super().__init__(**kwargs)

    @property
    def name(self):
        return f"Char{self.encoder.arch}"


class ExpertDictConfig(NestedOneHotConfig):
    """Config of an expert dictionary embedder.
    
    This embedder is designed for domain-specific expert dictionaries
    (e.g., medical terms, legal entities) to enhance NER performance
    by providing additional boundary and semantic information.
    
    Similar to SoftLexiconConfig but focused on expert-curated lexicons
    rather than general word segmentation lexicons.
    
    References
    ----------
    Inspired by Ma et al. (2020). Simplify the usage of lexicon in Chinese NER. ACL 2020.
    """

    def __init__(self, **kwargs):
        kwargs["field"] = kwargs.pop("field", "expert_dict")
        kwargs["num_channels"] = kwargs.pop("num_channels", 4)
        kwargs["squeeze"] = kwargs.pop("squeeze", False)

        kwargs["emb_dim"] = kwargs.pop("emb_dim", 50)
        kwargs["agg_mode"] = kwargs.pop("agg_mode", "wtd_mean_pooling")
        
        # BMES通道间注意力配置
        self.use_channel_attention = kwargs.pop("use_channel_attention", False)
        self.channel_attn_heads = kwargs.pop("channel_attn_heads", 4)
        self.channel_attn_dropout = kwargs.pop("channel_attn_dropout", 0.1)
        self.channel_attn_version = kwargs.pop("channel_attn_version", "v1")
        
        # 自动调整 num_heads 使其能整除 emb_dim
        if self.use_channel_attention:
            emb_dim = kwargs.get("emb_dim", 50)
            if emb_dim % self.channel_attn_heads != 0:
                # 寻找能整除 emb_dim 的最大头数（不超过原始设置）
                original_heads = self.channel_attn_heads
                for h in range(original_heads, 0, -1):
                    if emb_dim % h == 0:
                        self.channel_attn_heads = h
                        break
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"channel_attn_heads adjusted from {original_heads} to {self.channel_attn_heads} "
                    f"(emb_dim={emb_dim} must be divisible by num_heads)"
                )
        
        # BMES 跨位置编码器配置（方案B）
        self.use_cross_position_encoder = kwargs.pop("use_cross_position_encoder", False)
        self.cross_position_encoder_type = kwargs.pop("cross_position_encoder_type", "lstm")
        self.cross_position_encoder_dropout = kwargs.pop("cross_position_encoder_dropout", 0.1)
        
        super().__init__(**kwargs)

    def __repr__(self):
        repr_attr_dict = {
            key: getattr(self, key) for key in self.__dict__.keys() if key != "freqs"
        }
        return self._repr_non_config_attrs(repr_attr_dict)

    def build_freqs(self, *partitions):
        """Build frequency statistics for expert dictionary terms.
        
        The statistical data set is constructed from a combination
        of training and developing data of the task.
        """
        counter = Counter()
        for data in partitions:
            for data_entry in data:
                for inner_seq in self._inner_sequences(data_entry[self.tokens_key]):
                    counter.update(inner_seq)

        # NOTE: Set the minimum frequency as 1, to avoid OOV tokens being ignored
        self.freqs = {tok: 1 for tok in self.vocab.itos}
        self.freqs.update(counter)
        self.freqs["<pad>"] = 0

    def exemplify(self, tokens: TokenSequence):
        example = super().exemplify(tokens)

        inner_freqs_list = []
        for inner_seq in self._inner_sequences(tokens):
            inner_freqs_list.append(torch.tensor([self.freqs.get(x, 0) for x in inner_seq]))

        example["inner_freqs"] = inner_freqs_list
        return example

    def batchify(self, batch_ex: List[dict]):
        batch = super().batchify(batch_ex)

        batch_inner_freqs = [
            inner_freqs for ex in batch_ex for inner_freqs in ex["inner_freqs"]
        ]
        batch_inner_freqs = torch.nn.utils.rnn.pad_sequence(
            batch_inner_freqs, batch_first=True, padding_value=0
        )
        batch["inner_weight"] = batch_inner_freqs
        return batch
    
    def instantiate(self):
        """实例化ExpertDict嵌入器（支持通道注意力）"""
        if self.use_channel_attention:
            return ExpertDictWithChannelAttention(self)
        else:
            return NestedOneHotEmbedder(self)


class ExpertDictWithChannelAttention(NestedOneHotEmbedder):
    """带通道间注意力的ExpertDict嵌入器
    
    在标准的NestedOneHotEmbedder基础上,增加BMES通道间注意力机制
    """
    
    def __init__(self, config: ExpertDictConfig):
        super().__init__(config)
        
        from .channel_attention import BMESChannelAttention, BMESChannelAttentionV2, BMESCrossPositionEncoder
        
        if config.channel_attn_version == "v2":
            self.channel_attention = BMESChannelAttentionV2(
                emb_dim=config.emb_dim,
                num_channels=config.num_channels,
                dropout=config.channel_attn_dropout
            )
        else:
            self.channel_attention = BMESChannelAttention(
                emb_dim=config.emb_dim,
                num_channels=config.num_channels,
                num_heads=config.channel_attn_heads,
                dropout=config.channel_attn_dropout,
                use_channel_pos=True
            )
        
        # 方案B: 跨位置编码器
        if config.use_cross_position_encoder:
            self.cross_position_encoder = BMESCrossPositionEncoder(
                emb_dim=config.emb_dim,
                num_channels=config.num_channels,
                encoder_type=config.cross_position_encoder_type,
                dropout=config.cross_position_encoder_dropout,
            )
    
    def forward(
        self,
        inner_ids: torch.LongTensor,
        inner_mask: torch.BoolTensor,
        seq_lens: torch.LongTensor,
        inner_weight: Optional[torch.FloatTensor] = None,
    ):
        """前向传播（增加通道注意力 + 可选跨位置编码）
        
        流程:
        1. Embedding: (batch*step*4, inner_step) → (batch*step*4, inner_step, 50)
        2. Aggregation: (batch*step*4, inner_step, 50) → (batch*step*4, 50)
        3. 先恢复batch维度: (batch*step*4, 50) → (batch, step, 200)
        4. 重塑为通道维度: (batch, step, 200) → (batch, step, 4, 50)
        5. 通道内注意力: (batch, step, 4, 50) → (batch, step, 4, 50)
        6. [可选] 跨位置编码: (batch, step, 4, 50) → (batch, step, 4, 50)
        7. 展平通道维度: (batch, step, 4, 50) → (batch, step, 200)
        """
        assert (seq_lens * self.num_channels).sum().item() == inner_ids.size(0)
        
        embedded = self.embedding(inner_ids)
        
        if hasattr(self, "encoder"):
            hidden = self.encoder(embedded, inner_mask)
            agg_hidden = self.aggregating(hidden, inner_mask, weight=inner_weight)
        else:
            agg_hidden = self.aggregating(embedded, inner_mask, weight=inner_weight)
        
        batch_hidden = self._restore_outer_shapes(agg_hidden, seq_lens)
        
        batch_size, step_size = batch_hidden.size(0), batch_hidden.size(1)
        channel_hidden = batch_hidden.view(batch_size, step_size, self.num_channels, -1)
        
        channel_mask = (channel_hidden.abs().sum(dim=-1) > 0)
        
        # Step 5: 通道内注意力
        channel_hidden_flat = channel_hidden.view(-1, self.num_channels, channel_hidden.size(-1))
        channel_mask_flat = channel_mask.view(-1, self.num_channels)
        
        attended_hidden_flat = self.channel_attention(channel_hidden_flat, channel_mask_flat)
        
        if hasattr(self.channel_attention, "last_attn_weights"):
            attn_flat = self.channel_attention.last_attn_weights
            attn_heads = attn_flat.view(
                batch_size,
                step_size,
                self.channel_attention.num_heads,
                self.num_channels,
                self.num_channels,
            )
            self.last_channel_attn_weights = attn_heads

        attended_hidden = attended_hidden_flat.view(batch_size, step_size, self.num_channels, -1)
        
        # Step 6: 跨位置编码（方案B，如果启用）
        if hasattr(self, "cross_position_encoder"):
            token_mask = channel_mask.any(dim=-1)
            attended_hidden = self.cross_position_encoder(attended_hidden, mask=token_mask)
        
        self.last_channel_hidden = attended_hidden
        
        output = attended_hidden.view(batch_size, step_size, -1)
        
        return output
