# -*- coding: utf-8 -*-
import math
import torch

from ..config import Config
from ..nn.functional import mask2seq_lens
from ..nn.init import reinit_gru_, reinit_layer_, reinit_lstm_
from ..nn.modules import (
    CombinedDropout,
    ConvBlock,
    FeedForwardBlock,
    TransformerEncoderBlock,
)


class RotaryPositionEmbedding(torch.nn.Module):
    """Rotary Position Embedding (RoPE) from GTR-NNER.
    
    Applies relative position encoding through rotation matrices.
    Reference: Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding.
    """
    
    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Compute inverse frequency: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for efficiency
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Precompute cos and sin cache for positions."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        
        Returns:
            Tensor with RoPE applied, same shape as input
        """
        batch, seq_len, dim = x.shape
        
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        # Get cos and sin for current sequence length
        cos = self.cos_cached[:seq_len].unsqueeze(0)  # (1, seq_len, dim)
        sin = self.sin_cached[:seq_len].unsqueeze(0)  # (1, seq_len, dim)
        
        # Apply rotation
        x_rot = self._rotate_half(x)
        return x * cos + x_rot * sin
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)


class TriAffineAttention(torch.nn.Module):
    """TriAffine Attention Fusion from GTR-NNER.
    
    Three-way tensor fusion for combining multiple feature sources.
    Computes attention scores: score(h1, h2, h3) = h1 @ W @ h2.T + h3 @ U
    """
    
    def __init__(self, in_dim: int, hid_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        
        # Project to common dimension
        self.proj_q = torch.nn.Linear(in_dim, hid_dim)
        self.proj_k = torch.nn.Linear(in_dim, hid_dim)
        self.proj_v = torch.nn.Linear(in_dim, hid_dim)
        
        # TriAffine weight tensor
        self.triaffine_weight = torch.nn.Parameter(torch.zeros(hid_dim, hid_dim, hid_dim))
        torch.nn.init.xavier_uniform_(self.triaffine_weight)
        
        # Output projection
        self.out_proj = torch.nn.Linear(hid_dim, in_dim)
        self.dropout = torch.nn.Dropout(dropout)
        
        reinit_layer_(self.proj_q, "linear")
        reinit_layer_(self.proj_k, "linear")
        reinit_layer_(self.proj_v, "linear")
        reinit_layer_(self.out_proj, "linear")
    
    def forward(self, hidden: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """Apply TriAffine attention fusion.
        
        Args:
            hidden: Input tensor of shape (batch, seq_len, in_dim)
            mask: Optional mask tensor of shape (batch, seq_len)
        
        Returns:
            Fused tensor of shape (batch, seq_len, in_dim)
        """
        batch, seq_len, _ = hidden.shape
        
        # Project to common dimension
        q = self.proj_q(hidden)  # (batch, seq_len, hid_dim)
        k = self.proj_k(hidden)  # (batch, seq_len, hid_dim)
        v = self.proj_v(hidden)  # (batch, seq_len, hid_dim)
        
        # TriAffine computation: simplified bilinear attention
        # Use matrix multiplication approximation for efficiency
        W_mid = self.triaffine_weight.mean(dim=-1)  # (hid, hid) - average over third dimension
        attn_scores = torch.matmul(q, W_mid)  # (batch, seq, hid)
        attn_scores = torch.matmul(attn_scores, k.transpose(-1, -2))  # (batch, seq, seq)
        
        # Apply mask
        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, seq)
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.hid_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # (batch, seq, hid)
        
        # Output projection
        out = self.out_proj(attn_out)  # (batch, seq, in_dim)
        
        # Residual connection
        return hidden + self.dropout(out)


class EncoderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop("arch", "LSTM")
        self.in_dim = kwargs.pop("in_dim", None)
        self.in_proj = kwargs.pop("in_proj", False)
        self.shortcut = kwargs.pop("shortcut", False)
        self.shortcut_mode = kwargs.pop("shortcut_mode", "concat")  # "concat" or "add"
        
        # Self-rectified Gate (EPFD)
        self.use_srg = kwargs.pop("use_srg", False)
        self.srg_hid_dim = kwargs.pop("srg_hid_dim", 128)
        self.srg_dropout = kwargs.pop("srg_dropout", 0.2)
        
        # Rotary Position Embedding (RoPE) - GTR-NNER
        self.use_rope = kwargs.pop("use_rope", False)
        self.rope_base = kwargs.pop("rope_base", 10000)
        self.rope_max_seq_len = kwargs.pop("rope_max_seq_len", 512)
        
        # TriAffine Attention Fusion - GTR-NNER
        self.use_triaffine = kwargs.pop("use_triaffine", False)
        self.triaffine_hid_dim = kwargs.pop("triaffine_hid_dim", 128)

        if self.arch.lower() == "identity":
            self.in_drop_rates = kwargs.pop("in_drop_rates", (0.0, 0.0, 0.0))
            self.hid_drop_rate = kwargs.pop("hid_drop_rate", 0.0)

        else:
            self.hid_dim = kwargs.pop("hid_dim", 128)

            if self.arch.lower() == "ffn":
                self.num_layers = kwargs.pop("num_layers", 1)
                self.in_drop_rates = kwargs.pop("in_drop_rates", (0.5, 0.0, 0.0))
                self.hid_drop_rate = kwargs.pop("hid_drop_rate", 0.5)

            elif self.arch.lower() in ("lstm", "gru"):
                self.train_init_hidden = kwargs.pop("train_init_hidden", False)
                self.num_layers = kwargs.pop("num_layers", 1)
                self.in_drop_rates = kwargs.pop("in_drop_rates", (0.5, 0.0, 0.0))
                self.hid_drop_rate = kwargs.pop("hid_drop_rate", 0.5)

            elif self.arch.lower() in ("conv", "gehring"):
                self.kernel_size = kwargs.pop("kernel_size", 3)
                self.scale = kwargs.pop("scale", 0.5**0.5)
                self.num_layers = kwargs.pop("num_layers", 3)
                self.in_drop_rates = kwargs.pop("in_drop_rates", (0.25, 0.0, 0.0))
                self.hid_drop_rate = kwargs.pop("hid_drop_rate", 0.25)

            elif self.arch.lower() == "transformer":
                self.use_emb2init_hid = kwargs.pop("use_emb2init_hid", False)
                self.num_heads = kwargs.pop("num_heads", 8)
                self.ff_dim = kwargs.pop("ff_dim", 256)
                self.num_layers = kwargs.pop("num_layers", 3)
                self.in_drop_rates = kwargs.pop("in_drop_rates", (0.1, 0.0, 0.0))
                self.hid_drop_rate = kwargs.pop("hid_drop_rate", 0.1)

            else:
                raise ValueError(f"Invalid encoder architecture {self.arch}")

        super().__init__(**kwargs)

    @property
    def name(self):
        return self.arch

    @property
    def out_dim(self):
        if self.arch.lower() == "identity":
            out_dim = self.in_dim
        else:
            out_dim = self.hid_dim

        if self.shortcut:
            if self.shortcut_mode == "concat":
                out_dim = out_dim + self.in_dim
            # for "add" mode, out_dim remains the same (hid_dim)

        return out_dim

    def instantiate(self):
        if self.arch.lower() == "identity":
            return IdentityEncoder(self)
        elif self.arch.lower() == "ffn":
            return FFNEncoder(self)
        elif self.arch.lower() in ("lstm", "gru"):
            return RNNEncoder(self)
        elif self.arch.lower() == "conv":
            return ConvEncoder(self)
        elif self.arch.lower() == "gehring":
            return GehringConvEncoder(self)
        elif self.arch.lower() == "transformer":
            return TransformerEncoder(self)


class Encoder(torch.nn.Module):
    """`Encoder` forwards from embeddings to hidden states."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.dropout = CombinedDropout(*config.in_drop_rates)
        if config.in_proj:
            self.in_proj_layer = torch.nn.Linear(config.in_dim, config.in_dim)
            reinit_layer_(self.in_proj_layer, "linear")
        self.shortcut = config.shortcut
        self.shortcut_mode = getattr(config, "shortcut_mode", "concat")
        
        # For addition shortcut with different dimensions, need projection
        if self.shortcut and self.shortcut_mode == "add":
            if hasattr(config, "hid_dim") and config.hid_dim != config.in_dim:
                self.residual_proj = torch.nn.Linear(config.in_dim, config.hid_dim)
                reinit_layer_(self.residual_proj, "linear")
            else:
                self.residual_proj = None
        
        # Self-rectified Gate (EPFD)
        self.use_srg = getattr(config, "use_srg", False)
        if self.use_srg:
            # SRG input dim = encoder output dim
            srg_in_dim = self._get_output_dim(config)
            srg_hid_dim = getattr(config, "srg_hid_dim", 128)
            srg_dropout = getattr(config, "srg_dropout", 0.2)
            self.srg_fc1 = torch.nn.Linear(srg_in_dim, srg_hid_dim)
            self.srg_fc2 = torch.nn.Linear(srg_hid_dim, srg_in_dim)
            self.srg_dropout = torch.nn.Dropout(srg_dropout)
            reinit_layer_(self.srg_fc1, "linear")
            reinit_layer_(self.srg_fc2, "linear")
        
        # Rotary Position Embedding (RoPE) - GTR-NNER
        self.use_rope = getattr(config, "use_rope", False)
        if self.use_rope:
            rope_dim = self._get_output_dim(config)
            rope_base = getattr(config, "rope_base", 10000)
            rope_max_seq_len = getattr(config, "rope_max_seq_len", 512)
            self.rope = RotaryPositionEmbedding(
                dim=rope_dim, base=rope_base, max_seq_len=rope_max_seq_len
            )
        
        # TriAffine Attention Fusion - GTR-NNER
        self.use_triaffine = getattr(config, "use_triaffine", False)
        if self.use_triaffine:
            triaffine_in_dim = self._get_output_dim(config)
            triaffine_hid_dim = getattr(config, "triaffine_hid_dim", 128)
            self.triaffine = TriAffineAttention(
                in_dim=triaffine_in_dim, hid_dim=triaffine_hid_dim, dropout=0.2
            )
    
    def _get_output_dim(self, config):
        """Get encoder output dimension before SRG."""
        if hasattr(config, "hid_dim"):
            out_dim = config.hid_dim
            if config.shortcut:
                if config.shortcut_mode == "concat":
                    out_dim = out_dim + config.in_dim
            return out_dim
        else:
            return config.in_dim

    def embedded2hidden(
        self, embedded: torch.FloatTensor, mask: torch.BoolTensor = None
    ):
        raise NotImplementedError("Not Implemented `embedded2hidden`")

    def _apply_srg(self, hidden: torch.FloatTensor):
        """Apply Self-rectified Gate."""
        # g = σ(W_2 σ(W_1 h + b_1) + b_2)
        # corrected = g * hidden
        gate = torch.nn.functional.relu(self.srg_fc1(hidden))
        gate = self.srg_dropout(gate)
        gate = torch.sigmoid(self.srg_fc2(gate))
        return gate * hidden

    def forward(self, embedded: torch.FloatTensor, mask: torch.BoolTensor = None):
        # embedded: (batch, step, emb_dim)
        # hidden: (batch, step, hid_dim)
        if hasattr(self, "in_proj_layer"):
            hidden = self.embedded2hidden(
                self.in_proj_layer(self.dropout(embedded)), mask=mask
            )
        else:
            hidden = self.embedded2hidden(self.dropout(embedded), mask=mask)

        if self.shortcut:
            if self.shortcut_mode == "add":
                # Addition residual: o_i = l_i + v_i (EPFD style)
                if hasattr(self, "residual_proj") and self.residual_proj is not None:
                    residual = self.residual_proj(embedded)
                else:
                    residual = embedded
                hidden = hidden + residual
            else:
                # Concatenation (default eznlp style)
                hidden = torch.cat([hidden, embedded], dim=-1)
        
        # Apply Rotary Position Embedding (RoPE)
        if self.use_rope:
            hidden = self.rope(hidden)
        
        # Apply TriAffine Attention Fusion
        if self.use_triaffine:
            hidden = self.triaffine(hidden, mask=mask)
        
        # Apply Self-rectified Gate
        if self.use_srg:
            hidden = self._apply_srg(hidden)
        
        return hidden


class IdentityEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)

    def embedded2hidden(
        self, embedded: torch.FloatTensor, mask: torch.BoolTensor = None
    ):
        return embedded


class FFNEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        # NOTE: Only the first layer is differently configured, consistent to `torch.nn.RNN` modules
        self.ff_blocks = torch.nn.ModuleList(
            [
                FeedForwardBlock(
                    in_dim=(config.in_dim if k == 0 else config.hid_dim),
                    out_dim=config.hid_dim,
                    drop_rate=(0.0 if k == 0 else config.hid_drop_rate),
                )
                for k in range(config.num_layers)
            ]
        )

    def embedded2hidden(
        self, embedded: torch.FloatTensor, mask: torch.BoolTensor = None
    ):
        hidden = embedded
        for ff_block in self.ff_blocks:
            hidden = ff_block(hidden)

        return hidden


class RNNEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        rnn_config = {
            "input_size": config.in_dim,
            "hidden_size": config.hid_dim // 2,
            "num_layers": config.num_layers,
            "batch_first": True,
            "bidirectional": True,
            "dropout": 0.0 if config.num_layers <= 1 else config.hid_drop_rate,
        }

        if config.arch.lower() == "lstm":
            self.rnn = torch.nn.LSTM(**rnn_config)
            reinit_lstm_(self.rnn)
        elif config.arch.lower() == "gru":
            self.rnn = torch.nn.GRU(**rnn_config)
            reinit_gru_(self.rnn)

        # h_0/c_0: (layers*directions, batch, hid_dim/2)
        if config.train_init_hidden:
            self.h_0 = torch.nn.Parameter(
                torch.zeros(config.num_layers * 2, 1, config.hid_dim // 2)
            )
            if isinstance(self.rnn, torch.nn.LSTM):
                self.c_0 = torch.nn.Parameter(
                    torch.zeros(config.num_layers * 2, 1, config.hid_dim // 2)
                )

    def embedded2hidden(
        self,
        embedded: torch.FloatTensor,
        mask: torch.BoolTensor = None,
        return_last_hidden: bool = False,
    ):
        if hasattr(self, "h_0"):
            h_0 = self.h_0.expand(-1, embedded.size(0), -1)
            if hasattr(self, "c_0"):
                c_0 = self.c_0.expand(-1, embedded.size(0), -1)
                h_0 = (h_0, c_0)
        else:
            h_0 = None

        if mask is not None:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths=mask2seq_lens(mask).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

        # rnn_outs: (batch, step, hid_dim)
        if isinstance(self.rnn, torch.nn.LSTM):
            rnn_outs, (h_T, _) = self.rnn(embedded, h_0)
        else:
            rnn_outs, h_T = self.rnn(embedded, h_0)

        if mask is not None:
            rnn_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_outs, batch_first=True, padding_value=0
            )

        if return_last_hidden:
            # h_T: (layers*directions, batch, hid_dim/2)
            return rnn_outs, h_T
        else:
            return rnn_outs

    def forward(
        self,
        embedded: torch.FloatTensor,
        mask: torch.BoolTensor = None,
        return_last_hidden: bool = False,
    ):
        # embedded: (batch, step, emb_dim)
        # hidden: (batch, step, hid_dim)
        if hasattr(self, "in_proj_layer"):
            hidden = self.embedded2hidden(
                self.in_proj_layer(self.dropout(embedded)),
                mask=mask,
                return_last_hidden=return_last_hidden,
            )
        else:
            hidden = self.embedded2hidden(
                self.dropout(embedded), mask=mask, return_last_hidden=return_last_hidden
            )

        if self.shortcut:
            if self.shortcut_mode == "add":
                # Addition residual: o_i = l_i + v_i (EPFD style)
                if hasattr(self, "residual_proj") and self.residual_proj is not None:
                    residual = self.residual_proj(embedded)
                else:
                    residual = embedded
                if return_last_hidden:
                    hidden = (hidden[0] + residual, hidden[1])
                else:
                    hidden = hidden + residual
            else:
                # Concatenation (default eznlp style)
                if return_last_hidden:
                    hidden = (torch.cat([hidden[0], embedded], dim=-1), hidden[1])
                else:
                    hidden = torch.cat([hidden, embedded], dim=-1)
        
        # Apply Rotary Position Embedding (RoPE)
        if self.use_rope:
            if return_last_hidden:
                hidden = (self.rope(hidden[0]), hidden[1])
            else:
                hidden = self.rope(hidden)
        
        # Apply TriAffine Attention Fusion
        if self.use_triaffine:
            if return_last_hidden:
                hidden = (self.triaffine(hidden[0], mask=mask), hidden[1])
            else:
                hidden = self.triaffine(hidden, mask=mask)
        
        # Apply Self-rectified Gate
        if self.use_srg:
            if return_last_hidden:
                hidden = (self._apply_srg(hidden[0]), hidden[1])
            else:
                hidden = self._apply_srg(hidden)
        
        return hidden


class ConvEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        # NOTE: Only the first layer is differently configured, consistent to `torch.nn.RNN` modules
        self.conv_blocks = torch.nn.ModuleList(
            [
                ConvBlock(
                    in_dim=(config.in_dim if k == 0 else config.hid_dim),
                    out_dim=config.hid_dim,
                    kernel_size=config.kernel_size,
                    drop_rate=(0.0 if k == 0 else config.hid_drop_rate),
                    nonlinearity="relu",
                )
                for k in range(config.num_layers)
            ]
        )

    def embedded2hidden(
        self, embedded: torch.FloatTensor, mask: torch.BoolTensor = None
    ):
        # embedded: (batch, step, emb_dim) -> (batch, emb_dim, step)
        hidden = embedded.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            hidden = conv_block(hidden, mask=mask)

        # hidden: (batch, hid_dim, step) -> (batch, step, hid_dim)
        return hidden.permute(0, 2, 1)


class GehringConvEncoder(Encoder):
    """Convolutional sequence encoder by Gehring et al. (2017).

    References
    ----------
    Gehring, J., et al. 2017. Convolutional Sequence to Sequence Learning.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.emb2init_hid = torch.nn.Linear(config.in_dim, config.hid_dim * 2)
        self.glu = torch.nn.GLU(dim=-1)
        reinit_layer_(self.emb2init_hid, "glu")

        self.conv_blocks = torch.nn.ModuleList(
            [
                ConvBlock(
                    in_dim=config.hid_dim,
                    out_dim=config.hid_dim,
                    kernel_size=config.kernel_size,
                    drop_rate=config.hid_drop_rate,  # Note to apply dropout to `init_hidden`
                    nonlinearity="glu",
                )
                for k in range(config.num_layers)
            ]
        )
        self.scale = config.scale

    def embedded2hidden(
        self, embedded: torch.FloatTensor, mask: torch.BoolTensor = None
    ):
        init_hidden = self.glu(self.emb2init_hid(embedded))

        # hidden: (batch, step, hid_dim/channels) -> (batch, hid_dim/channels, step)
        hidden = init_hidden.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            conved = conv_block(hidden, mask=mask)
            hidden = (hidden + conved) * self.scale

        # hidden: (batch, hid_dim/channels, step) -> (batch, step, hid_dim/channels)
        final_hidden = hidden.permute(0, 2, 1)
        # Residual connection
        return (init_hidden + final_hidden) * self.scale


# TODO: Initialization with (truncated) normal distribution with standard deviation of 0.02?
class TransformerEncoder(Encoder):
    """Transformer encoder by Vaswani et al. (2017).

    References
    ----------
    Vaswani, A., et al. 2017. Attention is All You Need.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        if config.use_emb2init_hid:
            self.emb2init_hid = torch.nn.Linear(config.in_dim, config.hid_dim)
            self.relu = torch.nn.ReLU()
            reinit_layer_(self.emb2init_hid, "relu")
        else:
            assert config.hid_dim == config.in_dim

        self.tf_blocks = torch.nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hid_dim=config.hid_dim,
                    ff_dim=config.ff_dim,
                    num_heads=config.num_heads,
                    drop_rate=(
                        0.0
                        if (k == 0 and not config.use_emb2init_hid)
                        else config.hid_drop_rate
                    ),
                    nonlinearity="relu",
                )
                for k in range(config.num_layers)
            ]
        )

    def embedded2hidden(
        self, embedded: torch.FloatTensor, mask: torch.BoolTensor = None
    ):
        if hasattr(self, "emb2init_hid"):
            hidden = self.relu(self.emb2init_hid(embedded))
        else:
            hidden = embedded

        for tf_block in self.tf_blocks:
            hidden = tf_block(hidden, mask=mask)

        return hidden
