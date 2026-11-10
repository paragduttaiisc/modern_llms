
import torch
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from transformers.cache_utils import Cache

from .attention import MultiHeadedLatentAttention as Attention
from .feed_forward import FeedForward as MLP
from .feed_forward import MoE
from .hyper_connections import MHCRouter


def prefix_rms_norm(norm: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """RMSNorm over the first d' = x.shape[-1] dims (Matryoshka sub-width):
    statistics over those dims only, scaled by the first d' gains."""
    d = x.shape[-1]
    if d == norm.weight.shape[0]:
        return norm(x)
    return F.rms_norm(x, (d,), norm.weight[:d], norm.eps)


class ResidualConnection(nn.Module):
    """Residual connection around one sub-layer (attention or FFN).

    With `n_streams == 1` it is the ordinary residual: `collapse` returns
    the hidden state unchanged and `expand(x, y)` is `x + y`. With more
    streams it wraps an `MHCRouter` (hyper-connections): the hidden state
    is [B, T, S, C], `collapse` mixes the S streams into the sub-layer's
    [B, T, C] input, and `expand` mixes the streams and writes the
    sub-layer's output back into each of them.
    """
    def __init__(self, embedding_size: int, n_streams: int):
        super().__init__()

        self.n_streams = n_streams

        if n_streams == 1:
            self.mhc = None
        else:
            self.mhc = MHCRouter(
                embedding_size=embedding_size,
                n_streams=n_streams,
            )

    def collapse(self, x):
        """Sub-layer input from the residual state ([B, T, C])."""
        if self.mhc is None:
            return x
        return self.mhc.collapse(x)

    def expand(self, x, y):
        """New residual state from the old one `x` and sub-layer output
        `y` [B, T, C]."""
        if self.mhc is None:
            return x + y
        return self.mhc(x, y)


class Block(nn.Module):
    """One pre-norm transformer block:
        x = x + MLA(RMSNorm(x))
        x = x + FFN_or_MoE(RMSNorm(x))
    Pre-norm (normalizing the sub-layer input, not the residual) keeps an
    un-normalized path from the embeddings to the output, which makes deep
    stacks train stably. RMSNorm rescales by the root mean square only (no
    mean subtraction, no bias), which is cheaper than LayerNorm and works
    as well. The "+" is a `ResidualConnection` (plain add, or mHC streams).
    `n_experts == 1` selects the dense FFN, otherwise an MoE layer.
    """
    def __init__(
            self,
            embedding_size: int,
            sa_head_size: int,
            ff_hidden_size: int,
            n_experts: int,
            n_active_experts: int,
            rope_size: int,
            kv_latent_size: int,
            num_attn_heads: int,
            num_residual_streams: int,
            block_size: int,
            activation: str,
            dropout: float,
            exact_ffn_hidden: bool = False,
    ) -> None:
        super().__init__()

        self.sa_heads = Attention(
            num_heads=num_attn_heads,
            embed_dim=embedding_size,
            head_dim=sa_head_size,
            rope_dim=rope_size,
            kv_latent_dim=kv_latent_size,
            block_size=block_size,
            dropout=dropout,
        )

        if n_experts == 1:
            self.ffn = MLP(
                n_embed=embedding_size,
                hidden_size=ff_hidden_size,
                activation=activation,
                dropout=dropout,
                exact_hidden=exact_ffn_hidden,
            )
            self.is_moe = False
        else:
            self.ffn = MoE(
                n_embed=embedding_size,
                hidden_size=ff_hidden_size,
                n_experts=n_experts,
                top_k=n_active_experts,
                activation=activation,
                dropout=dropout,
                exact_hidden=exact_ffn_hidden,
            )
            self.is_moe = True

        self.rms_norm1 = nn.RMSNorm(embedding_size, eps=1e-6)
        self.rms_norm2 = nn.RMSNorm(embedding_size, eps=1e-6)

        self.attn_residual = ResidualConnection(
            embedding_size=embedding_size,
            n_streams=num_residual_streams,
        )
        self.ffn_residual = ResidualConnection(
            embedding_size=embedding_size,
            n_streams=num_residual_streams,
        )

    def forward(
            self,
            x: torch.Tensor,
            rotary_emb: RotaryEmbedding,
            past_key_values: Cache | None = None,
            past_length: int | None = 0,
            layer_idx: int | None = None,
            frac: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the block on x [B, T, C] (or [B, T, S, C] with mHC streams).

        Returns `(x, aux_loss)`: the new hidden state, same shape as the
        input, and the MoE's unscaled load-balancing loss (None for a dense
        FFN). The cache arguments are passed through to the attention (see
        `MultiHeadedLatentAttention.forward`).

        `frac` < 1: Matryoshka sub-block on the first d' = x.shape[-1]
        residual dims (heads / latent / FFN width scaled by `frac`)."""
        if frac < 1.0 and self.attn_residual.mhc is not None:
            raise NotImplementedError("Matryoshka nesting with mHC streams")
        attn_input = self.attn_residual.collapse(x)
        attn_output = self.sa_heads(
            prefix_rms_norm(self.rms_norm1, attn_input),
            rotary_emb,
            past_key_values,
            past_length,
            layer_idx,
            frac,
        )
        x = self.attn_residual.expand(x, attn_output)

        ffn_input = self.ffn_residual.collapse(x)
        ffn_output = self.ffn(prefix_rms_norm(self.rms_norm2, ffn_input), frac)

        x = self.ffn_residual.expand(x, ffn_output.value)

        return x, ffn_output.loss if self.is_moe else None
