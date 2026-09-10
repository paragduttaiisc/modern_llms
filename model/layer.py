
import torch
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from transformers.cache_utils import Cache

from .attention import MultiHeadedLatentAttention as Attention
from .feed_forward import FeedForward as MLP
from .feed_forward import MoE
from .hyper_connections import MHCRouter


class ResidualConnection(nn.Module):
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
        if self.mhc is None:
            return x
        return self.mhc.collapse(x)

    def expand(self, x, y):
        if self.mhc is None:
            return x + y
        return self.mhc(x, y)


class Block(nn.Module):
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_input = self.attn_residual.collapse(x)
        attn_output = self.sa_heads(
            self.rms_norm1(attn_input),
            rotary_emb,
            past_key_values,
            past_length,
            layer_idx,
        )
        x = self.attn_residual.expand(x, attn_output)

        ffn_input = self.ffn_residual.collapse(x)
        ffn_output = self.ffn(self.rms_norm2(ffn_input))

        x = self.ffn_residual.expand(x, ffn_output.value)

        return x, ffn_output.loss if self.is_moe else None
