import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from rotary_embedding_torch import RotaryEmbedding
from typing import Optional, Tuple

from .hyper_connections import MHCRouter
from .attention import MultiHeadedLatentAttention as Attention
from .feed_forward import FeedForward as MLP, MoE


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
            dropout=dropout
        )
        if n_experts == 1:
            self.ffn = MLP(
                n_embed=embedding_size,
                hidden_size=ff_hidden_size,
                activation=activation,
                dropout=dropout,
            )
        else:
            self.ffn = MoE(
                n_embed=embedding_size,
                hidden_size=ff_hidden_size,
                n_experts=n_experts,
                top_k=n_active_experts,
                activation=activation,
                dropout=dropout,
            )
        self.rms_norm1 = nn.RMSNorm(embedding_size, eps=1e-6)
        self.rms_norm2 = nn.RMSNorm(embedding_size, eps=1e-6)

        self.attn_mhc = MHCRouter(
            embedding_size=embedding_size,
            n_streams=num_residual_streams
        )
        self.ffn_mhc = MHCRouter(
            embedding_size=embedding_size,
            n_streams=num_residual_streams
        )

    def forward(
            self,
            x: torch.Tensor,
            rotary_emb: RotaryEmbedding,
            past_key_values: Optional[Cache] = None,
            past_length: Optional[int] = 0,
            layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.attn_mhc(x, self.sa_heads(
            self.rms_norm1(self.attn_mhc.collapse(x)),
            rotary_emb,
            past_key_values,
            past_length,
            layer_idx
        ))
        router_output = self.ffn(self.rms_norm2(self.ffn_mhc.collapse(x)))
        x = self.ffn_mhc(x, router_output.value)
        return x, router_output.loss
