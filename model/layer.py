import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from rotary_embedding_torch import RotaryEmbedding
from typing import Optional, Tuple

from .attention import MultiHeadedLatentAttention as Attention
from .feed_forward import MoE


class Block(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            head_size: int,
            n_experts: int,
            n_active_experts: int,
            rope_size: int,
            kv_latent_size: int,
            num_attn_heads: int,
            block_size: int,
            activation: str,
            dropout: float,
    ) -> None:
        super().__init__()
        self.sa_heads = Attention(
            num_heads=num_attn_heads,
            embed_dim=embedding_size,
            head_dim=head_size,
            rope_dim=rope_size,
            kv_latent_dim=kv_latent_size,
            block_size=block_size,
            dropout=dropout
        )
        self.moe = MoE(
            n_embed=embedding_size,
            n_experts=n_experts,
            top_k=n_active_experts,
            activation=activation,
            dropout=dropout
        )
        self.rms_norm1 = nn.RMSNorm(embedding_size, eps=1e-6)
        self.rms_norm2 = nn.RMSNorm(embedding_size, eps=1e-6)
    
    def forward(
            self,
            x: torch.Tensor,
            rotary_emb: RotaryEmbedding,
            past_key_values: Optional[Cache] = None,
            past_length: Optional[int] = 0,
            layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_outputs = self.sa_heads(
            self.rms_norm1(x),
            rotary_emb,
            past_key_values,
            past_length,
            layer_idx
        )
        x = x + attention_outputs
        router_outputs, router_loss = self.moe(self.rms_norm2(x))
        x = x + router_outputs
        return x, router_loss
