import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from rotary_embedding_torch import RotaryEmbedding
from typing import Optional


class MultiHeadedLatentAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            head_dim: int,
            rope_dim: int,
            kv_latent_dim: int,
            block_size: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.rope_dim = rope_dim
        self.nope_dim = head_dim - rope_dim
        self.q_proj = nn.Linear(
            in_features=embed_dim,
            out_features=num_heads * head_dim,
            bias=False
        )
        self.kv_down = nn.Linear(
            in_features=embed_dim,
            out_features=kv_latent_dim + rope_dim,
            bias=False
        )
        self.k_up = nn.Linear(
            in_features=kv_latent_dim,
            out_features=num_heads * self.nope_dim,
            bias=False
        )
        self.v_up = nn.Linear(
            in_features=kv_latent_dim,
            out_features=num_heads * head_dim,
            bias=False
        )

        self.proj = nn.Linear(
            in_features=num_heads * head_dim,
            out_features=embed_dim,
            bias=False
        )
        self.dropout = nn.Dropout(dropout)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                'mask', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(
            self,
            x: torch.Tensor,
            rotary_emb: RotaryEmbedding,
            past_key_values: Optional[Cache] = None,
            past_length: Optional[int] = 0,
            layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q_nope, q_rope = torch.split(
            self.q_proj(x).view(
                B, T, self.num_heads, self.head_dim).transpose(1, 2),
            [self.nope_dim, self.rope_dim],
            dim=-1
        )
        c, k_rope = torch.split(
            self.kv_down(x),
            [self.kv_latent_dim, self.rope_dim],
            dim=-1
        )
        k_rope = k_rope.view(B, T, 1, self.rope_dim).transpose(1, 2)

        q_rope = rotary_emb.rotate_queries_or_keys(q_rope, offset=past_length) # type: ignore
        k_rope = rotary_emb.rotate_queries_or_keys(k_rope, offset=past_length) # type: ignore

        if past_key_values is not None:
            c, k_rope = past_key_values.update(c, k_rope, layer_idx) # type: ignore

        T_k = c.shape[1]

        k_nope = self.k_up(c).view(
            B, T_k, self.num_heads, self.nope_dim).transpose(1, 2)
        k_rope = k_rope.view(
            B, T_k, 1, self.rope_dim).transpose(1, 2)
        v = self.v_up(c).view(
            B, T_k, self.num_heads, self.head_dim).transpose(1, 2)


        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat(
            [k_nope, k_rope.expand(-1, self.num_heads, -1, -1)], dim=-1)

        if self.flash: 
            is_causal = (T > 1) 
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=is_causal,
                dropout_p=self.dropout.p if self.training else 0.0)
        else: 
            att = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
            if T > 1:
                T_k = k.shape[-2]
                causal_mask = self.mask[T_k - T:T_k, :T_k] # type: ignore
                att = att.masked_fill(causal_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            out = att @ v
            
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(out)
