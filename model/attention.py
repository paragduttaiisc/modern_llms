
import torch
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from transformers.cache_utils import Cache


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
            past_key_values: Cache | None = None,
            past_length: int | None = 0,
            layer_idx: int | None = None,
            frac: float = 1.0,
    ) -> torch.Tensor:
        """`frac` < 1 runs the Matryoshka sub-attention: the input is the
        first d' = x.shape[-1] dims, and only the first `frac` of the heads
        and of the KV latent are used. All weights are prefix slices of the
        full ones (rows are head-major), so every matmul stays a plain GEMM.
        Head dim and RoPE dim are never nested."""
        B, T, d = x.shape
        n_heads = round(self.num_heads * frac)
        latent = round(self.kv_latent_dim * frac)

        q = F.linear(x, self.q_proj.weight[:n_heads * self.head_dim, :d])
        q_nope, q_rope = torch.split(
            q.view(B, T, n_heads, self.head_dim).transpose(1, 2),
            [self.nope_dim, self.rope_dim],
            dim=-1
        )
        if latent == self.kv_latent_dim:
            c, k_rope = torch.split(
                F.linear(x, self.kv_down.weight[:, :d]),
                [self.kv_latent_dim, self.rope_dim],
                dim=-1
            )
        else:
            # kv_down rows are [latent | rope]: nest only the latent part
            c = F.linear(x, self.kv_down.weight[:latent, :d])
            k_rope = F.linear(x, self.kv_down.weight[self.kv_latent_dim:, :d])
        k_rope = k_rope.view(B, T, 1, self.rope_dim).transpose(1, 2)

        q_rope = rotary_emb.rotate_queries_or_keys(q_rope, offset=past_length) # type: ignore
        k_rope = rotary_emb.rotate_queries_or_keys(k_rope, offset=past_length) # type: ignore

        if past_key_values is not None:
            c, k_rope = past_key_values.update(c, k_rope, layer_idx) # type: ignore

        T_k = c.shape[1]

        k_nope = F.linear(
            c, self.k_up.weight[:n_heads * self.nope_dim, :latent]).view(
            B, T_k, n_heads, self.nope_dim).transpose(1, 2)
        k_rope = k_rope.view(
            B, T_k, 1, self.rope_dim).transpose(1, 2)
        v = F.linear(
            c, self.v_up.weight[:n_heads * self.head_dim, :latent]).view(
            B, T_k, n_heads, self.head_dim).transpose(1, 2)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat(
            [k_nope, k_rope.expand(-1, n_heads, -1, -1)], dim=-1)

        if self.flash:
            # SDPA's is_causal aligns the mask top-left, which is only right
            # when T == T_k. With a cache and T > 1 (e.g. verifying several
            # draft tokens) the mask must be aligned bottom-right instead.
            attn_mask, is_causal = None, (T > 1)
            if T > 1 and T_k > T:
                attn_mask = torch.ones(
                    T, T_k, dtype=torch.bool, device=q.device,
                ).tril(diagonal=T_k - T)
                is_causal = False
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal,
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
        return F.linear(out, self.proj.weight[:d, :n_heads * self.head_dim])
