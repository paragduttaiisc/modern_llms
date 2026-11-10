
from typing import cast

import torch
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from rotary_embedding_torch.rotary_embedding_torch import rotate_half
from torch import nn
from transformers.cache_utils import Cache


def rope_fp32(
        rotary_emb: RotaryEmbedding, t: torch.Tensor,
        positions: torch.Tensor) -> torch.Tensor:
    """Rotate `t` [..., T, rope_dim] for `positions` [T], in fp32.

    The same rotation as `rotate_queries_or_keys`, except that positions,
    angles and the rotation are all computed in fp32. The library builds
    positions in the input's dtype, so with bf16 activations positions
    above 256 get rounded (only 641 distinct values among 0..2047). The
    stored frequencies are used as they are (bf16 after the model cast),
    which only shifts each frequency slightly and keeps relative positions
    exact. Returns `t`'s dtype.
    """
    angles = positions.float()[:, None] * rotary_emb.freqs.float()[None, :]
    angles = angles.repeat_interleave(2, dim=-1)  # [T, rope_dim], pairs
    x = t.float()
    return (x * angles.cos() + rotate_half(x) * angles.sin()).to(t.dtype)


class MultiHeadedLatentAttention(nn.Module):
    """Multi-Headed Latent Attention (MLA, from DeepSeek-V2) with RoPE.

    Standard multi-head attention caches a full key and value per head per
    token. MLA instead down-projects each token to a small latent `c`
    (`kv_latent_dim` wide) shared by all heads, and up-projects it to the
    per-head keys (`k_up`) and values (`v_up`) when attention is computed.
    Only `c` has to be cached, which makes the KV cache several times
    smaller.

    RoPE is position-dependent, so rotating keys up-projected from `c`
    would stop `k_up` from being absorbed into the query projection (the
    DeepSeek inference trick; this implementation re-materializes K and V
    instead, but keeps the same "decoupled RoPE" layout). Position is
    carried by a separate small part of each head instead: every head is
    `[nope | rope]`, `head_dim = nope_dim + rope_dim`. The nope parts of K
    come from `c`; the rope part of K is one extra `rope_dim` vector per
    token, projected straight from the input by `kv_down` and shared by all
    heads, so the cache grows by only `rope_dim` per token. Q has its own
    nope and rope parts per head.

    Weights (rows of `q_proj`, `k_up`, `v_up` are head-major):
        q_proj: [num_heads * head_dim, embed_dim]
        kv_down: [kv_latent_dim + rope_dim, embed_dim], rows [latent | rope]
        k_up: [num_heads * nope_dim, kv_latent_dim]
        v_up: [num_heads * head_dim, kv_latent_dim]
        proj: [embed_dim, num_heads * head_dim], the output projection
        q_norm, k_norm (with `qk_norm`): RMSNorm gains [head_dim], shared
            by all heads

    With `qk_norm`, each head's query and key are RMS-normalized over the
    whole head right before attention (after RoPE, which doesn't change the
    rope slice's norm). The cache is unchanged: it still holds `c` and the
    rotated `k_rope`, and K is normalized after it is re-materialized.
    """
    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            head_dim: int,
            rope_dim: int,
            kv_latent_dim: int,
            block_size: int,
            dropout: float,
            qk_norm: bool = False,
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
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else None

        # Fused SDPA (FlashAttention / memory-efficient kernels) when this
        # torch has it; otherwise an explicit causal mask buffer is needed.
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                'mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(
            self,
            x: torch.Tensor,
            rotary_emb: RotaryEmbedding,
            past_key_values: Cache | None = None,
            past_length: int = 0,
            layer_idx: int | None = None,
            frac: float = 1.0,
    ) -> torch.Tensor:
        """Causal self-attention over `x` [B, T, d], returns [B, T, d].

        `rotary_emb` is the model-wide RoPE module shared by every layer.
        With a `past_key_values` cache, `x` holds only the T new tokens:
        `past_length` (the number of tokens already cached) is the RoPE
        position offset, and this layer's latent `c` [B, T, latent] and
        rotated `k_rope` [B, 1, T, rope_dim] are appended to the cache at
        `layer_idx` (side effect), then the whole cached history is used
        as keys/values.

        `frac` < 1 runs the Matryoshka sub-attention: the input is the
        first d' = x.shape[-1] dims, and only the first `frac` of the heads
        and of the KV latent are used. All weights are prefix slices of the
        full ones (rows are head-major), so every matmul stays a plain GEMM.
        Head dim and RoPE dim are never nested."""
        B, T, d = x.shape
        n_heads = round(self.num_heads * frac)
        latent = round(self.kv_latent_dim * frac)

        # q: [B, n_heads, T, head_dim], split into its nope and rope parts
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
        # A single rope key "head" [B, 1, T, rope_dim], broadcast to all
        # query heads below (like multi-query attention for this part).
        k_rope = k_rope.view(B, T, 1, self.rope_dim).transpose(1, 2)

        # Rotate only the rope slices; the offset gives cached decoding the
        # absolute positions of the new tokens.
        # [T] absolute positions of the new tokens, static cache only
        positions: torch.Tensor | None = None
        # Set by Model from config.rope_fp32_positions (False for models
        # trained before the fix, so they keep the rotation they learned).
        fp32 = getattr(rotary_emb, "fp32_positions", False)
        if getattr(past_key_values, "is_static", False):
            # Static cache (CUDA-graph decoding, utils/fast_decode.py): the
            # positions live in a device tensor, so the offset can't be a
            # Python int.
            positions = cast(torch.Tensor, past_key_values.positions(T))  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            assert self.flash, "the static cache needs F.scaled_dot_product_attention"
        if fp32:
            pos = positions if positions is not None else past_length\
                + torch.arange(T, device=q_rope.device)
            q_rope = rope_fp32(rotary_emb, q_rope, pos)
            k_rope = rope_fp32(rotary_emb, k_rope, pos)
        elif positions is not None:
            # Reproduces rotate_queries_or_keys exactly, including its
            # positions in the query dtype.
            freqs = rotary_emb.forward(positions.to(q_rope.dtype))
            q_rope = apply_rotary_emb(freqs, q_rope)
            k_rope = apply_rotary_emb(freqs, k_rope)
        else:
            q_rope = rotary_emb.rotate_queries_or_keys(
                q_rope, offset=past_length)
            k_rope = rotary_emb.rotate_queries_or_keys(
                k_rope, offset=past_length)

        # The cache stores the latent c and the already-rotated k_rope (not
        # full per-head K/V); update() appends the new tokens and returns
        # the whole history: c [B, T_k, latent], k_rope [B, 1, T_k, rope].
        if past_key_values is not None:
            assert layer_idx is not None, "a KV cache needs the layer index"
            c, k_rope = past_key_values.update(c, k_rope, layer_idx)

        T_k = c.shape[1]

        # Re-materialize per-head K (nope part) and V from the latent for
        # every key position: [B, n_heads, T_k, nope_dim / head_dim].
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
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        if self.flash:
            # SDPA's is_causal aligns the mask top-left, which is only right
            # when T == T_k. With a cache and T > 1 (e.g. verifying several
            # draft tokens) the mask must be aligned bottom-right instead.
            # A single new token (T == 1) may see the whole cache: no mask.
            attn_mask, is_causal = None, (T > 1)
            if positions is not None:
                # The static cache spans its whole capacity; each new token
                # may see the keys up to and including its own position.
                attn_mask = torch.arange(T_k, device=q.device)[None, :]\
                    <= positions[:, None]
                is_causal = False
            elif T > 1 and T_k > T:
                attn_mask = torch.ones(
                    T, T_k, dtype=torch.bool, device=q.device,
                ).tril(diagonal=T_k - T)
                is_causal = False
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                dropout_p=self.dropout.p if self.training else 0.0)
        else:
            # Manual attention; the mask rows are taken from the bottom of
            # the tril so that, with a cache, query i sees keys up to
            # T_k - T + i (the same bottom-right alignment as above).
            att = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
            if T > 1:
                T_k = k.shape[-2]
                causal_mask = self.mask[T_k - T:T_k, :T_k]  # pyright: ignore[reportIndexIssue]
                att = att.masked_fill(causal_mask == 0, float('-inf'))
            # Softmax in fp32 for stability under bf16
            att = F.softmax(att.float(), dim=-1).to(v.dtype)
            att = self.dropout(att)
            out = att @ v

        # Concatenate heads back to [B, T, n_heads * head_dim], project to d
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return F.linear(out, self.proj.weight[:d, :n_heads * self.head_dim])
