"""CUDA-graph greedy and speculative decoding (batch size 1).

At batch size 1 a small model's forward pass costs about as much as a large
one's, because most of the time goes to launching hundreds of small kernels
from Python, not to the math. A CUDA graph records those kernels once and
replays them with a single launch, so a step's cost follows the model's size.
That matters for speculative decoding with a small draft model, which only
pays off when the draft's forward pass is actually cheaper.

A graph replays fixed shapes and fixed memory, so decoding needs:
- `StaticLatentCache`: preallocated per-layer buffers for the MLA latent `c`
  and the rotated `k_rope`, with the current length in a device tensor
  (attention reads positions and its mask from it; see
  `MultiHeadedLatentAttention.forward`);
- one captured graph per (model, number of new tokens): 1 for greedy, n for
  speculative verification, reading its input from a static buffer.

The prompt (variable length) runs eagerly through the same static cache.
Each step then copies the new tokens into the graph's input, replays it, and
advances the cache length; a rejected draft only moves the length back.
The decoders return the same `DecodeResult` as `utils.spec_decode` and
produce the same tokens as the eager versions (exactly in fp32; in bf16 the
masked attention kernel can round differently, which may flip a near-tie).

Limitations: dense models only on GPUs where torch's `grouped_mm` (used by
the MoE layers) synchronizes with the CPU, as on A100 and RTX 30xx; batch
size 1; the cache has a fixed capacity set from the prompt and
`max_new_tokens`.
"""
from collections.abc import Callable

import torch

from .spec_decode import DecodeResult, _done


class StaticLatentCache:
    """Fixed-size KV cache for MLA layers, usable inside a CUDA graph.

    Holds, for each attention layer (trunk blocks, then MTP heads), the
    latent `c` [1, max_len, latent] and the rotated rope key
    [1, 1, max_len, rope_dim]. `length` (a device tensor) is the number of
    valid positions; buffers past it hold stale values that attention masks
    out. Implements the two methods the model calls on a cache (`update`,
    `get_seq_length`) plus `positions`, which attention uses instead of an
    integer offset when `is_static` is set.
    """

    is_static = True

    def __init__(self, model, max_len: int):
        config = model.config
        n_layers = len(model.layers)
        if model.n_future_tokens > 1:
            n_layers += model.n_future_tokens  # one cache layer per head
        param = next(model.parameters())
        self.max_len = max_len
        self.c = [torch.zeros(1, max_len, config.kv_latent_size,
                              device=param.device, dtype=param.dtype)
                  for _ in range(n_layers)]
        self.k_rope = [torch.zeros(1, 1, max_len, config.rope_size,
                                   device=param.device, dtype=param.dtype)
                       for _ in range(n_layers)]
        self.length = torch.zeros((), dtype=torch.long, device=param.device)

    def positions(self, T: int) -> torch.Tensor:
        """Absolute positions [T] of the next T tokens."""
        return self.length + torch.arange(T, device=self.length.device)

    def update(self, c: torch.Tensor, k_rope: torch.Tensor, layer_idx: int):
        """Write the new tokens at their positions; return the full buffers
        (attention masks the positions past the new tokens)."""
        pos = self.positions(c.shape[1])
        self.c[layer_idx].index_copy_(1, pos, c)
        self.k_rope[layer_idx].index_copy_(2, pos, k_rope)
        return self.c[layer_idx], self.k_rope[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Unused by the static path (positions come from `length`); 0 keeps
        the model's integer `past_length` well defined without a GPU sync."""
        return 0

    def advance(self, n: int) -> None:
        """Mark `n` more positions as valid (after a forward pass)."""
        self.length += n

    def rewind(self, n: int) -> None:
        """Drop the last `n` positions (rejected drafts)."""
        self.length -= n


class GraphedStep:
    """A forward pass on a fixed number of new tokens, captured as a graph.

    `fn(ids)` runs the model on `ids` [1, T] with `cache` and returns a
    tensor; calling the object copies `ids` into the static input, replays
    the graph, advances the cache by T, and returns the (static) output,
    which the next call overwrites.
    """

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor],
                 cache: StaticLatentCache, T: int, warmup: int = 3):
        self.cache, self.T = cache, T
        self.ids = torch.zeros(1, T, dtype=torch.long,
                               device=cache.length.device)
        # Warm up on a side stream (as torch.cuda.graph requires), so lazy
        # initialization and kernel selection happen before capture. These
        # runs write into the cache at `length` without advancing it, so the
        # next real step overwrites those positions.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn(self.ids)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.graph):
                self.out = fn(self.ids)
        except RuntimeError as e:
            raise RuntimeError(
                "CUDA-graph capture failed. MoE layers can't be captured on "
                "GPUs where torch's grouped_mm copies its offsets to the CPU "
                "(seen on A100 and RTX 30xx); decode MoE models eagerly.") from e

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        self.ids.copy_(ids)
        self.graph.replay()
        self.cache.advance(self.T)
        return self.out


def _prefill(fn: Callable[[torch.Tensor], torch.Tensor],
             cache: StaticLatentCache, input_ids: torch.Tensor):
    """Run the prompt eagerly through the static cache."""
    out = fn(input_ids)
    cache.advance(input_ids.shape[1])
    return out


def _next_token_fn(model, cache):
    """ids [1, T] -> the model's greedy next token at each position [T]."""
    def fn(ids):
        logits = model(ids, past_key_values=cache, use_cache=True).logits
        return logits[0].argmax(-1)
    return fn


def _all_heads_fn(model, cache):
    """ids [1, T] -> every head's greedy prediction at each position [n, T]."""
    def fn(ids):
        return model.predict_all_heads(ids, cache)[:, 0]
    return fn


@torch.no_grad()
def graph_greedy_decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> DecodeResult:
    """`greedy_decode` with a static cache and a captured 1-token step."""
    assert input_ids.shape[0] == 1, "batch size 1 only"
    cache = StaticLatentCache(model, input_ids.shape[1] + max_new_tokens + 1)
    fn = _next_token_fn(model, cache)
    nxt = _prefill(fn, cache, input_ids)[-1:]
    step = GraphedStep(fn, cache, 1)
    n_forwards, tokens = 1, []
    while True:
        tokens.append(int(nxt))
        if _done(tokens, max_new_tokens, eos_token_id):
            break
        nxt = step(nxt[None, :])
        n_forwards += 1
    return DecodeResult(tokens=tokens, n_forwards=n_forwards)


@torch.no_grad()
def graph_speculative_decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    draft_model=None,
) -> DecodeResult:
    """`speculative_decode` with static caches and captured n-token steps.

    Same algorithm and output: with `draft_model=None` the model's own MTP
    heads draft (one cache, one graph); otherwise `draft_model`'s heads
    draft and `model` verifies (two caches, two graphs per step).
    """
    assert input_ids.shape[0] == 1, "batch size 1 only"
    drafter = model if draft_model is None else draft_model
    n = drafter.n_future_tokens
    if n < 2:
        raise ValueError("speculative decoding needs n_future_tokens > 1")
    max_len = input_ids.shape[1] + max_new_tokens + 2 * n

    draft_cache = StaticLatentCache(drafter, max_len)
    draft_fn = _all_heads_fn(drafter, draft_cache)
    if draft_model is None:
        cache, verify_fn = draft_cache, None
    else:
        cache = StaticLatentCache(model, max_len)
        verify_fn = _next_token_fn(model, cache)

    preds = _prefill(draft_fn, draft_cache, input_ids)  # [n, T]
    verify = preds[0] if verify_fn is None\
        else _prefill(verify_fn, cache, input_ids)
    draft_step = GraphedStep(draft_fn, draft_cache, n)
    verify_step = None if verify_fn is None\
        else GraphedStep(verify_fn, cache, n)

    n_forwards = 1
    block = torch.cat([verify[-1:], preds[1:, -1]])
    tokens: list[int] = []
    accepted_hist = [0] * n
    while True:
        tokens.append(int(block[0]))
        if _done(tokens, max_new_tokens, eos_token_id):
            break
        # The graphs' outputs are overwritten by the next replay; the
        # comparisons below copy what they need to the CPU first.
        preds = draft_step(block[None, :])
        verify = preds[0] if verify_step is None\
            else verify_step(block[None, :])
        n_forwards += 1
        verify_list, drafts = verify.tolist(), block.tolist()
        n_acc = 0
        while n_acc < n - 1 and verify_list[n_acc] == drafts[n_acc + 1]:
            n_acc += 1
            tokens.append(drafts[n_acc])
            if _done(tokens, max_new_tokens, eos_token_id):
                break
        accepted_hist[n_acc] += 1
        if _done(tokens, max_new_tokens, eos_token_id):
            break
        # Keep block[0..n_acc]; rewinding the length is all a crop needs.
        if n_acc + 1 < n:
            draft_cache.rewind(n - (n_acc + 1))
            if cache is not draft_cache:
                cache.rewind(n - (n_acc + 1))
        block = torch.cat([verify[n_acc:n_acc + 1], preds[1:, n_acc]])

    return DecodeResult(
        tokens=tokens[:max_new_tokens], n_forwards=n_forwards,
        accepted_hist=accepted_hist,
        n_draft_forwards=0 if draft_model is None else n_forwards)
