"""Greedy decoding and MTP self-speculative decoding (batch size 1).

Self-speculative decoding (blockwise parallel decoding, Stern et al. 2018;
Gloeckle et al. 2024) uses the model's own MTP heads as the drafter:

1. At the last accepted position, head 0 gives the next token (exact under
   greedy decoding) and heads 1..n-1 give drafts for the tokens after it.
2. Those n tokens are fed in one forward pass. Head 0's argmax at each
   position verifies the following draft; the longest matching prefix is
   accepted, the KV cache is cropped back to it, and step 1 repeats from the
   last accepted position.

Each forward pass therefore emits between 1 and n tokens, and the output is
identical to plain greedy decoding (up to floating-point ties).

The drafter can also be a separate model with MTP heads, such as a nested
Matryoshka model: it drafts the n-1 tokens after the verified one, and the
full model verifies them (`speculative_decode(..., draft_model=...)`).

Both decoders here are written for clarity, not speed: batch size 1, a
Python loop over steps, and an HF `DynamicCache`. `infer.py --speculative`
runs both and compares their tokens per forward pass.
"""
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class DecodeResult:
    """Output of a decoder: the generated tokens and how it got them.

    `tokens` excludes the prompt. `n_forwards` counts model forward passes,
    including the one over the prompt (prefill).
    """

    tokens: list[int]
    n_forwards: int
    # accepted_hist[a] = number of verify passes that accepted `a` drafts
    accepted_hist: list[int] = field(default_factory=list)
    # forward passes of a separate draft model (0 for self-speculation)
    n_draft_forwards: int = 0

    @property
    def tokens_per_forward(self) -> float:
        """Generated tokens per forward pass (1.0 for plain greedy)."""
        return len(self.tokens) / max(self.n_forwards, 1)


def _done(tokens: list[int], max_new_tokens: int, eos_token_id) -> bool:
    """True once `max_new_tokens` are generated or the last token is EOS."""
    return len(tokens) >= max_new_tokens or (
        eos_token_id is not None and bool(tokens)
        and tokens[-1] == eos_token_id)


@torch.no_grad()
def greedy_decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> DecodeResult:
    """Plain greedy decoding with the next-token head and a KV cache.

    The baseline for `speculative_decode`: one forward pass per token. The
    prompt `input_ids` (`[1, T]`) is run once to fill the cache, then each
    step feeds only the newest token. Only head 0 runs (the model's normal
    forward), so this also works for models without MTP heads.
    """
    assert input_ids.shape[0] == 1, "batch size 1 only"
    cache = DynamicCache()
    logits = model(input_ids, past_key_values=cache, use_cache=True).logits
    n_forwards, tokens = 1, []
    while True:
        tokens.append(int(logits[0, -1].argmax()))
        if _done(tokens, max_new_tokens, eos_token_id):
            break
        nxt = torch.tensor([[tokens[-1]]], device=input_ids.device)
        logits = model(nxt, past_key_values=cache, use_cache=True).logits
        n_forwards += 1
    return DecodeResult(tokens=tokens, n_forwards=n_forwards)


@torch.no_grad()
def speculative_decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    draft_model=None,
) -> DecodeResult:
    """Greedy speculative decoding with MTP heads as the drafter.

    Without `draft_model` this is self-speculative decoding: `model`'s own
    MTP heads draft and its head 0 verifies. With `draft_model` (e.g. a
    nested Matryoshka model extracted from `model`, trained with MTP heads),
    the drafts come from `draft_model`'s heads and `model`'s next-token head
    verifies; each model keeps its own KV cache, and both are cropped back
    after a rejection. Either way the output equals `greedy_decode(model)`
    (up to floating-point ties), with fewer forward passes of `model` when
    the drafts are right.

    Every forward of the drafter runs all its heads
    (`predict_all_heads`) on every fed position: each head has its own
    KV-cache layer, and all layers must cover the same positions so the
    cache can be cropped uniformly.

    `n_forwards` counts forwards of `model` and `n_draft_forwards` those of
    `draft_model` (0 for self-speculation). `accepted_hist[a]` counts verify
    passes that accepted exactly `a` of the n-1 drafts.
    """
    assert input_ids.shape[0] == 1, "batch size 1 only"
    drafter = model if draft_model is None else draft_model
    n = drafter.n_future_tokens
    if n < 2:
        raise ValueError("speculative decoding needs n_future_tokens > 1")

    cache = DynamicCache()
    draft_cache = cache if draft_model is None else DynamicCache()

    def run(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feed `ids` [1, L] to both models; return the verifier's next-token
        ids [L] and the drafter's per-head predictions [n, L]."""
        preds = drafter.predict_all_heads(ids, draft_cache)[:, 0]
        if draft_model is None:
            return preds[0], preds
        logits = model(ids, past_key_values=cache, use_cache=True).logits
        return logits[0].argmax(-1), preds

    verify, preds = run(input_ids)
    n_forwards = 1
    # block[0]: verified next token, block[1:]: drafts for the tokens after
    # it (head i at the last prompt position predicts token T+i)
    block = torch.cat([verify[-1:], preds[1:, -1]])
    tokens: list[int] = []
    accepted_hist = [0] * n

    while True:
        tokens.append(int(block[0]))
        if _done(tokens, max_new_tokens, eos_token_id):
            break

        # Feed the whole block [1, n] in one pass. block[0] is already
        # accepted; this pass both verifies the drafts and, at every
        # position, prepares the next block in case it ends up the last
        # accepted one.
        verify, preds = run(block[None, :])
        n_forwards += 1
        # verify[j] = the verifier's prediction for the token after block[j]
        verify_list = verify.tolist()
        drafts = block.tolist()
        # Accept drafts left to right while they equal what greedy decoding
        # would have produced; stop at the first mismatch. Under greedy
        # decoding, verify[j] is exactly the next token once block[0..j] is
        # accepted, so every accepted draft is a token greedy would emit.
        n_acc = 0
        while n_acc < n - 1 and verify_list[n_acc] == drafts[n_acc + 1]:
            n_acc += 1
            tokens.append(drafts[n_acc])
            if _done(tokens, max_new_tokens, eos_token_id):
                break
        accepted_hist[n_acc] += 1
        if _done(tokens, max_new_tokens, eos_token_id):
            break

        # Keep cache entries for block[0..n_acc]; drop rejected drafts.
        # A negative count removes that many tokens from the end of every
        # layer (trunk and all heads) in this transformers version.
        if n_acc + 1 < n:
            cache.crop(-(n - (n_acc + 1)))
            if draft_cache is not cache:
                draft_cache.crop(-(n - (n_acc + 1)))
        # The predictions at the last accepted position only attended to
        # accepted tokens (causal mask), so they stay valid after the crop:
        # the verifier gives the next token (the correction for the first
        # rejected draft, or a bonus token if all were accepted), and the
        # drafter's heads 1..n-1 give the new drafts. No extra forward is
        # needed.
        block = torch.cat([verify[n_acc:n_acc + 1], preds[1:, n_acc]])

    # _done is checked after every appended token, so the slice is only a
    # safeguard against overshooting max_new_tokens.
    return DecodeResult(
        tokens=tokens[:max_new_tokens], n_forwards=n_forwards,
        accepted_hist=accepted_hist,
        n_draft_forwards=0 if draft_model is None else n_forwards)
