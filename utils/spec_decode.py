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
) -> DecodeResult:
    """Greedy self-speculative decoding with the model's MTP heads.

    Produces the same tokens as `greedy_decode` but uses fewer forward
    passes when the extra heads' drafts are right. Every forward runs all
    heads (`model.predict_all_heads`) on every fed position: each head has
    its own KV-cache layer, and all layers must cover the same positions so
    the cache can be cropped uniformly after a rejection.

    `accepted_hist[a]` in the result counts verify passes that accepted
    exactly `a` of the n-1 drafts.
    """
    assert input_ids.shape[0] == 1, "batch size 1 only"
    n = model.n_future_tokens
    if n < 2:
        raise ValueError("speculative decoding needs n_future_tokens > 1")

    cache = DynamicCache()
    preds = model.predict_all_heads(input_ids, cache)  # [n, 1, T] token ids
    n_forwards = 1
    # block[0]: verified next token, block[1:]: drafts for the tokens after it
    # (head i at the last prompt position predicts token T+i)
    block = preds[:, 0, -1]
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
        preds = model.predict_all_heads(block[None, :], cache)
        n_forwards += 1
        # verify[j] = head-0 prediction for the token after block[j]
        verify = preds[0, 0].tolist()
        drafts = block.tolist()
        # Accept drafts left to right while they equal what greedy decoding
        # would have produced; stop at the first mismatch. Under greedy
        # decoding, verify[j] is exactly the next token once block[0..j] is
        # accepted, so every accepted draft is a token greedy would emit.
        n_acc = 0
        while n_acc < n - 1 and verify[n_acc] == drafts[n_acc + 1]:
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
        # The predictions at the last accepted position only attended to
        # accepted tokens (causal mask), so they stay valid after the crop:
        # head 0 gives the next verified token (the correction for the first
        # rejected draft, or a bonus token if all were accepted), and heads
        # 1..n-1 give the new drafts. No extra forward is needed.
        block = preds[:, 0, n_acc]

    # _done is checked after every appended token, so the slice is only a
    # safeguard against overshooting max_new_tokens.
    return DecodeResult(tokens=tokens[:max_new_tokens], n_forwards=n_forwards,
                        accepted_hist=accepted_hist)
