"""Generate text with a trained checkpoint and measure decoding throughput.

Two modes, plus an option that applies to both:

* Default: sample from the model through a HuggingFace `text-generation`
  pipeline (beam sampling with repetition penalties), then print the text and
  the tokens/second of a timed second run. `--use-cache` turns on the KV
  cache, which for MLA stores the compressed latent instead of full K/V.
* `--speculative` (multi-token prediction checkpoints only): compare plain
  greedy decoding against self-speculative decoding, where the extra MTP
  heads draft the next few tokens and one forward pass verifies them.
  Greedy verification makes both produce the same tokens (up to
  floating-point ties); the report shows how many tokens each forward pass
  yields and the speedup.
* `--matryoshka-frac=f` (Matryoshka checkpoints only): extract the nested
  model of width f first, then run either mode on it.

    python infer.py --save-dir=outputs/checkpoint-xxx --prompt "Hello" \
        --use-cache --max-new-tokens=256 --temperature=0.9 \
        --num-return-sequences=3

    # MTP checkpoints: greedy vs. self-speculative decoding
    python infer.py --save-dir=outputs/checkpoint-xxx --prompt "Hello" \
        --speculative --max-new-tokens=256
"""
import argparse
import time

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
    set_seed,
)

from model import Model, ModelConfig
from utils import get_tokenizer
from utils.fast_decode import (
    DecodeEngine,
    graph_greedy_decode,
    graph_speculative_decode,
)
from utils.spec_decode import greedy_decode, speculative_decode


def _timed(fn, device: str):
    """Run `fn()` and return `(result, wall-clock seconds)`.

    CUDA kernels run asynchronously, so the device is synchronized before
    and after; otherwise the timer would only measure kernel launches.
    """
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def run_speculative(model, tokenizer, args: argparse.Namespace,
                    draft=None) -> None:
    """Compare greedy decoding against MTP speculative decoding.

    Without `draft` the model drafts for itself (self-speculation); with
    `draft` (a nested Matryoshka model) the draft's MTP heads propose the
    tokens and `model` verifies them.
    """
    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], device=args.device)
    eos = None if args.ignore_eos else tokenizer.eos_token_id
    # --cuda-graphs: the same decoders with a static cache and captured
    # steps (utils/fast_decode.py), which removes the per-kernel launch
    # overhead that dominates batch-1 decoding.
    # --compile additionally fuses each step's kernels with torch.compile.
    # The engines keep the captured (and compiled) steps, so the warm-up
    # below pays for capture and compilation once.
    if args.cuda_graphs:
        n = (model if draft is None else draft).n_future_tokens
        max_len = input_ids.shape[1] + args.max_new_tokens + 2 * n
        engine = DecodeEngine(model, max_len, compile=args.compile)
        draft_engine = None if draft is None\
            else DecodeEngine(draft, max_len, compile=args.compile)
        run_greedy = lambda: graph_greedy_decode(  # noqa: E731
            model, input_ids, args.max_new_tokens, eos, engine=engine)
        run_spec = lambda: graph_speculative_decode(  # noqa: E731
            model, input_ids, args.max_new_tokens, eos, draft_model=draft,
            engine=engine, draft_engine=draft_engine)
    else:
        if args.compile:
            raise ValueError("--compile needs --cuda-graphs")
        run_greedy = lambda: greedy_decode(  # noqa: E731
            model, input_ids, args.max_new_tokens, eos)
        run_spec = lambda: speculative_decode(  # noqa: E731
            model, input_ids, args.max_new_tokens, eos, draft_model=draft)
    # Warm-up: the first calls pay for CUDA context setup, kernel selection
    # and allocator growth, which would distort the timings below.
    run_greedy(); run_spec()  # warm-up

    greedy, t_greedy = _timed(run_greedy, args.device)
    spec, t_spec = _timed(run_spec, args.device)

    print(tokenizer.decode(input_ids[0].tolist() + spec.tokens))
    print("-" * 40)
    # Length of the common prefix: speculative decoding with greedy
    # verification should reproduce greedy exactly; small differences can
    # still come from floating-point differences between the multi-token
    # verify pass and one-token steps (e.g. near-ties in the argmax).
    n_match = next((i for i, (a, b) in enumerate(
        zip(greedy.tokens, spec.tokens)) if a != b),
        min(len(greedy.tokens), len(spec.tokens)))
    print(f"Matches greedy:     {greedy.tokens == spec.tokens} "
          f"(first {n_match} tokens identical)")
    print(f"Greedy:             {len(greedy.tokens)} tokens, "
          f"{greedy.n_forwards} forwards, {t_greedy:.3f} s, "
          f"{len(greedy.tokens) / t_greedy:.1f} tok/s")
    print(f"Speculative:        {len(spec.tokens)} tokens, "
          f"{spec.n_forwards} forwards, {t_spec:.3f} s, "
          f"{len(spec.tokens) / t_spec:.1f} tok/s")
    drafter = model if draft is None else draft
    print(f"Tokens per forward: {spec.tokens_per_forward:.2f} "
          f"(max {drafter.n_future_tokens})")
    if draft is not None:
        ratio = sum(p.numel() for p in draft.parameters())\
            / sum(p.numel() for p in model.parameters())
        print(f"Draft model:        {spec.n_draft_forwards} forwards of a "
              f"model with {ratio:.0%} of the parameters")
    print(f"Speedup:            {t_greedy / t_spec:.2f}x")
    # Distribution of how many drafted tokens each verify pass accepted
    total = max(sum(spec.accepted_hist), 1)
    print("Accepted drafts per verify pass: " + ", ".join(
        f"{a}: {c / total:.0%}" for a, c in enumerate(spec.accepted_hist)))


def main(args: argparse.Namespace):
    """Load the checkpoint in `args.save_dir` and run the chosen mode."""
    # Register and load model. Registering the custom config/model classes
    # lets HF's Auto* machinery (used by the pipeline) recognize the
    # "modern_transformer" model type.
    AutoConfig.register("modern_transformer", ModelConfig)
    AutoModelForCausalLM.register(ModelConfig, Model)
    model = Model.from_pretrained(args.save_dir)
    draft = None
    if args.draft_frac > 0:
        if not args.speculative or args.matryoshka_frac < 1.0:
            raise ValueError("--draft-frac needs --speculative and the full "
                             "model (--matryoshka-frac=1)")
        # The nested model (with its MTP heads) drafts; the full model
        # verifies.
        draft = model.extract_matryoshka(args.draft_frac).to(args.device)
        draft.eval()
        print(f"Draft: Matryoshka size {args.draft_frac:g}, "
              f"{sum(p.numel() for p in draft.parameters()) / 1e6:.1f}M params")
    if args.matryoshka_frac < 1.0:
        # A standalone smaller model built from prefix slices of every weight
        model = model.extract_matryoshka(args.matryoshka_frac)
        print(f"Extracted Matryoshka size {args.matryoshka_frac:g}: "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    model = model.to(args.device)
    model.eval()
    tokenizer = get_tokenizer()
    if args.speculative:
        run_speculative(model, tokenizer, args, draft)
        return
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        num_beams=5,
        use_cache=args.use_cache,
        temperature=args.temperature,
        do_sample=True,
    )
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        clean_up_tokenization_spaces=False,
        device=args.device,
    )
    # The first call doubles as a warm-up; the timed run comes next.
    print("Generating text " + ("with" if args.use_cache else "without") + " cache...")
    outputs = generator(args.prompt, generation_config=generation_config)
    if args.device.startswith("cuda"):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            outputs = generator(args.prompt, generation_config=generation_config)
        end_event.record()

        # Force CPU to wait for GPU to finish
        torch.cuda.synchronize()
        elapsed_time_s = start_event.elapsed_time(end_event) / 1000.0
    else:
        # Fallback for CPU execution
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = generator(args.prompt, generation_config=generation_config)
        elapsed_time_s = time.perf_counter() - start_time

    # Count generated tokens by re-encoding the output text minus the prompt
    # (the pipeline returns text, not token ids).
    prompt_tokens = len(tokenizer.encode(args.prompt))
    total_generated_tokens = 0

    if args.num_return_sequences == 1:
        print(outputs[0]['generated_text'])
        total_generated_tokens = len(tokenizer.encode(outputs[0]['generated_text'])) - prompt_tokens
    else:
        for i, output in enumerate(outputs):
            print(f"Sequence {i+1}:\n{output['generated_text']}\n")
            seq_tokens = len(tokenizer.encode(output['generated_text'])) - prompt_tokens
            total_generated_tokens += seq_tokens

    print("-" * 40)
    print(f"Total Time:         {elapsed_time_s:.3f} seconds")
    print(f"Tokens Generated:   {total_generated_tokens} tokens")
    print(f"Throughput:         {total_generated_tokens / elapsed_time_s:.2f} tokens/second")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained transformer model")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="models/checkpoint-muon")
    parser.add_argument("--prompt", type=str, default="The river Mississippi")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num-return-sequences", type=int, default=3)
    parser.add_argument("--speculative", action="store_true", help="Benchmark greedy vs. MTP self-speculative decoding (needs an MTP checkpoint)")
    parser.add_argument("--ignore-eos", action="store_true", help="Keep generating past EOS (for fixed-length benchmarks)")
    parser.add_argument("--matryoshka-frac", type=float, default=1.0, help="Run a smaller nested size (extracted from a Matryoshka checkpoint)")
    parser.add_argument("--draft-frac", type=float, default=0.0, help="With --speculative: draft with this nested Matryoshka size's MTP heads and verify with the full model (0 = self-speculation)")
    parser.add_argument("--cuda-graphs", action="store_true", help="With --speculative: decode with a static cache and CUDA graphs (dense models; much less launch overhead)")
    parser.add_argument("--compile", action="store_true", help="With --cuda-graphs: also fuse each decoding step's kernels with torch.compile (slow first run)")
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
