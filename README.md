# Modern LLMs

<div align="center">

Extending [nanoGPT](https://github.com/karpathy/nanoGPT) to train modern billion-parameter LLMs from scratch on local/multi-GPU hardware.

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0--1.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/Accelerate-DDP-orange.svg)](https://huggingface.co/docs/accelerate)

</div>

<!-- ---

## 📺 Video Lectures

> A companion video lecture series walking through the architecture, training tricks, and code — from first principles to a modern transformer.

| # | Topic | Status |
|---|-------|--------|
| 1 | Project overview & setup | 🔜 Coming soon |
| 2 | Multi-Headed Latent Attention (MLA) | 🔜 Coming soon |
| 3 | Mixture of Experts (MoE) training | 🔜 Coming soon |
| 4 | Dual optimizer (Muon + AdamW) | 🔜 Coming soon |
| 5 | Training at scale with Accelerate | 🔜 Coming soon |

📎 [Link to playlist](https://youtube.com/playlist?list=YOUR_PLAYLIST_ID) — *placeholder* -->

---

## 🏗️ Architecture

```
train.py / infer.py          ← Entry points
model/
  config.py                  ← ModelConfig (transformers PreTrainedConfig)
  model.py                   ← Model (PreTrainedModel + GenerationMixin)
  layer.py                   ← Block (MLA + FFN/MoE + mHC with residual connections)
  attention.py               ← MultiHeadedLatentAttention (MLA core)
  feed_forward.py            ← FeedForward (SwiGLU / GELU / SqReLU) + MoE (top-k routing)
  hyper_connections.py       ← MHCRouter (hyper-connection routing with Sinkhorn-Knopp)
utils/
  data_utils.py              ← TokenDataset (IterableDataset), HellaswagDataset
  train_utils.py             ← LLMTrainer (HuggingFace Trainer subclass)
  spec_decode.py             ← Greedy + MTP self-speculative decoding
  misc_utils.py              ← human_readable_numbers helper
scripts/
  token_frequencies.py       ← Token counts for the MTP draft vocabulary
  extract_matryoshka.py      ← Slice a smaller standalone model out of a Matryoshka checkpoint
tokenizer/                   ← Custom tokenizer (starcoderbase base + special tokens)
data -> /home/parag/Data/llm_data   ← Symlink to shard data
```

### Block diagram

```
Input Embedding
       │
       ▼
┌───────────────────────────────────────┐
│           Transformer Block           │
│                                       │
│   ┌─────────────┐    ┌────────────┐   │
│   │ RMSNorm  ──►│───►│ MLA        │   │
│   └─────────────┘    │ (latent    │   │
│                      │  K/V)      │   │
│                      └────────────┘   │
│              ▲          │             │
│              │     residual           │
│              │          ▼             │
│   ┌─────────────┐    ┌────────────┐   │
│   │ RMSNorm  ──►│───►│ FFN / MoE  │   │
│   └─────────────┘    └────────────┘   │
│              ▲          │             │
│              │     residual           │
│              │          ▼             │
│   ┌────────────────────────────────┐  │
│   │      mHC Router (Sinkhorn)     │  │
│   └────────────────────────────────┘  │
└───────────────────────────────────────┘
       │
       ▼
   LM Head
```

### Key architectural details

<details>
<summary><b>Click to expand architectural deep-dive</b></summary>

- **MLA Attention** — K and V projections share a latent bottleneck (`kv_latent_dim`). K is split into nope (up-projected via `k_up`) and rope components; V is up-projected via `v_up`. Q is split into nope + rope. RoPE is applied only to the rope components. This is the core memory-saving innovation.
- **Block structure**: Pre-norm RMSNorm → MLA → residual → RMSNorm → FFN → residual. The FFN is either a plain MLP (when `n_experts=1`) or a sparse MoE (when `n_experts>1`). Gated SwiGLU or standard activation.
- **MoE** — Each FFN layer is replaced by a sparse Mixture-of-Experts with top-k gating. Switch-style load-balancing auxiliary loss (`n_experts · Σ load·importance`, equal to 1 when perfectly balanced) is averaged over layers and added to the total loss scaled by `router_loss_coef`. Experts are stored as stacked 3-D weights and run with grouped matmuls.
- **Hyper-connections (mHC)** — Each layer has a learnable multi-stream routing mechanism (`MHCRouter`) that expands the input into `num_residual_streams` parallel streams. Each stream goes through its own MLA + FFN path, and streams are mixed via Sinkhorn-Knopp soft attention before being collapsed back. Uses learnable pre-mixing (`h_pre`), post-mixing (`h_post`), and residual mixing (`h_res`) parameters.
- **Multi-token prediction (MTP)** — With `--n-future-tokens=n > 1`, the last block is replaced by `n` parallel single-block heads on a shared trunk (Gloeckle et al., 2024); head `i` predicts token `t+1+i` and all heads share the tied unembedding. Loss = head-0 loss + `mtp_loss_weight` × mean of the other heads. Each head runs under activation checkpointing so only one head's `[B, T, V]` logits are alive at a time. Eval loss, perplexity and HellaSwag use head 0 only. To cut the cost of the extra heads, they can predict over only the most frequent tokens (`draft_vocab_size`; targets outside it are ignored) and compute their loss on a random fraction of positions (`mtp_position_frac`). Head 0 always uses the full vocabulary and every position.
- **Self-speculative decoding** — The MTP heads draft the next `n-1` tokens; one forward pass verifies them with head 0 and keeps the longest matching prefix (greedy, output identical to plain greedy decoding).
- **Matryoshka (nested-width) training** — With `--matryoshka-fracs=1,0.5,0.25`, every smaller size `f` is a prefix slice of the full model: the first `f` of d_model, attention heads, MLA KV latent and FFN width in every block (incl. MTP heads), the first `f·d` columns of the tied embedding, and RMSNorm over the prefix dims. Head and RoPE dims are not nested. By default each micro-batch trains one size (the full model with probability `--matryoshka-full-prob`; with `--matryoshka-full-per-step`, every optimizer step includes a full-model micro-batch). With `--matryoshka-all-sizes`, every micro-batch trains every size on the same tokens and the loss is their sum, so the full model sees the same data as a run without Matryoshka. Every size is evaluated separately. `scripts/extract_matryoshka.py` slices out a standalone smaller checkpoint whose outputs equal the parent at that size.
- **Hybrid optimizer** — A single [muonium](https://pypi.org/project/muonium/) `Muon` optimizer with per-group algorithms: Muon for hidden 2-D weights (head-wise for Q / K-up / V-up) and 3-D MoE expert weights; AdamW for embeddings, the LM head, routers, mHC parameters and 1-D params. Both share one 3-phase LR schedule (warmup → cosine decay → constant plateau at 0.1×), with separate base LRs.

</details>

<!-- ---

## 📚 Paper References

Techniques used in this project, with links to the original papers:

| Technique | Paper |
|-----------|-------|
| **Multi-Headed Latent Attention (MLA)** | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) |
| **Mixture of Experts (MoE)** | [Mixtral of Experts](https://arxiv.org/abs/2401.04088) |
| **Muon Optimizer** | [Muon: Momentum Orthogonalized NeuralOptimizer](https://arxiv.org/abs/2411.02884) |
| **Flash Attention** | [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) |
| **Rotary Positional Embeddings (RoPE)** | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) |
| **Gated Linear Units (GLU)** | [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) |
| **SwiGLU Activation** | [GLU Variants Improve Transformer](https://arxiv.org/abs/2105.09118) |
| **RMSNorm** | [Root Mean Layer Normalization](https://arxiv.org/abs/1910.07467) |
| **nanoGPT** | [nanoGPT — The simplest best tutorial/training codebase for training LLMs](https://github.com/karpathy/nanoGPT) |

--- -->

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU(s) (tested on H200, A100, RTX 4090)
- 80GB+ VRAM per GPU for full-scale training (less for smaller configs)

### Installation

```bash
git clone git@github.com:paragduttaiisc/modern_llms.git
cd modern_llms
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🎯 Training

### MoE (default)

```bash
# Single GPU / debug
accelerate launch train.py --use-bf16 --batch-size=24 --n-layers=12 --n-heads=12 \
    --block-size=2048 --n-iters=25400 --save-dir=models/your_model

# Multi-GPU (SLURM via run.sh)
sbatch run.sh
```

The SLURM script (`run.sh`) is configured for 4 H200 GPUs with bf16, batch=80, and ~15K iterations (approx 10B tokens) targeting ~500K effective tokens per iteration.

### Multi-token prediction

Add `--n-future-tokens=4` to train 4 parallel prediction heads (`--mtp-loss-weight` scales the extra heads; `--no-mtp-checkpoint` trades memory for speed). Alongside the total `loss`, training logs the next-token loss as `lm_loss`, the mean of the extra heads as `mtp_loss`, and each head as `mtp_loss_1..n`.

To make the extra heads cheaper, first count token frequencies on the training data, then restrict heads 2..n to the most frequent tokens and a random half of the positions:

```bash
python scripts/token_frequencies.py --shard-list-file=data/web_small_10B.json --out=data/token_counts_web_small_10B.npy
accelerate launch --num_processes=8 --mixed_precision=bf16 train.py --n-experts=1 --n-future-tokens=4 --no-mtp-checkpoint \
    --draft-vocab-size=16384 --mtp-position-frac=0.5 --wandb-run-name="<RunName>"
```

On a 1B-token sample of `web_small_10B`, the top 8,192 tokens cover 95.6% of tokens and the top 16,384 cover 99.25%.

### Matryoshka (nested sizes)

Train the full model and nested 1/2- and 1/4-width models together (combinable with MTP), then extract a smaller standalone model:

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 train.py --n-experts=1 --matryoshka-fracs=1,0.5,0.25 \
    --matryoshka-full-per-step --wandb-run-name="<RunName>"
# or train every size on every micro-batch:
accelerate launch --num_processes=8 --mixed_precision=bf16 train.py --n-experts=1 --matryoshka-fracs=1,0.25 \
    --matryoshka-all-sizes --wandb-run-name="<RunName>"
python scripts/extract_matryoshka.py --save-dir=outputs/checkpoint-xxx --frac=0.25 --out=outputs/checkpoint-xxx-f0.25
```

Every size must give whole-number dimensions (e.g. 768·f, 12·f heads, 128·f latent, FFN width·f). Per-size losses are logged with an `_f<size>` suffix (e.g. `lm_loss_f0.25`, `eval_f0.25_loss`).

### Plain MLP

Set `--n-experts=1` to disable MoE and train with a standard feed-forward network (no router loss):

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 train.py --use-bf16 --n-experts=1 \
    --router-loss-weight=0.0 --batch-size=16 --n-iters=38140 --warmup-iters=500 \
    --last-decay-iter=37000 --eval-interval=1000 --save-interval=5000 \
    --wandb-run-name="<RunName>"
```

### Larger models (ZeRO-2)

`--fsdp` shards gradients and optimizer state across GPUs (FSDP2 `shard_grad_op`, i.e. ZeRO-2) while each GPU keeps a full copy of the weights during the step. Weights and gradients stay bf16 (mixed precision is turned off so Accelerate does not upcast them to fp32); Muon momentum and AdamW moments are fp32. `--activation-checkpointing` recomputes every transformer block in backward. Example: a GPT-3 Large-shaped MoE (24 layers, d_model 1536, 16 heads of 96) with 12 experts, 3 active, for 2.95B total / 0.91B active parameters:

```bash
accelerate launch --num_processes=8 train.py --fsdp --embed-size=1536 --n-layers=24 --n-heads=16 \
    --attn-head-size=96 --rope-size=32 --kv-latent-size=384 --mlp-hidden-size=3072 \
    --n-experts=12 --n-active-experts=3 --batch-size=4 --shard-list-file=data/llm_mix_100B.json \
    --wandb-run-name="<RunName>"
```

### Key arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `--block-size` | 2048 | Sequence length |
| `--n-layers` | 12 | Transformer layers |
| `--n-heads` | 12 | Attention heads |
| `--attn-head-size` | 64 | Per-head dimension |
| `--kv-latent-size` | 128 | MLA latent bottleneck |
| `--rope-size` | 16 | RoPE dimension (subset of head_size) |
| `--embed-size` | 768 | Model hidden dimension |
| `--mlp-hidden-size` | 3072 | FFN hidden dimension |
| `--non-linearity` | SwiGLU | FFN activation (GELU / SwiGLU / SqReLU) |
| `--n-experts` | 8 | Number of MoE experts per layer (**1 = plain MLP**) |
| `--n-active-experts` | 2 | Top-k experts active per token |
| `--router-loss-weight` | 0.01 | Load balancing auxiliary loss coefficient (ignored when `n_experts=1`) |
| `--n-future-tokens` | 1 | Multi-token prediction heads (1 = standard next-token LM) |
| `--mtp-loss-weight` | 1.0 | Weight of the mean loss of MTP heads 2..n |
| `--draft-vocab-size` | 0 | Heads 2..n predict only the N most frequent tokens (0 = full vocab) |
| `--draft-vocab-file` | `data/token_counts_web_small_10B.npy` | Token counts from `scripts/token_frequencies.py` |
| `--mtp-position-frac` | 1.0 | Fraction of positions used for the loss of heads 2..n |
| `--matryoshka-fracs` | `1` | Comma-separated nested widths trained jointly (1 = no nesting) |
| `--matryoshka-full-prob` | 0.5 | Probability a micro-batch trains the full model (else a smaller size, uniformly) |
| `--matryoshka-full-per-step` | off | First micro-batch of every optimizer step is the full model (the overall full share stays at `--matryoshka-full-prob`). Recommended: without it, steps with no full-size gradient made training diverge |
| `--matryoshka-all-sizes` | off | Train every size on every micro-batch (loss = sum over sizes) instead of sampling one; `--matryoshka-full-prob` is then unused, and it cannot be combined with `--matryoshka-full-per-step` |
| `--fsdp` | off | ZeRO-2 via FSDP2: shard gradients and optimizer state across GPUs; weights and gradients stay bf16 |
| `--activation-checkpointing` | off | Recompute every transformer block in backward (saves activation memory, costs extra compute) |
| `--n-res-streams` | 1 | Number of residual streams for mHC routing (1 = plain residual) |
| `--effective-tokens-target` | 2^19 | Target tokens for gradient accumulation calculation |
| `--muon-lr` | 0.02 | Muon optimizer learning rate |
| `--adamw-lr` | 1e-3 | AdamW optimizer learning rate |

---

## 🔍 Inference

```bash
python infer.py --save-dir=models/checkpoint-xxx --prompt "Hello" --use-cache \
    --max-new-tokens=1000 --temperature=0.9 --num-return-sequences=3
```

Add `--matryoshka-frac=0.25` to extract and run a smaller nested size of a Matryoshka checkpoint (works with `--speculative`). Add `--use-cache` to enable KV caching (speeds up autoregressive generation). Throughput is benchmarked automatically.

For an MTP checkpoint, compare greedy decoding with self-speculative decoding (reports tokens per forward pass, draft acceptance and speedup):

```bash
python infer.py --save-dir=models/checkpoint-xxx --prompt "Hello" --speculative --max-new-tokens=256
```

---

## 📊 Evaluation

The training loop automatically evaluates:

- **Perplexity** on the validation set at each eval interval
- **HellaSwag accuracy** via `evaluate_hellaswag()` in `LLMTrainer`

Logs are sent to WandB (online/offline) or TensorBoard, tracking loss, perplexity, tokens/sec, iters/sec, per-optimizer LR, and per-expert load/importance statistics.

---

## 🗺️ Roadmap

### Completed ✅

- [x] Character-level bigram encoder → base GPT-2 architecture (nanoGPT)
- [x] HuggingFace model classes (`PreTrainedModel` + `GenerationMixin`) with beam decoding and repetition penalty
- [x] Accelerate + HuggingFace Trainer for multi-GPU DDP training with BF16 mixed precision, gradient accumulation
- [x] WandB + TensorBoard logging
- [x] Flash attention (`F.scaled_dot_product_attention`)
- [x] Starcoderbase tokenizer with custom special tokens for sub-word language modeling
- [x] Custom dataset of 100B tokens (Fineweb-edu, Redpajama-Github, Cosmopedia, OpenWeb-math)
- [x] Sharded dataset support (`IterableDataset`)
- [x] Hybrid Muon optimizer (2D params) + AdamW (1D params)
- [x] 3-phase LR schedule (warmup → cosine decay → constant plateau)
- [x] Weight initialization, RoPE, KV Cache, RMSNorm
- [x] Gated Linear Unit (GLU) activation (SqReLU / SwiGLU)
- [x] HellaSwag evaluation
- [x] Multi-Headed Latent Attention (MLA) with DeepSeek-style RoPE
- [x] Mixture of Experts (MoE) — sparse top-k routing with load balancing auxiliary loss
- [x] MLP/MoE toggle — `--n-experts=1` for plain MLP, `n_experts > 1` for MoE
- [x] Hyper-connections (mHC) — learnable multi-stream routing with Sinkhorn-Knopp attention
- [x] Multi-token prediction (parallel heads) with self-speculative decoding
- [x] Matryoshka nested-width training (d_model, heads, MLA latent, FFN) with extraction of smaller models

### Planned 🚧

- [ ] **Sparse attention via token clustering** — Group every N tokens and attend only to representative cluster centroids, reducing attention from O(T²) to O(T·C).
- [ ] **Hybrid SSM / linear-attention layers** — Replace most attention layers with Gated DeltaNet or Mamba-2 layers, keeping about 1 MLA layer in 4 (as in Qwen3-Next, Kimi Linear, Nemotron-H). Each such layer keeps a fixed-size state instead of a per-token KV cache, so decoding costs the same per token at any context length; past ~1k tokens the state is smaller than MLA's cache. Needs long-context training (16k+) to show the benefit; speculative decoding must save and restore recurrent states instead of cropping the cache.
- [ ] **Looped (recurrent-depth) transformers** — Reuse a block several times per token, with the number of loops chosen at inference: more loops for hard problems, early exit for easy tokens, and few-loop drafts verified by full loops for self-speculative decoding.
- [ ] **Engrams** — Learnable external memory bank (key-value store) for long-range information retrieval beyond the context window.
- [ ] **Supervised fine-tuning (SFT)** — Instruction-following training pipeline with formatted prompt templates.
- [ ] **RLHF / DPO / GRPO** — Preference optimization without a separate reward model.
- [ ] **LoRA** — Low-rank adaptation for efficient fine-tuning.
- [ ] **RAG integration** — Vector database integration for external knowledge retrieval.
- [ ] **vLLM serving** — High-throughput inference with PagedAttention and continuous batching.
- [ ] **Quantization** — INT8/INT4/GGUF for deployment on consumer GPUs.

---

## 📝 License

[CC0-1.0](LICENSE) — Creative Commons Zero v1.0 Universal

---

> **Built with ❤️ for the open-source ML community.** Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT).
