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
  optimizer_utils.py         ← MultiOptimizer, MultiScheduler wrappers
  misc_utils.py              ← human_readable_numbers helper
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
- **MoE** — Each FFN layer is replaced by a sparse Mixture-of-Experts with top-k gating. Load balancing auxiliary loss (MSE between expert load/importance and uniform) is tracked and added to the total loss via `router_loss_coef`.
- **Hyper-connections (mHC)** — Each layer has a learnable multi-stream routing mechanism (`MHCRouter`) that expands the input into `num_residual_streams` parallel streams. Each stream goes through its own MLA + FFN path, and streams are mixed via Sinkhorn-Knopp soft attention before being collapsed back. Uses learnable pre-mixing (`h_pre`), post-mixing (`h_post`), and residual mixing (`h_res`) parameters.
- **Dual optimizer** — Muon optimizer for 2D+ parameters (weights), AdamW for 1D params (biases, embeddings). Separate LR schedules: Muon gets warmup → cosine decay → constant plateau; AdamW gets the same 3-phase schedule.

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

### Plain MLP

Set `--n-experts=1` to disable MoE and train with a standard feed-forward network (no router loss):

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 train.py --use-bf16 --n-experts=1 \
    --router-loss-weight=0.0 --batch-size=16 --n-iters=38140 --warmup-iters=500 \
    --last-decay-iter=37000 --eval-interval=1000 --save-interval=5000 \
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
| `--non-linearity` | SqReLU | FFN activation (GELU / SwiGLU / SqReLU) |
| `--n-experts` | 8 | Number of MoE experts per layer (**1 = plain MLP**) |
| `--n-active-experts` | 2 | Top-k experts active per token |
| `--router-loss-weight` | 0.01 | Load balancing auxiliary loss coefficient (ignored when `n_experts=1`) |
| `--n-res-streams` | 4 | Number of residual streams for mHC routing |
| `--effective-tokens-target` | 2^19 | Target tokens for gradient accumulation calculation |
| `--muon-lr` | 0.02 | Muon optimizer learning rate |
| `--adamw-lr` | 1e-3 | AdamW optimizer learning rate |

---

## 🔍 Inference

```bash
python infer.py --save-dir=models/checkpoint-xxx --prompt "Hello" --use-cache \
    --max-new-tokens=1000 --temperature=0.9 --num-return-sequences=3
```

Add `--use-cache` to enable KV caching (speeds up autoregressive generation). Throughput is benchmarked automatically.

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

### Planned 🚧

- [ ] **Sparse attention via token clustering** — Group every N tokens and attend only to representative cluster centroids, reducing attention from O(T²) to O(T·C).
- [ ] **Linear attention / State Space Models** — Replace softmax attention with SSM-based mixing (RWKV, Mamba-style) for O(T) sequence complexity.
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
