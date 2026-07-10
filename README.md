# Modern LLMs

Extending [nanoGPT](https://github.com/karpathy/nanoGPT) to train modern billion-parameter LLMs from scratch on local/multi-GPU hardware.

## Architecture at a Glance

```
train.py / infer.py          ← Entry points
model/
  config.py                  ← ModelConfig (transformers PreTrainedConfig)
  model.py                   ← Model (PreTrainedModel + GenerationMixin)
  layer.py                   ← Block (MLA + FeedForward with residual connections)
  attention.py               ← MultiHeadedLatentAttention (MLA core)
  feed_forward.py            ← FeedForward (SwiGLU / GELU / SqReLU)
utils/
  data_utils.py              ← TokenDataset (IterableDataset), HellaswagDataset
  train_utils.py             ← LLMTrainer (HuggingFace Trainer subclass)
  optimizer_utils.py         ← MultiOptimizer, MultiScheduler wrappers
  misc_utils.py              ← human_readable_numbers helper
tokenizer/                   ← Custom tokenizer (starcoderbase base + special tokens)
data -> /home/parag/Data/llm_data   ← Symlink to shard data
```

### Key architectural details

- **MLA Attention** ([attention.py](model/attention.py)): K and V projections share a latent bottleneck (`kv_latent_dim`). K is split into nope (up-projected via `k_up`) and rope components; V is up-projected via `v_up`. Q is split into nope + rope. RoPE is applied only to the rope components. This is the core memory-saving innovation.
- **Block structure**: Pre-norm RMSNorm → MLA → residual → RMSNorm → FeedForward → residual. Gated SwiGLU or standard activation in FFN.
- **Dual optimizer**: Muon optimizer for 2D+ parameters (weights), AdamW for 1D params (biases, embeddings). Separate LR schedules: Muon gets warmup → cosine decay → constant plateau; AdamW gets the same 3-phase schedule.

## Getting Started

- Clone the repo
- Create a Python virtual environment and install requirements

```bash
git clone git@github.com:paragduttaiisc/modern_llms.git
cd modern_llms
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Single GPU / debug
accelerate launch train.py --use-bf16 --batch-size=24 --n-layers=12 --n-heads=12 \
    --block-size=2048 --n-iters=25400 --save-dir=models/your_model

# Multi-GPU (SLURM via run.sh)
sbatch run.sh
```

The SLURM script (`run.sh`) is configured for 4 H200 GPUs with bf16, batch=80, and ~15K iterations (approx 10B tokens) targeting ~500K effective tokens per iteration.

### Key arguments

| Argument | Default | Purpose |
|---|---|---|
| `--block-size` | 2048 | Sequence length |
| `--n-layers` | 12 | Transformer layers |
| `--n-heads` | 12 | Attention heads |
| `--head-size` | 64 | Per-head dimension |
| `--kv-latent-size` | 128 | MLA latent bottleneck |
| `--rope-size` | 16 | RoPE dimension (subset of head_size) |
| `--embed-size` | 768 | Model hidden dimension |
| `--non-linearity` | SwiGLU | FFN activation (GELU/SwGLU/SqReLU) |
| `--effective-tokens-target` | 2^19 | Target tokens for grad accum calc |
| `--muon-lr` | 0.02 | Muon optimizer LR |
| `--adamw-lr` | 1e-3 | AdamW optimizer LR |

## Inference

```bash
python infer.py --save-dir=models/checkpoint-xxx --prompt "Hello" --use-cache \
    --max-new-tokens=1000 --temperature=0.9 --num-return-sequences=3
```

Add `--use-cache` to enable KV caching (speeds up autoregressive generation). Throughput is benchmarked automatically.

## Development Notes

- **Data format**: Sharded `.npy` token files referenced via JSON shard list (`data/web_small_10B.json`). `TokenDataset` is an `IterableDataset` that streams examples from these files.
- **Tokenizer**: Starcoderbase base with custom special tokens added (markdown, code, Jupyter, GitHub markers). Located in `tokenizer/`.
- **Evaluation**: Hellaswag accuracy is computed at every eval interval via `evaluate_hellaswag()` in `LLMTrainer`.
- **Logging**: WandB (online/offline) or tensorboard. Logs include loss, perplexity, tokens/sec, iters/sec, and per-optimizer LR tracking.
- **Weight init**: Xavier normal for Linear, normal for Embedding, ones for RMSNorm. Applied via `model.apply(model.weight_init)` before training.
- **Flash attention**: Uses `F.scaled_dot_product_attention` with `is_causal=True` when `T > 1`. Falls back to manual softmax attention with causal mask for older PyTorch.
- **Multi-GPU**: Uses HuggingFace Accelerate with DDP. Gradient accumulation is auto-computed from `effective_tokens_target`.

## TODO

### Completed

- [x] Started with a Character-level bigram encoder
- [x] Implemented a base GPT-2 architecture from nanoGPT for character-level modeling
- [x] HuggingFace model classes (`PreTrainedModel` + `GenerationMixin`) for easy inference with HuggingFace `pipeline` including beam decoding and repetition penalty
- [x] Accelerate and HuggingFace Trainer for multi-GPU distributed training (DDP) with BF16 / FP8 mixed precision training, Gradient accumulation, etc.
- [x] WandB + TensorBoard logging
- [x] Flash attention (`F.scaled_dot_product_attention`)
- [x] Used Starcoderbase tokenizer with custom special tokens for sub-word Language Modeling
- [x] Custom dataset of 100B tokens (sharded `.npy` files) for training (contains Fineweb-edu, Redpajama-Github, Cosmopedia, and OpenWeb-math with approx 40-25-20-15 mixing fractions)
- [x] Smaller 10B token dataset for debugging and testing with Fineweb-edu
- [x] Sharded dataset support (`IterableDataset`)
- [x] Hybrid Muon optimizer (for 2D params) and AdamW (for 1D params)
- [x] 3-phase LR schedule (warmup → cosine decay → constant plateau)
- [x] Weight initialization
- [x] Rotary Positional Embeddings (RoPE)
- [x] KV Cache
- [x] RMS Norm (replacing LayerNorm)
- [x] Gated Linear Unit (GLU) activation (SqReLU / SwiGLU)
- [x] HellaSwag evaluation
- [x] Grouped-Query Attention (GQA) instead of standard multi-head attention
- [x] Updated attention to Multi-headed Latent Attention (MLA) along with DeepSeek-style RoPE embeddings

### Training: Architecture Innovations

- [ ] **Mixture of Experts (MoE)** — Replace dense FFN with sparse MoE FFN (top-k routing) to increase parameter count without increasing per-token compute. Enables scaling to 10B+ with fixed FLOPs.
- [ ] **Sparse attention via token clustering** — Group every N tokens (e.g. 8) and attend only to representative cluster centroids, reducing attention complexity from O(T²) to O(T·C) where C << T.
- [ ] **Linear attention / State Space Models** — Replace softmax attention with SSM-based mixing (e.g. RWKV, Mamba-style) for O(T) sequence complexity. Consider Hyena operator or RetNet-style recurrence as alternatives.
- [ ] **Engrams** — Add a learnable external memory bank (key-value store) that the model can read from/write to during generation. Enables long-range information retrieval beyond the context window.
- [ ] **Hyper connections (residual gating)** — Add learnable residual paths (similar to mHC / hypernetwork-style gating) that let the model dynamically route information across layers, improving gradient flow and expressivity.

<!-- ### Training: Next-Gen Model Paradigms

- [ ] **Diffusion Language Models** — Shift from autoregressive (causal) LM to non-autoregressive diffusion over tokens. Sample all positions in parallel; denoising process replaces next-token prediction. Enables better mode coverage and avoids error accumulation.
- [ ] **Neural Operator token mixing** — Replace transformer attention with neural operator layers (e.g. FNO / DeepONet style) for continuous-domain token mixing. More expressive than attention for certain patterns; scales sub-quadratically with sequence length. -->

### Inference: Fine-Tuning & Alignment

- [ ] **Supervised fine-tuning (SFT / instruct)** — Training pipeline for instruction-following datasets ( Alpaca, UltraChat, etc.). Add formatted prompt templates and supervised loss on response tokens.
- [ ] **RLHF (Reinforcement Learning from Human Feedback)** — PPO-based reward modeling and policy optimization on top of the SFT model. Requires a separate reward model and reference policy.
- [ ] **RLVR (Reinforcement Learning from Verifiable Rewards)** — Use deterministic verification signals (code execution, math solutions) as rewards instead of human preferences. Cheaper and more reliable for technical domains.
- [ ] **DPO (Direct Preference Optimization)** — Pairwise preference optimization without a separate reward model. Simpler than PPO, trains directly on preference pairs (chosen vs rejected).
- [ ] **GRPO (Group Relative Policy Optimization)** — Group-based RL optimization from the R1 paper. Normalize rewards within a group of responses, eliminating the need for a critic/reward model.

### Inference: Parameter-Efficient Fine-Tuning

- [ ] **LoRA (Low-Rank Adaptation)** — Inject low-rank decomposition matrices into attention and FFN layers. Enables fine-tuning large models with minimal memory (only train ~1% of parameters). Include rank selection, alpha scaling, and dropout.
- [ ] **LoRA inference integration** — Merge trained LoRA weights back into the base model for deployment, or implement runtime LoRA switching for multi-task models.

### Deployment & Integration

- [ ] **RAG (Retrieval-Augmented Generation)** — Vector database integration (e.g., FAISS, Chroma) for external knowledge retrieval at inference time. Combine with the model's parametric knowledge to reduce hallucinations and enable up-to-date responses without retraining.
- [ ] **vLLM integration** — Optimize inference pipeline with vLLM's PagedAttention and continuous batching. Enable high-throughput serving with Tensor Parallelism.
- [ ] **OpenCode / OpenClaw agent integration** — Package the trained model for use in agent frameworks. Provide API endpoints (REST/gRPC) and structured prompt templates for tool-use and code-generation tasks.
- [ ] **Deployable model for local GPU inference** — Quantization (INT8/INT4/GGUF), ONNX export, and benchmarking for deployment on consumer GPUs.

## License

[CC0-1.0](LICENSE) — Creative Commons Zero v1.0 Universal
