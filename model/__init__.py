"""The model package: a nanoGPT-style decoder-only transformer built on
HuggingFace `PreTrainedModel`, extended with modern components.

- `config.py`: `ModelConfig`, every architecture hyperparameter.
- `model.py`: `Model`, embeddings + stack of `Block`s + tied LM head, with
  multi-token prediction heads and Matryoshka (nested-width) support.
- `layer.py`: `Block` (pre-norm RMSNorm -> attention -> RMSNorm -> FFN/MoE)
  and the residual connection wrapper.
- `attention.py`: Multi-Headed Latent Attention (MLA) with RoPE and a
  compressed (latent) KV cache.
- `feed_forward.py`: dense FFN (GELU / SqReLU / SwiGLU) and Mixture of
  Experts.
- `hyper_connections.py`: manifold-constrained hyper-connections (mHC),
  multiple residual streams mixed by a doubly stochastic matrix.
"""
from .config import ModelConfig
from .model import Model

__all__ = [
    "Model",
    "ModelConfig"
]
