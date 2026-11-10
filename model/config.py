from transformers import PreTrainedConfig


class ModelConfig(PreTrainedConfig):
    model_type = "modern_transformer"

    def __init__(
            self,
            vocab_size: int = 65,
            block_size: int = 256,
            embedding_size: int = 384,
            sa_head_size: int = 64,
            ff_hidden_size: int = 1536,
            experts: int = 8,
            active_experts: int = 2,
            router_loss_coef: float = 0.01,
            rope_size: int = 16,
            kv_latent_size: int = 96,
            num_hidden_layers: int = 6,
            num_attention_heads: int = 4,
            num_residual_streams: int = 4,
            non_linearity: str = "GELU",
            dropout: float = 0.2,
            n_future_tokens: int = 1,
            mtp_loss_weight: float = 1.0,
            mtp_checkpoint: bool = True,
            block_checkpoint: bool = False,
            draft_vocab_size: int = 0,
            mtp_position_frac: float = 1.0,
            matryoshka_fracs: list[float] | None = None,
            ff_hidden_exact: bool = False,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Matryoshka (nested-width) sizes trained jointly; 1.0 = full model.
        # A size f uses the first f of d_model, attention heads, KV latent
        # and FFN width of every block (head / RoPE dims are not nested).
        self.matryoshka_fracs = sorted(
            {1.0, *(matryoshka_fracs or [])}, reverse=True)
        # Use ff_hidden_size as the exact FFN width (no SwiGLU 2/3 + round);
        # set by Matryoshka extraction.
        self.ff_hidden_exact = ff_hidden_exact
        # Cheaper extra MTP heads (heads 1..n-1; head 0 is never changed):
        # predict over only the `draft_vocab_size` most frequent tokens
        # (0 = full vocab), and compute their training loss on a random
        # `mtp_position_frac` of the positions.
        self.draft_vocab_size = draft_vocab_size
        self.mtp_position_frac = mtp_position_frac
        # Multi-token prediction: number of parallel heads (1 = standard LM),
        # weight of the mean extra-head loss, and whether each head runs
        # under activation checkpointing (keeps one head's logits in memory).
        self.n_future_tokens = n_future_tokens
        self.mtp_loss_weight = mtp_loss_weight
        self.mtp_checkpoint = mtp_checkpoint
        # Recompute each trunk block in backward (activation checkpointing)
        self.block_checkpoint = block_checkpoint
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout = dropout
        self.non_linearity = non_linearity
        self.num_hidden_layers = num_hidden_layers
        self.embedding_size = embedding_size
        self.sa_head_size = sa_head_size
        self.ff_hidden_size = ff_hidden_size
        self.experts = experts
        self.active_experts = active_experts
        self.router_loss_coef = router_loss_coef
        self.rope_size = rope_size
        self.kv_latent_size = kv_latent_size
        self.num_attention_heads = num_attention_heads
        self.num_residual_streams = num_residual_streams

    def matryoshka_dims(self, frac: float) -> dict[str, int]:
        """Nested sizes at width `frac`; raises if any is not a whole number."""
        from .feed_forward import ffn_hidden_size
        full = {
            "embedding_size": self.embedding_size,
            "num_attention_heads": self.num_attention_heads,
            "kv_latent_size": self.kv_latent_size,
            "ffn_hidden": ffn_hidden_size(
                self.ff_hidden_size, self.non_linearity, self.ff_hidden_exact),
        }
        dims = {}
        for name, size in full.items():
            nested = size * frac
            if not 0 < frac <= 1 or abs(nested - round(nested)) > 1e-6\
                    or round(nested) < 1:
                raise ValueError(
                    f"Matryoshka size {frac} gives {name}={nested:g} "
                    f"(full {size}); it must be a positive whole number")
            dims[name] = round(nested)
        return dims
