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
            draft_vocab_size: int = 0,
            mtp_position_frac: float = 1.0,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
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
