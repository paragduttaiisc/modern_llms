from transformers import PreTrainedConfig


class ModelConfig(PreTrainedConfig):
    model_type = "modern_transformer"

    def __init__(
            self,
            vocab_size: int = 65,
            block_size: int = 256,
            embedding_size: int = 384,
            head_size: int = 64,
            experts: int = 8,
            active_experts: int = 2,
            router_loss_coef: float = 0.01,
            rope_size: int = 16,
            kv_latent_size: int = 96,
            num_hidden_layers: int = 6,
            num_attention_heads: int = 4,
            non_linearity: str = "GELU",
            dropout: float = 0.2,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout = dropout
        self.non_linearity = non_linearity
        self.num_hidden_layers = num_hidden_layers
        self.embedding_size = embedding_size
        self.head_size = head_size
        self.experts = experts
        self.active_experts = active_experts
        self.router_loss_coef = router_loss_coef
        self.rope_size = rope_size
        self.kv_latent_size = kv_latent_size
        self.num_attention_heads = num_attention_heads
