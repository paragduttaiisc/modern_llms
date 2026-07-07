import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from rotary_embedding_torch import RotaryEmbedding
from typing import Optional

from .config import ModelConfig
from .layer import Block


class Model(PreTrainedModel, GenerationMixin):
    config_class = ModelConfig
    base_model_prefix = "modern_transformer"
    _tied_weights_keys = {"lm_head.weight": "tok_emb_table.weight"}

    def __init__(
            self,
            config: ModelConfig,
    ) -> None:
        super().__init__(config)
        self.tok_emb_table = nn.Embedding(
            config.vocab_size, config.embedding_size)
        self.emb_drop = nn.Dropout(config.dropout)
        self.rotary_emb = RotaryEmbedding(dim=config.rope_size)
        self.layers = nn.ModuleList([
            Block(
                embedding_size=config.embedding_size,
                head_size=config.head_size,
                n_experts=config.experts,
                n_active_experts=config.active_experts,
                rope_size=config.rope_size,
                kv_latent_size=config.kv_latent_size,
                num_attn_heads=config.num_attention_heads,
                block_size=config.block_size,
                activation=config.non_linearity,
                dropout=config.dropout,
            ) for _ in range(config.num_hidden_layers)
        ])
        self.rms_norm_f = nn.RMSNorm(config.embedding_size, eps=1e-6)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.block_size = config.block_size
        
        self.config.tie_word_embeddings = True
        self.post_init()

        self.last_lm_loss = None
        self.last_router_loss = None
    
    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_emb_table

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.tok_emb_table = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        **kwargs
    ):
        past_length = 0

        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()

        if past_length > 0:
            input_ids = input_ids[:, -1:]
        else:
            input_ids = input_ids[:, -self.block_size:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[int] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        return_per_sample_loss: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        B, T = input_ids.shape

        use_cache = use_cache if use_cache is not None\
            else getattr(self.config, "use_cache", True)
        
        x = self.emb_drop(self.tok_emb_table(input_ids))

        x = x.to(next(self.parameters()).dtype)

        past_length = past_key_values.get_seq_length()\
            if past_key_values is not None else 0

        aux_loss = 0.0
        for i, layer in enumerate(self.layers):
            x, router_loss = layer(
                x,
                self.rotary_emb,
                past_key_values=past_key_values if use_cache else None,
                past_length=past_length,
                layer_idx=i,
            )
            aux_loss += router_loss
        aux_loss /= len(self.layers)

        x = self.rms_norm_f(x)
        logits = self.lm_head(x)

        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape

            if attention_mask is not None:
                labels[attention_mask == 0] = -100
            
            lm_loss = F.cross_entropy(
                logits.view(-1, C),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )

            if return_per_sample_loss:
                lm_loss = lm_loss.view(B, T)
                loss = lm_loss.sum(dim=1) / (labels != -100).sum(dim=1).clamp_min(1)
            elif num_items_in_batch is not None:
                lm_loss = lm_loss.sum() / num_items_in_batch
                loss = lm_loss + self.config.router_loss_coef * aux_loss
            else:
                lm_loss = lm_loss.mean()
                loss = lm_loss + self.config.router_loss_coef * aux_loss

            self.last_lm_loss = lm_loss.detach()
            self.last_router_loss = router_loss.detach()

        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss, # type: ignore
            past_key_values=past_key_values if use_cache else None, # type: ignore
        )