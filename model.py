import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from rotary_embedding_torch import RotaryEmbedding
from typing import Optional


class ModelConfig(PreTrainedConfig):
    model_type = "modern_transformer"

    def __init__(
            self,
            vocab_size: int = 65,
            block_size: int = 256,
            hidden_size: int = 384,
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
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            n_embed: int,
            block_size: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_embed // num_heads

        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=False)

        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                'mask', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(
            self,
            x: torch.Tensor,
            rotary_emb: RotaryEmbedding,
            past_key_values: Optional[Cache] = None,
            layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(
            B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(
            B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(
            B, T, self.num_heads, self.head_size).transpose(1, 2)

        past_length = past_key_values.get_seq_length()\
                        if past_key_values is not None else 0

        q = rotary_emb.rotate_queries_or_keys(q, offset=past_length)
        k = rotary_emb.rotate_queries_or_keys(k, offset=past_length)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx) # type: ignore

        if self.flash: 
            is_causal = (T > 1) 
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=is_causal,
                dropout_p=self.dropout.p if self.training else 0.0)
        else: 
            att = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
            if T > 1:
                T_k = k.shape[-2]
                causal_mask = self.mask[T_k - T:T_k, :T_k] # type: ignore
                att = att.masked_fill(causal_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            out = att @ v
            
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, activation: str, dropout: float):
        super().__init__()
        assert activation in ["GELU", "SwiGLU", "SqReLU"], "Unsupported activation"
        hidden_size = 4 * n_embed
        self.act = nn.GELU(approximate="tanh") if activation == "GELU"\
                    else lambda x: F.relu(x).square()
        self.forward = self._forward_standard
        if activation == "SwiGLU":
            hidden_size = 8 * n_embed // 3
            hidden_size = 256 * ((hidden_size + 255) // 256) # for efficiency
            self.gate_proj = nn.Linear(n_embed, hidden_size, bias=False)
            self.forward = self._forward_SwiGLU
        self.up_proj = nn.Linear(n_embed, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _forward_SwiGLU(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)

    def _forward_standard(self, x):
        x = self.act(self.up_proj(x))
        x = self.down_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(
            self,
            n_embed: int,
            num_heads: int,
            block_size: int,
            activation: str,
            dropout: float,
    ) -> None:
        super().__init__()
        self.sa_heads = MultiHeadAttention(
            num_heads, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, activation, dropout)
        self.rms_norm1 = nn.RMSNorm(n_embed, eps=1e-6)
        self.rms_norm2 = nn.RMSNorm(n_embed, eps=1e-6)
    
    def forward(
            self,
            x: torch.Tensor,
            rotary_emb: RotaryEmbedding,
            past_key_values: Optional[Cache] = None,
            layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        x = x + self.sa_heads(
            self.rms_norm1(x), rotary_emb, past_key_values, layer_idx)
        x = x + self.ffwd(self.rms_norm2(x))
        return x 


class Model(PreTrainedModel, GenerationMixin):
    config_class = ModelConfig
    base_model_prefix = "modern_transformer"
    _tied_weights_keys = {"lm_head.weight": "tok_emb_table.weight"}

    def __init__(
            self,
            config: ModelConfig,
    ) -> None:
        super().__init__(config)
        self.tok_emb_table = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_drop = nn.Dropout(config.dropout)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        self.layers = nn.ModuleList([
            Block(
                config.hidden_size, config.num_attention_heads,
                config.block_size, config.non_linearity, config.dropout
            ) for _ in range(config.num_hidden_layers)
        ])
        self.rms_norm_f = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.block_size = config.block_size
        
        self.config.tie_word_embeddings = True
        self.post_init()
    
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
        **kwargs,
    ) -> CausalLMOutputWithPast:

        use_cache = (
            use_cache
            if use_cache is not None
            else getattr(self.config, "use_cache", True)
        )

        x = self.emb_drop(self.tok_emb_table(input_ids))

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                self.rotary_emb,
                past_key_values=past_key_values if use_cache else None,
                layer_idx=i,
            )

        x = self.rms_norm_f(x)
        logits = self.lm_head(x)

        if labels is None:
            loss = None
        else:
            _, _, C = logits.shape

            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = F.cross_entropy(
                logits.view(-1, C), labels.view(-1), reduction=reduction)
            if num_items_in_batch is not None:
                loss = loss / num_items_in_batch

        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss, # type: ignore
            past_key_values=past_key_values if use_cache else None,
        )