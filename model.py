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
            embedding_size: int = 384,
            head_size: int = 64,
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
        self.kv_latent_size = kv_latent_size
        self.num_attention_heads = num_attention_heads


class MultiHeadedLatentAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            head_dim: int,
            kv_latent_dim: int,
            block_size: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim

        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.kv_down = nn.Linear(embed_dim, kv_latent_dim, bias=False)
        self.k_up = nn.Linear(kv_latent_dim, num_heads * head_dim, bias=False)
        self.v_up = nn.Linear(kv_latent_dim, num_heads * head_dim, bias=False)

        self.proj = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(
            self,
            x: torch.Tensor,
            # rotary_emb: RotaryEmbedding,
            past_key_values: Optional[Cache] = None,
            past_length: Optional[int] = 0,
            layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        c = self.kv_down(x)

        if past_key_values is not None:
            c, _ = past_key_values.update(c, torch.empty_like(c), layer_idx) # type: ignore

        Wk = self.k_up.weight.view(
            self.num_heads, self.head_dim, self.kv_latent_dim)
        Wv = self.v_up.weight.view(
            self.num_heads, self.head_dim, self.kv_latent_dim)

        q_latent = torch.einsum("bhtd,hdr->bhtr", q, Wk)

        att = torch.einsum("bhtr,bsr->bhts", q_latent, c)
        # att = att * (self.kv_latent_dim ** 0.5)
        att = att * (self.head_dim ** -0.5)

        if T > 1:
            S = c.shape[1]
            causal_mask = self.mask[S - T:S, :S] # type: ignore
            att = att.masked_fill(causal_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        latent_out = torch.einsum("bhts,bsr->bhtr", att, c)

        out = torch.einsum("bhtr,hdr->bhtd", latent_out, Wv)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, activation: str, dropout: float):
        super().__init__()
        assert activation in ["GELU", "SwiGLU", "SqReLU"],\
            "Unsupported activation"
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
            embedding_size: int,
            head_size: int,
            kv_latent_size: int,
            num_attn_heads: int,
            block_size: int,
            activation: str,
            dropout: float,
    ) -> None:
        super().__init__()
        self.sa_heads = MultiHeadedLatentAttention(
            num_heads=num_attn_heads,
            embed_dim=embedding_size,
            head_dim=head_size,
            kv_latent_dim=kv_latent_size,
            block_size=block_size,
            dropout=dropout
        )
        self.ffwd = FeedForward(embedding_size, activation, dropout)
        self.rms_norm1 = nn.RMSNorm(embedding_size, eps=1e-6)
        self.rms_norm2 = nn.RMSNorm(embedding_size, eps=1e-6)
    
    def forward(
            self,
            x: torch.Tensor,
            # rotary_emb: RotaryEmbedding,
            past_key_values: Optional[Cache] = None,
            past_length: Optional[int] = 0,
            layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        x = x + self.sa_heads(
            self.rms_norm1(x),
            # rotary_emb,
            past_key_values,
            past_length,
            layer_idx
        )
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
        self.tok_emb_table = nn.Embedding(
            config.vocab_size, config.embedding_size)
        self.pos_emb_table = nn.Embedding(
            config.block_size, config.embedding_size)
        self.emb_drop = nn.Dropout(config.dropout)
        # self.rotary_emb = RotaryEmbedding(dim=config.head_size)
        self.layers = nn.ModuleList([
            Block(
                embedding_size=config.embedding_size,
                head_size=config.head_size,
                kv_latent_size=config.kv_latent_size,
                num_attn_heads=config.num_attention_heads,
                block_size=config.block_size,
                activation=config.non_linearity,
                dropout=config.dropout
            ) for _ in range(config.num_hidden_layers)
        ])
        self.rms_norm_f = nn.RMSNorm(config.embedding_size, eps=1e-6)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
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
        return_per_sample_loss: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        B, T = input_ids.shape

        use_cache = use_cache if use_cache is not None\
            else getattr(self.config, "use_cache", True)
        
        tok_emb = self.tok_emb_table(input_ids)
        pos_emb = self.pos_emb_table(torch.arange(T, device=input_ids.device))
        x = self.emb_drop(tok_emb + pos_emb)

        past_length = past_key_values.get_seq_length()\
            if past_key_values is not None else 0

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                # self.rotary_emb,
                past_key_values=past_key_values if use_cache else None,
                past_length=past_length,
                layer_idx=i,
            )

        x = self.rms_norm_f(x)
        logits = self.lm_head(x)

        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape

            if attention_mask is not None:
                labels[attention_mask == 0] = -100
            
            loss = F.cross_entropy(
                logits.view(-1, C),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )

            if return_per_sample_loss:
                loss = loss.view(B, T)
                loss = loss.sum(dim=1) / (labels != -100).sum(dim=1).clamp_min(1)
            elif num_items_in_batch is not None:
                loss = loss.sum() / num_items_in_batch
            else:
                loss = loss.mean()

        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss, # type: ignore
            past_key_values=past_key_values if use_cache else None, # type: ignore
        )