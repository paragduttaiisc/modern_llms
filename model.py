import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
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
            dropout: float = 0.2,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads


class MultiHeadAttention(nn.Module):
    def __init__(
            self, num_heads: int, n_embed: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_embed // num_heads

        self.attn_mat = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.attn_mat(x).chunk(3, dim=-1)  # Each: B, T, n_embed
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        if self.flash: # Flash attention using PyTorch 2.0's built-in function
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0)
        else: # Fallback to manual SDPA implementation
            att = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
            att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf')) # type: ignore
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))  # B, T, n_embed


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(
            self, n_embed: int, num_heads: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.sa_heads = MultiHeadAttention(
            num_heads, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Model(PreTrainedModel, GenerationMixin):
    config_class = ModelConfig
    base_model_prefix = "modern_transformer"
    _tied_weights_keys = {"lm_head.weight": "tok_emb_table.weight"}

    def __init__(self, config) -> None:
        super().__init__(config)
        self.tok_emb_table = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb_table = nn.Embedding(config.block_size, config.hidden_size)
        self.emb_drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            Block(
                config.hidden_size, config.num_attention_heads,
                config.block_size, config.dropout
            ) for _ in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.block_size = config.block_size
        
        self.config.tie_word_embeddings = True
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_emb_table

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.tok_emb_table = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids[:, -self.block_size:]}

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            num_items_in_batch: Optional[int] = None,
            **kwargs
    ) -> CausalLMOutput:
        _, T = input_ids.shape
        tok_embs = self.tok_emb_table(input_ids)
        pos_embs = self.pos_emb_table(torch.arange(T, device=input_ids.device))
        x = self.emb_drop(tok_embs + pos_embs)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.ln_f(x))

        if labels is None:
            loss = None
        else:
            _, _, C = logits.shape
            if num_items_in_batch is not None:
                loss = F.cross_entropy(
                    logits.view(-1, C), labels.view(-1), reduction="sum")
                loss = loss / num_items_in_batch
            else:
                loss = F.cross_entropy(
                    logits.view(-1, C), labels.view(-1), reduction="mean")
        return CausalLMOutput(logits=logits, loss=loss)  # type: ignore