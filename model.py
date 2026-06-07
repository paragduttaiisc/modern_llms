import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from utils import decode_text


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, n_embed: int, block_size: int, dropout: float) -> None:
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

        if self.flash: # Optimized attention computation using PyTorch's built-in function
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else: # Fallback to manual SDPA implementation
            att = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
            att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
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
    def __init__(self, n_embed: int, num_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.sa_heads = MultiHeadAttention(num_heads, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(self, vocab_size: int, args) -> None:
        super().__init__()
        self.tok_emb_table = nn.Embedding(vocab_size, args.n_embed)
        self.pos_emb_table = nn.Embedding(args.block_size, args.n_embed)
        self.layers = nn.ModuleList(
            [Block(args.n_embed, args.n_heads, args.block_size, args.dropout) for _ in range(args.n_layers)])
        self.ln_f = nn.LayerNorm(args.n_embed)
        self.lm_head = nn.Linear(args.n_embed, vocab_size)
        self.block_size = args.block_size

    def forward(
            self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        B, T = idx.shape
        tok_embs = self.tok_emb_table(idx)
        pos_embs = self.pos_emb_table(torch.arange(T, device=idx.device))
        x = tok_embs + pos_embs
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.ln_f(x))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, prompt: torch.Tensor, idx_to_token: dict, max_new_tokens: int) -> None:
        print("Generating text...")
        generated = prompt.clone()
        print(decode_text(generated[0], idx_to_token), end='', flush=True)
        prompt = prompt[:, -self.block_size:]
        for _ in range(max_new_tokens):
            T = prompt.shape[1]
            tok_embs = self.tok_emb_table(prompt)  # B, T, C
            pos_embs = self.pos_emb_table(torch.arange(T, device=prompt.device))  # T, C
            x = tok_embs + pos_embs  # B, T, C

            for layer in self.layers:
                x = layer(x)
            logits = self.lm_head(self.ln_f(x))  # B, T, vocab_size
            logits = logits[:, -1, :]  # B, vocab_size (take the last timestep)

            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            next_idx = torch.multinomial(probs, num_samples=1)  # Sample the next token (B,1)
            generated = torch.cat([generated, next_idx], dim=1)
            prompt = torch.cat([prompt, next_idx], dim=1)[:, -self.block_size:]
            print(decode_text(next_idx[0], idx_to_token), end='', flush=True)
        print()
