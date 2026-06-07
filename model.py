import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from utils import decode_text


class Head(nn.Module):
    def __init__(self, n_embed: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        k = self.key(x)   # B, T, H
        q = self.query(x) # B, T, H
        v = self.value(x) # B, T, H

        # Scaled Dot-product Attention (SDPA)
        head_size = k.shape[-1]
        att = q @ k.transpose(-2, -1) * head_size ** -0.5
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf')) # type: ignore
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        return att @ v  # B, T, H


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, n_embed: int, block_size: int, dropout: float) -> None:
        super().__init__()
        head_size = n_embed // num_heads
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1)  # B, T, num_heads * head_size
        return self.proj(x)  # B, T, n_embed


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
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
