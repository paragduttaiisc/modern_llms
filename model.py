import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class Model(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(
            self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        logits = self.token_embedding_table(idx) # B, T, C
        if targets is None: return logits
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self.token_embedding_table(idx) # B, T, C
            logits = logits[:, -1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1) # Convert logits to probabilities
            next_idx = torch.multinomial(probs, num_samples=1) # Sample the next token
            idx = torch.cat([idx, next_idx], dim=1) # B, T+1
        return idx
