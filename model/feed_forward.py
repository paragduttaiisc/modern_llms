import torch.nn as nn
import torch.nn.functional as F


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
