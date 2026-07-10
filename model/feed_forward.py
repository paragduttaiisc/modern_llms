import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, Optional


class MLPOutput(NamedTuple):
    value: torch.Tensor
    loss: Optional[torch.Tensor] = None


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, hidden_size: int, activation: str, dropout: float) -> None:
        super().__init__()
        assert activation in ["GELU", "SwiGLU", "SqReLU"],\
            "Unsupported activation"
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

    def _forward_SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.up_proj(x))
        x = self.down_proj(x)
        return self.dropout(x)

    def __call__(self, x: torch.Tensor) -> MLPOutput:
        return MLPOutput(value=self.forward(x))


class MoE(nn.Module):
    def __init__(
        self,
        n_embed: int,
        hidden_size: int,
        n_experts: int = 8,
        top_k: int = 2,
        activation: str = "SwiGLU",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_experts = n_experts
        self.top_k = top_k

        self.router = nn.Linear(n_embed, n_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(
                n_embed=n_embed,
                hidden_size=hidden_size,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(n_experts)
        ])

        self.last_load = None
        self.last_importance = None
        self.last_aux_loss = None
        self.last_num_tokens = None

    def forward(self, x: torch.Tensor) -> MLPOutput:

        B, T, C = x.shape

        router_logits = self.router(x) # (B,T,E)
        router_probs = F.softmax(router_logits, dim=-1)
        importance = router_probs.float().mean(dim=(0, 1))

        topk_probs, indices = torch.topk(router_probs, self.top_k, dim=-1)
        weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        flat_indices = indices.reshape(-1, self.top_k)
        flat_weights = topk_probs.float().reshape(-1, self.top_k)
        load = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        load.scatter_add_(0, flat_indices.reshape(-1), flat_weights.reshape(-1))
        load /= load.sum()

        target = torch.full(
            (self.num_experts,),
            1.0 / self.num_experts,
            device=x.device,
            dtype=torch.float32,
        )
        aux_loss = F.mse_loss(load, target) + F.mse_loss(importance, target)

        self.last_load = load.detach()
        self.last_importance = importance.detach()
        self.last_aux_loss = aux_loss.detach()
        self.last_num_tokens = torch.bincount(
            flat_indices.reshape(-1), minlength=self.num_experts).detach()

        x_flat = x.reshape(-1, C)
        out = torch.zeros_like(x_flat)
        flat_weights = weights.reshape(-1, self.top_k)

        for expert_id, expert in enumerate(self.experts):
            token_idx, kth = torch.where(flat_indices == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_input = x_flat[token_idx]
            expert_output = expert.forward(expert_input).to(out.dtype)
            expert_output *= flat_weights[token_idx, kth].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_output)
        out = out.view(B, T, C)

        return MLPOutput(value=out, loss=aux_loss)