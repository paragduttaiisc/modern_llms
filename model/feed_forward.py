import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class MLPOutput(NamedTuple):
    """Output of an FFN / MoE layer: `value` [B, T, C] and an optional
    auxiliary `loss` (the MoE load-balancing loss; None for a dense FFN)."""
    value: torch.Tensor
    loss: torch.Tensor | None = None


def ffn_hidden_size(hidden_size: int, activation: str, exact: bool) -> int:
    """Actual FFN width. SwiGLU uses 2/3 of the nominal width (same params
    as a 2-matrix FFN), rounded up to a multiple of 256, unless `exact`
    (used by Matryoshka-extracted models, whose width is already sliced)."""
    if activation != "SwiGLU" or exact:
        return hidden_size
    hidden_size = 2 * hidden_size // 3
    return 256 * ((hidden_size + 255) // 256)  # for efficiency


class FeedForward(nn.Module):
    """Dense FFN. `frac` < 1 runs the Matryoshka sub-FFN: input is the first
    d' = x.shape[-1] dims and only the first `frac` of the hidden units are
    used (prefix slices of the full weights).

    Two variants, chosen once in `__init__` by binding `self.forward`:
    - GELU / SqReLU: down(act(up(x))), the classic 2-matrix FFN. SqReLU is
      relu(x)^2 (Primer).
    - SwiGLU: down(silu(gate(x)) * up(x)), a gated FFN: one projection,
      passed through SiLU, multiplies the other elementwise. It has a third
      matrix, hence the narrower hidden width from `ffn_hidden_size`.
    Called as `ffn(x, frac)` with x [B, T, d], returns `MLPOutput` with no
    loss.
    """
    def __init__(self, n_embed: int, hidden_size: int, activation: str, dropout: float,
                 exact_hidden: bool = False) -> None:
        super().__init__()
        assert activation in ["GELU", "SwiGLU", "SqReLU"],\
            "Unsupported activation"
        # Unused by SwiGLU (it applies SiLU itself)
        self.act = nn.GELU(approximate="tanh") if activation == "GELU"\
                    else lambda x: F.relu(x).square()
        self.forward = self._forward_standard
        hidden_size = ffn_hidden_size(hidden_size, activation, exact_hidden)
        self.hidden_size = hidden_size
        if activation == "SwiGLU":
            self.gate_proj = nn.Linear(n_embed, hidden_size, bias=False)
            self.forward = self._forward_SwiGLU
        self.up_proj = nn.Linear(n_embed, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _forward_SwiGLU(self, x: torch.Tensor, frac: float) -> torch.Tensor:
        d, h = x.shape[-1], round(self.hidden_size * frac)
        x = F.silu(F.linear(x, self.gate_proj.weight[:h, :d]))\
            * F.linear(x, self.up_proj.weight[:h, :d])
        x = F.linear(x, self.down_proj.weight[:d, :h])
        return self.dropout(x)

    def _forward_standard(self, x: torch.Tensor, frac: float) -> torch.Tensor:
        d, h = x.shape[-1], round(self.hidden_size * frac)
        x = self.act(F.linear(x, self.up_proj.weight[:h, :d]))
        x = F.linear(x, self.down_proj.weight[:d, :h])
        return self.dropout(x)

    # Overrides nn.Module.__call__ (so no forward hooks run) to wrap the
    # bound forward variant in MLPOutput.
    def __call__(self, x: torch.Tensor, frac: float = 1.0) -> MLPOutput:
        return MLPOutput(value=self.forward(x, frac))


class MoE(nn.Module):
    """Mixture of Experts FFN with top-k token-choice routing.

    Each block holds `n_experts` independent FFNs (experts) and a linear
    router. For every token the router gives a softmax over experts; the
    token is sent to its `top_k` most probable experts and the outputs are
    summed, weighted by those probabilities renormalized to sum to 1. So
    parameters grow with `n_experts` while compute per token grows only
    with `top_k`.

    Expert weights are stacked into 3-D parameters so that all experts run
    in one `grouped_mm` call instead of a Python loop:
        SwiGLU: gate_up_weight [E, 2H, C] (rows [gate (H) | up (H)]),
                down_weight [E, C, H]
        GELU / SqReLU: up_weight [E, H, C], down_weight [E, C, H]
    where H is the per-expert hidden width after `ffn_hidden_size`.
    `grouped_mm` needs bf16, so the expert matmuls run in bf16.

    Side effects of `forward` (for logging per-expert stats in the
    trainer), all detached: `last_load` [E] fraction of routed token slots
    each expert received, `last_importance` [E] mean router probability
    per expert, `last_num_tokens` [E] token-slot counts, and
    `last_aux_loss` the scalar balance loss.
    """
    def __init__(
        self,
        n_embed: int,
        hidden_size: int,
        n_experts: int = 8,
        top_k: int = 2,
        activation: str = "SwiGLU",
        dropout: float = 0.0,
        exact_hidden: bool = False,
    ) -> None:
        super().__init__()

        if activation not in {"GELU", "SwiGLU", "SqReLU"}:
            raise ValueError(f"Unsupported activation: {activation}")

        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {n_experts}")

        if not 1 <= top_k <= n_experts:
            raise ValueError(
                f"top_k must satisfy 1 <= top_k <= n_experts, "
                f"got top_k={top_k}, n_experts={n_experts}"
            )

        self.n_embed = n_embed
        self.num_experts = n_experts
        self.top_k = top_k
        self.activation = activation
        self.dropout_p = dropout

        self.router = nn.Linear(n_embed, n_experts, bias=False)

        expert_hidden_size = ffn_hidden_size(
            hidden_size, activation, exact_hidden)
        self.hidden_size = expert_hidden_size

        if activation == "SwiGLU":
            self.gate_up_weight = nn.Parameter(
                torch.empty(n_experts, 2 * expert_hidden_size, n_embed))
            self.down_weight = nn.Parameter(
                torch.empty(n_experts, n_embed, expert_hidden_size))
            self.up_weight = None
        else:  # activation GELU and SqReLU
            self.up_weight = nn.Parameter(
                torch.empty(n_experts, expert_hidden_size, n_embed))
            self.down_weight = nn.Parameter(
                torch.empty(n_experts, n_embed, expert_hidden_size))
            self.gate_up_weight = None
        self._init_expert_params()

        self.last_load: torch.Tensor | None = None
        self.last_importance: torch.Tensor | None = None
        self.last_aux_loss: torch.Tensor | None = None
        self.last_num_tokens: torch.Tensor | None = None

    @torch.no_grad()
    def _init_expert_params(self) -> None:
        """Xavier-normal init of each expert matrix (std sqrt(2 / (in +
        out))). The 3-D expert weights are plain Parameters, not Linear
        layers, so `Model.weight_init` does not reach them; the gate and up
        halves of `gate_up_weight` are initialized as separate matrices."""
        n_embed = self.n_embed
        hidden = self.hidden_size
        up_std = math.sqrt(2.0 / float(n_embed + hidden))
        down_std = math.sqrt(2.0 / float(hidden + n_embed))

        if self.activation == "SwiGLU":
            assert self.gate_up_weight is not None
            self.gate_up_weight[:, :hidden, :].normal_(mean=0.0, std=up_std)
            self.gate_up_weight[:, hidden:, :].normal_(mean=0.0, std=up_std)
        else:
            assert self.up_weight is not None
            self.up_weight.normal_(mean=0.0, std=up_std)
        self.down_weight.normal_(mean=0.0, std=down_std)

    def _count_per_expert(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """Number of routed slots per expert [E] (int64).

        Same result as `torch.bincount(..., minlength=E)`, but bincount on
        CUDA reads the maximum index back to the CPU to size its output,
        which breaks CUDA-graph capture (utils/fast_decode.py) and forces a
        sync every forward pass. A fixed-size scatter-add needs neither.
        """
        counts = torch.zeros(
            self.num_experts, dtype=torch.long, device=expert_indices.device)
        return counts.scatter_add_(
            0, expert_indices, torch.ones_like(expert_indices))

    @staticmethod
    def _grouped_mm(
        x: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert matmul over expert-sorted rows.

        x: [N_routed, in], rows grouped by expert (expert 0's rows first).
        weight: [E, in, out] (callers pass the stored [E, out, in] expert
        weights transposed). offsets: [E] int32, the cumulative row counts,
        so expert e owns rows offsets[e-1]:offsets[e]. Returns
        [N_routed, out] in bf16, row r multiplied by its expert's matrix.
        """
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        if hasattr(torch.nn.functional, "grouped_mm"):
            return torch.nn.functional.grouped_mm(x, weight, offs=offsets)

        if hasattr(torch, "_grouped_mm"):
            return torch._grouped_mm(  # pyright: ignore[reportPrivateImportUsage]
                x, weight, offsets)

        # Fallback for non-grouped_mm implementations
        output_dim = weight.shape[-1]
        out = torch.empty(
            x.shape[0], output_dim, device=x.device, dtype=x.dtype)
        start = 0
        for expert_id in range(weight.shape[0]):
            end = int(offsets[expert_id].item())
            if end <= start:
                continue
            out[start:end] = (x[start:end] @ weight[expert_id])
            start = end
        return out

    def forward(self, x: torch.Tensor, frac: float = 1.0) -> MLPOutput:
        """Route each token of x [B, T, C] to its top-k experts; returns
        `MLPOutput(value [B, T, C], loss=aux_loss)`. The aux loss is
        unscaled; `Model.forward` multiplies it by `router_loss_coef`.
        Also sets the `last_*` stats (see the class docstring).

        `frac` < 1 runs the Matryoshka sub-MoE: input is the first
        C = x.shape[-1] dims, every expert uses only its first `frac` of
        hidden units, and the router sees only the first C dims. All experts
        are kept (expert count is not nested)."""
        B, T, C = x.shape
        if C > self.n_embed:
            raise ValueError(f"Expected input dim <= {self.n_embed}, got {C}")
        N = B * T
        H = self.hidden_size
        h = round(H * frac)

        x_flat = x.reshape(N, C)
        router_logits = F.linear(x_flat, self.router.weight[:, :C])
        router_probs = F.softmax(router_logits.float(), dim=-1)  # [N, E]

        topk_probs, topk_indices = torch.topk(
            router_probs, k=self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # importance: mean router probability per expert (differentiable);
        # load: fraction of the N * top_k routed slots each expert got
        # (a count, so no gradient flows through it).
        importance = router_probs.mean(dim=0)
        flat_expert_indices = topk_indices.reshape(-1)

        num_tokens_per_expert = self._count_per_expert(flat_expert_indices)
        load = num_tokens_per_expert.float() / float(N * self.top_k)

        # Unscaled Switch-style balance loss (== 1.0 when perfectly balanced);
        # Model.forward applies config.router_loss_coef.
        aux_loss = self.num_experts * torch.sum(load * importance)

        self.last_load = load.detach()
        self.last_importance = importance.detach()
        self.last_aux_loss = aux_loss.detach()
        self.last_num_tokens = num_tokens_per_expert.detach()

        # Flatten the N x top_k (token, expert) assignments, then sort them
        # by expert so each expert's tokens form one contiguous run of rows:
        # that is the layout grouped_mm needs. A token routed to k experts
        # appears k times.
        token_indices = torch.arange(N, device=x.device)\
            .unsqueeze(1).expand(N, self.top_k).reshape(-1)
        routing_weights = topk_weights.reshape(-1)

        _, sort_order = torch.sort(flat_expert_indices)
        sorted_token_indices = token_indices[sort_order]
        sorted_routing_weights = routing_weights[sort_order]
        x_sorted = x_flat[sorted_token_indices]

        tokens_per_expert = num_tokens_per_expert
        # End row of each expert's run in the sorted order
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

        if self.activation == "SwiGLU":
            assert self.gate_up_weight is not None
            if h == H and C == self.n_embed:
                gate_up = self._grouped_mm(
                    x_sorted, self.gate_up_weight.transpose(-1, -2), offsets)
                gate, up = gate_up[:, :H], gate_up[:, H:]
            else:
                # gate_up rows are [gate (H) | up (H)]: nest each half
                gate = self._grouped_mm(
                    x_sorted,
                    self.gate_up_weight[:, :h, :C].transpose(-1, -2), offsets)
                up = self._grouped_mm(
                    x_sorted,
                    self.gate_up_weight[:, H:H + h, :C].transpose(-1, -2),
                    offsets)
            hidden = F.silu(gate) * up
        else:  # GELU / SqReLU
            assert self.up_weight is not None
            hidden = self._grouped_mm(
                x_sorted, self.up_weight[:, :h, :C].transpose(-1, -2), offsets)
            hidden = F.gelu(hidden, approximate="tanh")\
                if self.activation == "GELU" else F.relu(hidden).square()
        expert_output = self._grouped_mm(
            hidden, self.down_weight[:, :C, :h].transpose(-1, -2), offsets)

        # Weight each (token, expert) output by its renormalized router prob;
        # the LM loss reaches the router only through these weights.
        expert_output *= \
            sorted_routing_weights.to(expert_output.dtype).unsqueeze(-1)

        # Undo the sort: back to (token, slot) order, so the top_k outputs of
        # each token are adjacent and can be summed.
        inverse_order = torch.empty_like(sort_order)
        inverse_order[sort_order] = torch.arange(
            sort_order.numel(), device=x.device)
        expert_output = expert_output[inverse_order]

        out = expert_output.view(N, self.top_k, C)\
            .float().sum(dim=1).to(x.dtype)

        if self.dropout_p > 0.0:
            out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = out.view(B, T, C)

        return MLPOutput(value=out, loss=aux_loss)
