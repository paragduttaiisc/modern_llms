import torch
import torch.nn as nn


class MHCRouter(nn.Module):
    def __init__(self, n_streams: int = 4, init_scale: float = 0.01):
        super().__init__()
        self.n_streams = n_streams

        self.h_pre_raw = nn.Parameter(torch.zeros(n_streams))
        self.h_post_raw = nn.Parameter(torch.zeros(n_streams))
        self.h_res_raw = nn.Parameter(torch.randn(n_streams, n_streams) * init_scale)
    
    @staticmethod
    def sinkhorn_knopp(log_alpha: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
        alpha = torch.exp(log_alpha)
        for _ in range(n_iters):
            alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-12)
            alpha = alpha / (alpha.sum(dim=-2, keepdim=True) + 1e-12)
        return alpha

    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        B, T, S, C = x.shape

        h_res = self.sinkhorn_knopp(self.h_res_raw, n_iters=5) # [S, S] # type: ignore
        h_post = torch.sigmoid(self.h_post_raw) # [S]

        x_flat = x.view(-1, S, C) # [B*T, S, C]
        x_mixed = torch.bmm(h_res.unsqueeze(0).expand(B * T, -1, -1), x_flat)
        x_mixed = x_mixed.view(B, T, S, C)

        expanded_output = layer_output.unsqueeze(2) * h_post.view(1, 1, S, 1)
        return x_mixed + expanded_output

    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        h_pre = torch.softmax(self.h_pre_raw, dim=0) # [S]
        return torch.sum(x * h_pre.view(1, 1, -1, 1), dim=2)