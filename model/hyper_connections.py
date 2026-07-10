import torch
import torch.nn as nn
import torch.nn.functional as F

class MHCRouter(nn.Module):
    def __init__(self, embedding_size: int, n_streams: int = 4):
        super().__init__()
        self.n_streams = n_streams
        self.embedding_size = embedding_size

        self.phi_pre = nn.Linear(embedding_size, n_streams, bias=True)
        self.phi_post = nn.Linear(embedding_size, n_streams, bias=True)
        self.phi_res = nn.Linear(embedding_size, n_streams * n_streams, bias=True)

        self.alpha_pre = nn.Parameter(torch.ones(1))
        self.alpha_post = nn.Parameter(torch.ones(1))
        self.alpha_res = nn.Parameter(torch.ones(1))
        
        self.rms_norm = nn.RMSNorm(embedding_size, eps=1e-6)

    @staticmethod
    def sinkhorn_knopp(log_alpha: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
        log_alpha = log_alpha - torch.max(log_alpha, dim=-1, keepdim=True)[0]
        alpha = torch.exp(log_alpha)
        for _ in range(n_iters):
            alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-6)
            alpha = alpha / (alpha.sum(dim=-2, keepdim=True) + 1e-6)
        return alpha

    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        B, T, S, _ = x.shape

        x_mean = self.rms_norm(x.mean(dim=2)) # [B, T, C]

        h_post_raw = self.alpha_post * self.phi_post(x_mean)
        h_post = 2.0 * torch.sigmoid(h_post_raw) # [B, T, S]

        h_res_raw = self.alpha_res * self.phi_res(x_mean)
        h_res_raw = h_res_raw.view(B, T, S, S)
        h_res = self.sinkhorn_knopp(h_res_raw, n_iters=5) # [B, T, S, S]

        x_mixed = torch.einsum("btij,btjc->btic", h_res, x)
        expanded_output = layer_output.unsqueeze(2) * h_post.unsqueeze(-1) # [B, T, S, C]

        return x_mixed + expanded_output

    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = self.rms_norm(x.mean(dim=2))
        
        h_pre_raw = self.alpha_pre * self.phi_pre(x_mean)
        h_pre = torch.sigmoid(h_pre_raw) # [B, T, S]
        
        h_pre = h_pre / (h_pre.sum(dim=-1, keepdim=True) + 1e-6)
        
        return torch.sum(x * h_pre.unsqueeze(-1), dim=2)