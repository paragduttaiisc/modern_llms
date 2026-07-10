import torch
import torch.nn as nn
import torch.nn.functional as F

class MHCRouter(nn.Module):
    def __init__(self, embedding_size: int, n_streams: int = 4):
        super().__init__()
        self.n_streams = n_streams
        self.embedding_size = embedding_size

        # Projections to dynamically generate coefficients from the hidden state dim (C)
        # To make it efficient, we project from the token embedding dimension
        self.phi_pre = nn.Linear(embedding_size, n_streams, bias=True)
        self.phi_post = nn.Linear(embedding_size, n_streams, bias=True)
        self.phi_res = nn.Linear(embedding_size, n_streams * n_streams, bias=True)

        # Learnable scale parameters (alpha)
        self.alpha_pre = nn.Parameter(torch.ones(1))
        self.alpha_post = nn.Parameter(torch.ones(1))
        self.alpha_res = nn.Parameter(torch.ones(1))
        
        # Norm applied to the stream before generating hyper-connections
        self.rms_norm = nn.RMSNorm(embedding_size, eps=1e-6)

    @staticmethod
    def sinkhorn_knopp(log_alpha: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
        alpha = torch.exp(log_alpha)
        for _ in range(n_iters):
            alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-12)
            alpha = alpha / (alpha.sum(dim=-2, keepdim=True) + 1e-12)
        return alpha

    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Residual stream tensor [B, T, S, C]
            layer_output: Output from sub-layer [B, T, C]
        """
        B, T, S, C = x.shape

        # 1. Compute token-wise representation to drive the hyper-connections
        # We mean-pool across streams to get a clean [B, T, C] token feature
        x_mean = self.rms_norm(x.mean(dim=2)) # [B, T, C]

        # 2. Dynamically generate H_post using 2 * sigmoid
        # phi_post(x_mean) -> [B, T, S]
        h_post_raw = self.alpha_post * self.phi_post(x_mean)
        h_post = 2.0 * torch.sigmoid(h_post_raw) # [B, T, S]

        # 3. Dynamically generate H_res and project via Sinkhorn-Knopp
        # phi_res(x_mean) -> [B, T, S*S] -> reshape to [B, T, S, S]
        h_res_raw = self.alpha_res * self.phi_res(x_mean)
        h_res_raw = h_res_raw.view(B, T, S, S)
        h_res = self.sinkhorn_knopp(h_res_raw, n_iters=5) # [B, T, S, S]

        # 4. Mix the parallel streams using the dynamic H_res matrix
        # x: [B, T, S, C], h_res: [B, T, S, S]
        # We perform batch matrix multiplication over the stream dimension for every token
        x_mixed = torch.einsum("btij,btjc->btic", h_res, x)

        # 5. Scale and add the block's sub-layer output
        # layer_output: [B, T, C] scaled token-wise by h_post: [B, T, S]
        expanded_output = layer_output.unsqueeze(2) * h_post.unsqueeze(-1) # [B, T, S, C]

        return x_mixed + expanded_output

    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        """Dynamically compresses the multi-stream state into a 1D representation for sub-layer processing."""
        B, T, S, C = x.shape
        x_mean = self.rms_norm(x.mean(dim=2))
        
        # Dynamically generate H_pre using standard sigmoid
        h_pre_raw = self.alpha_pre * self.phi_pre(x_mean)
        h_pre = torch.sigmoid(h_pre_raw) # [B, T, S]
        
        # Normalize weights across the stream dimension so they sum to 1
        h_pre = h_pre / (h_pre.sum(dim=-1, keepdim=True) + 1e-12)
        
        return torch.sum(x * h_pre.unsqueeze(-1), dim=2)