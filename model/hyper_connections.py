import torch
from torch import nn


class MHCRouter(nn.Module):
    """Manifold-constrained hyper-connections (mHC) for one sub-layer.

    Hyper-connections replace the single residual stream with S parallel
    streams, hidden state [B, T, S, C], and learn how the sub-layer reads
    from and writes to them. Every mixing weight is computed per token
    from the stream mean (dynamic), via a linear projection scaled by a
    learned scalar `alpha_*`:
    - h_pre [B, T, S] (`collapse`): sigmoid weights, normalized to sum to
      1, that combine the streams into the sub-layer input [B, T, C].
    - h_res [B, T, S, S] (`forward`): stream-to-stream mixing of the
      residual. "Manifold-constrained" means it is projected onto (an
      approximation of) the doubly stochastic matrices with Sinkhorn-Knopp:
      non-negative, rows and columns summing to 1. Each output stream is
      then a weighted average of the input streams, so repeated mixing
      across layers cannot blow up the residual signal (unconstrained
      hyper-connections can).
    - h_post [B, T, S] (`forward`): 2 * sigmoid gains, in (0, 2), with
      which the sub-layer output is added to each stream.

    `Model` replicates the embeddings into the S streams and averages them
    before the final norm. Not in active use (default one stream).
    """
    def __init__(self, embedding_size: int, n_streams: int = 4):
        super().__init__()
        self.n_streams = n_streams
        self.embedding_size = embedding_size

        self.phi_pre = nn.Linear(embedding_size, n_streams, bias=True)
        self.phi_post = nn.Linear(embedding_size, n_streams, bias=True)
        self.phi_res = nn.Linear(embedding_size, n_streams * n_streams, bias=True)
        # Tags read by Model.weight_init (small init; phi_res bias = 0.1 * I)
        for proj in (self.phi_pre, self.phi_post, self.phi_res):
            proj.is_mhc_projection = True  # pyright: ignore[reportArgumentType]
        self.phi_res.is_mhc_residual = True  # pyright: ignore[reportArgumentType]

        self.alpha_pre = nn.Parameter(torch.ones(1))
        self.alpha_post = nn.Parameter(torch.ones(1))
        self.alpha_res = nn.Parameter(torch.ones(1))

        self.rms_norm = nn.RMSNorm(embedding_size, eps=1e-6)

    @staticmethod
    def sinkhorn_knopp(log_alpha: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
        """Approximately doubly stochastic matrix from logits [..., S, S].

        exp() makes every entry positive; alternately normalizing rows and
        columns to sum to 1 converges to a doubly stochastic matrix
        (Sinkhorn's theorem). After `n_iters` rounds the columns sum to 1
        exactly and the rows approximately.
        """
        # Subtract the row max before exp() to avoid overflow
        log_alpha = log_alpha - torch.max(log_alpha, dim=-1, keepdim=True)[0]
        alpha = torch.exp(log_alpha)
        for _ in range(n_iters):
            alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-6)
            alpha = alpha / (alpha.sum(dim=-2, keepdim=True) + 1e-6)
        return alpha

    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        """New residual streams: h_res @ x + h_post * layer_output.

        x: residual streams [B, T, S, C]; layer_output: the sub-layer's
        output [B, T, C]. Returns [B, T, S, C].
        """
        # Mixing weights are predicted from a normalized summary of the
        # streams (their mean).
        B, T, S, _ = x.shape

        x_mean = self.rms_norm(x.mean(dim=2)) # [B, T, C]

        h_post_raw = self.alpha_post * self.phi_post(x_mean)
        h_post = 2.0 * torch.sigmoid(h_post_raw) # [B, T, S]

        h_res_raw = self.alpha_res * self.phi_res(x_mean)
        h_res_raw = h_res_raw.view(B, T, S, S)
        h_res = self.sinkhorn_knopp(h_res_raw, n_iters=5) # [B, T, S, S]

        # Stream i of the output is sum_j h_res[i, j] * stream j
        x_mixed = torch.einsum("btij,btjc->btic", h_res, x)
        expanded_output = layer_output.unsqueeze(2) * h_post.unsqueeze(-1) # [B, T, S, C]

        return x_mixed + expanded_output

    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        """Sub-layer input [B, T, C]: a per-token convex combination (weights
        h_pre, summing to 1) of the S streams of x [B, T, S, C]."""
        x_mean = self.rms_norm(x.mean(dim=2))

        h_pre_raw = self.alpha_pre * self.phi_pre(x_mean)
        h_pre = torch.sigmoid(h_pre_raw) # [B, T, S]

        h_pre = h_pre / (h_pre.sum(dim=-1, keepdim=True) + 1e-6)

        return torch.sum(x * h_pre.unsqueeze(-1), dim=2)
