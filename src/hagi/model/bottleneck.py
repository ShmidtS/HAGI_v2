"""InformationBottleneck — auxiliary variational semantic rate (off the main path).

The IB computes KL / distortion on the context hidden as an auxiliary
regularizer. It does NOT intercept the LM signal — inserting it into the main
path deadlocks from-scratch training (CE stalls at ~ln(V)).

  * The stochastic encoder (``to_mu`` / ``to_logvar``) and the source decoder
    (``decompress``) are FP32 ``nn.Linear`` (KL numerical stability; rate-
    critical). They route to AdamW, not Muon (the optimizer picks them by type:
    they are not BitLinear).
  * Rate = ``KL[q(z|h) || N(0,I)]`` averaged, with a per-dim ``free_bits`` floor.
  * Distortion = normalized reconstruction ``||h_ctx - h_hat||^2`` (denominator
    detached so the gradient pushes ``h_hat -> h_ctx`` only). Beta-annealed over
    warmup by the loss aggregator so it does not dominate CE at init.

At eval (``not self.training``) ``z = mu`` deterministically (reparam is train).
"""

from __future__ import annotations

import torch
from torch import nn

from hagi.config import BottleneckConfig
from hagi.model.norms import RMSNorm


class InformationBottleneck(nn.Module):
    """H -> C variational stochastic encoder + C -> H source decoder.

    forward(h) -> info dict carrying 'mu', 'rate', 'distortion' scalars.
    """

    FP32_PARAM_NAMES = ("to_mu", "to_logvar", "decompress")

    def __init__(self, hidden_size: int, cfg: BottleneckConfig, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim = cfg.dim
        self.cfg = cfg
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.to_mu = nn.Linear(hidden_size, cfg.dim, bias=False)
        self.to_logvar = nn.Linear(hidden_size, cfg.dim, bias=False)
        self.decompress = nn.Linear(cfg.dim, hidden_size, bias=False)
        nn.init.normal_(self.decompress.weight, std=1.0 / (cfg.dim**0.5))

    @staticmethod
    def kl_rate(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
        """KL[N(mu, var) || N(0, I)] averaged, with per-dim free-bits floor."""
        per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        per_dim = torch.clamp(per_dim, min=free_bits)
        return per_dim.mean()

    @staticmethod
    def distortion_penalty(h_ctx: torch.Tensor, h_hat: torch.Tensor, eps: float) -> torch.Tensor:
        """Normalized, scale-invariant RD distortion (detached denominator)."""
        h_f = h_ctx.float()
        denom = h_f.pow(2).mean().detach() + eps
        return (h_f - h_hat.float()).pow(2).mean() / denom

    def forward(self, h: torch.Tensor) -> dict:
        """Compute rate / distortion on ``h`` (off the main path).

        Args:
            h: ``[B, T, H]`` context hidden.

        Returns:
            dict with 'mu', 'logvar', 'rate', 'distortion'.
        """
        h_n = self.norm(h)
        mu = self.to_mu(h_n)
        logvar = torch.clamp(self.to_logvar(h_n), self.cfg.logvar_clamp[0], self.cfg.logvar_clamp[1])
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        h_hat = self.decompress(z)
        return {
            "mu": mu,
            "logvar": logvar,
            "rate": self.kl_rate(mu, logvar, self.cfg.kl_free_bits),
            "distortion": self.distortion_penalty(h, h_hat, self.cfg.distortion_eps),
        }
