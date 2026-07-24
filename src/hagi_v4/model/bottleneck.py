"""InformationBottleneck — auxiliary variational semantic rate (off the main path).

The IB computes KL/distortion/perception on the context hidden as an
auxiliary regularizer. It does NOT intercept the LM signal — inserting it
into the main path deadlocks from-scratch training.

  * The stochastic encoder (``to_mu`` / ``to_logvar``) and the source
    decoder (``decompress``) are FP32 ``nn.Linear`` — they are NOT ternary
    (KL numerical stability; rate-critical). The corresponding parameter
    names must be routed to AdamW, not Muon (added to ``_MUON_EXCLUDE``).
  * The ONLY rate notion is ``KL[q(z|h)||N(0,I)]`` with a per-dim
    ``free_bits`` floor.
  * Distortion is the normalized reconstruction ``||h_ctx - h_hat||^2``,
    denominator detached so the gradient pushes ``h_hat -> h_ctx`` only.
  * Perception is the lag-1 channel-axis cosine-autocorrelation of the
    residual — the RDP third axis.

At eval (``not self.training``) ``z = mu`` deterministically (reparam trick
is train-only).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.norms import RMSNorm


@dataclass
class BottleneckConfig:
    """Variational information bottleneck parameters."""

    dim: int = 192  # C: must satisfy C < H (real compression)
    ib_beta: float = 0.001  # beta: rate Lagrangian (the ONLY rate notion)
    distortion_weight: float = 1.0
    perception_weight: float = 0.01
    kl_free_bits: float = 0.5
    logvar_clamp: tuple[float, float] = (-10.0, 10.0)
    distortion_eps: float = 1e-6


class InformationBottleneck(nn.Module):
    """H -> C variational stochastic encoder + C -> H source decoder.

    forward(h) -> (z, info) where ``z`` is the bottleneck latent in R^C and
    ``info`` carries 'z', 'mu', 'rate', 'distortion', 'perception' scalars.
    """

    # Names of the FP32 rate-critical masters that must NOT be ternarized
    # and must be routed to AdamW (added to ``_MUON_EXCLUDE`` upstream).
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

    def encode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h -> (z, mu, logvar). Reparam at train; z = mu at eval."""
        h_n = self.norm(h)
        mu = self.to_mu(h_n)
        logvar = torch.clamp(self.to_logvar(h_n), self.cfg.logvar_clamp[0], self.cfg.logvar_clamp[1])
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decompress(z)

    @staticmethod
    def kl_rate(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
        """KL[N(mu, var) || N(0, I)] averaged, with per-dim free bits floor."""
        per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        per_dim = torch.clamp(per_dim, min=free_bits)
        return per_dim.mean()

    @staticmethod
    def distortion_penalty(h_ctx: torch.Tensor, h_hat: torch.Tensor, eps: float) -> torch.Tensor:
        """Normalized, scale-invariant RD distortion (detached denominator)."""
        h_f = h_ctx.float()
        denom = h_f.pow(2).mean().detach() + eps
        return (h_f - h_hat.float()).pow(2).mean() / denom

    @staticmethod
    def perception_penalty(h_ctx: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
        """RDP perception axis: lag-1 channel-axis autocorr of the residual."""
        residual = (h_ctx - h_hat).float()
        if residual.size(-1) < 2:
            return residual.new_zeros(())
        cos = F.cosine_similarity(residual[..., :-1], residual[..., 1:], dim=-1)
        return cos.abs().mean()

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict]:
        z, mu, logvar = self.encode(h)
        h_hat = self.decode(z)
        info = {
            "z": z,
            "mu": mu,
            "rate": self.kl_rate(mu, logvar, self.cfg.kl_free_bits),
            "distortion": self.distortion_penalty(h, h_hat, self.cfg.distortion_eps),
            "perception": self.perception_penalty(h, h_hat),
        }
        return z, info
