"""InformationBottleneck — variational rate-distortion compression (§3.1).

Self-contained additive module (no dependency on the V21 config dataclasses).
Compresses h_ctx in R^H to a stochastic latent z in R^C while retaining
task-relevant information. The only "rate" is KL[q(z|h)||N(0,I)] — a genuine
rate-distortion quantity, learned via the encoder log-variance. No AWGN.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.norms import RMSNorm


@dataclass
class BottleneckConfig:
    """Variational information bottleneck parameters (§3.1)."""

    dim: int = 192  # C < H: real compression
    ib_beta: float = 1e-3  # β: rate Lagrangian (small -> fidelity-led)
    distortion_weight: float = 1.0  # λ_D: reconstruction error weight
    perception_weight: float = 0.01  # λ_P: residual decorrelation (RDP axis)
    kl_free_bits: float = 0.5  # per-dim free bits (prevent rate collapse to 0)


class InformationBottleneck(nn.Module):
    """H -> C variational bottleneck producing z, μ, logσ², and losses.

    forward(h) -> (z, info) where info holds:
      - rate:       KL[q(z|h) || N(0,I)]  (mean over batch/positions)
      - distortion: ||h - ĥ||² reconstruction error
      - perception: lag-1 autocorrelation of the bottleneck residual

    At inference (eval mode), z = μ (deterministic).
    """

    def __init__(self, hidden_size: int, cfg: BottleneckConfig, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim = cfg.dim
        self.cfg = cfg
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        # Encoder: h -> (μ, log σ²) in R^C.
        self.to_mu = nn.Linear(hidden_size, cfg.dim, bias=False)
        self.to_logvar = nn.Linear(hidden_size, cfg.dim, bias=False)
        # Decoder (source decoder side): z -> ĥ in R^H.
        self.decompress = nn.Linear(cfg.dim, hidden_size, bias=False)
        nn.init.normal_(self.decompress.weight, std=1.0 / (cfg.dim**0.5))

    def encode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h -> (z, μ, logσ²). z is sampled at train time, = μ at eval."""
        h_n = self.norm(h)
        mu = self.to_mu(h_n)
        logvar = self.to_logvar(h_n)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        return z, mu, logvar

    @staticmethod
    def kl_rate(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
        """KL[ N(μ,σ²) || N(0,I) ] per position, with per-dim free bits.

        Free bits: floor each dimension's KL contribution at `free_bits` to
        prevent the rate collapsing to 0 (posterior -> prior) before any
        information is retained.
        """
        per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        per_dim = torch.clamp(per_dim, min=free_bits)
        return per_dim.mean()

    @staticmethod
    def perception_penalty(h_ctx: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
        """RDP perception axis: lag-1 autocorrelation of the channel residual.

        High autocorrelation = structured residual (distributional mismatch);
        minimizing it pushes the reconstruction toward the data distribution.
        """
        residual = (h_ctx - h_hat).float()
        if residual.size(-1) < 2:
            return residual.new_zeros(())
        r_left = residual[..., :-1]
        r_right = residual[..., 1:]
        cos = F.cosine_similarity(r_left, r_right, dim=-1)
        return cos.abs().mean()

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Encode h -> z and compute RD/perception terms."""
        z, mu, logvar = self.encode(h)
        h_hat = self.decompress(z)
        rate = self.kl_rate(mu, logvar, self.cfg.kl_free_bits)
        # Normalized distortion (relative MSE / "1 - explained variance").
        # ||h-ĥ||².mean() over the UN-normalized h_ctx scales with ||h_ctx||²
        # (~306 at init vs CE ~11) and grows as activations grow during training
        # — that 30x scale mismatch is the V24 instability root cause (grad norm
        # climbing, distortion Goodhart-delayed by the warmup but not removed).
        # Normalizing by ||h||².mean() (detached) makes it scale-invariant and
        # bounded in [0, ~1], so the weight is meaningful and it cannot dominate
        # CE or run away. Detaching the denominator keeps the gradient pushing
        # ĥ->h only, not inflating ||h||.
        h_f = h.float()
        denom = h_f.pow(2).mean().detach() + 1e-6
        distortion = (h_f - h_hat.float()).pow(2).mean() / denom
        perception = self.perception_penalty(h, h_hat)
        info = {
            "z": z,
            "mu": mu,
            "rate": rate,
            "distortion": distortion,
            "perception": perception,
        }
        return z, info
