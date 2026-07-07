"""Variational Information Bottleneck — optimal source coding.

Shannon analogy: Source coding theorem says optimal compression
achieves rate R = H(X). The Information Bottleneck (Tishby) formalizes
this as: min I(X;Z) - beta * I(Y;Z).

V5 uses a deterministic linear bottleneck: z = Linear(h). This has no
explicit control over I(X;Z) — compression is implicit.

V6 uses a variational bottleneck: z = mu + eps * sigma, where
mu = Linear_mu(h), logvar = Linear_logvar(h). The KL divergence
KL(q(z|x) || N(0,1)) provides an analytical upper bound on I(X;Z),
making the IB objective explicit and differentiable.

This is analogous to VAE-based compression: the encoder produces a
distribution (not a point), and the KL term controls the compression
rate. Lower KL = more compression = less rate = closer to IB bound.

Reparameterization trick: z = mu + eps * exp(0.5 * logvar)
allows gradients to flow through the stochastic sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.model.norms import RMSNorm


class VariationalBottleneck(nn.Module):
    """Variational IB bottleneck with learned compression.

    Encodes h [B, T, H] into compressed latent z [B, T, C] via:
      mu = Linear_mu(h)
      logvar = Linear_logvar(h)
      z = mu + eps * exp(0.5 * logvar)  (reparameterization)

    KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    At inference: z = mu (deterministic, no sampling).
    """

    def __init__(
        self,
        input_dim: int,
        compressed_dim: int,
        kl_weight: float = 0.01,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.kl_weight = kl_weight

        self.encoder_mu = nn.Linear(input_dim, compressed_dim, bias=False)
        self.encoder_logvar = nn.Linear(input_dim, compressed_dim, bias=False)
        self.decoder = nn.Linear(compressed_dim, input_dim, bias=False)
        self.norm = RMSNorm(compressed_dim, eps=norm_eps)

        nn.init.normal_(self.encoder_mu.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder_logvar.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)

    def encode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode h -> (z, mu, logvar).

        Returns:
            z: [B, T, C] compressed latent (sampled if training, mu if eval).
            mu: [B, T, C] mean of posterior.
            logvar: [B, T, C] log-variance of posterior.
        """
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu

        z = self.norm(z)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode z -> h_reconstructed [B, T, H]."""
        return self.decoder(z)

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)).

        This is the variational upper bound on I(X;Z).
        Minimizing KL = minimizing complexity = maximizing compression.
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return kl

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Full encode-decode pass.

        Returns:
            z: compressed latent [B, T, C].
            h_recon: reconstructed hidden [B, T, H].
            kl_loss: KL divergence (None if eval mode).
        """
        z, mu, logvar = self.encode(h)
        h_recon = self.decode(z)

        if self.training:
            kl = self.kl_divergence(mu, logvar)
        else:
            kl = None

        return z, h_recon, kl
