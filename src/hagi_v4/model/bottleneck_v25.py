"""InformationBottleneck (V25) — the only semantic rate (§3.2 of ARCHITECTURE_V25.md).

V25 keeps the variational IB core of the V24 module but makes the module
self-contained and explicit about the V25 invariants:

  * The stochastic encoder (``to_mu`` / ``to_logvar``) and the source
    decoder (``decompress``) are FP32 ``nn.Linear`` — they are NOT ternary
    (KL numerical stability; rate-critical). The corresponding parameter
    names must be added to ``_MUON_EXCLUDE`` so the FP masters flow to
    AdamW, not Muon.
  * The ONLY rate notion is ``KL[q(z|h)||N(0,I)]`` with a per-dim
    ``free_bits`` floor. This replaces the V23 AWGN capacity invariants.
  * Distortion is the normalized reconstruction ``||h_ctx - h_hat||²``,
    denominator detached so the gradient pushes ``h_hat -> h_ctx`` only.
  * Perception is the lag-1 channel-axis cosine-autocorrelation of the
    residual — the RDP third axis (NOW WIRED; was dead in V24).

At eval (``not self.training``) ``z = mu`` deterministically (reparam trick
is train-only).

``C < H`` (real compression) is enforced by config validation in
``config.py`` (``model.core_hidden_size <= hidden_size``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.norms import RMSNorm


@dataclass
class BottleneckConfig:
    """Variational information bottleneck parameters (V25 §3.2).

    The four scalar weights here mirror the loss-term weights in the
    contract (``w_v24_rate`` / ``w_v24_distortion`` / ``w_v24_perception``)
    and the doc (``ib_beta`` / ``distortion_weight`` / ``perception_weight``).
    They are stored for introspection / serialization; the actual loss
    aggregation happens in ``train/losses.py`` which reads the
    ``info['rate']`` / ``info['distortion']`` / ``info['perception']``
    scalars returned by ``forward``.
    """

    dim: int = 192  # C: must satisfy C < H (real compression)
    ib_beta: float = 1e-3  # β: rate Lagrangian (the ONLY rate notion)
    distortion_weight: float = 1.0  # λ_D: reconstruction fidelity weight
    perception_weight: float = 0.01  # λ_P: residual autocorrelation (RDP axis)
    kl_free_bits: float = 0.5  # per-dim free bits (prevent rate collapse -> 0)
    logvar_clamp: tuple[float, float] = (-10.0, 10.0)  # clamp before exp
    distortion_eps: float = 1e-6  # numerical floor on the detached denominator


class InformationBottleneck(nn.Module):
    """H -> C variational stochastic encoder + C -> H source decoder.

    forward(h) -> (z, info) where:

      * ``z`` is the bottleneck latent in ``R^C`` (``[..., C]``).
      * ``info`` is a dict with keys ``'z'``, ``'mu'``, ``'rate'`` (scalar),
        ``'distortion'`` (scalar, normalized), ``'perception'`` (scalar).

    The encoder path is::

        μ       = to_mu( LN(h_ctx) )        # [..., C]
        log σ²  = clamp( to_logvar( LN(h_ctx) ), -10, 10 )
        z       = μ + σ ⊙ ε                 # train; z = μ at eval

    All three linears (``to_mu`` / ``to_logvar`` / ``decompress``) stay FP32
    and are excluded from Muon — they are rate-critical and feed a KL whose
    numerical stability depends on FP master arithmetic.

    Args:
        hidden_size: H (context hidden dim, input to the bottleneck).
        cfg: ``BottleneckConfig`` (``cfg.dim`` is C and must be < H).
        norm_eps: RMSNorm epsilon.
    """

    # Names of the FP32 rate-critical masters that must NOT be ternarized
    # and must be routed to AdamW (added to ``_MUON_EXCLUDE`` upstream).
    FP32_PARAM_NAMES = ("to_mu", "to_logvar", "decompress")

    def __init__(self, hidden_size: int, cfg: BottleneckConfig, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim = cfg.dim
        self.cfg = cfg
        # FP32 RMSNorm pre-norm on the context hidden before parametrising the
        # posterior. Gain stays FP (1D Parameter, excluded from ternary/Muon).
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        # Encoder: h -> (μ, log σ²) in R^C. FP32 nn.Linear (rate-critical).
        self.to_mu = nn.Linear(hidden_size, cfg.dim, bias=False)
        self.to_logvar = nn.Linear(hidden_size, cfg.dim, bias=False)
        # Source decoder: z -> ĥ in R^H. FP32 nn.Linear. Small init so the
        # distortion term starts near the (scale-invariant) reference and the
        # gradient signal to ĥ->h is well-conditioned.
        self.decompress = nn.Linear(cfg.dim, hidden_size, bias=False)
        nn.init.normal_(self.decompress.weight, std=1.0 / (cfg.dim**0.5))

    # ------------------------------------------------------------------ #
    # Stochastic encode / decode
    # ------------------------------------------------------------------ #
    def encode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h -> (z, μ, logσ²).

        Reparameterization trick at train time (``z = μ + σ⊙ε``); at eval
        (``not self.training``) ``z = μ`` deterministically.
        """
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
        """z -> ĥ (source decoder reconstruction in R^H)."""
        return self.decompress(z)

    # ------------------------------------------------------------------ #
    # RD / perception helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def kl_rate(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
        """KL[ N(μ, σ²) || N(0, I) ] averaged over positions, with per-dim free bits.

        Per-dim KL ``0.5 (μ² + σ² − log σ² − 1)`` is clamped at ``free_bits``
        to prevent posterior -> prior collapse (rate -> 0) before information
        is retained. Returns a scalar (mean over batch / positions / dims).

        Args:
            mu: posterior mean ``[..., C]``.
            logvar: posterior log-variance ``[..., C]``.
            free_bits: per-dim KL floor.
        """
        per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        per_dim = torch.clamp(per_dim, min=free_bits)
        return per_dim.mean()

    @staticmethod
    def distortion_penalty(h_ctx: torch.Tensor, h_hat: torch.Tensor, eps: float) -> torch.Tensor:
        """Normalized, scale-invariant RD distortion.

        ``D = ||h_ctx − ĥ||².mean() / (||h_ctx||².mean().detach() + eps)``.

        The denominator is DETACHED so the gradient pushes ``ĥ -> h_ctx``
        only and never inflates ``||h_ctx||``. The quantity is bounded in
        ``[0, ~1]`` so the ``distortion_weight`` is meaningful and cannot
        dominate CE (~30× scale mismatch was the V24 instability root).

        Args:
            h_ctx: context hidden ``[..., H]`` (the encode target).
            h_hat: reconstruction ``[..., H]``.
            eps: numerical floor on the denominator.
        """
        h_f = h_ctx.float()
        denom = h_f.pow(2).mean().detach() + eps
        return (h_f - h_hat.float()).pow(2).mean() / denom

    @staticmethod
    def perception_penalty(h_ctx: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
        """RDP perception axis: lag-1 channel-axis autocorr of the residual.

        ``P = | cos( residual[..., :-1] , residual[..., 1:] ) |.mean()``.

        Minimizing this pushes the reconstruction residual toward white
        (distribution match). Computed in float32 for cosine stability.

        Args:
            h_ctx: context hidden ``[..., H]``.
            h_hat: reconstruction ``[..., H]``.
        """
        residual = (h_ctx - h_hat).float()
        if residual.size(-1) < 2:
            return residual.new_zeros(())
        r_left = residual[..., :-1]
        r_right = residual[..., 1:]
        cos = F.cosine_similarity(r_left, r_right, dim=-1)
        return cos.abs().mean()

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Encode ``h`` -> ``z`` and compute RD / perception terms.

        Args:
            h: context hidden state ``h_ctx`` of shape ``[..., H]``
               (post-context-stack, pre-predictive-decode).

        Returns:
            Tuple ``(z, info)`` where ``z`` is ``[..., C]`` and ``info`` is a
            dict with keys ``'z'``, ``'mu'``, ``'rate'`` (scalar),
            ``'distortion'`` (scalar, normalized), ``'perception'`` (scalar).
        """
        z, mu, logvar = self.encode(h)
        h_hat = self.decode(z)

        rate = self.kl_rate(mu, logvar, self.cfg.kl_free_bits)
        distortion = self.distortion_penalty(h, h_hat, self.cfg.distortion_eps)
        perception = self.perception_penalty(h, h_hat)

        info = {
            "z": z,
            "mu": mu,
            "rate": rate,
            "distortion": distortion,
            "perception": perception,
        }
        return z, info
