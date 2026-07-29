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
  * Iterative latent refinement (opt-in): when ``ib_max_iters > 1``, the
    bottleneck runs multiple passes feeding each reconstruction back as input.
    EXIT-halt via SNR of the latent state (``ib_snr_threshold``) or distortion
    stall (``ib_distortion_epsilon``).

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
    When ``cfg.iterative_ib_enabled`` and ``cfg.ib_max_iters > 1``, runs
    iterative latent refinement with EXIT-halt gating.
    """

    FP32_PARAM_NAMES = ("to_mu", "to_logvar", "decompress")

    def ensure_fp32(self) -> None:
        """Convert KL-critical parameters to fp32 regardless of model-wide dtype.

        The to_mu/to_logvar KL rate and decompress distortion decoder are
        numerically sensitive: bf16 logvar clamps cause floating-point collapse
        of the rate regularizer, leading to unbounded rate growth and CE
        gradient conflict. This method is called after the bf16 model cast.
        """
        for name in self.FP32_PARAM_NAMES:
            mod = getattr(self, name, None)
            if mod is not None:
                mod.to(torch.float32)

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

    def _single_pass(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One encode-decode pass through the bottleneck.

        Args:
            h: ``[B, T, H]`` input hidden.

        Returns:
            ``(h_hat, mu, logvar, z)`` — reconstruction, latent params, latent sample.
        """
        h_n = self.norm(h)
        h_n_f = h_n.float()
        mu = self.to_mu(h_n_f)
        logvar = torch.clamp(self.to_logvar(h_n_f), self.cfg.logvar_clamp[0], self.cfg.logvar_clamp[1])
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        h_hat = self.decompress(z)
        return h_hat, mu, logvar, z

    @staticmethod
    def _latent_snr(mu: torch.Tensor, logvar: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Per-token SNR: ``||mu||^2 / mean(exp(logvar))``.

        Signal = latent mean (structured code), noise = average variance across
        dimensions. At random init mu ~ N(0,~1) / sqrt(dim) so ||mu||^2 is tiny
        and the noise floor dominates — SNR is low. As the IB learns structure,
        ||mu|| rises relative to the noise, and SNR grows.

        Higher SNR = the latent code carries real signal. Used as an EXIT-halt
        gate: if SNR exceeds the threshold, further iterations are diminishing
        returns.
        """
        noise = torch.exp(logvar.float()).mean() + eps
        signal = mu.float().pow(2).sum(dim=-1).mean()
        return signal / noise

    def forward(self, h: torch.Tensor) -> dict:
        """Compute rate / distortion on ``h`` (off the main path).

        Args:
            h: ``[B, T, H]`` context hidden.

        Returns:
            dict with 'mu', 'logvar', 'rate', 'distortion', and diagnostic
            'ib_iters' (number of iterations used, for histogram logging).
        """
        use_iterative = (
            self.cfg.iterative_ib_enabled
            and self.cfg.ib_max_iters > 1
            and self.training
        )
        max_iters = self.cfg.ib_max_iters if use_iterative else 1

        h_input = h
        distortion_prev: torch.Tensor | None = None
        iters_used = 1

        for it in range(max_iters):
            h_hat, mu, logvar, z = self._single_pass(h_input)

            if use_iterative and it < max_iters - 1:
                # EXIT-halt gates skip iteration 0 (baseline pass) — they
                # only gate after at least one refinement iteration.
                if it > 0:
                    # EXIT-halt gate: SNR of the latent state.
                    if self.cfg.ib_snr_threshold > 0.0:
                        snr_val = self._latent_snr(mu, logvar)
                        if snr_val > self.cfg.ib_snr_threshold:
                            iters_used = it + 1
                            break

                    # EXIT-halt gate: distortion stall.
                    if self.cfg.ib_distortion_epsilon > 0.0:
                        d_curr = self.distortion_penalty(h, h_hat, self.cfg.distortion_eps)
                        if distortion_prev is not None:
                            delta = (distortion_prev - d_curr).abs()
                            if delta < self.cfg.ib_distortion_epsilon:
                                iters_used = it + 1
                                break
                        distortion_prev = d_curr.detach()

                h_input = h_hat  # feed reconstruction as next input
        else:
            iters_used = max_iters  # no early break: completed all iterations

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "rate": self.kl_rate(mu, logvar, self.cfg.kl_free_bits),
            "distortion": self.distortion_penalty(h, h_hat, self.cfg.distortion_eps),
            "ib_iters": iters_used,
        }
