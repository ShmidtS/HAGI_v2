"""SwiGLU channel mixer.

The per-position nonlinear map. SwiGLU's gate is a learned, input-dependent
attenuation: ``silu(W_g x)`` decides how much of ``W_u x`` reaches the output
per channel, which is a soft per-channel power control on the mixer's input.
Empirically it beats a same-parameter-count two-matrix MLP, and with the 8/3
expansion the three matrices cost the same as a 4x MLP's two.

Down-projection init is scaled by ``residual_scale`` so the summed variance of
all residual branches stays O(1) as depth grows. Without it a deep pre-norm
stack starts with a residual stream whose variance grows linearly in depth,
which forces the first thousand steps to be spent shrinking output weights
rather than learning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi.model.norms import RMSNorm
from hagi.model.ternary import BitLinear


class BranchScale(nn.Module):
    """Learnable cap on a residual branch's output norm.

    A pre-norm stack keeps the residual stream an O(1) sum of increments, but
    nothing bounds each increment itself. Under a matrix-sign optimizer (Muon)
    the down/out projections grow without the ``1/||W||`` brake plain SGD has;
    measured on the V31 1B run the ``down`` weights grew ~24x and the branch
    output dominated the stream (std 24-35 against a target of O(1)), which
    destroyed the input statistics of every following layer. One learnable
    scalar, initialized to ``residual_scale`` and clamped to ``[0.5, 2]x`` of
    that scale, caps the branch so it cannot capture the stream regardless of
    how large the underlying weights get. The gain still learns — a shallow
    feature that genuinely needs a large increment can ask for it — but it is
    bounded, and ``keep_fp32`` keeps its tiny updates from rounding to zero
    under bf16.

    Args:
        residual_scale: init (``1/sqrt(2L)`` at construction) and clamp center.
        clamp_ratio: branch scale stays in ``[residual_scale/clamp_ratio,
            residual_scale*clamp_ratio]``.
    """

    def __init__(self, residual_scale: float, clamp_ratio: float = 2.0) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.clamp_ratio = float(clamp_ratio)
        self.keep_fp32 = True
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo = self.residual_scale / self.clamp_ratio
        hi = self.residual_scale * self.clamp_ratio
        s = self.scale.clamp(lo, hi).to(x.dtype)
        return x * s

    def extra_repr(self) -> str:
        return (
            f"residual_scale={self.residual_scale:.4f}, "
            f"clamp=[{self.residual_scale / self.clamp_ratio:.4f}, "
            f"{self.residual_scale * self.clamp_ratio:.4f}]"
        )


def linear(in_features: int, out_features: int, use_ternary: bool) -> nn.Module:
    """2D channel weight: ternary master or plain floating-point linear.

    The returned module carries ``is_channel_weight = True``. That marker is what
    the optimizer partitions on (see :func:`~hagi.train.optim._muon_parameters`),
    and it is deliberately independent of ``use_ternary``: Muon's argument is
    about the geometry of a hidden-mixing matrix, not about its storage rate, so
    disabling quantization for an ablation must not also change which optimizer
    every matrix in the body rides.
    """
    module = BitLinear(in_features, out_features, bias=False) if use_ternary else nn.Linear(
        in_features, out_features, bias=False
    )
    if not use_ternary:
        nn.init.normal_(module.weight, std=in_features**-0.5)
    module.is_channel_weight = True
    return module


class SwiGLU(nn.Module):
    """``down(silu(gate(x)) * up(x))`` with no internal residual or norm.

    Used both as the dense mixer body (wrapped by :class:`FeedForward`) and as a
    single MoE expert, which is why normalization and the residual live in the
    caller: an expert must not re-normalize a hidden state that the router
    already normalized.

    Args:
        hidden_size: H.
        intermediate_size: mixer width.
        use_ternary: quantize the 2D weights.
        residual_scale: init scale on ``down``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_ternary: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.gate = linear(hidden_size, intermediate_size, use_ternary)
        self.up = linear(hidden_size, intermediate_size, use_ternary)
        self.down = linear(intermediate_size, hidden_size, use_ternary)
        nn.init.normal_(self.down.weight, std=residual_scale / intermediate_size**0.5)
        self.branch_scale = BranchScale(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down(F.silu(self.gate(x)) * self.up(x))
        return self.branch_scale(out)


class FeedForward(nn.Module):
    """Pre-norm SwiGLU mixer branch (norm + SwiGLU; residual added by the block).

    Args:
        hidden_size: H.
        intermediate_size: mixer width.
        norm_eps: RMSNorm epsilon.
        use_ternary: quantize the 2D weights.
        residual_scale: init scale on the down-projection.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        norm_eps: float = 1e-5,
        use_ternary: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.mixer = SwiGLU(hidden_size, intermediate_size, use_ternary, residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x))
