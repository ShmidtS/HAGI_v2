"""Ternary quantization util + BitLinear (BitNet b1.58) for HAGI V25.

The 2D hidden mixing matrices of the channel body are ternarized. This is
rate-distortion of weight storage at a fixed log2(3) ~= 1.585 bits/weight
(vs FP32's 32 bits/weight). The quantization noise is the genuine discrete
channel impairment in V25 (replaces V23's self-inflicted AWGN).

Scheme: per-OUTPUT-channel absmean scale (the published BitNet b1.58 minimizer)::

    scale          = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)  # [out, 1]
    qweight        = round(clamp(weight / scale, -1, 1))                    # {-1, 0, +1}
    effective_w    = qweight * scale                                        # {-scale, 0, +scale}

The zero bin is IMPLICIT (round sends |w/scale| < 0.5 to 0) -- NOT a TWN
explicit threshold.

Identity STE (gradient flows straight to the FP master; saturated-region
gradients are NOT zeroed -- critical for Muon's Newton-Schulz)::

    w_ste = weight + (effective_weight - weight).detach()
    y     = F.linear(x, w_ste, bias)

``self.weight`` is the FP master trained by Muon; the {-1,0,1} values are
recomputed from the master every forward -- never cached as a Parameter. WD
acts only on the FP master. Quantization is loss-free at this scale
(ternary15M: +0.0104 val loss) so no auxiliary quantization loss term is
needed.

INVARIANTS (enforced):
  * weight MUST be 2D so per-output-channel absmean is well-defined.
  * BitLinear is ONLY for 2D hidden weights (never 1D bias/gate or the
    source codebook). Bias/gate stay FP in the caller.
  * ternarize math runs in the master (weight) dtype; F.linear casts to
    x.dtype. bf16 autocast is mandatory in the caller.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def ternarize(
    weight: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BitNet b1.58 ternarization of a 2D weight matrix.

    Args:
        weight: 2D FP tensor of shape ``[out_features, in_features]``.
        eps: lower bound on the per-output-channel scale.

    Returns:
        ``(effective_weight, scale)`` where ``effective_weight`` has shape
        ``[out, in]`` with values in ``{-scale, 0, +scale}`` and ``scale``
        has shape ``[out, 1]``.

    Raises:
        ValueError: if ``weight`` is not 2D.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"ternarize expects a 2D weight [out, in], got shape {tuple(weight.shape)}"
        )
    # Per-OUTPUT-channel absmean scale. math in the master dtype so the
    # scale tracks the FP latent, not a downcast.
    scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)  # [out, 1]
    # round(clamp(w/scale, -1, 1)) -> {-1, 0, +1}. clamp guards the rare
    # scale<=|w| edge (only possible when all-but-one entry of a row is ~0).
    qweight = (weight / scale).clamp(-1.0, 1.0).round()
    effective_weight = qweight * scale
    return effective_weight, scale


class BitLinear(nn.Module):
    """Drop-in 2D linear whose weight is ternarized in the forward pass.

    Stores an FP master ``self.weight`` (shape ``[out_features, in_features]``,
    init ``N(0, 0.02)``). On every forward the effective ternary weight is
    recomputed from the master and applied via identity STE, so gradients
    flow straight to the master (saturated-region gradients are NOT zeroed).

    This MUST be used only for 2D hidden weights. Biases and 1D gates/codebooks
    stay as plain ``nn.Linear`` / ``nn.Parameter`` in the caller.

    Args:
        in_features: input dimension (last dim of ``x``).
        out_features: output dimension.
        bias: if True, learn a FP bias (default False).
        eps: lower bound on the per-output-channel ternary scale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # no_grad / inference fast path: ternarize and go -- the STE identity
        # is a no-op under inference_mode / eval (detach contributes nothing
        # to a graphless backward) but skipping it avoids building the add.
        if not torch.is_grad_enabled():
            eff_weight, _scale = ternarize(self.weight, self.eps)
            # ternary math ran in the master dtype; cast to x.dtype for the matmul.
            return F.linear(x, eff_weight.to(x.dtype), self.bias)

        eff_weight, _scale = ternarize(self.weight, self.eps)
        # Pure identity STE: forward uses the ternary effective weight,
        # backward flows straight to the FP master. Saturated-region
        # gradients (|w/scale| >= 1) are NOT zeroed -- capacity-matched
        # gradient transport for Muon's Newton-Schulz.
        w_ste = self.weight + (eff_weight - self.weight).detach()
        return F.linear(x, w_ste, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, eps={self.eps}, ternary=BitNet-b1.58"
        )
