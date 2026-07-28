"""Ternary quantization util + BitLinear (BitNet b1.58).

The 2D hidden mixing matrices of the channel body are ternarized. This is
rate-distortion of weight storage at a fixed log2(3) ~= 1.585 bits/weight
(vs FP32's 32 bits/weight). The quantization noise is the genuine discrete
channel impairment (there is no self-inflicted AWGN).

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

import torch
import torch.nn.functional as F
from torch import nn


def ternarize(
    weight: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
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


class _TernarizeSTE(torch.autograd.Function):
    """Identity straight-through estimator for BitNet b1.58 ternarization.

    Forward: ternarize the FP master into the effective {-scale,0,+scale}
    weight. Backward: pass grad through unchanged. Saturated-region gradients
    (|w/scale| >= 1) are NOT zeroed -- capacity-matched gradient transport for
    Muon's Newton-Schulz.

    Replaces the graph-building ``w + (eff - w).detach()`` form, which emits a
    sub/add/detach per BitLinear forward (~17k detach/step on the small model,
    aten::detach 2.76% + detach 1.41% self CUDA in the profiler) plus the
    MulBackward0 nodes that dominate CPU-total. Function builds no autograd
    graph for the ternarize math -- one launch in, one grad out.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, eps: float) -> torch.Tensor:
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
        qweight = (weight / scale).clamp(-1.0, 1.0).round()
        return qweight * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


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

        # Pure identity STE via autograd.Function: forward ternarizes, backward
        # passes grad through unchanged (saturated-region gradients NOT zeroed
        # -- capacity-matched transport for Muon's Newton-Schulz). Replaces the
        # graph-building w + (eff - w).detach() form that emitted sub/add/detach
        # per forward (~17k detach/step + MulBackward0 nodes in the profiler).
        w_ste = _TernarizeSTE.apply(self.weight, self.eps)
        return F.linear(x, w_ste, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, eps={self.eps}, ternary=BitNet-b1.58"
        )
