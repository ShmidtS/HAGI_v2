"""Ternary weight quantization (BitNet b1.58) — the channel rate constraint.

Storing a weight as one of ``{-s, 0, +s}`` fixes the per-weight storage rate at
``log2(3) = 1.585`` bits, against 16 for bf16. This is rate-distortion applied
to the *parameters*: the quantizer is the rate constraint, and the resulting
weight error is the channel's intrinsic distortion. It is the only noise in the
model — nothing artificial is injected.

Scheme (the published b1.58 minimizer, per output channel)::

    s   = mean(|W|, dim=1)                # [out, 1]
    Q   = round(clamp(W / s, -1, +1))     # {-1, 0, +1}
    W~  = Q * s

The zero bin is implicit: ``round`` sends ``|w/s| < 0.5`` to 0. ``s`` is
recomputed from the fp master every forward and never stored as a parameter, so
weight decay and the optimizer act on the master only.

The straight-through estimator is the *identity* — saturated entries are not
zeroed. Zeroing them would erase the gradient of exactly the largest-magnitude
weights, which is where a matrix-sign optimizer (Muon) gets most of its signal.

A useful invariant, verified by simulation over 20k Muon steps: because ``s``
is the per-row absmean of the master, any uniform outward drift of ``||W||``
cancels in ``W/s``. The effective weight RMS is therefore self-stabilizing,
which is why the ternary body does not need a spectral cap.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def ternarize(weight: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to ``{-s, 0, +s}`` per output channel.

    Args:
        weight: ``[out, in]`` floating-point master.
        eps: floor on the per-channel scale (guards all-zero rows).

    Returns:
        ``(effective_weight, scale)`` with shapes ``[out, in]`` and ``[out, 1]``.

    Raises:
        ValueError: if ``weight`` is not 2D.
    """
    if weight.dim() != 2:
        raise ValueError(f"ternarize expects a 2D weight [out, in], got {tuple(weight.shape)}")
    scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    qweight = (weight / scale).clamp(-1.0, 1.0).round()
    return qweight * scale, scale


class _TernarizeSTE(torch.autograd.Function):
    """Identity straight-through estimator; builds no graph for the quantizer."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, eps: float) -> torch.Tensor:  # type: ignore[override]
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
        return (weight / scale).clamp(-1.0, 1.0).round() * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output, None


class BitLinear(nn.Module):
    """Linear layer whose 2D weight is ternarized in the forward pass.

    Only for 2D hidden-mixing weights. Codebooks, biases and 1D gains stay
    floating point in the caller — quantizing the source codebook destroys the
    token identity code, and quantizing a 1D gain has no rate benefit.

    **Step cache (OFDM coherence interval).** Within one optimizer step the
    master ``W`` is constant across ``grad_accum`` microbatches. Recomputing the
    ternary map on every microbatch is pure waste: the quantizer is a slow
    function of ``W`` (sign flips ~0.05%/step). :meth:`cache_quantized` freezes
    ``Q*s`` for the step; the forward then uses the classic STE rewrite
    ``W + (Q - W).detach()`` so the matmul sees ``Q`` while the gradient still
    flows to the master as the identity. Cleared by :meth:`clear_quantized`
    after the optimizer step.

    Args:
        in_features: input width.
        out_features: output width.
        bias: learn a floating-point bias.
        eps: floor on the per-channel ternary scale.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = float(eps)
        # Marks this weight as a 2D hidden-mixing matrix for the optimizer
        # partition; see hagi.model.ffn.linear for why the marker rather than the
        # module type is what the partition reads.
        self.is_channel_weight = True
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=in_features**-0.5)
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        # Per-step quantized weight (set by the trainer around grad accumulation).
        self._step_q: torch.Tensor | None = None

    def cache_quantized(self) -> None:
        """Freeze the ternary map for the current optimizer step."""
        with torch.no_grad():
            self._step_q, _ = ternarize(self.weight, self.eps)

    def clear_quantized(self) -> None:
        """Drop the step cache (call after the optimizer step)."""
        self._step_q = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._step_q is not None:
            # STE with a precomputed Q: forward equals Q, backward dL/dW = dL/dQ.
            q = self._step_q.to(dtype=self.weight.dtype, device=self.weight.device)
            eff = self.weight + (q - self.weight).detach()
            return F.linear(x, eff.to(x.dtype), self.bias)
        if not torch.is_grad_enabled():
            eff, _ = ternarize(self.weight, self.eps)
            return F.linear(x, eff.to(x.dtype), self.bias)
        return F.linear(x, _TernarizeSTE.apply(self.weight, self.eps), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant=b1.58"
        )


def cache_ternary_weights(model: nn.Module) -> None:
    """Freeze every BitLinear's ternary map for one optimizer step."""
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.cache_quantized()


def clear_ternary_weights(model: nn.Module) -> None:
    """Clear every BitLinear step cache."""
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.clear_quantized()
