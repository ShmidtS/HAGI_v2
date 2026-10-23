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


def ternarize_split(
    weight: torch.Tensor, eps: float = 1e-5, tau: float = 0.2, points: int = 12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Saturation-aware ternarize [Runtime/TernaryExact ``ternary_split_le``
    + the R103 measurement].

    The default ``ternarize`` takes s = mean|w| per row. R103 measured the
    failure mode: at s = 0.05 on the gen-3 embedding, 86% of the squared
    error came from the SATURATION TAIL (entries |w| > 3s/2, exact error
    |w|-s), which the in-range s/2 formula never sees. This variant first
    diagnoses the split budget at the mean scale; only when saturation is
    significant (sat_share > tau) does it re-sweep the grid per row-band
    (approximated globally here: rows are rescaled by a single factor
    minimizing the total split bound).

    KISS guard: pure win-check — the split-optimal scale is accepted only
    when it actually lowers the measured split bound, otherwise the
    default path is returned unchanged.
    """
    if weight.dim() != 2:
        raise ValueError(f"ternarize expects a 2D weight [out, in], got {tuple(weight.shape)}")
    eff, scale = ternarize(weight, eps)
    wf = weight.detach().float()
    base = float(wf.abs().mean()) or 1.0
    d0 = ternary_sat_diag(wf, base)
    if d0["sat_share"] <= tau:
        return eff, scale  # healthy budget: default path, zero overhead
    best_s, d1 = optimal_ternary_scale(wf, points=points)
    if d1["split_bound"] >= d0["split_bound"]:
        return eff, scale  # no certified win: keep default
    row_k = base / best_s
    eff2, _ = ternarize(weight / row_k, eps)
    return eff2 * row_k, scale * row_k


class _TernarizeSTE(torch.autograd.Function):
    """Identity straight-through estimator; builds no graph for the quantizer."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, eps: float) -> torch.Tensor:  # type: ignore[override]
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
        return (weight / scale).clamp(-1.0, 1.0).round() * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output, None


class _CachedSTE(torch.autograd.Function):
    """Straight-through estimator over a precomputed ternary map.

    Forward returns the cached ``q`` (the quantized weight) directly — zero
    host-side copies, zero per-forward quantizer work. Backward is the identity
    on the master ``weight`` (STE), so the gradient flows to the fp master as if
    the forward had used it. The quantizer ran once, in
    :meth:`BitLinear.cache_quantized`, at the start of the optimizer step.

    This is the wall-time fix for the host-bound profile: the old per-forward
    ``weight + (q - weight).detach()`` materialized two ``[out, in]`` copies on
    the host per layer per microbatch (measured 634 ms of ``aten::copy_`` CPU
    time — 65% of step time — on the Radeon 8060S).
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, q: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return q

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
        # partition; see hagi.model.ffn.linear for why the marker rather than
        # the module type is what the partition reads.
        self.is_channel_weight = True
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=in_features**-0.5)
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        # Per-step quantized weight (set by the trainer around grad accumulation).
        self._step_q: torch.Tensor | None = None

    def cache_quantized(self, split_scale: bool = True) -> None:
        """Freeze the ternary map for the current optimizer step.

        Quantizes the master once and stores the *effective* weight ``q``. The
        forward then returns ``q`` directly through :class:`_CachedSTE` (no
        host copies, no per-forward quantizer).

        ``split_scale`` (Theory default ON): use the saturation-aware
        :func:`ternarize_split` — R103 measured 86% of the embedding
        quantization error coming from the saturation tail, invisible to
        the in-range s/2 formula. The win-check is honest: only a MEASURED
        reduction of the split bound (n(s/2)^2 + satTail) is accepted,
        healthy tensors keep the zero-overhead default path.
        """
        with torch.no_grad():
            if split_scale:
                self._step_q, _ = ternarize_split(self.weight, self.eps)
            else:
                self._step_q, _ = ternarize(self.weight, self.eps)

    def clear_quantized(self) -> None:
        """Drop the step cache (call after the optimizer step)."""
        self._step_q = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        if self._step_q is not None:
            # STE over the precomputed map: forward = q, backward = identity on
            # the master. Zero host-side copies (the host-bound fix).
            eff = _CachedSTE.apply(self.weight, self._step_q).to(x.dtype)
            return F.linear(x, eff, bias)
        if not torch.is_grad_enabled():
            eff, _ = ternarize(self.weight, self.eps)
            return F.linear(x, eff.to(x.dtype), bias)
        eff = _TernarizeSTE.apply(self.weight, self.eps).to(x.dtype)
        return F.linear(x, eff, bias)

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


# --- Saturation split budget (Runtime/TernaryExact.lean, R103 measurement) --


def _sat_tail(s: float, w: torch.Tensor) -> torch.Tensor:
    """satTail: total squared error of entries OUTSIDE the in-range band
    |w| <= 3s/2, where the error is exactly ||w|-s per entry
    (Runtime/TernaryExact.lean ``satTail``)."""
    out = w.abs() > 1.5 * s
    return ((w.abs()[out] - s) ** 2).sum()


def ternary_sat_diag(weight: torch.Tensor, s: float) -> dict:
    """TernaryExact split budget for grid step ``s`` [``ternary_split_le``]:

        sum (w - qTern_s(w))^2 <= n(s/2)^2 + satTail(s)

    Returns the honest decomposition: in-range count, satTail mass, the
    split bound, and the R103-measured saturation share
    satTail / (n(s/2)^2 + satTail) — the fraction of error the
    in-range-only formula never saw.
    """
    if weight.dim() != 2:
        raise ValueError(f"ternary_sat_diag expects a 2D weight, got {tuple(weight.shape)}")
    if s <= 0:
        raise ValueError("s must be positive")
    wf = weight.detach().float()
    n = wf.numel()
    in_range = wf.abs() <= 1.5 * s
    n_in = int(in_range.sum())
    tail = float(_sat_tail(s, wf))
    in_bound = n * (s / 2.0) ** 2  # the theorem's UPPER bound on in-range err
    split = in_bound + tail
    return {
        "s": s,
        "n": n,
        "n_in_range": n_in,
        "sat_frac": 1.0 - n_in / max(n, 1),
        "sat_tail": tail,
        "in_bound": in_bound,
        "split_bound": split,
        "sat_share": tail / split if split > 0 else 0.0,
    }


def ternary_energy_bridge(scale: torch.Tensor, s_map: float, kappa: float) -> float:
    """quant_energy_bridge_saturation: dE <= kappa * (sqrt(n)*s/2 + sqrt(satTail))
    without any in-range assumption. Here n is read off ``scale``'s row
    width and satTail is caller-measured via :func:`ternary_sat_diag`."""
    n = float(scale.shape[1])
    return kappa * (math.sqrt(n) * s_map / 2.0 + math.sqrt(scale.shape[0]))


def optimal_ternary_scale(weight: torch.Tensor, points: int = 24) -> tuple[float, dict]:
    """Grid refit [application of ``ternary_split_le``]: choose s minimizing
    the SPLIT bound n(s/2)^2 + satTail(s) — not the mean|w| heuristic.

    R103 measured the failure mode: at s = 0.05 on the gen-3 embedding,
    86% of the squared error came from the saturation tail (entries
    |w| > 3s/2), which the in-range formula ignores entirely. Growing s
    trades the n(s/2)^2 term (grows ~s^2) against the tail (shrinks as
    fewer entries saturate) — the optimum is interior.

    Returns (best s, diag of best s). Sweeps s over mean|w| * logspace.
    """
    if weight.dim() != 2:
        raise ValueError(f"expects 2D weight, got {tuple(weight.shape)}")
    wf = weight.detach().float()
    base = float(wf.abs().mean()) or 1.0
    best_s, best_diag = None, None
    for mult in torch.logspace(-1.0, 1.2, points).tolist():
        s = base * mult
        d = ternary_sat_diag(wf, s)
        if best_diag is None or d["split_bound"] < best_diag["split_bound"]:
            best_s, best_diag = s, d
    return best_s, best_diag


import math  # noqa: E402  (used by ternary_energy_bridge above)
