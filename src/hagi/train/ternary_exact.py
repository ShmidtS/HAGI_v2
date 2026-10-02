"""Honest ternary compression: the saturation tail is measured (R103).

``Hagi/Runtime/TernaryExact.lean``. R102's FIX 1 restored the ``sqrt(n)``
that was hiding behind ``Real.sqrt 1``, but the compression bound still
carried a premise this project cannot honour: ``∀i |w_i − q_i| ≤ s/2``.
That is the IN-RANGE guarantee, and it is false for a real saturating
quantiser as soon as a coordinate exceeds ``3s/2``.

R103 removes the premise instead of defending it.

``qTern_error_out``
    For ``|x| > 3s/2`` the error is exactly ``|x| − s`` -- the saturating
    law. This is precisely where the old hypothesis broke, so it is
    stated as a law rather than as an exception nobody expected.

``ternary_split_bound``
    An IDENTITY, not a bound: the total squared error equals the
    in-range part plus ``satTail``, the sum of squared overshoots. Every
    nat of the error is accounted for by name.

``satTail``
    ``Σ_{|w| > 3s/2} (|w| − s)²`` -- computable from a checkpoint before
    quantising anything.

``quant_energy_bridge_saturation``
    ``ΔE ≤ κ(√n·s/2 + √satTail)`` with NO in-range assumption.

``no_saturation_recover``
    When ``satTail = 0`` the old hypothesis is DERIVED, not assumed. So
    the earlier bridge was not wrong, it was conditional -- and this
    says exactly what the condition was.

The operational consequence, and the reason this matters more than the
arithmetic: comparing ``κ·√satTail`` against the cost of widening the
grid, clipping, or splitting the group turns "should I re-calibrate the
quantiser?" into a comparison on a checkpoint that already exists,
instead of a sweep of GPU experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def in_range(scale: float, x: float) -> bool:
    """``|x| ≤ 3s/2`` -- in range for nearest-grid ternary rounding.

    The coordinate is within half a step of the endpoint level ``s``, so
    the saturating quantiser still covers it.

    Raises:
        ValueError: on a non-positive scale.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return abs(x) <= 1.5 * scale


def qtern_error(scale: float, x: float) -> float:
    """``|x − qTern(s, x)|`` -- the exact per-coordinate error.

    Two regimes, both exact:

    - in range (``|x| ≤ 3s/2``): at most ``s/2``;
    - saturated (``|x| > 3s/2``): exactly ``|x| − s``, because the
      quantiser clips to the endpoint level.

    Raises:
        ValueError: on a non-positive scale.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if in_range(scale, x):
        # nearest ternary level in {-1,0,1}*s
        ratio = x / scale
        level = round(ratio)
        level = max(-1.0, min(1.0, level))
        return abs(x - level * scale)
    return abs(abs(x) - scale)


def sat_tail(weights: torch.Tensor, scale: float) -> float:
    """``Σ_{|w| > 3s/2} (|w| − s)²`` -- the saturation mass.

    Computable from a checkpoint BEFORE quantising, which is what makes
    it a diagnostic rather than a post-mortem.

    Args:
        weights: the weight tensor.
        scale: the grid step ``s > 0``.

    Returns:
        The squared overshoot mass, non-negative.

    Raises:
        ValueError: on a non-positive scale.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    w = weights.double().flatten()
    mag = w.abs()
    out = mag > 1.5 * scale
    if not bool(out.any()):
        return 0.0
    over = (mag[out] - scale) ** 2
    return float(over.sum())


@dataclass(frozen=True)
class QuantSplit:
    """The exact decomposition of a quantiser's squared error.

    Attributes:
        total: ``Σ (w − q)²`` over every coordinate.
        in_range: the share covered by the half-step guarantee.
        tail: the saturation share, ``satTail``.
    """

    total: float
    in_range: float
    tail: float

    @property
    def splits_exactly(self) -> bool:
        """``total == in_range + tail`` -- the identity, checked."""
        return abs(self.total - (self.in_range + self.tail)) <= 1e-9 * max(
            1.0, self.total
        )


def ternary_split(weights: torch.Tensor, scale: float) -> QuantSplit:
    """``ternary_split_bound``: total error = in-range part + satTail.

    An IDENTITY, computed both sides independently so the check is real.

    Args:
        weights: the weight tensor.
        scale: the grid step.

    Returns:
        The decomposition.

    Raises:
        ValueError: on a non-positive scale.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    w = weights.double().flatten()
    total = 0.0
    inr = 0.0
    for v in w.tolist():
        e = qtern_error(scale, v)
        total += e * e
        if in_range(scale, v):
            inr += e * e
    tail = sat_tail(weights, scale)
    return QuantSplit(total=total, in_range=inr, tail=tail)


def compression_cost(weights: torch.Tensor, scale: float,
                     kappa: float) -> float:
    """``κ(√n·s/2 + √satTail)`` -- the honest energy charge.

    ``quant_energy_bridge_saturation``, with NO in-range assumption. The
    old form was ``κ√n·s/2``, which silently charged nothing for
    saturation -- i.e. it was right exactly when ``satTail = 0``.

    Args:
        weights: the weight tensor, whose size gives ``n``.
        scale: the grid step.
        kappa: the compression constant.

    Returns:
        The energy charge, non-negative.

    Raises:
        ValueError: on a non-positive scale or a negative kappa.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    n = int(weights.numel())
    return kappa * (math.sqrt(n) * scale / 2.0
                     + math.sqrt(sat_tail(weights, scale)))


def no_saturation_recover(weights: torch.Tensor, scale: float) -> bool:
    """``no_saturation_recover``: the old hypothesis is DERIVED here.

    When nothing saturates, ``satTail = 0`` and the R70 bridge's premise
    ``∀i |w_i − q_i| ≤ s/2`` genuinely holds. So the earlier bound was
    not wrong, it was conditional -- and this reports whether the
    condition is met.

    Args:
        weights: the weight tensor.
        scale: the grid step.

    Returns:
        True iff every coordinate is inside ``3s/2``.
    """
    w = weights.double().flatten()
    return bool((w.abs() <= 1.5 * scale).all())


def recalibrate_pays(weights: torch.Tensor, scale: float, kappa: float,
                     new_scale: float) -> tuple[bool, float, float]:
    """Recalibrate the grid, or live with the tail?

    The runtime decision the theorem enables, as a comparison rather
    than a sweep. Both costs are computable from the checkpoint that
    exists:

    - live with it: ``κ·√satTail(s)`` at the current grid;
    - recalibrate: ``κ·√satTail(s')`` at a wider grid, where the tail
      is recomputed against ``s'``.

    Returns ``(recalibrate, cost_now, cost_new)`` -- True iff widening
    the grid is strictly cheaper.

    Args:
        weights: the weight tensor.
        scale: the current grid step.
        kappa: the compression constant.
        new_scale: the candidate wider grid step.

    Returns:
        The decision and both costs.

    Raises:
        ValueError: on a non-positive scale or new_scale.
    """
    if scale <= 0.0 or new_scale <= 0.0:
        raise ValueError("scales must be positive")
    now = kappa * math.sqrt(sat_tail(weights, scale))
    new = kappa * math.sqrt(sat_tail(weights, new_scale))
    return new < now, now, new