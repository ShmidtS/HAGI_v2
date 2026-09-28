"""Formalization-derived utilities (Hagi Lean program -> code).

Each function is the executable form of a proven theorem; the docstring
cites the source. No fitted constants live here: every threshold comes
from a proof (see .omc/attempts/growth_gate_v2_round14.md).

1. Mix.lean ``mixed_invisible``: a row rotation Q applied to the
   concat hidden stream is invisible to the logits iff its column
   sums reproduce the heads (for a shared head: sums == 1). A mixer
   violating this transfers mass between leaves -- the regime where
   it must be TRAINED, not trusted.
2. KVWater.lean ``waterfilling_bound``: under a fixed bit budget the
   total KV residual sum c_i * exp(-kappa * b_i) is minimized by
   equalizing the marginal residuals: bits grow LOGARITHMICALLY in
   the sensitivity c_i (b*_i - b*_j = (log c_i - log c_j)/kappa),
   NOT proportionally. Closed form -> no allocation sweep.
3. Select.lean ``certifiedGain``: (M - c)/(N+1), the certified new
   mean-CE bound gain of adding a candidate -- used by
   scripts/growth_gate.py; re-exported here as the canonical home.
"""
from __future__ import annotations

import math

import torch


def mixer_invisible_condition(q: torch.Tensor) -> bool:
    """Mix.lean mixed_invisible: check the visibility invariant.

    Args:
        q: the ``[n, n]`` rotation applied to the concat stream's
            block axis (e.g. ``H_n / sqrt(n)`` from the Hadamard
            mixer, an F3 character matrix, any learned rotation).

    Returns:
        True iff the rotation is *logit-invisible* for a shared head:
        every column of Q sums to 1, i.e. the head's per-leaf readout
        sees each leaf's own contribution exactly. Formally:
        ``forall b, sum_a Q[a, b] = 1`` (the common-head case of
        ``sum_a Q[a, b] * W_a = W_b`` -- when all W_a are the same
        weight, this reduces to the column sums being 1).

    Notes:
        The Sylvester Hadamard ``H_n / sqrt(n)`` FAILS this for
        n > 1 (columns sum to +-1/sqrt(n) != 1) -- which is exactly
        why the Hadamard mixer needs the pre-rotated head ``WQ`` and
        cannot be trusted bare on the ensemble path. The identity and
        the parent-preserving lifts PASS. Use this to gate any mixer
        that claims to preserve the free-ensemble logits.
    """
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError(f"expected a square rotation, got {q.shape}")
    sums = q.sum(dim=0)
    return bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))


def kv_waterfill_bits(
    sensitivities: torch.Tensor,
    total_bits: float,
    kappa: float,
) -> torch.Tensor:
    """KVWater.lean waterfilling_bound: the optimal bit allocation.

    Minimizes ``sum_i c_i * exp(-kappa * b_i)`` subject to
    ``sum_i b_i = total_bits`` (the theorem's marginal-residual
    equalization): the optimum equalizes ``c_i * exp(-kappa * b_i)``,
    so ``b*_i = (log c_i - log lambda) / kappa`` with the water
    level ``log lambda`` fixed by the budget.

    Args:
        sensitivities: ``c_i >= 0`` -- per-position KV sensitivity
            (e.g. softmax mass * value norm; the theorem's
            contribution-weighted scale).
        total_bits: the total bit budget across positions.
        kappa: the residual decay rate per bit (quantization error
            decays exponentially with precision).

    Returns:
        The optimal per-entry bit allocation ``b*_i`` (float; floor to
        integer grids at the call site). Entries with ``c_i = 0``
        get 0 bits -- zero sensitivity carries zero residual weight.
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if total_bits < 0:
        raise ValueError("total_bits must be nonnegative")
    c = sensitivities.float().clamp_min(0)
    n = c.numel()
    if n == 0 or total_bits == 0.0:
        return torch.zeros_like(c)
    log_c = torch.where(c > 0, c.log(), torch.full_like(c, -math.inf))
    # b_i = (log c_i - log lambda) / kappa; budget fixes lambda:
    # sum b_i = B  =>  (sum log c_i - n log lambda)/kappa = B.
    finite = torch.isfinite(log_c)
    k = int(finite.sum())
    if k == 0:
        # all-zero sensitivities: uniform split is optimal (all
        # marginal residuals equal at any allocation).
        return torch.full_like(c, total_bits / n)
    s = float(log_c[finite].sum())
    log_lambda = (s - kappa * total_bits) / k
    b = (log_c - log_lambda) / kappa
    # zero-sensitivity entries: their residual weight is 0 at any b;
    # the theorem's infimum puts their bits at 0 (transfer to useful).
    b = torch.where(finite, b, torch.zeros_like(b))
    b = b.clamp_min(0)
    return b


def certified_gain(n: int, mean_ce: float, candidate_ce: float) -> float:
    """Select.lean newBound / certifiedGain: the pre-step bound.

    Adding a candidate with standalone CE ``c`` to a pool of ``n``
    leaves with mean-CE ``M`` certifies the new bound
    ``(n*M + c)/(n+1)``; the gain over the current bound is exactly
    ``(M - c)/(n+1)``. Monotone in c (certifiedGain_monotone), so the
    best candidate certifies for all (grow_epsilon_stop).
    """
    return (mean_ce - candidate_ce) / (n + 1)
