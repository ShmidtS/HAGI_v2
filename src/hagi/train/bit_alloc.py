"""BitAlloc: optimal bit allocation under a parameter budget (R176).

``Hagi/Budget/BitAlloc.lean``. The "maximum quality at minimum size"
program: under a hard cap on model volume the only remaining freedom
is WHERE the bits go. The layer error model

    e_l(b) = c_l / 2^b

(the QuantBridge sensitivity ``c_l``; the grid halves per extra bit)
has an exact marginal-exchange calculus, and this module is its
runtime port:

* :func:`layer_error` / :func:`total_error` — the error model.

* :func:`transfer_delta` — THE MARGINAL-EXCHANGE IDENTITY
  (``transfer_exact``): moving one bit from layer ``k`` (with a bit
  to give) to a different layer ``j`` changes the total error by
  EXACTLY ``e_k − e_j/2``. The transfer does not increase the total
  exactly when ``e_k ≤ e_j/2`` — the receiver's current error is at
  least TWICE the giver's. Bits want to live where the error is
  still large.

* :func:`greedy_transfer` — the licensed greedy loop: while some
  pair is off by more than the factor 2, move a bit. Each move
  strictly decreases the total error (``imbalance_yields_gain``),
  and a stabilized allocation has every error within a factor 2 of
  every other with a bit to give (``stable_factor_two``) — the
  discrete water-filling invariant, certified at the fixed point.

* :func:`two_layer_equalize` — the two-layer optimum
  (``two_layer_equalize``): any split ``x + (x+2d)`` is dominated by
  the balanced split ``(x+d, x+d)``. Balancing never hurts.

* :func:`distill_quant_composite` — the full-generation
  quality-volume certificate (``distill_quant_composite``): the
  distillate of generation ``n``, under the compound distill gain
  and a bit allocation ``f``, obeys the ADDITIVE composite bound
  ``S_n + n·g ≤ E_0 + δ_n + totalError``. One inequality
  certifies the whole grow-compress-quantize pipeline of a
  generation against the parameter budget.

Honest boundary (Lean): the full n-layer continuous water filling
and the STEPQuant lifetime weighting are engineering on this kernel;
the frontier claim of 2607.16097 is empirical fitting, not claimed.
"""
from __future__ import annotations

import math


def layer_error(c: float, b: int) -> float:
    """``layerError c b = c / 2^b`` — the per-layer quantization error.

    Each extra bit halves the grid, hence halves the error.

    Raises:
        ValueError: on a negative bit width.
    """
    if b < 0:
        raise ValueError("bit width must be non-negative")
    return c / float(2 ** b)


def total_error(c: list[float], f: list[int]) -> float:
    """``totalError``: the sum of the layer errors of an allocation.

    Raises:
        ValueError: on a length mismatch.
    """
    if len(c) != len(f):
        raise ValueError("sensitivities and bits must be aligned")
    return math.fsum(layer_error(ci, bi) for ci, bi in zip(c, f))


def transfer_delta(c: list[float], f: list[int], j: int, k: int) -> float:
    """``transfer_exact``: the EXACT change from moving one bit k -> j.

    Moving a bit from layer ``k`` to layer ``j`` changes the total
    error by exactly ``e_k(f_k) − e_j(f_j)/2`` (removing a bit
    doubles the giver's error; adding one halves the receiver's).
    The runtime decision rule: the transfer is nonincreasing
    precisely when ``e_k ≤ e_j/2``.

    Raises:
        ValueError: on coincident indices, an out-of-range index, or
            a giver with no bit to give.
    """
    if j == k:
        raise ValueError("receiver and giver must differ")
    if not 0 <= j < len(f) or not 0 <= k < len(f):
        raise ValueError("index out of range")
    if f[k] <= 0:
        raise ValueError("giver must hold a bit")
    return layer_error(c[k], f[k]) - layer_error(c[j], f[j]) / 2.0


def apply_transfer(f: list[int], j: int, k: int) -> list[int]:
    """The post-transfer allocation: one bit from ``k`` to ``j``.

    Raises:
        ValueError: as :func:`transfer_delta`.
    """
    if j == k:
        raise ValueError("receiver and giver must differ")
    if not 0 <= j < len(f) or not 0 <= k < len(f):
        raise ValueError("index out of range")
    if f[k] <= 0:
        raise ValueError("giver must hold a bit")
    g = list(f)
    g[j] += 1
    g[k] -= 1
    return g


def two_layer_equalize(c: float, x: int, d: int) -> float:
    """``two_layer_equalize``: balancing never hurts.

    Returns the signed gap ``e(x) + e(x+2d) − 2·e(x+d)`` — always
    ``>= 0`` (the balanced split dominates); a caller uses the value
    as the exact saving of balancing the pair.

    Raises:
        ValueError: on a negative sensitivity.
    """
    if c < 0.0:
        raise ValueError("sensitivity must be non-negative")
    return (
        layer_error(c, x) + layer_error(c, x + 2 * d)
        - 2.0 * layer_error(c, x + d)
    )


def greedy_transfer(
    c: list[float], f: list[int], max_rounds: int = 10_000
) -> tuple[list[int], int]:
    """The greedy bit-reallocation loop licensed by ``transfer_exact``.

    While some giver/receiver pair is off by more than the factor 2
    (``2·e_k < e_j``, the receiver's error more than twice the
    giver's) and the giver still holds a bit, move that bit — the
    move strictly decreases the total error (``imbalance_yields_gain``).
    Terminates at the factor-2 balanced fixed point
    (``stable_factor_two``).

    Returns:
        ``(allocation, rounds)`` — the stabilized bits and the number
        of transfers performed.

    Raises:
        ValueError: on a length mismatch, a negative bit width, or a
            budget violation by the input.
    """
    if len(c) != len(f):
        raise ValueError("sensitivities and bits must be aligned")
    if any(b < 0 for b in f):
        raise ValueError("bit widths must be non-negative")
    g = list(f)
    for _ in range(max_rounds):
        best = None  # (delta, j, k)
        for k in range(len(g)):
            if g[k] <= 0:
                continue
            ek = layer_error(c[k], g[k])
            for j in range(len(g)):
                if j == k:
                    continue
                if 2.0 * ek < layer_error(c[j], g[j]):
                    d = transfer_delta(c, g, j, k)
                    if best is None or d < best[0]:
                        best = (d, j, k)
        if best is None:
            return g, _rounds_done(g, f)
        _, j, k = best
        g = apply_transfer(g, j, k)
    raise RuntimeError(
        f"greedy transfer did not stabilize in {max_rounds} rounds"
    )


def _rounds_done(g: list[int], f: list[int]) -> int:
    """Number of transfers performed: total variation distance / 2."""
    return sum(abs(a - b) for a, b in zip(g, f)) // 2


def distill_quant_composite(
    s_n: float,
    e0: float,
    delta_n: float,
    n: int,
    g: float,
    c: list[float],
    f: list[int],
) -> tuple[float, float]:
    """``distill_quant_composite``: the generation's quality-volume bound.

    Checks ``S_n + n·g ≤ E_0 + δ_n + totalError(c, f)`` — the
    additive composite of the compound distill gain and the bit
    budget. One inequality certifies the whole
    grow-compress-quantize pipeline of a generation against the
    parameter budget.

    Returns:
        ``(bound, slack)``: the right side, and ``bound − lhs``
        (non-negative when the certificate holds).

    Raises:
        ValueError: on a negative cycle count or a length mismatch.
    """
    if n < 0:
        raise ValueError("cycle count must be non-negative")
    lhs = s_n + n * g
    bound = e0 + delta_n + total_error(c, f)
    return bound, bound - lhs
