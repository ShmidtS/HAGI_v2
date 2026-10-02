"""Adaptive SafeQP: the direction chosen AFTER seeing the noise.

``Hagi/Step/StochasticSafeQP.lean`` (R92) and its adaptive extension
(R97). R92 conditions the safety margin on a FIXED direction ``d``:

    eps_noise = sigma * ||d|| * sqrt(2 * log(|K| / delta) / m)

That covers a controller which picks its direction in advance. A real
controller does not: it solves the QP against the minibatch estimates
``ĝ_i`` and therefore selects ``d*(ω)`` AFTER seeing the noise. R92
stated this transfer a posteriori, honestly, as an open gap. R97 closes
it.

The obstacle is real: the concentration in R92 is a statement about one
fixed ``d``, and a ``d*`` that depends on the data is not covered by it.
The standard route is a covering number:

1. Build an explicit integer-lattice ``eps_dir``-net of the candidate
   directions (directions with ``||d|| <= D`` and entries in ``[-M, M]``
   live on a grid of ``(2M+1)^n`` points).
2. Prove concentration UNIFORMLY over the grid by a union bound over
   ``2 |K| (2M+1)^n`` events.
3. Transfer from a grid point to the adaptively chosen ``d*`` by
   Lipschitz continuity, which costs a further ``sigma * eps_dir``.

The result is the margin that actually holds for an adaptive step:

    eps_net = sigma * (D + eps_dir)
                * sqrt(2 * log(2 |K| (2M+1)^n / delta) / m)
              + sigma * eps_dir

and the trade-off is explicit. A finer grid (``eps_dir`` small) shrinks
the Lipschitz slack ``sigma * eps_dir`` but inflates the logarithm by
``n * log(1/eps_dir)``, which raises the required minibatch ``m``. So
``eps_dir`` is a computable parameter, not a tolerance to guess --
:func:`optimal_grid_step` picks it by searching that explicit trade-off.

This matters because the gap is not academic: it is the difference
between a controller whose certificate covers what it actually does and
one that is honest only in a regime it never enters.
"""

from __future__ import annotations

import math


def net_log_term(
    n_domains: int, dim: int, bound: int, eps_dir: float, delta: float
) -> float:
    """``log(2 |K| (2M+1)^n / delta)`` -- the union bound over the grid.

    Computed in log space (``n * log(2M+1)``) rather than as
    ``(2M+1)**n``. The direct form overflows a float for the dimensions
    this project actually uses: at n=1152 and M=100 the count is about
    10^1900, so the naive expression raises OverflowError instead of
    returning a number. The logarithm is all that is ever needed.

    Args:
        n_domains: ``K``, the protected domains.
        dim: ``n``, the dimension of the direction space.
        bound: ``M``, the integer grid half-width (entries in ``[-M, M]``).
        eps_dir: the net resolution (present for signature symmetry; the
            grid count depends on ``M``, and ``eps_dir`` enters through
            the Lipschitz term instead).
        delta: the confidence level in ``(0, 1)``.

    Returns:
        The logarithm, as a float.

    Raises:
        ValueError: on non-positive grid parameters or a bad delta.
    """
    if n_domains <= 0 or dim <= 0 or bound < 0:
        raise ValueError("net_log_term needs positive K, n and non-negative M")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if eps_dir <= 0.0:
        raise ValueError("eps_dir must be positive")
    return math.log(2.0 * n_domains / delta) + dim * math.log(2.0 * bound + 1.0)


def adaptive_noise_epsilon(
    noise_bound: float,
    direction_bound: float,
    eps_dir: float,
    minibatch: int,
    n_domains: int,
    dim: int,
    bound: int,
    delta: float,
) -> float:
    """``eps_net`` -- the safety inflation for an ADAPTIVELY chosen d*.

    ``sigma (D + eps_dir) sqrt(2 log(2|K|(2M+1)^n / delta) / m)
     + sigma * eps_dir``

    The first term is the uniform-over-grid concentration; the second is
    the Lipschitz transfer from a grid point to whatever ``d*`` the
    controller actually selected.

    Args:
        noise_bound: ``sigma``, the per-sample noise bound.
        direction_bound: ``D``, an upper bound on ``||d||``.
        eps_dir: the net resolution; smaller is tighter but costs a
            larger grid (see :func:`optimal_grid_step`).
        minibatch: ``m``.
        n_domains: ``K``.
        dim: ``n``, the direction dimension.
        bound: ``M``, the grid half-width.
        delta: the confidence level.

    Returns:
        The required safety inflation.

    Raises:
        ValueError: on non-positive inputs.
    """
    if noise_bound <= 0.0 or direction_bound <= 0.0:
        raise ValueError("adaptive_noise_epsilon needs positive sigma and D")
    if minibatch <= 0:
        raise ValueError("adaptive_noise_epsilon needs a positive minibatch")
    log_term = net_log_term(n_domains, dim, bound, eps_dir, delta)
    concentration = (
        noise_bound
        * (direction_bound + eps_dir)
        * math.sqrt(2.0 * log_term / minibatch)
    )
    return concentration + noise_bound * eps_dir


def fixed_noise_epsilon(
    noise_bound: float, direction_bound: float, minibatch: int,
    n_domains: int, delta: float,
) -> float:
    """R92's margin for a FIXED direction, for side-by-side comparison."""
    return noise_bound * direction_bound * math.sqrt(
        2.0 * math.log(n_domains / delta) / minibatch
    )


def minibatch_for_adaptive_margin(
    noise_bound: float,
    direction_bound: float,
    eps_dir: float,
    n_domains: int,
    dim: int,
    bound: int,
    delta: float,
    target_epsilon: float,
    max_minibatch: int = 1_000_000,
) -> tuple[int, bool]:
    """Smallest ``m`` meeting ``eps_net <= target`` for an adaptive step.

    Solved by bisection on ``m`` rather than algebra: ``eps_net`` mixes
    ``1/sqrt(m)`` with an additive ``sigma*eps_dir`` floor, so the
    equation has no clean closed form once ``eps_dir > 0``. The floor is
    reported honestly: if ``sigma * eps_dir >= target`` no minibatch can
    work, and the caller must coarsen the grid.

    Returns:
        ``(minibatch, achievable)`` as in the R92 port.
    """
    if target_epsilon <= 0.0:
        raise ValueError("minibatch_for_adaptive_margin needs a positive target")
    floor = noise_bound * eps_dir
    if floor >= target_epsilon:
        return 0, False
    lo, hi = 1, max_minibatch
    if adaptive_noise_epsilon(
        noise_bound, direction_bound, eps_dir, hi, n_domains, dim, bound, delta
    ) > target_epsilon:
        return max_minibatch, False
    while lo < hi:
        mid = (lo + hi) // 2
        if adaptive_noise_epsilon(
            noise_bound, direction_bound, eps_dir, mid, n_domains, dim, bound, delta
        ) <= target_epsilon:
            hi = mid
        else:
            lo = mid + 1
    return lo, True


def optimal_grid_step(
    noise_bound: float,
    direction_bound: float,
    n_domains: int,
    dim: int,
    bound: int,
    delta: float,
    target_epsilon: float,
    candidates: tuple[float, ...] = (0.5, 0.25, 0.125, 0.0625, 0.03125,
                                     0.015625, 0.0078125),
) -> tuple[float, int, bool]:
    """Pick ``eps_dir`` by the explicit trade-off, and the ``m`` it needs.

    Two forces pull in opposite directions:

    * a COARSE grid (large ``eps_dir``) shrinks the grid count and so
      the logarithm, but pays a larger Lipschitz slack ``sigma*eps_dir``;
    * a FINE grid shrinks that slack but inflates ``n*log(1/eps_dir)``.

    Neither extreme is right, so this evaluates the candidates and
    returns the feasible one needing the SMALLEST minibatch -- the
    cheapest certificate for a fixed accuracy. This replaces manual
    tuning of a safety margin with arithmetic.

    Returns:
        ``(eps_dir, minibatch, achieved)``. ``achieved`` is False when
        no candidate is feasible, in which case the coarsest candidate
        and its minibatch are still reported for diagnosis.
    """
    best: tuple[float, int, bool] | None = None
    for eps_dir in sorted(candidates, reverse=True):
        m, ok = minibatch_for_adaptive_margin(
            noise_bound, direction_bound, eps_dir, n_domains, dim, bound,
            delta, target_epsilon,
        )
        if ok and (best is None or m < best[1]):
            best = (eps_dir, m, True)
        if best is None and not ok and eps_dir == max(candidates):
            return eps_dir, m, False
    if best is None:
        coarsest = max(candidates)
        m, _ = minibatch_for_adaptive_margin(
            noise_bound, direction_bound, coarsest, n_domains, dim, bound,
            delta, target_epsilon,
        )
        return coarsest, m, False
    return best
