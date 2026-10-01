"""The data axis: when consensus stalls, fresh data reopens the floor.

``Hagi/Step/JointPreserve.lean`` and ``Hagi/Unified/Liveness.lean``.

``diversity_floor_strict_pos``
    If the diversity obeys ``D (t+1) >= rho * D t + inj - xi`` with
    ``0 < rho``, ``0 <= xi < inj`` and ``T != 0``, then

        D T >= rho^T * D 0 + (inj - xi) * sum_{i<T} rho^i

    The closed form of that geometric sum is the module's payoff:
    ``sum_{i<T} rho^i = (1 - rho^T)/(1 - rho)`` for ``rho != 1`` and
    ``T`` at ``rho == 1``. The floor is STRICTLY POSITIVE whenever the
    injection ``inj`` exceeds the decay ``xi`` -- over any number of
    generations, no matter how small ``inj - xi`` is.

``liveness_data_axis``
    Under ``0 < rho`` the result strengthens to ``0 < D T``: a strictly
    positive diversity on every generation.

``liveness_two_axis``
    The loop cannot freeze while EITHER axis is alive -- the merge axis
    (``disagreement => certified gain > 0``) or the data axis
    (``inj > xi => D_T > 0``). Freezing requires BOTH consensus
    (``G = 0``) AND ``inj <= xi``.

This is the operational content of ``ALGORITHMS.md`` §4: when the
certified gain goes to zero, the honest response is not to keep trying
the same action but to inject fresh independent data until the floor
is positive again. :func:`needs_data_injection` is the gate;
:func:`diversity_after` is the forecast.

The decay ``xi`` is the half-life the current corpus mix imposes -- new
data from the same distribution decays as the model learns it, so a
same-domain corpus does NOT satisfy ``inj > xi``. This is the measured
round-66/67 finding (sibling residual CE = ln V, pure noise), and it is
why the criterion is stated on ``inj - xi`` rather than on raw volume.
"""

from __future__ import annotations

import math


def geometric_sum(rho: float, terms: int) -> float:
    """``sum_{i<terms} rho^i``, closed form (the ``diversity_floor`` sum).

    Uses ``(1 - rho^T)/(1 - rho)`` off ``rho == 1`` and ``T`` on it, so the
    caller never has to iterate. At ``rho = 1`` the naive expression is
    0/0, which is exactly the case a fresh-data run sits in (no
    contraction, no expansion).
    """
    if terms <= 0:
        return 0.0
    if abs(rho - 1.0) < 1e-15:
        return float(terms)
    if abs(rho) < 1e-15:
        return 1.0 if terms >= 1 else 0.0
    return (1.0 - rho**terms) / (1.0 - rho)


def diversity_after(
    d0: float, rho: float, injection: float, decay: float, generations: int
) -> float:
    """``D_T >= rho^T D_0 + (inj - xi) * sum_{i<T} rho^i`` (floor).

    Args:
        d0: the initial diversity ``D 0``.
        rho: the per-generation retention ``0 < rho``.
        injection: ``inj`` -- the fresh diversity injected per generation.
        decay: ``xi`` -- the diversity lost per generation.
        generations: ``T``, strictly positive.

    Returns:
        The certified LOWER bound on diversity after ``generations``.

    Raises:
        ValueError: when ``generations <= 0`` or ``rho <= 0``.
    """
    if generations <= 0:
        raise ValueError("diversity_after needs a positive generation count")
    if rho <= 0.0:
        raise ValueError("diversity_after needs a positive rho (liveness_data_axis)")
    return rho**generations * d0 + (injection - decay) * geometric_sum(rho, generations)


def needs_data_injection(gain: float, injection: float, decay: float) -> bool:
    """The §4 gate: consensus AND ``inj <= xi`` means the loop is frozen.

    Args:
        gain: the certified gain ``G`` of the current action set.
        injection: ``inj``.
        decay: ``xi``.

    Returns:
        True iff the honest stopping condition of ``liveness_two_axis``
        holds and fresh independent data is required to reopen the
        merge axis. Two axes must BOTH be dead: a positive gain means
        the merge axis is still alive and no injection is needed.
    """
    return gain <= 0.0 and injection <= decay


def generations_to_reopen(
    target: float, rho: float, injection: float, decay: float, cap: int = 100_000
) -> int:
    """Smallest ``T`` with the diversity floor above ``target``.

    Binary search over the closed form rather than simulating the
    recursion: the floor is increasing in ``T`` whenever ``inj > xi``,
    so the search is well posed.

    Returns:
        The generation count, or ``-1`` when ``inj <= xi`` (no finite T
        suffices -- the honest answer is that this axis is dead and no
        amount of waiting reopens it).

    Raises:
        ValueError: on non-positive ``rho``.
    """
    if rho <= 0.0:
        raise ValueError("generations_to_reopen needs a positive rho")
    if injection <= decay:
        return -1
    lo, hi = 1, cap
    if diversity_after(0.0, rho, injection, decay, hi) < target:
        return -1
    while lo < hi:
        mid = (lo + hi) // 2
        if diversity_after(0.0, rho, injection, decay, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo
