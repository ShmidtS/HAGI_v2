"""R94: top-k routing is optimal, and the geometric pool is honest.

``Hagi/Unified/RecursiveGrowth.lean`` + ``Hagi/Energy/FreeEnergy.lean``.

This round was announced with "the brute-force proof is incomplete" and
carried over several sessions as an open item. Reading the Lean source
shows both theorems are in fact closed there, so the port can be exact
rather than a reconstruction.

``topk_routing_optimal``
    Given coefficients ``c`` and a set ``s`` of size ``k`` such that every
    selected coordinate dominates every dropped one (``c_j² ≤ c_i²`` for
    ``i ∈ s``, ``j ∉ s``), any other set ``s'`` of size ``k`` has at
    least as large a complement:

        sum over sᶜ of c²  <=  sum over s'ᶜ of c²

    The exchange argument: both sets drop the same number of
    coordinates, and swapping shows the top-k complement carries the
    least energy. So dropping the smallest ``|c|`` is optimal, and no
    router can do better at a fixed budget.

``geometric_pool_identity_nonneg``
    A weighted geometric pool of distributions satisfies

        sum_i w_i KL(q || p_i)
          = KL(q || [prod_i p_i^w_i / sum_u prod_i p_i^u])
            - log(sum_u prod_i p_i^u)

    The right-hand bracket is ``log`` of the pool's normaliser, which is
    the KL between the arithmetic and geometric means of the pool --
    non-negative. So pooling costs at least the excess of the geometric
    over the arithmetic mixture: the ``0`` is achieved only when the
    members agree.

Both are the same shape of statement this project keeps needing: a
mixture's cost is not free, and the amount by which it is not free is a
named, non-negative quantity.
"""

from __future__ import annotations

import math
from itertools import combinations


def tail_energy(coeffs: list[float], keep: set[int]) -> float:
    """``sum over the complement of c²`` -- the energy a subset drops.

    Args:
        coeffs: ``[c_0, ..., c_{n-1}]``; only ``c²`` matters, so the sign
            is irrelevant and the theorem is stated on squares.
        keep: the retained indices.

    Returns:
        The dropped squared energy.

    Raises:
        ValueError: on an out-of-range index.
    """
    n = len(coeffs)
    for i in keep:
        if not 0 <= i < n:
            raise ValueError(f"index {i} out of range for {n} coefficients")
    return sum(c * c for i, c in enumerate(coeffs) if i not in keep)


def is_dominating(coeffs: list[float], keep: set[int]) -> bool:
    """``∀ i ∈ s, ∀ j ∉ s : c_j² ≤ c_i²``.

    The hypothesis of ``topk_routing_optimal``, checked on the numbers
    rather than assumed -- the optimality is only meaningful when the
    retained set really does dominate.

    Raises:
        ValueError: on an out-of-range index.
    """
    n = len(coeffs)
    for i in keep:
        if not 0 <= i < n:
            raise ValueError(f"index {i} out of range for {n} coefficients")
    for i in keep:
        for j in range(n):
            if j in keep:
                continue
            if coeffs[j] ** 2 > coeffs[i] ** 2:
                return False
    return True


def topk_tail(coeffs: list[float], k: int) -> float:
    """The minimal tail energy over all subsets of size ``k``.

    Brute force over every subset, which is the check the Lean proof's
    exchange argument replaces: for a handful of coordinates this
    enumerates them all, and it is the port's own verification of the
    theorem rather than a restatement of it.

    Args:
        coeffs: the coefficients.
        k: how many to retain.

    Returns:
        ``(minimal tail energy, the optimal subset)``.

    Raises:
        ValueError: on an out-of-range ``k``.
    """
    n = len(coeffs)
    if k <= 0 or k > n:
        raise ValueError(f"k must lie in (0, {n}], got {k}")
    best = None
    best_set: set[int] = set()
    for combo in combinations(range(n), k):
        tail = tail_energy(coeffs, set(combo))
        if best is None or tail < best - 1e-15:
            best = tail
            best_set = set(combo)
    assert best is not None
    return best, best_set


def topk_routing_optimal(coeffs: list[float], keep: set[int]) -> bool:
    """``topk_routing_optimal``, checked by exhaustive comparison.

    Returns True iff the retained set dominates AND no other subset of
    the same size leaves less energy behind. The second half is what the
    theorem actually asserts; the first is its hypothesis, and returning
    False when it fails is the port refusing to certify anything.

    Raises:
        ValueError: on an out-of-range index.
    """
    n = len(coeffs)
    if not keep or len(keep) > n:
        return False
    if not is_dominating(coeffs, keep):
        return False
    best, best_set = topk_tail(coeffs, len(keep))
    return abs(tail_energy(coeffs, keep) - best) <= 1e-12


def minimal_topk_tail(coeffs: list[float], k: int) -> float:
    """The routing cost of a budget ``k``: ``sum of the k smallest squares``.

    Convenience wrapper over :func:`topk_tail` for callers that only want
    the number -- the gate-law module uses it the same way.

    Raises:
        ValueError: on an out-of-range ``k``.
    """
    return topk_tail(coeffs, k)[0]


# --- geometric pool (R94) ------------------------------------------------


def kl(p: list[float], q: list[float]) -> float:
    """``KL(p || q)`` in nats.

    Raises:
        ValueError: on a length mismatch or a zero entry in ``q``.
    """
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    total = 0.0
    for pi, qi in zip(p, q):
        if qi <= 0.0:
            raise ValueError("KL needs strictly positive q everywhere")
        if pi > 0.0:
            total += pi * math.log(pi / qi)
    return total


def geometric_pool(pools: list[list[float]], weights: list[float]
                   ) -> tuple[list[float], float]:
    """The geometric mean of the pool and its normaliser.

    Args:
        pools: the member distributions, each summing to 1.
        weights: the mixing weights, summing to 1 and non-negative.

    Returns:
        ``(pooled distribution, log normaliser)``.

    Raises:
        ValueError: on a malformed pool, zero weights, or a zero entry.
    """
    if not pools or len(pools) != len(weights):
        raise ValueError("pools and weights must be non-empty and aligned")
    v = len(pools[0])
    for p in pools:
        if len(p) != v:
            raise ValueError("all pools must have the same support size")
        for x in p:
            if x <= 0.0:
                raise ValueError("the pool requires strictly positive entries")
    if any(w < 0.0 for w in weights):
        raise ValueError("weights must be non-negative")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {sum(weights)}")

    unnormalised = []
    for j in range(v):
        prod = 1.0
        for p, w in zip(pools, weights):
            prod *= p[j] ** w
        unnormalised.append(prod)
    norm = sum(unnormalised)
    log_norm = math.log(norm)
    return [u / norm for u in unnormalised], log_norm


def pool_excess(q: list[float], pools: list[list[float]],
                weights: list[float]) -> float:
    """``sum_i w_i KL(q || p_i) − KL(q || pooled)``.

    ``geometric_pool_identity_nonneg``: this equals the log of the pool's
    normaliser, i.e. ``KL(arithmetic || geometric)`` -- the price of
    pooling. Non-negative, and zero exactly when the members agree.

    Raises:
        ValueError: on a malformed pool or a zero entry in ``q``.
    """
    for x in q:
        if x <= 0.0:
            raise ValueError("the reference q must be strictly positive")
    weighted = sum(w * kl(q, p) for p, w in zip(pools, weights))
    pooled, _ = geometric_pool(pools, weights)
    return weighted - kl(q, pooled)


def pool_is_free(weights: list[float], pools: list[list[float]],
                 tol: float = 1e-12) -> bool:
    """Is pooling free? Only when every member is the same distribution.

    Raises:
        ValueError: on a malformed pool.
    """
    pooled, log_norm = geometric_pool(pools, weights)
    return abs(log_norm) <= tol