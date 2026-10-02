"""Tests for top-k routing optimality and the geometric pool (R94).

Both theorems were carried as "the brute-force proof is incomplete" for
several sessions. Reading the Lean source showed them closed there, so
this port is exact rather than a reconstruction -- and one of these
tests IS the brute force, used to verify the exchange argument rather
than to replace it.
"""

from __future__ import annotations

import math
import random
from itertools import combinations

import pytest

from hagi.train.routing_optimal import (
    geometric_pool,
    is_dominating,
    kl,
    minimal_topk_tail,
    pool_excess,
    pool_is_free,
    tail_energy,
    topk_routing_optimal,
    topk_tail,
)


def sq(vals):
    return [v * v for v in vals]


# --- tail_energy ---------------------------------------------------------


def test_the_tail_is_the_dropped_squares():
    c = [3.0, 1.0, 4.0, 1.0]
    assert tail_energy(c, {0, 2}) == pytest.approx(1.0 + 1.0)


def test_keeping_everything_leaves_no_tail():
    assert tail_energy([1.0, 2.0], {0, 1}) == 0.0


def test_the_sign_of_a_coefficient_does_not_matter():
    """The theorem is stated on squares; only ``|c|`` is used."""
    assert tail_energy([3.0, -1.0], {0}) == pytest.approx(1.0)


def test_an_out_of_range_index_is_refused():
    with pytest.raises(ValueError):
        tail_energy([1.0, 2.0], {5})


# --- the optimality theorem, verified by brute force --------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_top_k_is_optimal_for_every_subset_size(k):
    """``topk_routing_optimal``, checked by enumerating all subsets."""
    c = [5.0, 1.0, 4.0, 2.0, 3.0]
    best, best_set = topk_tail(c, k)
    assert topk_routing_optimal(c, best_set)
    assert tail_energy(c, best_set) == pytest.approx(best)
    # and no other subset of size k does better
    for combo in combinations(range(len(c)), k):
        assert tail_energy(c, set(combo)) >= best - 1e-12


@pytest.mark.parametrize("seed", range(12))
def test_top_k_is_optimal_on_random_coefficients(seed):
    rng = random.Random(seed)
    c = [rng.uniform(-5, 5) for _ in range(6)]
    for k in (1, 2, 3):
        _, best_set = topk_tail(c, k)
        assert topk_routing_optimal(c, best_set)


def test_a_non_dominating_set_is_not_certified():
    """The theorem is conditional on domination; the port refuses without it."""
    c = [5.0, 1.0, 4.0]
    assert is_dominating(c, {0}) is True
    # keeping the smallest is dominated by the largest
    assert is_dominating(c, {1}) is False
    assert topk_routing_optimal(c, {1}) is False


def test_a_wrong_size_set_is_not_certified():
    c = [5.0, 1.0, 4.0]
    assert topk_routing_optimal(c, set()) is False


def test_the_optimal_tail_decreases_with_the_budget():
    """More branches kept means strictly less energy dropped."""
    c = [5.0, 1.0, 4.0, 2.0, 3.0]
    tails = [minimal_topk_tail(c, k) for k in range(1, 6)]
    assert tails == sorted(tails, reverse=True)
    assert tails[-1] == pytest.approx(0.0)


def test_ties_are_broken_deterministically():
    c = [1.0, 1.0, 1.0, 2.0]
    a = topk_tail(c, 2)
    b = topk_tail(c, 2)
    assert a == b


def test_an_out_of_range_budget_is_refused():
    with pytest.raises(ValueError):
        topk_tail([1.0, 2.0], 0)
    with pytest.raises(ValueError):
        topk_tail([1.0, 2.0], 3)


def test_the_optimal_set_drops_the_smallest_magnitudes():
    c = [5.0, 1.0, 4.0, 2.0]
    _, best_set = topk_tail(c, 2)
    assert best_set == {0, 2}


# --- KL -----------------------------------------------------------------


def test_kl_of_a_distribution_with_itself_is_zero():
    p = [0.2, 0.3, 0.5]
    assert kl(p, p) == pytest.approx(0.0, abs=1e-15)


def test_kl_is_non_negative():
    p = [0.5, 0.5]
    q = [0.25, 0.75]
    assert kl(p, q) >= 0.0


def test_kl_is_asymmetric():
    p = [0.9, 0.1]
    q = [0.5, 0.5]
    assert kl(p, q) != pytest.approx(kl(q, p))


def test_a_zero_in_the_reference_is_refused():
    with pytest.raises(ValueError):
        kl([0.5, 0.5], [0.0, 1.0])


def test_a_length_mismatch_is_refused():
    with pytest.raises(ValueError):
        kl([0.5, 0.5], [1.0])


# --- the geometric pool -------------------------------------------------


def test_a_pool_of_identical_members_is_free():
    p = [0.25, 0.75]
    pooled, log_norm = geometric_pool([p, p, p], [1 / 3, 1 / 3, 1 / 3])
    assert log_norm == pytest.approx(0.0, abs=1e-15)
    assert pool_is_free([1 / 3] * 3, [p, p, p]) is True


def test_a_pool_of_different_members_costs_something():
    """Pooling is not free, and the SIGN is what AM-GM fixes.

    ``Z = sum_u prod_i p_i[u]^w_i <= 1`` by the arithmetic-geometric
    mean inequality, so ``log Z <= 0``. The non-negative quantity the
    Lean name refers to is therefore the EXCESS
    ``sum w_i KL(q||p_i) - KL(q||pooled) = -log Z``, not ``log Z``
    itself. Checking ``log_norm > 0`` would assert the opposite of the
    theorem.
    """
    pools = [[0.2, 0.8], [0.6, 0.4]]
    _, log_norm = geometric_pool(pools, [0.5, 0.5])
    assert log_norm < 0.0                      # Z < 1 by AM-GM
    assert pool_excess([0.4, 0.6], pools, [0.5, 0.5]) == \
        pytest.approx(-log_norm)
    assert pool_is_free([0.5, 0.5], pools) is False


def test_the_pool_is_normalised():
    pooled, _ = geometric_pool([[0.2, 0.8], [0.6, 0.4]], [0.5, 0.5])
    assert sum(pooled) == pytest.approx(1.0, abs=1e-15)


def test_a_zero_weight_member_is_ignored():
    """``p^0 = 1``: a zero weight must drop its member from the product."""
    p = [0.25, 0.75]
    _, log_norm = geometric_pool([p, [0.5, 0.5]], [1.0, 0.0])
    assert log_norm == pytest.approx(0.0, abs=1e-15)


# --- the identity: the excess is non-negative ---------------------------


@pytest.mark.parametrize("seed", range(10))
def test_the_pooling_excess_is_non_negative(seed):
    """``geometric_pool_identity_nonneg``: the pool never costs less than
    the weighted average of its members' KL."""
    rng = random.Random(seed)
    v = 4
    pools = []
    for _ in range(3):
        raw = [rng.uniform(0.05, 1.0) for _ in range(v)]
        s = sum(raw)
        pools.append([x / s for x in raw])
    q = [rng.uniform(0.05, 1.0) for _ in range(v)]
    qs = sum(q)
    q = [x / qs for x in q]
    w = [0.5, 0.3, 0.2]
    assert pool_excess(q, pools, w) >= -1e-12


def test_the_excess_vanishes_when_the_members_agree():
    p = [0.25, 0.75]
    assert pool_excess([0.4, 0.6], [p, p], [0.5, 0.5]) == \
        pytest.approx(0.0, abs=1e-14)


def test_a_single_member_pool_is_free():
    p = [0.3, 0.7]
    assert pool_excess([0.5, 0.5], [p], [1.0]) == pytest.approx(0.0, abs=1e-14)


def test_malformed_weights_are_refused():
    p = [0.5, 0.5]
    with pytest.raises(ValueError, match="sum to 1"):
        geometric_pool([p, p], [0.5, 0.7])
    with pytest.raises(ValueError, match="non-negative"):
        geometric_pool([p, p], [1.5, -0.5])


def test_a_zero_entry_in_a_member_is_refused():
    """The premise ``∀ i v, 0 < p i v`` is enforced, not assumed."""
    with pytest.raises(ValueError, match="strictly positive"):
        geometric_pool([[0.0, 1.0], [0.5, 0.5]], [0.5, 0.5])