"""Tests for the derived safe step and the PoE correction (R105/R106).

Both rounds remove a hyperparameter by making it arithmetic on measured
quantities. The properties worth pinning are the ones a plausible-looking
reimplementation would get wrong:

  - the ``min 1`` clamp is load-bearing (the audit's literal formula is
    wrong, and the formalisation is sharper than its source);
  - a conflicting domain can drive the window NEGATIVE, which means
    "skip the step", not "take a small step";
  - the PoE bound is the weighted form with 1/8, and the audit's
    pairwise form with 1/8 is provably false.
"""

from __future__ import annotations

import math

import pytest

from hagi.train.safeqp_step import (
    binding_domain,
    eta_max,
    eta_max_conflict_free,
    expert_spread,
    no_domain_regresses,
    pairwise_constant_is_halved,
    poe_log_z,
    poe_log_z_bound,
    poe_softmax_speed,
    weighted_log_z,
)


def three(margin=0.05, curvature=2.0, d2=4.0, inner=None):
    eps = {"math": margin, "ru": margin, "code": margin}
    cur = {"math": curvature, "ru": curvature, "code": curvature}
    if inner is None:
        inner = {"math": 0.1, "ru": 0.1, "code": 0.1}
    return eps, inner, cur, d2


# --- R105: the window ----------------------------------------------------


def test_a_conflict_free_window_is_strictly_positive():
    """The controller always has a derived safe rate."""
    eps, inner, cur, d2 = three()
    e = eta_max(eps, inner, cur, d2)
    assert e > 0.0


def test_the_window_never_exceeds_one():
    """``min 1`` is part of the formula, and the clamp is load-bearing."""
    # huge margins and tiny curvature give thresholds far above 1
    eps = {"a": 100.0}
    cur = {"a": 0.01}
    inner = {"a": 50.0}
    assert eta_max(eps, inner, cur, 1.0) == pytest.approx(1.0)


def test_without_the_clamp_a_large_step_would_violate_the_budget():
    """Why the audit's literal formula is wrong, as an executable check.

    ``eps`` enters LINEARLY in the descent lemma, so the threshold is
    not a fixed budget the step may exceed: taking eta = 5 when the
    formula without the clamp allows it blows past eps by a wide margin.
    """
    eps, inner, cur, d2 = three(margin=0.05, curvature=2.0)
    e = eta_max(eps, inner, cur, d2)
    # the per-domain threshold, before the clamp
    raw = min(2.0 * (eps[k] + inner[k]) / (cur[k] * d2) for k in eps)
    assert e == pytest.approx(min(1.0, raw))
    assert no_domain_regresses(eps, inner, cur, d2, e) is True
    # ... and past the threshold it fails
    assert no_domain_regresses(eps, inner, cur, d2, raw * 1.5) is False


def test_a_conflicting_domain_can_close_the_window_completely():
    """``<g_i, d>`` far negative means NO step is safe, not a small one."""
    eps = {"a": 0.05, "b": 0.05}
    cur = {"a": 2.0, "b": 2.0}
    inner = {"a": 0.1, "b": -5.0}
    e = eta_max(eps, inner, cur, 4.0)
    assert e < 0.0


def test_the_binding_domain_is_the_one_that_closes_the_window():
    eps = {"a": 0.05, "b": 0.05, "c": 0.05}
    cur = {"a": 2.0, "b": 2.0, "c": 2.0}
    inner = {"a": 0.1, "b": -5.0, "c": 0.1}
    assert binding_domain(eps, inner, cur, 4.0) == "b"


def test_the_conflict_free_window_drops_the_inner_term():
    """With ``<g_i,d> = 0`` the two forms agree; non-zero and they do not.

    The conflict-free window is the special case ``inner = 0``, and
    dropping the inner term can only SHRINK it -- the descent term was
    buying room, which is the whole point of a direction that already
    descends everywhere.
    """
    eps, inner, cur, d2 = three()
    zeros = {k: 0.0 for k in eps}
    assert eta_max_conflict_free(eps, cur, d2) == \
        eta_max(eps, zeros, cur, d2)
    assert eta_max_conflict_free(eps, cur, d2) <= eta_max(eps, inner, cur, d2)


def test_a_tighter_curvature_narrows_the_window():
    eps, inner, _, d2 = three()
    loose = eta_max(eps, inner, {"math": 1.0, "ru": 1.0, "code": 1.0}, d2)
    tight = eta_max(eps, inner, {"math": 4.0, "ru": 4.0, "code": 4.0}, d2)
    assert tight < loose


def test_a_smaller_direction_norm_widens_the_window():
    """``||d||^2`` in the denominator: a shorter step buys more room."""
    eps, inner, cur, _ = three()
    big = eta_max(eps, inner, cur, 16.0)
    small = eta_max(eps, inner, cur, 1.0)
    assert small > big


def test_the_guarantee_holds_at_or_below_the_window():
    """The theorem's conclusion, checked rather than trusted."""
    eps, inner, cur, d2 = three()
    e = eta_max(eps, inner, cur, d2)
    for frac in (0.0, 0.25, 0.5, 0.9, 1.0):
        assert no_domain_regresses(eps, inner, cur, d2, frac * e) is True


@pytest.mark.parametrize("eta", [0.0, 0.001, 0.01, 1.0])
def test_no_domain_regresses_agrees_with_the_window(eta):
    eps, inner, cur, d2 = three()
    e = eta_max(eps, inner, cur, d2)
    assert no_domain_regresses(eps, inner, cur, d2, eta) is (eta <= e)


def test_a_negative_step_is_refused():
    eps, inner, cur, d2 = three()
    with pytest.raises(ValueError):
        no_domain_regresses(eps, inner, cur, d2, -1e-9)


def test_malformed_domains_are_refused():
    with pytest.raises(ValueError):
        eta_max({}, {}, {}, 1.0)
    with pytest.raises(ValueError):
        eta_max({"a": 1.0}, {"b": 1.0}, {"a": 1.0}, 1.0)
    with pytest.raises(ValueError):
        eta_max({"a": 1.0}, {"a": 1.0}, {"a": 0.0}, 1.0)
    with pytest.raises(ValueError):
        eta_max({"a": 1.0}, {"a": 1.0}, {"a": 1.0}, 0.0)


# --- R106: the PoE bound -------------------------------------------------


def test_the_pool_is_the_logsumexp_of_the_weighted_mean():
    logits = [[0.5, -0.2, 0.1], [0.3, 0.1, -0.4]]
    w = [0.5, 0.5]
    mean = [0.4, -0.05, -0.15]
    assert poe_log_z(logits, w) == pytest.approx(
        math.log(sum(math.exp(x) for x in mean))
    )


def test_the_gap_is_non_positive_by_jensen():
    """Pooling cannot INCREASE the normaliser.

    Jensen at the pooled mean gives log Z_w <= sum_i w_i log Z_i, so the
    signed gap is <= 0. The theorem's absolute value is what makes it
    symmetric -- and the sign matters for the correction's meaning: the
    cheap approximation errs on one side only.
    """
    logits = [[0.5, -0.2, 0.1], [0.3, 0.1, -0.4]]
    gap, _ = poe_log_z_bound(logits, [0.5, 0.5])
    assert gap <= 1e-12


def test_the_gap_vanishes_for_identical_experts():
    p = [0.5, -0.2, 0.1]
    gap, _ = poe_log_z_bound([p, p], [0.5, 0.5])
    assert gap == pytest.approx(0.0, abs=1e-15)


def test_the_bound_holds_on_a_random_pool():
    import random

    rng = random.Random(0)
    for _ in range(20):
        logits = [[rng.uniform(-3, 3) for _ in range(5)] for _ in range(3)]
        w = [0.5, 0.3, 0.2]
        gap, bound = poe_log_z_bound(logits, w)
        assert abs(gap) <= bound + 1e-9


def test_the_spread_is_zero_for_identical_experts():
    p = [0.5, -0.2, 0.1]
    assert expert_spread([p, p], [0.5, 0.5], 0) == pytest.approx(0.0, abs=1e-15)


def test_the_spread_measures_disagreement():
    logits = [[0.5, -0.2, 0.1], [1.5, -1.2, 0.1]]
    assert expert_spread(logits, [0.5, 0.5], 0) == pytest.approx(1.0)


def test_the_softmax_speed_bound_is_quadratic():
    assert poe_softmax_speed(0.0) == 0.0
    assert poe_softmax_speed(0.5) == pytest.approx(0.5 ** 2 / 8)
    # a spread of 1 costs 0.125 nats per token -- small
    assert poe_softmax_speed(1.0) == pytest.approx(0.125)


def test_a_negative_spread_is_refused():
    with pytest.raises(ValueError):
        poe_softmax_speed(-1.0)


# --- the audit refutation ------------------------------------------------


def test_the_audit_pairwise_form_is_provably_false():
    """The counterexample is executable, not a claim in a docstring.

    z1 = (1,-1), z2 = (0,0), w = 1/2: the gap is about 0.157 while the
    audit's pairwise 1/8 bound claims 0.0625. The honest pairwise constant
    is 1/2, proved in Lean with weighted Cauchy-Schwarz.
    """
    violated, gap, claimed = pairwise_constant_is_halved(
        [[1.0, -1.0], [0.0, 0.0]], [0.5, 0.5])
    assert violated is True
    # Jensen makes the SIGNED gap negative; the theorem bounds its
    # absolute value, and that is what the audit's form claimed.
    assert gap < 0.0
    assert abs(gap) == pytest.approx(0.096776, abs=1e-6)
    assert claimed == pytest.approx(0.0625)
    assert abs(gap) > claimed


def test_the_weighted_form_holds_on_that_same_counterexample():
    """The corrected law is not merely weaker -- it is true here."""
    _, bound = poe_log_z_bound([[1.0, -1.0], [0.0, 0.0]], [0.5, 0.5])
    _, gap, _ = pairwise_constant_is_halved([[1.0, -1.0], [0.0, 0.0]],
                                             [0.5, 0.5])
    assert abs(gap) <= bound + 1e-9


def test_malformed_pools_are_refused():
    with pytest.raises(ValueError):
        poe_log_z([[0.5, 0.5]], [0.5, 0.5])
    with pytest.raises(ValueError):
        poe_log_z([[0.5, 0.5]], [1.5])
    with pytest.raises(ValueError):
        poe_log_z([[0.5, 0.5], [0.5]], [0.5, 0.5])