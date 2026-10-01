"""Tests for the stochastic SafeQP margin (StochasticSafeQP.lean, R92)."""

from __future__ import annotations

import math
import random

import pytest

from hagi.train.stochastic_safeqp import (
    certified_feasible,
    feasible_margin,
    hoeffding_tail_probability,
    minibatch_for_margin,
    noise_epsilon,
)


def test_noise_epsilon_matches_the_closed_form() -> None:
    """``eps_noise = sigma*||d||*sqrt(2 log(K/delta)/m)``."""
    assert noise_epsilon(1.0, 2.0, 16, 8, 0.05) == pytest.approx(
        1.0 * 2.0 * math.sqrt(2.0 * math.log(8 / 0.05) / 16)
    )


def test_noise_epsilon_decays_as_one_over_sqrt_m() -> None:
    """The ``m`` in the exponent is the minibatch variance reduction."""
    base = noise_epsilon(1.0, 2.0, 16, 8, 0.05)
    # Four times the minibatch halves the required inflation.
    assert noise_epsilon(1.0, 2.0, 64, 8, 0.05) == pytest.approx(base / 2.0, rel=1e-12)
    assert noise_epsilon(1.0, 2.0, 256, 8, 0.05) == pytest.approx(base / 4.0, rel=1e-12)


def test_union_bound_widens_the_margin_with_domain_count() -> None:
    """More protected domains means a larger ``log(K/delta)`` factor."""
    assert noise_epsilon(1.0, 1.0, 16, 1, 0.05) < noise_epsilon(1.0, 1.0, 16, 16, 0.05)


def test_minibatch_for_margin_inverts_exactly_and_minimally() -> None:
    """The returned size meets the target, and one less would not."""
    random.seed(20)
    for _ in range(5000):
        sigma = random.uniform(0.01, 5.0)
        d_norm = random.uniform(0.01, 10.0)
        k = random.randint(1, 16)
        delta = random.uniform(1e-3, 0.5)
        target = random.uniform(0.001, 10.0)
        m, ok = minibatch_for_margin(sigma, d_norm, k, delta, target)
        if not ok:
            continue
        assert noise_epsilon(sigma, d_norm, m, k, delta) <= target * (1 + 1e-9)
        if m > 1:
            assert noise_epsilon(sigma, d_norm, m - 1, k, delta) > target * (1 + 1e-9)


def test_minibatch_reports_unreachable_targets() -> None:
    """A target no minibatch can reach is reported, not silently clamped."""
    m, ok = minibatch_for_margin(
        noise_bound=3.0, d_norm=8.0, n_domains=16, delta=0.4,
        target_epsilon=0.001, max_minibatch=1000,
    )
    assert ok is False
    assert m == 1000


def test_hoeffding_tail_decays_exponentially() -> None:
    """``exp(-m*threshold^2/(2R^2))`` -- the exponent is LINEAR in m.

    With threshold=1 and R=2 the exponent is ``-m/8``, so doubling m
    multiplies the exponent by two: the log-ratio is exactly 4 here.
    """
    a = hoeffding_tail_probability(1.0, 32, 2.0)
    b = hoeffding_tail_probability(1.0, 64, 2.0)
    assert a > b > 0.0
    assert math.log(a / b) == pytest.approx(4.0, rel=1e-6)
    # The tail falls below any fixed delta once m is large enough.
    assert hoeffding_tail_probability(1.0, 4096, 2.0) < 1e-6


def test_feasible_margin_is_the_inflated_constraint() -> None:
    """True gradients satisfy ``-(eps_i + eps_noise)``."""
    assert feasible_margin(0.05, 0.02) == pytest.approx(-0.07)


def test_certified_feasible_accepts_the_stochastic_constraint() -> None:
    """The check is on the minibatch inner product, at the given delta."""
    assert certified_feasible(
        inner_hat=-0.01, epsilon_i=0.05, noise_bound=1.0, d_norm=2.0,
        minibatch=64, n_domains=8, delta=0.05,
    )
    assert not certified_feasible(
        inner_hat=-0.9, epsilon_i=0.05, noise_bound=1.0, d_norm=2.0,
        minibatch=64, n_domains=8, delta=0.05,
    )


def test_smaller_minibatch_still_passes_when_the_direction_is_tight() -> None:
    """A tiny minibatch is admissible when the constraint is met easily."""
    assert certified_feasible(
        inner_hat=0.0, epsilon_i=1.0, noise_bound=0.001, d_norm=0.001,
        minibatch=1, n_domains=2, delta=0.1,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(minibatch=0), dict(n_domains=0), dict(delta=0.0), dict(delta=1.0),
    ],
)
def test_noise_epsilon_rejects_bad_inputs(kwargs: dict) -> None:
    base = dict(
        noise_bound=1.0, d_norm=1.0, minibatch=8, n_domains=4, delta=0.05
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        noise_epsilon(**base)


def test_hoeffding_rejects_nonpositive_radius() -> None:
    with pytest.raises(ValueError):
        hoeffding_tail_probability(1.0, 8, 0.0)
