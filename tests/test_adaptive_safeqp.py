"""Tests for the adaptive SafeQP margin (R97, covering-number route).

R92 conditions the margin on a FIXED direction. A real controller solves
the QP against the minibatch estimates and so selects d*(omega) AFTER
seeing the noise; R92 documented that transfer as an open gap. R97
closes it with an eps-net, union bound and Lipschitz transfer.
"""

from __future__ import annotations

import math

import pytest

from hagi.train.adaptive_safeqp import (
    adaptive_noise_epsilon,
    fixed_noise_epsilon,
    minibatch_for_adaptive_margin,
    net_log_term,
    optimal_grid_step,
)


def test_net_log_term_matches_the_closed_form_when_it_fits():
    K, n, M, delta = 4, 3, 10, 0.05
    expected = math.log(2.0 * K * (2.0 * M + 1.0) ** n / delta)
    assert net_log_term(K, n, M, 0.5, delta) == pytest.approx(expected, rel=1e-12)


def test_net_log_term_does_not_overflow_at_project_dimensions():
    """The direct form overflows a float at n=1152; log space does not.

    (2*100+1)**1152 is around 1e1900, so writing the grid count and then
    taking a log raises OverflowError. This is the project's own width.
    """
    value = net_log_term(8, 1152, 100, 0.0625, 0.05)
    assert math.isfinite(value)
    assert value == pytest.approx(
        math.log(2.0 * 8 / 0.05) + 1152 * math.log(201.0), rel=1e-12
    )
    assert math.isfinite(net_log_term(8, 4096, 100, 0.0625, 0.05))


def test_adaptive_costs_more_than_fixed_and_grows_with_dimension():
    """The price of adaptivity, stated rather than hidden."""
    s, D, K, M, m, delta = 1.0, 10.0, 8, 100, 256, 0.05
    fixed = fixed_noise_epsilon(s, D, m, K, delta)
    previous = 0.0
    for n in (16, 1152, 4096):
        adaptive = adaptive_noise_epsilon(s, D, 0.0625, m, K, n, M, delta)
        assert adaptive > fixed
        assert adaptive > previous
        previous = adaptive


def test_finer_grid_shrinks_the_margin_monotonically():
    s, D, K, M, m, delta, n = 1.0, 10.0, 8, 100, 256, 0.05, 1152
    values = [
        adaptive_noise_epsilon(s, D, e, m, K, n, M, delta)
        for e in (0.5, 0.25, 0.125, 0.0625)
    ]
    assert values == sorted(values, reverse=True)


def test_minibatch_for_adaptive_margin_is_exact_and_minimal():
    s, D, K, M, delta, n = 1.0, 10.0, 8, 100, 0.05, 64
    eps_dir, target = 0.0625, 2.0
    m, ok = minibatch_for_adaptive_margin(
        s, D, eps_dir, K, n, M, delta, target
    )
    assert ok
    assert adaptive_noise_epsilon(s, D, eps_dir, m, K, n, M, delta) <= target
    assert adaptive_noise_epsilon(s, D, eps_dir, m - 1, K, n, M, delta) > target


def test_coarse_grid_floor_is_reported_not_silently_wrong():
    """sigma*eps_dir >= target means NO minibatch works."""
    m, ok = minibatch_for_adaptive_margin(
        1.0, 10.0, 0.5, 8, 64, 100, 0.05, target_epsilon=0.4
    )
    assert not ok
    assert m == 0


def test_optimal_grid_step_picks_a_feasible_minimum():
    """The trade-off is resolved by arithmetic, not by tuning."""
    eps_dir, m, ok = optimal_grid_step(
        1.0, 10.0, 8, 1152, 100, 0.05, target_epsilon=2.0
    )
    assert ok
    assert adaptive_noise_epsilon(1.0, 10.0, eps_dir, m, 8, 1152, 100, 0.05) <= 2.0
    # No other candidate is feasible with fewer samples.
    for other in (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
        if other == eps_dir:
            continue
        m_other, ok_other = minibatch_for_adaptive_margin(
            1.0, 10.0, other, 8, 1152, 100, 0.05, 2.0
        )
        if ok_other:
            assert m_other >= m


def test_optimal_grid_step_reports_infeasible_targets():
    eps_dir, m, ok = optimal_grid_step(
        1.0, 10.0, 8, 1152, 100, 0.05, target_epsilon=0.01
    )
    assert not ok


def test_adaptive_margin_decays_as_one_over_sqrt_m():
    s, D, K, M, delta, n, e = 1.0, 10.0, 8, 100, 0.05, 64, 0.0625
    base = adaptive_noise_epsilon(s, D, e, 256, K, n, M, delta)
    assert adaptive_noise_epsilon(s, D, e, 1024, K, n, M, delta) < base


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(noise_bound=0.0, direction_bound=1.0),
        dict(noise_bound=1.0, direction_bound=0.0),
        dict(minibatch=0),
    ],
)
def test_adaptive_noise_epsilon_rejects_bad_inputs(kwargs: dict):
    base = dict(
        noise_bound=1.0, direction_bound=1.0, eps_dir=0.1, minibatch=16,
        n_domains=4, dim=8, bound=10, delta=0.05,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        adaptive_noise_epsilon(**base)


def test_net_log_term_rejects_bad_grid():
    with pytest.raises(ValueError):
        net_log_term(0, 8, 10, 0.1, 0.05)
    with pytest.raises(ValueError):
        net_log_term(4, 8, 10, 0.1, 1.5)
    with pytest.raises(ValueError):
        net_log_term(4, 8, 10, 0.0, 0.05)
