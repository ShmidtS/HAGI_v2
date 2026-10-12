"""Tests for the theorem-derived optimizer components.

Each test is the executable form of a Lean statement. The source
theorem is cited in the test name so a failure points at the
proposition that broke, not just at a line number.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from hagi.train.analytic_step import (
    analytic_step,
    guaranteed_descent,
    select_step,
)


# --- RecursiveGrowth.lean ----------------------------------------------


def test_optimal_step_unconstrained_descent_is_bounded() -> None:
    """``forall eta', eta'*inner - L*dn2*eta'^2/2 <= inner^2/(2*L*dn2)``."""
    random.seed(0)
    for _ in range(2000):
        L = random.uniform(0.1, 10.0)
        dn2 = random.uniform(1e-6, 10.0)
        inner = random.uniform(-10.0, 10.0)
        bound = inner**2 / (2.0 * L * dn2)
        for _ in range(3):
            eta = random.uniform(-50.0, 50.0)
            assert guaranteed_descent(eta, inner, dn2, L) <= bound + 1e-9


def test_optimal_step_value_is_the_maximum() -> None:
    """``g(eta*) == inner^2/(2*L*dn2)``."""
    random.seed(1)
    for _ in range(2000):
        L = random.uniform(0.1, 10.0)
        dn2 = random.uniform(1e-6, 10.0)
        inner = random.uniform(-10.0, 10.0)
        eta = analytic_step(inner, dn2, L)
        assert guaranteed_descent(eta, inner, dn2, L) == pytest.approx(
            inner**2 / (2.0 * L * dn2), rel=1e-9
        )


def test_optimal_step_ge_recip_and_clip_is_exact() -> None:
    """``inner >= dn2 => eta* >= 1/L``; the clip then returns exactly 1/L."""
    random.seed(2)
    for _ in range(2000):
        L = random.uniform(0.1, 10.0)
        dn2 = random.uniform(1e-6, 10.0)
        inner = dn2 * random.uniform(1.0, 5.0)
        assert analytic_step(inner, dn2, L) >= 1.0 / L - 1e-9
        assert select_step(inner, dn2, L) == pytest.approx(1.0 / L, rel=1e-12)


def test_select_step_never_ascends() -> None:
    """A direction that ascends yields a zero step, not a negative one."""
    assert select_step(inner=-1.0, dn2=1.0, smoothness=2.0) == 0.0
    assert select_step(inner=-1.0, dn2=1.0, smoothness=2.0, base_lr=3e-4) == 0.0


def test_select_step_respects_lr_ceiling() -> None:
    """The schedule LR stays a ceiling, so the step cannot exceed it."""
    assert select_step(100.0, 1.0, 0.01, base_lr=3e-4) == pytest.approx(3e-4)


def test_zero_ceiling_does_not_unlock_the_unclipped_step() -> None:
    """Regression: a base_lr of 0 must not let the ceiling be skipped.

    The first A/B run reached CE 21.2 because linear warmup returns
    exactly 0 from the schedule, and the guard ``base_lr > 0`` skipped
    the ceiling -- letting an unclipped eta* 700x the tuned LR through
    at step 0. Callers must pass the configured base LR, but the
    predicate itself is also documented here: 0 means "no step".
    """
    # A zero ceiling yields no step -- never the unclipped eta*.
    assert select_step(1.0, 1.0, 4.541, base_lr=0.0, clip_to_lr=True) == 0.0
    # And the configured ceiling binds regardless of warmup arithmetic.
    assert select_step(1.0, 1.0, 4.541, base_lr=3e-4, clip_to_lr=True) == pytest.approx(3e-4)


@pytest.mark.parametrize("bad", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_analytic_step_rejects_nonpositive_inputs(bad: tuple[float, float]) -> None:
    """The theorem assumes ``0 < L`` and ``0 < dn2``; the code must too."""
    with pytest.raises(ValueError):
        analytic_step(1.0, bad[0], bad[1])


