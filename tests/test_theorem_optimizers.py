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
from hagi.train.controller_policy import (
    Candidate,
    certified_gain,
    select_action,
)
from hagi.train.hedge import (
    hedge_potential,
    hedge_weights,
    minimal_topk,
    min_regret,
    optimal_hedge_eta,
    regret_bound,
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


@pytest.mark.parametrize("bad", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_analytic_step_rejects_nonpositive_inputs(bad: tuple[float, float]) -> None:
    """The theorem assumes ``0 < L`` and ``0 < dn2``; the code must too."""
    with pytest.raises(ValueError):
        analytic_step(1.0, bad[0], bad[1])


# --- Hedge.lean ---------------------------------------------------------


def test_router_regret_bound_is_minimized_at_eta_star() -> None:
    """``ln K/eta + eta*T/2`` is convex; no eta beats eta* within the cap.

    The comparison runs over the feasible region ``(0, 1]`` -- the
    domain ``hedge_step`` actually certifies -- so the clamp is part of
    the search space rather than an unchecked liberty.
    """
    for K in (2, 3, 8, 16):
        for T in (1, 10, 100, 1000):
            eta = optimal_hedge_eta(K, T)
            best = regret_bound(K, T, eta)
            for other in (eta * 0.5, eta * 0.9, eta, eta * 1.1, eta * 2.0):
                if other <= 1.0:  # inside the certified region
                    assert regret_bound(K, T, other) >= best - 1e-9


def test_eta_star_attains_the_closed_form_regret() -> None:
    """At ``eta*`` the bound equals the closed form ``sqrt(2*T*ln K)``.

    Only when the optimum fits inside ``hedge_step``'s ``eta <= 1``
    hypothesis. For small T the unconstrained ``eta* = sqrt(2 ln K / T)``
    exceeds 1, so the code clamps to 1 and the certified regret is the
    clamped value -- HIGHER than the unconstrained minimum, never lower.
    """
    for K in (2, 3, 8, 16):
        for T in (1, 10, 100, 1000):
            eta = optimal_hedge_eta(K, T)
            certified = regret_bound(K, T, eta)
            # The clamp can only cost, never gain.
            assert certified >= min_regret(K, T) - 1e-9
            if eta < 1.0:  # unclamped: the bound is attained exactly
                assert certified == pytest.approx(min_regret(K, T), rel=1e-9)


def test_eta_star_is_clamped_to_the_hedge_step_hypothesis() -> None:
    """``hedge_step`` assumes ``0 <= eta <= 1``; the cap must hold."""
    for K in (2, 3, 8, 16, 64):
        for T in (1, 2, 5, 10, 100):
            assert 0.0 < optimal_hedge_eta(K, T) <= 1.0


def test_hedge_step_potential_bound() -> None:
    """``sum p_i e^(-eta l_i) <= 1 - eta*Lbar + eta^2/2`` for ``eta <= 1``."""
    random.seed(3)
    for _ in range(5000):
        K = random.randint(2, 8)
        eta = random.uniform(0.0, 1.0)
        p = torch.softmax(torch.randn(K), dim=0)
        losses = torch.rand(K)
        potential, bound = hedge_potential(p, losses, eta)
        assert potential <= bound + 1e-6


def test_hedge_weights_normalize_and_shrink_toward_the_best() -> None:
    """Exponential weights up-weight the lowest-loss expert."""
    losses = torch.tensor([0.0, 5.0, 5.0])
    p = hedge_weights(losses, eta=1.0)
    assert float(p.sum()) == pytest.approx(1.0, rel=1e-6)
    assert int(p.argmax()) == 0
    assert float(p[0]) > float(p[1])


def test_minimal_topk_is_minimal_and_within_budget() -> None:
    """``min k : sum_{i>k} c_i^2 <= eps`` -- read off, not searched."""
    random.seed(4)
    for _ in range(500):
        K = random.randint(2, 20)
        c = torch.rand(K, dtype=torch.float64)
        eps = random.uniform(1e-4, 1.0)
        k = minimal_topk(c, eps)
        s, _ = torch.sort(c, descending=True)
        tail = float((s[k:] ** 2).sum())
        assert tail <= eps or k == K
        if k > 1:
            assert float((s[k - 1 :] ** 2).sum()) > eps


# --- ControllerPolicy.lean ---------------------------------------------


def test_budget_allocation_dominance() -> None:
    """Concentrating the budget beats EVERY split allocation."""
    random.seed(5)
    for _ in range(2000):
        K = random.randint(1, 5)
        cands = [
            Candidate(
                name=random.choice(["merge", "joint", "prune"]),
                gain=random.uniform(0.0, 10.0),
                cost=random.uniform(0.1, 10.0),
            )
            for _ in range(K)
        ]
        budget = random.uniform(0.0, 100.0)
        best = select_action(cands)
        if best is None:
            continue
        concentrated = best.gain * (budget / best.cost)
        # ratio_dominance: no single alternative action beats it.
        for c in cands:
            assert c.gain * (budget / c.cost) <= concentrated + 1e-9
        # budget_allocation_dominance: no split allocation beats it.
        for _ in range(20):
            alloc = [random.uniform(0.0, budget) for _ in cands]
            if sum(a * c.cost for a, c in zip(alloc, cands)) > budget:
                continue
            assert sum(c.gain * a for c, a in zip(cands, alloc)) <= concentrated + 1e-9


def test_select_action_prefers_the_best_ratio() -> None:
    cands = [Candidate("merge", 8.0, 2.0), Candidate("joint", 3.0, 1.0)]
    assert select_action(cands).name == "merge"
    assert certified_gain(cands, 10.0) == pytest.approx(40.0)


def test_select_action_refuses_without_certified_progress() -> None:
    """All ratios <= 0 means no action certifies progress: do nothing."""
    cands = [Candidate("merge", 0.0, 1.0), Candidate("joint", -1.0, 2.0)]
    assert select_action(cands) is None
    assert certified_gain(cands, 100.0) == 0.0


def test_candidate_rejects_nonpositive_cost() -> None:
    with pytest.raises(ValueError):
        Candidate("merge", 1.0, 0.0)
    with pytest.raises(ValueError):
        Candidate("merge", math.inf, 1.0)
