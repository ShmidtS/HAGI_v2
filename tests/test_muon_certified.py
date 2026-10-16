"""Tests: certified Muon extensions (MuonStableLR, NewtonPolar R379–R382)."""
import math

import pytest
import torch

from hagi.train.optim import min_ns_iterations, muon_eta_max_stable, newton_schulz


def test_min_ns_iterations_boundary():
    # already at 1 (σ=1): zero iterations needed
    assert min_ns_iterations(1.0, 1e-3) == 0
    # barely beyond: one halving suffices when 2·r ≤ eps
    assert min_ns_iterations(1.0 + 1e-6, 1e-3) == 0


def test_min_ns_iterations_exact_formula():
    sigma, eps = 2.5, 1e-3
    r = (sigma - 1.0) / 2.0
    T = min_ns_iterations(sigma, eps)
    assert 2.0 * r ** (2 ** T) <= eps  # certificate holds at T
    assert T == 0 or 2.0 * r ** (2 ** (T - 1)) > eps  # T is minimal


def test_min_ns_iterations_out_of_basin():
    with pytest.raises(ValueError):
        min_ns_iterations(3.0, 1e-3)  # polar basin is [1, 3)


def test_min_ns_iterations_monotone():
    prev = -1
    for eps in [1e-1, 1e-2, 1e-3, 1e-6]:
        t = min_ns_iterations(2.9, eps)
        assert t >= prev
        prev = t


def test_min_ns_iterations_validation():
    with pytest.raises(ValueError):
        min_ns_iterations(3.0, 0.0)
    with pytest.raises(ValueError):
        min_ns_iterations(0.5, 1e-3)  # below unit norm


def test_eta_max_conservative_vs_max():
    torch.manual_seed(0)
    G = torch.randn(16, 8)
    O = newton_schulz(G)
    lam = 10.0
    eta = muon_eta_max_stable(G, O, lam)
    sigma_max = torch.linalg.svdvals(G.float()).max().item()
    # muon_avg_not_max: Σσ/m ≤ σ_max  =>  eta ≤ 2σ_max/λ
    assert eta <= 2.0 * sigma_max / lam + 1e-4


def test_eta_max_positive_scalar():
    torch.manual_seed(1)
    G = torch.randn(8, 8)
    O = newton_schulz(G)
    eta = muon_eta_max_stable(G, O, 5.0)
    assert eta > 0
    with pytest.raises(ValueError):
        muon_eta_max_stable(G, O, -1.0)


def test_default_ns_steps_unchanged():
    # regression: the live path still runs fixed ns (5 square / 3+ non-square)
    torch.manual_seed(2)
    G = torch.randn(8, 8)
    O5 = newton_schulz(G, steps=5)
    Odef = newton_schulz(G)  # default
    assert torch.allclose(O5, Odef)
    Gt = torch.randn(16, 8)
    assert torch.allclose(newton_schulz(Gt, 3), newton_schulz(Gt, steps=3))
