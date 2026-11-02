# Wave port: DescentCoreDiag + QuantEnergyBridge (2026-11-02).
# Lean SSOT:
#   primes Hagi/Unified/DescentCoreDiag.lean (diagE, diagGrad,
#     diag_descent_eq, diag_descent_le, cMax);
#   primes Hagi/Unified/EFArc.lean, section QuantEnergyBridge
#     (gridQ_energy_bound).
# Pattern follows tests/test_primes_wave_r541.py.

import math

import pytest

from hagi.train import diag_quad
from hagi.train.lip_energy import gridq_energy_bound


# --- DescentCoreDiag: exact expansion identity -----------------------------


def test_diag_descent_eq_exact_identity():
    c, a, x, d = [2.0, 0.5, 3.0], [1.0, -1.0, 2.0], [0.0, 1.0, 4.0], [0.3, -0.7, 0.1]
    e0, inner, quad = diag_quad.diag_descent_expansion(c, a, x, d)
    e1 = diag_quad.diag_energy(c, a, [xi + di for xi, di in zip(x, d)])
    assert e1 == pytest.approx(e0 + inner + quad)


def test_diag_grad_matches_energy_difference():
    # grad is the exact derivative: dE/dx_i = c_i (x_i - a_i)
    c, a, x = [2.0, 0.5, 3.0], [1.0, -1.0, 2.0], [0.0, 1.0, 4.0]
    eps = 1e-6
    g = diag_quad.diag_grad(c, a, x)
    for i in range(len(x)):
        xp = list(x)
        xp[i] += eps
        fd = (diag_quad.diag_energy(c, a, xp) - diag_quad.diag_energy(c, a, x)) / eps
        assert fd == pytest.approx(g[i], abs=1e-3)


# --- diag_descent_le: derived smoothness L = cMax ---------------------------


def test_diag_descent_le_bound_and_tightness():
    c, a, x, d = [2.0, 0.5, 3.0], [1.0, -1.0, 2.0], [0.0, 1.0, 4.0], [0.3, -0.7, 0.1]
    L = diag_quad.c_max(c)
    e0, inner, quad = diag_quad.diag_descent_expansion(c, a, x, d)
    bound = diag_quad.diag_descent_le(c, a, x, d, L)
    e1 = e0 + inner + quad
    assert e1 <= bound  # the model dominates the exact value
    # tightness: with anisotropic curvature the gap is exactly
    # (L/2 - per-coordinate c_i/2) mass of ||d||^2
    gap = bound - e1
    expected = 0.5 * sum((L - ci) * di * di for ci, di in zip(c, d))
    assert gap == pytest.approx(expected)


def test_c_max_positive_curvature_required_for_witness():
    c, a, x = [2.0, 4.0], [1.0, 3.0], [0.0, 0.0]
    x1 = diag_quad.diag_one_step(c, a, x)
    assert diag_quad.diag_energy(c, a, x1) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        diag_quad.diag_one_step([1.0, 0.0], a, x)


# --- QuantEnergyBridge: gridQ_energy_bound ----------------------------------


def test_gridq_energy_bound_formula():
    assert gridq_energy_bound(2.0, 9, 0.4) == pytest.approx(2.0 * 3.0 * 0.2)


def test_gridq_energy_bound_dominates_measured_distortion():
    # per-coordinate rounding to grid delta gives |err_i| <= delta/2,
    # so the Euclidean distortion <= sqrt(n) * delta / 2, and the energy
    # change under an L-Lipschitz energy is dominated by the bridge.
    delta, n, L = 0.1, 16, 1.5
    errs = [delta / 2] * n  # worst case
    dist = math.sqrt(sum(e * e for e in errs))
    exact = L * dist
    assert exact <= gridq_energy_bound(L, n, delta) + 1e-12


def test_gridq_energy_bound_validation():
    with pytest.raises(ValueError):
        gridq_energy_bound(-1.0, 4, 0.1)
    with pytest.raises(ValueError):
        gridq_energy_bound(1.0, 0, 0.1)
    with pytest.raises(ValueError):
        gridq_energy_bound(1.0, 4, 0.0)
