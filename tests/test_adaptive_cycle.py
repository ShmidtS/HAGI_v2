"""Tests for hagi.train.adaptive_cycle (ports of
Optimization/{DirectionPacking,SubspaceGrowth,InexactContraction,
SphericalDamping,AdaptiveCycle}.lean)."""
from __future__ import annotations

import math

import pytest
import torch

from hagi.train.adaptive_cycle import (
    angular_damping_window,
    chordal_residual,
    contraction_floor,
    growth_finite,
    inexact_contraction_halt_k,
    proj_residual,
    proj_residual_ratio,
    welch_capacity_full,
    welch_min_dimension,
    welch_required_dimension,
)

# --- Welch packing -------------------------------------------------------


def test_welch_boundary_mu_zero_and_one():
    # mu = 0: orthonormal -> need d >= M
    assert welch_min_dimension(5, 0.0, 5)
    assert not welch_min_dimension(5, 0.0, 4)
    assert welch_required_dimension(5, 0.0) == 5
    # mu = 1: all directions coincide -> d >= 1
    assert welch_min_dimension(5, 1.0, 1)
    assert welch_required_dimension(5, 1.0) == 1


def test_welch_against_direct_formula():
    for m in (2, 3, 7, 16):
        for mu in (0.1, 0.3, 0.5):
            lhs = m / (1.0 + (m - 1) * mu * mu)
            assert welch_min_dimension(m, mu, math.ceil(lhs))
            if lhs > 1:
                assert not welch_min_dimension(m, mu, math.ceil(lhs) - 1)
            assert welch_required_dimension(m, mu) == math.ceil(lhs)


def test_welch_capacity_full_is_contraposition():
    # capacity full: NOT (M+1)/(1+M*mu^2) <= d
    for m in (1, 2, 7):
        for mu in (0.0, 0.2, 0.6):
            for d in range(1, m + 2):
                lhs = (m + 1) / (1.0 + m * mu * mu)
                assert welch_capacity_full(m, mu, d) == (lhs > d)
    # growth trigger: full at d=M, mu=0
    assert welch_capacity_full(5, 0.0, 5)
    assert not welch_capacity_full(5, 0.0, 6)


# --- proj residual -------------------------------------------------------


def test_proj_residual_orthogonal_to_basis_and_ratio_le_one():
    g = torch.Generator().manual_seed(0)
    u = torch.randn(4, 16, generator=g)
    v = torch.randn(16, generator=g)
    r = proj_residual(u, v)
    # orthogonality to the QR-orthonormalized basis rows
    q, _ = torch.linalg.qr(u.mT)
    assert torch.allclose(q.mT @ r, torch.zeros(4), atol=1e-5)
    # reconstruction: v lies in span(r, basis)
    assert torch.allclose(v, r + q @ (q.mT @ v), atol=1e-5)
    ratio = proj_residual_ratio(u, v)
    assert 0.0 <= ratio <= 1.0 + 1e-6
    assert math.isclose(ratio, float(r.norm() / v.norm()), rel_tol=1e-6)


def test_proj_residual_exact_and_full_cases():
    v = torch.tensor([1.0, 2.0, 3.0])
    # empty basis: residual is v itself
    r0 = proj_residual(torch.zeros(0, 3), v)
    assert torch.allclose(r0, v)
    assert proj_residual_ratio(torch.zeros(0, 3), v) == pytest.approx(1.0)
    # v in span of basis: residual ~ 0
    basis = torch.stack([torch.tensor([1.0, 0.0, 0.0]),
                         torch.tensor([0.0, 1.0, 0.0])])
    r = proj_residual(basis, v)
    assert torch.allclose(r, torch.tensor([0.0, 0.0, 3.0]), atol=1e-6)
    assert proj_residual_ratio(basis, v) == pytest.approx(
        3.0 / v.norm(), rel=1e-5)


# --- growth finiteness ---------------------------------------------------


def test_growth_finite_floor():
    assert growth_finite(10.0, 3.0) == 3
    assert growth_finite(9.999, 3.0) == 3  # k*delta <= c0
    assert growth_finite(10.0, 4.0) == 2
    assert growth_finite(0.5, 1.0) == 0
    with pytest.raises(ValueError):
        growth_finite(1.0, 0.0)


# --- inexact contraction -------------------------------------------------


def test_contraction_floor():
    assert contraction_floor(0.5, 0.1) == pytest.approx(0.2)
    assert contraction_floor(0.9, 0.01) == pytest.approx(0.1)


def test_halt_k_none_when_floor_ge_tau():
    assert inexact_contraction_halt_k(0.5, 0.2, 0.4, 1.0) is None
    assert inexact_contraction_halt_k(0.5, 0.2, 0.399, 1.0) is None


def test_halt_k_synthetic_recursion_reaches_tau():
    q, eps, tau, b = 0.6, 0.05, 0.5, 1.0
    k = inexact_contraction_halt_k(q, eps, tau, b)
    assert k is not None
    # minimal: q^K < bound, q^(K-1) >= bound
    bound = (tau - eps / (1 - q)) / b
    assert q**k < bound
    assert k == 0 or q ** (k - 1) >= bound
    # direct run: x(0)=B, x(k+1)=q*x+eps -> x_K < tau (conservative)
    x = b
    for _ in range(k):
        x = q * x + eps
    assert x < tau + 1e-12


def test_halt_k_direct_recursion_marginal_case():
    # margin/bound close to 1 -> K = 0 admissible when q^0=1 < bound
    q, eps, tau, b = 0.5, 0.1, 10.0, 1.0
    k = inexact_contraction_halt_k(q, eps, tau, b)
    assert k == 0  # bound = (10 - 0.2)/1 > 1, q^0 = 1 < bound


# --- spherical damping ---------------------------------------------------


def test_angular_damping_window():
    for zeta in (-1.0, 0.0, 0.5, 0.9):
        lo, hi = angular_damping_window(zeta)
        assert lo == 0.0
        assert hi == pytest.approx(2.0 / (1.0 - zeta))
    with pytest.raises(ValueError):
        angular_damping_window(1.0)


def test_chordal_residual_equals_half_sq_norm():
    g = torch.Generator().manual_seed(1)
    u = torch.randn(8, generator=g)
    v = torch.randn(8, generator=g)
    u = u / u.norm()
    v = v / v.norm()
    rho = chordal_residual(u, v)
    assert rho == pytest.approx(1.0 - float(torch.dot(u, v)))
    assert rho == pytest.approx(0.5 * float((u - v).norm() ** 2))
    assert chordal_residual(u, u) == pytest.approx(0.0)
