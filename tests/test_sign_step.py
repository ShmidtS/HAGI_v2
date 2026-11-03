# §2b SignDescent port tests. Lean SSOT: primes Hagi/Unified/SignDescent.lean
# (signScale, sign_scaled_aligned, sign_descent, scaled_sign_aligned_of_coherence,
# lion_descent, sign_is_pointwise_max, sam_descent_bridge/flat).

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.sign_step import (  # noqa: E402
    lion_coherence,
    lion_descent_bound,
    sam_descent_transfer,
    scaled_sign_direction,
    sgn1,
    sign_alignment_residual,
    sign_descent_bound,
    sign_scale,
)


def test_sign_scale_is_mean_absolute_value():
    assert sign_scale([3.0, -1.0, 2.0, 0.0]) == pytest.approx(6.0 / 4.0)
    assert sign_scale([0.0, 0.0]) == 0.0
    with pytest.raises(ValueError):
        sign_scale([])


def test_scaled_sign_is_aligned_always():
    # Lean sign_scaled_aligned: residual <g,d> - ||d||^2 >= 0 for ANY g
    for g in ([3.0, -1.0, 2.0, 0.0], [1e-3, -5.0], [7.0] * 10):
        d = scaled_sign_direction(g)
        assert sign_alignment_residual(g, d) >= -1e-12


def test_raw_sign_can_violate_alignment_scale_is_mandatory():
    # measured counterexample: g = (-1,-1,0.5), raw sign step residual < 0
    g = [-1.0, -1.0, 0.5]
    raw = [sgn1(x) for x in g]
    assert sign_alignment_residual(g, raw) < 0


def test_sign_descent_bound_monotone_in_eta_and_matches_lean_form():
    f0, n2, L = 10.0, 2.0, 1.0
    b_half = sign_descent_bound(f0, 0.0, n2, 0.5, L)
    assert b_half == pytest.approx(f0 - 0.5 * (1 - 0.25) * n2)
    # full step eta = 1/L is the strongest certified bound
    assert sign_descent_bound(f0, 0.0, n2, 1.0, L) == pytest.approx(f0 - 0.5 * n2)
    with pytest.raises(ValueError):
        sign_descent_bound(f0, 0.0, n2, 2.0, L)
    with pytest.raises(ValueError):
        sign_descent_bound(f0, 0.0, n2, 0.5, 0.0)


def test_lion_coherence_gate_and_descent():
    g = [2.0, 3.0, -4.0, 1.0]
    m = [1.0, 1.0, -1.0, 0.0]  # moment consistent with g's signs
    assert lion_coherence(g, m) <= 0.0  # coherence holds
    # fully flipped moment: sign disagreement on every nonzero coordinate
    m_flip = [-1.0, -1.0, 1.0, 0.0]
    assert lion_coherence(g, m_flip) > 0.0
    # descent bound has the same certified form
    n2 = 2.0
    assert lion_descent_bound(5.0, n2, 1.0, 1.0) == pytest.approx(5.0 - 0.5 * n2)


def test_sam_descent_transfer_bridge_and_flat():
    assert sam_descent_transfer(0.5, 0.2) == pytest.approx(0.3)
    assert sam_descent_transfer(0.5, 0.0, flat=True) == pytest.approx(0.5)
    # flat=True with nonzero sharp is a contract violation
    with pytest.raises(ValueError):
        sam_descent_transfer(0.5, 0.2, flat=True)
