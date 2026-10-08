"""Test for the §20 fiber-geometry 90%-energy computation."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "growth"
for _p in (str(_SCRIPTS),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from measure_fiber_geometry import energy_profile  # noqa: E402


def test_energy_profile_pins_the_90_percent_computation() -> None:
    # An exactly-rank-2 matrix: W = diag-basis u*s0 + v*s1 (orthogonal
    # rows), so the spectrum is exactly {s0, s1, 0, ...} and the 90%
    # intrinsic dim is computable by hand.
    n = 6
    s0, s1 = 3.0, 1.0
    w = torch.zeros(n, n)
    w[0, 0] = s0
    w[1, 1] = s1
    p = energy_profile(w)
    total = s0 ** 2 + s1 ** 2  # 10
    assert p["rank"] == 2
    assert abs(p["energy_top1"] - 9.0 / total) < 1e-9
    # top1 share is 0.9 exactly: one component ALREADY reaches 90%
    assert p["d90"] == 1


def test_energy_profile_d90_two_components_when_needed() -> None:
    # energy shares 0.6 / 0.25 / 0.15: cum = 0.6, 0.85, 1.0 -- the 90%
    # target needs THREE components; one (0.6) and two (0.85) both
    # fall short.
    w = torch.zeros(4, 3)
    w[0, 0] = math_sqrt(0.6)
    w[1, 1] = math_sqrt(0.25)
    w[2, 2] = math_sqrt(0.15)
    p = energy_profile(w)
    assert p["d90"] == 3
    assert p["rank"] == 3


def math_sqrt(x: float) -> float:
    return x ** 0.5


def test_energy_profile_zero_matrix_does_not_crash() -> None:
    p = energy_profile(torch.zeros(3, 3))
    assert p["d90"] == 0
    assert p["energy_top1"] == 0.0


def test_energy_profile_rejects_non_2d() -> None:
    import pytest

    with pytest.raises(ValueError):
        energy_profile(torch.zeros(3))
