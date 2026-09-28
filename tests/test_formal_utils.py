"""Tests for hagi.model.formal -- the Lean-program utilities.

Each test pins the executable form of a proven theorem:
- mixer_invisible_condition: Mix.lean mixed_invisible (column sums).
- kv_waterfill_bits: KVWater.lean waterfilling_bound (log allocation,
  budget conservation, degenerate cases).
- certified_gain: Select.lean newBound (bound algebra).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hagi.model.formal import (  # noqa: E402
    certified_gain,
    kv_waterfill_bits,
    mixer_invisible_condition,
)


def _hadamard(n: int) -> torch.Tensor:
    """Unused placeholder -- replaced by _sylvester below."""
    raise NotImplementedError


def _sylvester(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        m = h.shape[0]
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(h.shape[0])


def test_identity_is_invisible():
    assert mixer_invisible_condition(torch.eye(4))


def test_hadamard_is_visible():
    # The Sylvester H/sqrt(n) has column sums +-1/sqrt(n) != 1 for n>1:
    # the theorem predicts the Hadamard mixer is NOT logit-invisible --
    # which is why the ensemble path never trusted it bare.
    for n in (2, 4, 8):
        assert not mixer_invisible_condition(_sylvester(n))


def test_column_sum_one_matrix_is_invisible():
    # Any column-stochastic-preserving rotation (sums == 1) is invisible.
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert mixer_invisible_condition(q)
    q2 = torch.tensor([[0.6, 0.2], [0.4, 0.8]])  # col sums = 1
    assert mixer_invisible_condition(q2)


def test_kv_waterfill_budget_conserved():
    c = torch.tensor([0.9, 0.5, 0.1, 0.05])
    b = kv_waterfill_bits(c, total_bits=32.0, kappa=0.5)
    assert abs(float(b.sum()) - 32.0) < 1e-3
    # Higher sensitivity -> more bits (monotone in c).
    assert bool((b[0] >= b[1]).all()) and bool((b[1] >= b[2]).all())
    # Logarithmic, not proportional: b_i - b_j = (log c_i - log c_j)/kappa.
    assert abs(float(b[0] - b[1]) - (math.log(0.9) - math.log(0.5)) / 0.5) < 1e-3


def test_kv_waterfill_zero_sensitivity_gets_zero_bits():
    c = torch.tensor([1.0, 0.0, 0.3])
    b = kv_waterfill_bits(c, total_bits=8.0, kappa=1.0)
    assert float(b[1]) == 0.0
    assert abs(float(b.sum()) - 8.0) < 1e-3


def test_kv_waterfill_uniform_sensitivity_uniform_split():
    c = torch.tensor([0.5, 0.5, 0.5, 0.5])
    b = kv_waterfill_bits(c, total_bits=12.0, kappa=0.7)
    assert torch.allclose(b, torch.full_like(b, 3.0), atol=1e-3)


def test_certified_gain_algebra():
    # new bound = (N*M + c)/(N+1); gain over M is (M-c)/(N+1).
    n, m, c = 9, 6.44, 6.06
    assert abs(certified_gain(n, m, c) - (m - c) / (n + 1)) < 1e-12
    new_bound = (n * m + c) / (n + 1)
    assert abs((m - certified_gain(n, m, c)) - new_bound) < 1e-12
    # Monotone in c: a better (smaller) candidate certifies a bigger gain.
    assert certified_gain(n, m, 5.5) > certified_gain(n, m, 6.5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all formal tests passed")
