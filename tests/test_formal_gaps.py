"""Theory-gap closure: Freedman tail (Probability) and Eckart-Young
certificate (Spectral) — executable forms of the Lean theorems."""
import math

import pytest
import torch

from src.hagi.model.formal import (
    eckart_young_frobenius_error,
    freedman_tail_low,
)


def azuma_tail(delta: float, n: int) -> float:
    """Azuma form exp(-delta^2/(2n)) for comparison (b=1 steps)."""
    return math.exp(-(delta * delta) / (2.0 * n))


class TestFreedmanTail:
    def test_stricter_than_azuma_when_V_below_n(self):
        delta, V, b, n = 0.5, 10.0, 1.0, 100
        assert freedman_tail_low(delta, V, b) < azuma_tail(delta, n)

    def test_reduces_to_azuma_as_b_goes_to_zero(self):
        delta, V, n = 0.5, 100.0, 100
        tiny_b = freedman_tail_low(delta, V, 1e-9)
        assert tiny_b == pytest.approx(azuma_tail(delta, n), rel=1e-9)

    def test_monotone_in_delta(self):
        assert (
            freedman_tail_low(1.0, 5.0, 1.0)
            < freedman_tail_low(0.5, 5.0, 1.0)
        )

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            freedman_tail_low(-1.0, 5.0, 1.0)
        with pytest.raises(ValueError):
            freedman_tail_low(1.0, 5.0, 0.0)


class TestEckartYoung:
    def test_full_norm_at_k0_and_zero_at_full_rank(self):
        s = torch.tensor([3.0, 2.0, 1.0])
        full = torch.sqrt(torch.sum(s**2))
        assert torch.allclose(
            eckart_young_frobenius_error(s, 0), full, atol=1e-6
        )
        assert eckart_young_frobenius_error(s, 3).item() == pytest.approx(
            0.0, abs=1e-6
        )

    def test_monotone_decreasing_in_k(self):
        torch.manual_seed(0)
        s = torch.linalg.svdvals(torch.randn(8, 6))
        errs = [
            eckart_young_frobenius_error(s, k).item() for k in range(7)
        ]
        assert all(
            errs[k + 1] <= errs[k] + 1e-7 for k in range(len(errs) - 1)
        )

    def test_matches_torch_svd_truncation(self):
        torch.manual_seed(0)
        a = torch.randn(8, 6)
        u, sv, vh = torch.linalg.svd(a)
        k = 3
        approx = (u[:, :k] * sv[:k]) @ vh[:k]
        residual = torch.linalg.matrix_norm(a - approx, ord="fro")
        assert eckart_young_frobenius_error(sv, k).item() == pytest.approx(
            residual.item(), rel=1e-5
        )
