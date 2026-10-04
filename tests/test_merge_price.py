"""T2: тесты MergePrice Σ (содержательный R129, 2607.09202).

Каждый тест пришпиливает runtime-порт к Lean-теореме T2 планa:

* ``merge_price`` -- Th.1 ΔL = ½·devᵀΣ·dev (квадратичная цена в
  метрике задачи, не линейная норма);
* ``ker_violation`` -- Th.4 ΔL = 0 ⟺ dev ∈ ker Σ;
* ``merge_gate`` -- слияние выгодно только когда twoGap покрывает
  Σ-цену + цену сжатия GapLaw;
* ``distortion_floor`` -- Th.5 D ≥ ¼σ_uδ²;
* ``sigma_gram`` -- эмпирическая Σ = XᵀX/n из активаций.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.merge_price import (  # noqa: E402
    distortion_floor,
    ker_violation,
    merge_gate,
    merge_price,
    sigma_gram,
)


def _sigma(seed: int = 0, d: int = 8, live: int = 5) -> torch.Tensor:
    """Σ float64 с rank=live: d−live нулевых собственных значений.

    eigh возвращает λ по ВОЗРАСТАНИЮ, ker Σ = span ПЕРВЫХ d−live
    собственных векторов (зануляем младшие).
    """
    torch.manual_seed(seed)
    a = torch.randn(d, d, dtype=torch.float64)
    sym = a @ a.T
    lam, vecs = torch.linalg.eigh(sym)
    lam[: d - live] = 0.0  # ker Σ = span первых d-live собственных векторов
    return (vecs * lam) @ vecs.T


class TestTh1MergePrice:
    def test_quadratic_exact_on_identity(self):
        # Σ = I: ΔL = ½‖dev‖² — классическая квадратика
        dev = torch.tensor([1.0, 2.0, -1.0])
        sig = torch.eye(3)
        assert merge_price(dev, sig) == pytest.approx(0.5 * 6.0)

    def test_metric_dependent(self):
        # Th.1 суть: цена зависит от Σ, не только от ‖dev‖ — одинаковые
        # ‖dev‖ в разных метриках дают разную цену. dev=[3,4]:
        # diag(0.01,10): 0.5*(0.09+160)=80.0 (дорогая по 2-й оси)
        # diag(10,0.01): 0.5*(90+0.16)=45.1 (дешёвая по 2-й оси)
        dev = torch.tensor([3.0, 4.0])
        dear = torch.diag(torch.tensor([0.01, 10.0]))  # большой вес на 4
        cheap = torch.diag(torch.tensor([10.0, 0.01]))  # малый вес на 4
        assert merge_price(dev, dear) > merge_price(dev, cheap)

    def test_sign_invariant(self):
        dev = torch.randn(8)
        sig = _sigma()
        assert merge_price(dev, sig) == pytest.approx(merge_price(-dev, sig))

    def test_zero_dev_zero_price(self):
        dev = torch.zeros(8)
        assert merge_price(dev, _sigma()) == 0.0

    def test_shape_checks(self):
        with pytest.raises(ValueError):
            merge_price(torch.randn(3, 3), torch.eye(3))
        with pytest.raises(ValueError):
            merge_price(torch.randn(3), torch.randn(3, 4))


class TestTh4KerViolation:
    def test_dev_in_ker_sigma_price_zero(self):
        sig = _sigma(d=8, live=5)
        lam, vecs = torch.linalg.eigh(sig)
        ker_basis = vecs[:, :3]  # первые 3 = нулевые λ (ascending)
        dev = ker_basis @ torch.randn(3, dtype=torch.float64)
        assert ker_violation(dev, sig) == pytest.approx(0.0, abs=1e-10)
        assert merge_price(dev, sig) == pytest.approx(0.0, abs=1e-10)


class TestMergeGate:
    def test_gate_opens_when_gap_covers_price(self):
        # twoGap=0.5, цена=0.1+0.035: гейт открыт
        assert merge_gate(
            two_gap=0.5, prices=[0.1], kappa=0.05, s=1.0, n=100
        )

    def test_gate_closes_when_price_exceeds_gap(self):
        # twoGap=0.1, цена=0.3+0.035: закрыт
        assert not merge_gate(
            two_gap=0.1, prices=[0.3], kappa=0.05, s=1.0, n=100
        )

    def test_compression_cost_enters_the_bar(self):
        # one price flips the gate, same gap, higher kappa
        open_ = merge_gate(two_gap=0.2, prices=[0.1], kappa=0.0, s=1.0, n=100)
        closed = merge_gate(two_gap=0.2, prices=[0.1], kappa=0.1, s=1.0, n=100)
        assert open_ and not closed


class TestTh5DistortionFloor:
    def test_floor_quadratic_in_delta(self):
        assert distortion_floor(2.0, 1.0) == pytest.approx(0.5)
        assert distortion_floor(2.0, 2.0) == pytest.approx(2.0)  # 4x

    def test_floor_zero_when_no_disagreement(self):
        assert distortion_floor(3.0, 0.0) == 0.0


class TestSigmaGram:
    def test_gram_of_orthonormal_is_identity(self):
        # Q с ортонормальными колонками: Σ = Q^TQ/n = I/n
        q, _ = torch.linalg.qr(torch.randn(50, 6, dtype=torch.float64))
        sig = sigma_gram(q)
        n = q.shape[0]
        assert torch.allclose(sig * n, torch.eye(6, dtype=sig.dtype), atol=1e-10)

    def test_energy_preserved(self):
        x = torch.randn(100, 4)
        sig = sigma_gram(x)
        energy = float((x.double() ** 2).sum()) / 100
        assert float(sig.trace()) == pytest.approx(energy, rel=1e-9)
