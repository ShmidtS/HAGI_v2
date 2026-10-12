"""R366 grokking-расписание — исполняемая форма теоремы.

Каждый тест — исполняемая форма утверждения primes R366
(``grokking_threshold_time``, arXiv:2609.26679):
T = ln(r0/eps)/(η·λ), фазовая безопасность η·λ ≤ 1.
"""

from __future__ import annotations

import math

import pytest

from hagi.train.grokking_schedule import (
    grokking_lambda,
    grokking_threshold_time,
    grokking_weight_decay,
)


class TestR366ThresholdTime:
    def test_closed_form_matches_theorem(self) -> None:
        """R366: T = ln(r0/eps)/(eta*lambda) — точная форма."""
        for r0, eps, el in [(10.0, 0.1, 0.05), (2.0, 1e-3, 0.2),
                            (5.0, 0.5, 1.0)]:
            assert grokking_threshold_time(r0, eps, el) == pytest.approx(
                math.log(r0 / eps) / el
            )

    def test_residual_reaches_eps_by_T(self) -> None:
        """R366 (динамика): r0*(1−ηλ)^T ≤ eps при T из формулы."""
        r0, eps, el = 4.0, 0.01, 0.1
        T = grokking_threshold_time(r0, eps, el)
        assert r0 * (1.0 - el) ** T <= eps + 1e-12


class TestR366Lambda:
    def test_round_trip_recovers_T_star(self) -> None:
        """R366 (обратная форма): λ(T*) даёт порог ровно к T*."""
        for r0, eps, eta, T in [(8.0, 0.2, 0.5, 100.0),
                                (3.0, 0.01, 0.1, 500.0)]:
            lam = grokking_lambda(r0, eps, eta, T)
            assert eta * lam <= 1.0  # фазовая безопасность
            assert grokking_threshold_time(r0, eps, eta * lam) == pytest.approx(T)

    def test_phase_safety_raises(self) -> None:
        """R366 (условие): ln(r0/eps)/T* > eta — расходимость, ValueError."""
        # ln(1e6)/10 ≈ 1.38 > eta=1.0
        with pytest.raises(ValueError, match="phase divergence"):
            grokking_lambda(1e6, 1.0, 1.0, 10.0)

    def test_domain_errors(self) -> None:
        """Посылки теоремы: r0>eps>0, eta>0, T*>0, eta_lambda>0."""
        with pytest.raises(ValueError):
            grokking_threshold_time(0.0, 0.1, 0.1)
        with pytest.raises(ValueError):
            grokking_threshold_time(1.0, 0.0, 0.1)
        with pytest.raises(ValueError):
            grokking_threshold_time(0.1, 0.1, 0.1)  # eps >= r0
        with pytest.raises(ValueError):
            grokking_threshold_time(1.0, 0.1, 0.0)
        with pytest.raises(ValueError):
            grokking_lambda(1.0, 0.1, 0.0, 10.0)
        with pytest.raises(ValueError):
            grokking_lambda(1.0, 0.1, 0.5, 0.0)


class TestR366WeightDecay:
    def test_muon_mapping(self) -> None:
        """Мост в Muon: eta·wd·fan == eta·λ_theorem (optim.py decay)."""
        lam = grokking_lambda(10.0, 0.1, 0.5, 200.0)
        for fan in (1.0, 1.5, 2.0):
            wd = grokking_weight_decay(10.0, 0.1, 0.5, 200.0, fan_ratio=fan)
            assert 0.5 * wd * fan == pytest.approx(0.5 * lam)

    def test_fan_ratio_domain(self) -> None:
        with pytest.raises(ValueError):
            grokking_weight_decay(10.0, 0.1, 0.5, 200.0, fan_ratio=0.0)
