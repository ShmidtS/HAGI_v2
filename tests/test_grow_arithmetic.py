"""Арифметика решений роста (волна перечитки 2026-10-24): порты
Growth/Grow, Ensemble/Hoeffding, NoiseDisentangle, SeedOnly, BatchCap."""

from __future__ import annotations

import math

import pytest

from hagi.train.grow_arithmetic import (
    batch_noise_variance,
    certified_gain,
    critical_batch,
    disagreement_gain_floor,
    disagreement_noise_floor,
    grow_epsilon_stop,
    merge_skip_certificate,
    seed_only_dispersion,
    two_gap_ceiling,
)


class TestCertifiedGain:
    def test_exact_arithmetic(self):
        # N=6, M=5.5, c=5.0: (M−c)/(N+1) = 0.5/7
        assert certified_gain(6, 5.5, 5.0) == pytest.approx(0.5 / 7)

    def test_positive_iff_better_than_mean(self):
        assert certified_gain(3, 5.0, 4.9) > 0
        assert certified_gain(3, 5.0, 5.1) < 0

    def test_epsilon_stop(self):
        # лучший кандидат 5.49 при среднем 5.5 и N=10: выигрыш 0.01/11 < 0.01
        assert grow_epsilon_stop(10, 5.5, 5.49, eps=0.01)
        # сильный кандидат: выигрыш 0.5/11 > 0.01 — продолжать
        assert not grow_epsilon_stop(10, 5.5, 5.0, eps=0.01)


class TestMergeSkip:
    def test_ceiling_m2_over_4(self):
        # ½ log cosh M ≤ M²/4, равенство в нуле
        for m in (0.0, 0.5, 2.0, 8.0):
            assert two_gap_ceiling(m) <= m * m / 4.0 + 1e-12

    def test_skip_when_small_diameter(self):
        # ε=0.01: диаметр 0.1 → 0.0025 < 0.01 → SKIP
        assert merge_skip_certificate(0.1, eps=0.01)
        # диаметр 0.5 → 0.0625 > 0.01 → не скипать
        assert not merge_skip_certificate(0.5, eps=0.01)

    def test_transport_efficiency(self):
        # η=0.5 сокращает потолок вдвое: M=0.4 → 0.04/2 = 0.02 < 0.03
        assert merge_skip_certificate(0.4, eps=0.03, transport_efficiency=0.5)
        assert not merge_skip_certificate(0.4, eps=0.03, transport_efficiency=1.0)


class TestDisagreementFloors:
    def test_gain_floor_positive(self):
        floor = disagreement_gain_floor(p_u=0.4, p_v=0.3, delta=1.0)
        assert floor > 0
        # точная формула
        expected = 0.5 * math.log(1.0 + 0.12 * (math.cosh(1.0) - 1.0))
        assert floor == pytest.approx(expected)

    def test_gain_floor_grows_with_delta(self):
        small = disagreement_gain_floor(0.4, 0.3, 0.2)
        big = disagreement_gain_floor(0.4, 0.3, 4.0)
        assert big > small

    def test_noise_floor(self):
        assert disagreement_noise_floor(0.02, 6) == pytest.approx(0.02 / 6)
        # twoGap 0.001 при σ²/N = 0.0033 — шум, не сигнал
        assert 0.001 < disagreement_noise_floor(0.02, 6)


class TestSeedOnlyDispersion:
    def test_geometric_decay(self):
        # a=0.5: сумма → 2; дисперсия ограничена
        total = seed_only_dispersion(disp0=1.0, rate=0.5, steps=50)
        assert total == pytest.approx(2.0, rel=1e-6)
        assert seed_only_dispersion(1.0, 0.5, 3) == pytest.approx(1.75)  # 1+0.5+0.25

    def test_linear_when_no_decay(self):
        assert seed_only_dispersion(1.0, 1.0, 10) == 10.0


class TestCriticalBatch:
    def test_formula(self):
        # Y=1, eps=3, Z=0: b* = 9/(4·9) = 0.25
        assert critical_batch(1.0, 3.0) == pytest.approx(0.25)
        # Y=2, eps=3: 9·4/(4·9) = 1.0
        assert critical_batch(2.0, 3.0) == pytest.approx(1.0)

    def test_bias_reduces_denominator(self):
        assert critical_batch(1.0, 3.0, bias_term=1.0) == pytest.approx(9.0 / 16.0)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            critical_batch(1.0, 1.0, bias_term=1.0)  # eps <= bias
        with pytest.raises(ValueError):
            critical_batch(-1.0, 3.0)


class TestBatchNoise:
    def test_per_token(self):
        assert batch_noise_variance(0.5, 1024) == pytest.approx(0.5 / 1024)
