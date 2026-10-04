"""T5: Pinsker anti-collapse — замена опровергнутой hcert (R108).

Рецензия: «KL <= delta ⟹ H(q) >= H(m) - delta» опровергнута
контрпримером (m=(0.9,0.1), q=(0.99,0.01): KL=0.14, потеря 0.27).
План T5: |H(p)-H(q)| <= tau*log(V-1) + h2(tau), tau <= sqrt(KL/2) —
сублинейная замена. Тесты пришпиливают каждый член цепочки.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.distill_recursion import (  # noqa: E402
    corrected_entropy_floor,
    entropy_tv_modulus,
    pinsker_floor_correction,
    pinsker_tv_bound,
    shannon_entropy,
    tv_distance,
)

V = 100


class TestTVDistance:
    def test_identical_is_zero(self):
        p = [0.2, 0.3, 0.5]
        assert tv_distance(p, p) == pytest.approx(0.0)

    def test_disjoint_support_is_one(self):
        assert tv_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_half_l1(self):
        # |0.6-0.4| + |0.4-0.6| = 0.4; TV = 0.2
        assert tv_distance([0.6, 0.4], [0.4, 0.6]) == pytest.approx(0.2)


class TestPinskerTVBound:
    def test_zero_kl_zero_tv(self):
        assert pinsker_tv_bound(0.0) == pytest.approx(0.0)

    def test_sqrt_half(self):
        assert pinsker_tv_bound(2.0) == pytest.approx(1.0)

    def test_negative_kl_raises(self):
        with pytest.raises(ValueError):
            pinsker_tv_bound(-0.1)


class TestEntropyTVModulus:
    def test_zero_tv_zero_modulus(self):
        assert entropy_tv_modulus(0.0, V) == pytest.approx(0.0)

    def test_tau_one_is_log_v_minus_one(self):
        # tau=1: h2(1)=0, модуль = log(V-1) — точная граница
        assert entropy_tv_modulus(1.0, V) == pytest.approx(math.log(V - 1))

    def test_monotone_in_tau(self):
        assert entropy_tv_modulus(0.2, V) < entropy_tv_modulus(0.4, V)

    def test_sharp_pair_saturates(self):
        # Ho-Yeung худший случай: q вырожден, p=(1-tau, tau/(V-1),...)
        tau = 0.3
        p = [1.0 - tau] + [tau / (V - 1)] * (V - 1)
        q = [1.0] + [0.0] * (V - 1)
        loss = shannon_entropy(p) - shannon_entropy(q)
        bound = entropy_tv_modulus(tv_distance(p, q), V)
        assert bound >= loss - 1e-12  # граница доминирует худший случай

    def test_v_below_two_raises(self):
        with pytest.raises(ValueError):
            entropy_tv_modulus(0.5, 1)


class TestPinskerFloorCorrection:
    def test_counterexample_loss_is_covered(self):
        # линейная ставка провалилась: KL=0.14 -> потеря 0.27 > delta.
        # Сублинейная должна покрывать: tau=0.09 (точный TV),
        # V=2: 0.09*log(1) + h2(0.09) >= 0.27? h2(0.09)=0.31 >= 0.27 OK
        tv = tv_distance([0.9, 0.1], [0.99, 0.01])
        hm = shannon_entropy([0.9, 0.1])
        hq = shannon_entropy([0.99, 0.01])
        loss = hm - hq
        assert entropy_tv_modulus(tv, 2) >= loss - 1e-12

    def test_sublinear_beats_linear_on_counterexample(self):
        # при KL=0.1445 линейный delta=KL < потери 0.27; сублинейный покрывает
        kl = 0.1445
        assert pinsker_floor_correction(kl, 2) >= 0.27

    def test_zero_kl_zero_correction(self):
        assert pinsker_floor_correction(0.0, V) == pytest.approx(0.0)


class TestCorrectedEntropyFloor:
    def test_zero_kl_exact_floor(self):
        # KL=0: коррекция 0, floor = h_data
        assert corrected_entropy_floor(0.5, 0.0, V, 6.5) == pytest.approx(6.5)

    def test_positive_kl_lowers_floor(self):
        hi = corrected_entropy_floor(0.5, 0.1, V, 6.5)
        lo = corrected_entropy_floor(0.5, 0.0, V, 6.5)
        assert hi < lo

    def test_floor_finite_and_below_data(self):
        f = corrected_entropy_floor(0.1, 0.1, 32768, 6.5)
        assert math.isfinite(f)
        assert f < 6.5
