"""Tests: exact layer/embedding tunings (Layers.lean wave-5,
Transfer.lean wave-4 ports)."""
import numpy as np
import pytest

from hagi.model.layer_tuning import (
    admission_variance_ratio,
    branchscale_minmax_floor,
    branchscale_variance_init,
    dead_layer_stop,
    vocab_critical_point,
)


def test_branchscale_uniform_allocation():
    # equal variances -> equal scales, budget preserved
    v = np.array([1.0, 1.0, 1.0, 1.0])
    s = branchscale_variance_init(v)
    assert np.allclose(s, s[0])
    assert np.isclose(np.sum(s**2), 0.5)


def test_branchscale_minmax_product_uniform():
    # unequal variances: s_l^2 * v_l must be constant (the floor)
    v = np.array([4.0, 1.0, 0.25])
    s = branchscale_variance_init(v)
    prod = s**2 * v
    assert np.allclose(prod, prod[0])
    # and the attained floor equals the proved bound B/sum(1/v)
    assert np.isclose(prod[0], branchscale_minmax_floor(v, 0.5))


def test_branchscale_floor_is_lower_bound():
    v = np.array([3.0, 0.5, 2.0])
    B = 0.5
    floor = branchscale_minmax_floor(v, B)
    # any allocation, e.g. uniform, has worst contribution >= floor
    s_uni = np.sqrt(B / 3) * np.ones(3)
    assert np.max(s_uni**2 * v) >= floor - 1e-12


def test_dead_layer_stop_healthy():
    v = np.array([0.9, 0.7, 0.5, 0.2, 0.05])
    assert dead_layer_stop(v, lam=0.1) == 4  # V_5=0.05 <= 0.1
    assert dead_layer_stop(v, lam=0.6) == 2


def test_dead_layer_stop_all_dead():
    assert dead_layer_stop(np.array([0.01, 0.005]), lam=0.1) == 0


def test_dead_layer_nonmonotone_raises():
    with pytest.raises(ValueError, match="non-monotone"):
        dead_layer_stop(np.array([0.1, 0.5, 0.2]), lam=0.05)


def test_vocab_critical_point_recovers_c():
    c_true = 32768.0
    vs = np.array([16384.0, 32768.0, 65536.0])
    ls = np.log(vs) + c_true / vs  # exact model
    c, vstar = vocab_critical_point(vs, ls)
    assert c == pytest.approx(c_true)
    assert vstar == pytest.approx(c_true)


def test_vocab_critical_point_noisy_fit():
    c_true = 30000.0
    vs = np.array([8192.0, 16384.0, 32768.0])
    noise = np.array([0.05, -0.05, 0.02])
    ls = np.log(vs) + c_true / vs + noise
    c, _ = vocab_critical_point(vs, ls)
    assert abs(c - c_true) / c_true < 0.05  # within 5%


def test_admission_variance_ratio():
    # same ordering as gap/cost (the 1/2 commutes)
    dv = np.array([2.0, 1.0, 3.0])
    dc = np.array([1.0, 1.0, 10.0])
    assert admission_variance_ratio(dv, dc) == 0  # 2/1 beats 3/10
    with pytest.raises(ValueError):
        admission_variance_ratio(dv, np.array([1.0, 0.0, 1.0]))
