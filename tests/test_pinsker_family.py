# §18/§19 ports: Pinsker family + TV duality + DV/Cramér.
# Lean SSOT: primes Hagi/Probability/PinskerTV.lean, ChernoffCGF.lean.
# Numerical verification on distributions where the Lean theorems apply
# (strictly positive p, q summing to 1).

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.pinsker_family import (  # noqa: E402
    bhattacharyya,
    kl,
    kl_variational_lower,
    mean_shift_tv_bound,
    neg_log_bc_le_half_kl,
    one_minus_bc_le_half_kl,
    pinsker_sqrt_kl,
    subgauss_dev_kl,
    tv,
    tv_dual_abs,
    tv_le_pos_part_kl,
    tv_le_sqrt_one_minus_exp,
)


def _p() -> list[float]:
    return [0.5, 0.3, 0.15, 0.05]


def _q() -> list[float]:
    return [0.2, 0.3, 0.3, 0.2]


def _mixture(lam: float) -> list[float]:
    return [lam * p + (1 - lam) * q for p, q in zip(_p(), _q())]


def test_bhattacharyya_cauchy_bound():
    assert bhattacharyya(_p(), _q()) <= 1.0 + 1e-12
    assert bhattacharyya(_p(), _p()) == pytest.approx(1.0)


def test_one_minus_bc_le_half_kl():
    assert 1 - bhattacharyya(_p(), _q()) <= one_minus_bc_le_half_kl(_p(), _q()) + 1e-12


def test_pinsker_sqrt_kl_binds_tv():
    v = tv(_p(), _q())
    bound = pinsker_sqrt_kl(_p(), _q())
    assert v <= bound + 1e-12
    # Pinsker is the sqrt form: linear KL >= TV is FALSE in general
    # (Lean counterexample 0.368 < 0.4); here KL=0.285 < TV=0.3 demonstrates it
    assert kl(_p(), _q()) < v


def test_neg_log_bc_le_half_kl():
    bc = bhattacharyya(_p(), _q())
    assert -math.log(bc) <= neg_log_bc_le_half_kl(_p(), _q()) + 1e-12
    # BC >= exp(-KL/2)
    assert bc >= math.exp(-kl(_p(), _q()) / 2.0) - 1e-12


def test_bretagnolle_huber_tighter_at_large_kl():
    # disjoint-ish distributions: large KL
    p = [0.999, 0.001]
    q = [0.001, 0.999]
    k = kl(p, q)
    bh = tv_le_sqrt_one_minus_exp(p, q)
    assert tv(p, q) <= bh + 1e-12
    assert bh < math.sqrt(k)  # tighter than plain Pinsker at large KL


def test_tv_le_pos_part_kl():
    # symmetric KL halves: the positive-part sum >= TV for any pair
    for lam in (0.2, 0.5, 0.8):
        r = _mixture(lam)
        assert tv(r, _q()) <= tv_le_pos_part_kl(r, _q()) + 1e-12


def test_tv_dual_abs():
    f = [1.0, 0.0, 0.0, 0.0]
    gap = tv_dual_abs(_p(), _q(), f)
    assert gap == pytest.approx(abs(_p()[0] - _q()[0]))
    assert gap <= tv(_p(), _q()) + 1e-12
    with pytest.raises(ValueError):
        tv_dual_abs(_p(), _q(), [1.5, 0, 0, 0])


def test_dv_variational_lower_bound():
    # E_p f − ln E_q e^f <= KL for ANY f
    for f in ([0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.2, 0.5, -0.3, 1.0]):
        assert kl_variational_lower(_p(), _q(), f) <= kl(_p(), _q()) + 1e-12
    # attained at f = ln(p/q)
    f_star = [math.log(pi / qi) for pi, qi in zip(_p(), _q())]
    assert kl_variational_lower(_p(), _q(), f_star) >= kl(_p(), _q()) - 1e-9


def test_subgauss_dev_and_mean_shift():
    assert subgauss_dev_kl(2.0, 1.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        subgauss_dev_kl(1.0, 0.0)
    assert mean_shift_tv_bound(2.0, 0.0, 0.1) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        mean_shift_tv_bound(0.0, 1.0, 0.1)
