# §17 Active-Inference port tests. Lean SSOT: primes
# Hagi/Probability/GibbsConcentration.lean (proxy_cert_preserved/lost,
# efe_ceiling, gap_decomposition/gapII_nonpos/gap_graceful,
# compounding_bound/sq, amortized_policy_bound).

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.active_inference import (  # noqa: E402
    amortized_policy_slack,
    compounding_bound,
    compounding_sq,
    efe_ceiling,
    gap_graceful,
    mpc_horizon_one_resets,
    proxy_cert_lost_bound,
    proxy_cert_preserved,
)


def test_lower_proxy_preserves_certificate_with_slack():
    # s = 1, h = 2, ĥ = 1.5 (lower bound): margin E − ĥ = 1.5 >= s
    assert proxy_cert_preserved(1.0, 2.0, 1.5) == pytest.approx(1.5)
    assert proxy_cert_preserved(1.0, 2.0, 1.5) >= 1.0
    # tight proxy ĥ = h: margin exactly s
    assert proxy_cert_preserved(1.0, 2.0, 2.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        proxy_cert_preserved(1.0, 2.0, 2.5)


def test_upper_proxy_loses_certificate():
    # ĥ > h: E − ĥ = s − (ĥ − h) strictly below s
    m = proxy_cert_lost_bound(1.0, 2.0, 3.0)
    assert m == pytest.approx(0.0)
    assert m < 1.0
    with pytest.raises(ValueError):
        proxy_cert_lost_bound(1.0, 2.0, 2.0)  # not strictly upper


def test_efe_ceiling():
    assert efe_ceiling(3.0, 1.0) == pytest.approx(2.0)
    assert efe_ceiling(3.0, 0.0) == pytest.approx(3.0)  # ceiling is P
    with pytest.raises(ValueError):
        efe_ceiling(3.0, -1.0)


def test_gap_graceful_linear_degradation():
    assert gap_graceful(0.2, -0.1, 0.05, 0.3, 0.2, 0.1) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        gap_graceful(0.5, 0.0, 0.0, 0.3, 0.2, 0.1)  # component exceeds slack


def test_compounding_bounds_and_mpc_reset():
    assert compounding_bound(0.5, 2.0, 3) == pytest.approx(0.25)
    assert compounding_sq(0.5, 2.0, 2) == pytest.approx((0.25**2) * 4.0)
    with pytest.raises(ValueError):
        compounding_bound(-1.0, 1.0, 2)
    # horizon-one MPC: chain restarts from the current error
    assert mpc_horizon_one_resets(0.7) == 0.7


def test_amortized_policy_two_c_slack():
    assert amortized_policy_slack(0.3) == pytest.approx(0.6)
    assert amortized_policy_slack(0.0) == 0.0
    with pytest.raises(ValueError):
        amortized_policy_slack(-0.1)
