"""Property tests for the theory-derived optimizer primitives
(warmup_eta / armijo_accept / wd_norm_bound / adaptive warmup)."""

import math

import pytest

from hagi.train.optim import (
    armijo_accept, learning_rate_at, set_learning_rate, warmup_eta,
    wd_norm_bound,
)


def test_warmup_eta_monotone_and_cap():
    # WarmupSchedule.warmupEta: c*min{1/K0, Delta^-rho/(3^rho K_rho)}
    assert math.isclose(warmup_eta(1.0, c=0.5, k0=1.0, k_rho=1.0, rho=1.0),
                        0.5 * min(1.0, 1.0 / 3.0))
    # monotone: smaller gap -> larger eta (warmup_eta_monotone)
    assert warmup_eta(10.0) <= warmup_eta(1.0) <= warmup_eta(0.1)
    # zero gap -> K0 branch (safe cap)
    assert math.isclose(warmup_eta(0.0, c=0.5, k0=2.0), 0.25)


def test_armijo_accept_cases():
    # ArmijoAcceptance.armijo_sufficient_decrease
    ok, req = armijo_accept(5.0, 4.9, -2.0, 1.0, 0.1)
    assert ok and math.isclose(req, 0.01, abs_tol=1e-12)
    # insufficient decrease -> reject
    ok2, _ = armijo_accept(5.0, 4.9999, -2.0, 1.0, 0.1)
    assert not ok2
    # direction premise violated (g,d > -c1||d||^2) -> cannot certify
    ok3, _ = armijo_accept(5.0, 4.0, -0.05, 1.0, 0.1)
    assert not ok3
    # positive L shrinks the required decrease (c1 - L*alpha/2)
    _, req_l = armijo_accept(5.0, 4.9, -2.0, 1.0, 0.1, lipschitz_l=1.0)
    assert req_l < req


def test_wd_norm_bound():
    # WeightDecay.wd_norm_bound: ||w|| <= max(||w0||, G/lam)
    assert math.isclose(wd_norm_bound(0.5, 0.01), 50.0)
    with pytest.raises(ValueError):
        wd_norm_bound(1.0, 0.0)


def _cfg():
    from hagi.config import Config, TrainConfig, ScheduleConfig
    return Config(train=TrainConfig(
        schedule=ScheduleConfig(warmup_steps=2000)))


def test_schedule_regression():
    # adaptive_warmup off (default): identical linear-ramp schedule
    lr_plain = learning_rate_at(100, 0.1, _cfg())
    assert math.isclose(lr_plain, 0.1 * 100 / 2000)


def test_cfg_like():
    cfg = _cfg()
    lr = learning_rate_at(100, 0.1, cfg)
    assert math.isclose(lr, 0.1 * 100 / 2000)
    # adaptive path: early step with large gap -> capped at 2*linear
    cfg.train.schedule.adaptive_warmup = True
    lr_ad = learning_rate_at(100, 0.1, cfg, loss_gap=5.0)
    assert lr_ad <= 0.1 * 2.0 * (100 / 2000) + 1e-15
    assert lr_ad > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
