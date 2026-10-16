"""Property tests for src/hagi/train/round_viability.py — mirror
Lean theorems GainOperator/RoundViability/WallClockTakeoff."""

import math

import pytest
import torch

from hagi.train.round_viability import (
    advise, cycle_criterion, dev_energy, gainop_holds, k_eff,
    round_exhausted, round_viable, wallclock_takeoff,
)


def _viable_case(dev_scale: float):
    # hand-computed viable configuration (verified below premise-by-premise)
    torch.manual_seed(0)
    dev = torch.randn(4, 8) * dev_scale
    gamma, rho, k, xi = 0.1, 0.5, 2.0, 0.01
    c, d = 1.0, 4.0           # hcone: kC=2 <= D=4
    c_prime = c + gamma * d   # hstep: 1.4
    e = dev_energy(dev)
    # choose eta_p*eta_s so that the threshold premise holds with margin
    thresh = gamma * k * k * c + (1 - rho) * k * c + xi
    eta = (thresh + 0.5) / e if e > 0 else 1.0
    d_prime = rho * d + eta * e - xi + 0.1
    return (c, d, c_prime, d_prime, gamma, rho, k, xi, eta, dev)


def test_dev_energy_sum_of_sq_norms():
    dev = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    assert math.isclose(dev_energy(dev), 1 + 4 + 9.0)
    assert dev_energy(torch.zeros(3, 3)) == 0.0


def test_round_viable_accepts_and_rejects():
    c, d, cp, dp, gamma, rho, k, xi, eta, dev = _viable_case(1.0)
    ok, cert = round_viable(c, d, cp, dp, gamma, rho, k, xi,
                            eta, 1.0, dev)
    assert ok, cert
    assert cert["conclusion"] == "k*C' <= D'"
    # break the cone premise (kC > D) → not viable
    ok2, cert2 = round_viable(c, d=1.0, c_prime=c + gamma * 1.0,
                              d_prime=dp, gamma=gamma, rho=rho, k=k,
                              xi=xi, eta_p=eta, eta_s=1.0, dev=dev)
    assert not ok2 and not cert2["hcone"]
    # break the step premise
    ok3, cert3 = round_viable(c, d, cp + 1.0, dp, gamma, rho, k, xi,
                              eta, 1.0, dev)
    assert not ok3 and not cert3["hstep"]


def test_zero_dev_energy_exhausted():
    dev = torch.zeros(4, 8)
    assert round_exhausted(dev)
    assert cycle_criterion(dev) == "STOP_round_exhausted"
    assert advise(dev, {"gen": 42})["criterion"].startswith("STOP")
    dev2 = torch.ones(2, 2)
    assert cycle_criterion(dev2) == "CONTINUE"
    assert not round_exhausted(dev2)


def test_wallclock_bound_below_exact_product():
    # theorem: C0·exp(kΣτ − (k²/2)Στ²) ≤ C0·Π(1+kτ)
    c0, k = 2.0, 0.3
    taus = [0.1, 0.4, 1.0, 2.5]
    bound = wallclock_takeoff(c0, k, taus)
    prod = c0
    for t in taus:
        prod *= 1 + k * t
    assert bound <= prod + 1e-9
    assert bound > 0
    # single tiny step: bound ≈ product (slack second-order)
    b1 = wallclock_takeoff(c0, k, [1e-4])
    assert math.isclose(b1, c0 * (1 + k * 1e-4), rel_tol=1e-6)


def test_k_eff_formula():
    assert math.isclose(k_eff(0.5, math.e - 1, 2.0), 0.25)
    # p0·ln(1+α)/τ monotone in α and p0
    assert k_eff(0.5, 0.2, 1.0) < k_eff(0.5, 0.5, 1.0)


def test_gainop_predicate():
    assert gainop_holds(1.0, 2.0, e_dev=1.0, rho=0.5, eta=1.0, xi=0.1)
    assert not gainop_holds(1.0, 1.0, e_dev=1.0, rho=0.5, eta=1.0, xi=0.1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
