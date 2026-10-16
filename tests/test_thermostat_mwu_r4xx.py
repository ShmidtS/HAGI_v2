"""Property tests for the R407-R412 thermostat port and the R432-R443
MWU/Gibbs port + R419 admission margin soundness."""

import math

import numpy as np
import pytest

from hagi.train.thermostat import (
    ThermostatConfig, consolidation_certificate, gibbs_share, phase_point,
    temperature,
)
from hagi.train.mwu import (
    eta_star, gibbs_hedge, mwu_double_upper, potential_sandwich,
    static_regret,
)
from hagi.train.memory_cycle import (
    AdmissionMeasurements, AdmissionThresholds, admission_margin_sound,
)

CFG = ThermostatConfig(t0=1.0, alpha=0.9, theta=0.05)


def test_phase_point_unified_horizon():
    # thermostat_phase_point (R407): both crossings at/ before N
    delta, j = 0.5, 0.3
    n = phase_point(CFG, delta, j)
    assert n >= 0
    tt = temperature(CFG, n)
    assert math.exp(-delta / tt) <= CFG.theta + 1e-9   # exploration stopped
    assert CFG.theta * tt <= j + 1e-12                 # merge gate open
    # monotone: past N both keep holding (alpha<1 cooling)
    tt2 = temperature(CFG, n + 10)
    assert math.exp(-delta / tt2) <= CFG.theta
    assert CFG.theta * tt2 <= j


def test_gibbs_share_concentration():
    # phase_concentration (R408): share >= 1 - C*theta past N
    energies = [0.0, 0.5, 0.9, 2.0]  # v*=0 minimizer, gap delta=0.5
    n = phase_point(CFG, 0.5, 0.3)
    tt = temperature(CFG, n)
    share = gibbs_share(energies, 0, tt)
    assert share >= 1.0 - 3 * CFG.theta - 1e-9


def test_consolidation_certificate_3in1():
    # consolidation_certificate (R412): all three at once past N
    energies = [0.0, 0.6, 1.2, 3.0]
    cert = consolidation_certificate(CFG, delta=0.6, j_gap=0.3,
                                     energies=energies, v_star=0,
                                     t=phase_point(CFG, 0.6, 0.3) + 5)
    assert cert["past_phase_point"]
    assert cert["consolidation"] or cert["share_v_star"] > 0.5
    # before the phase point the certificate is honestly not claimed
    early = consolidation_certificate(CFG, delta=0.6, j_gap=0.3,
                                      energies=energies, v_star=0, t=0)
    assert not early["past_phase_point"]


def test_mwu_sandwich():
    # mwu_potential_squeeze (R435/R434)
    ls = [0.1, 0.4, 0.7, 1.0]
    lo, z, hi = potential_sandwich(ls, 0.3)
    assert lo <= z + 1e-12 <= hi
    # double upper (R443): both bounds hold
    b1, z2, b2 = mwu_double_upper(ls, 0.3)
    assert math.isclose(z, z2, rel_tol=1e-12)
    assert z2 <= b1 + 1e-12 and z2 <= b2 + 1e-12


def test_eta_star_equalizes():
    # R443: at eta* the two competing bounds are equal
    ls = [0.2, 0.5, 0.9, 1.3]
    e = eta_star(ls)
    n = len(ls)
    mean = sum(ls) / n
    assert math.isclose(2 * e * mean, math.log(n) + e * min(ls),
                        rel_tol=1e-9)


def test_gibbs_hedge_bridge():
    # gibbs_hedge_squeeze (R441): eta = 1/T recovers the sandwich
    es = [0.0, 0.3, 0.8]
    lo, z, hi = gibbs_hedge(es, temp=4.0)
    assert lo <= z + 1e-12 <= hi
    # direct statistic check: -ln((1/n) sum e^{-E/T})
    stat = -math.log(sum(math.exp(-e / 4.0) for e in es) / 3)
    assert math.isclose(z, stat, rel_tol=1e-9)


def test_static_regret_telescope():
    # mwu_regret_static (R440) via R438/R439 telescoping
    seqs = [[0.1, 0.2, 0.3], [0.4, 0.1, 0.2], [0.2, 0.5, 0.1]]
    lo, z, hi = static_regret(seqs, eta=0.5)
    assert lo <= z + 1e-12 <= hi


def test_admission_margin_sound():
    # MemoryController.admission_margin_sound (R419)
    t = AdmissionThresholds(tau=0.1, lam_m=1.0, lam_r=2.0, eps_cert=0.05)
    m = AdmissionMeasurements(gain_hat=1.0, delta_c_state=0.3,
                              delta_r_protected=0.2)
    # need = 0.3+0.4=0.7; admit needs gain_hat >= 0.75 — holds
    ok, covered = admission_margin_sound(t, m, gain_true=0.96)
    assert ok and covered >= 0
    # true gain below need but within eps of hat — premise broken or
    # admit premise fails: not certified
    ok2, cov2 = admission_margin_sound(t, m, gain_true=0.68)
    assert not ok2 and cov2 < 0
    # estimate too far from truth — no soundness claim
    ok3, _ = admission_margin_sound(t, m, gain_true=1.5)
    assert not ok3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
