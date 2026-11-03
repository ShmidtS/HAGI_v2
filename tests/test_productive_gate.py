# Productive-transition gate (audit §3.2 / P0).
# Problem SSOT: primes Hagi/Unified/Assumptions.lean trunc_idle_witness —
# the premise package admits a fully idle trajectory; soundness ≠ progress.
# Runtime: an ACCEPT must be sound AND productive (measured metric moved
# beyond the noise floor). Wiring target: stage_telescope / accept gates.

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.productive_gate import (  # noqa: E402
    gate_accept,
    productive,
    productive_with_noise_floor,
)


def test_productive_requires_measured_movement():
    # capability path
    assert productive(0.05, 0.0, 0.01, 0.01)
    # loss path (loss fell by more than δ_L)
    assert productive(0.0, -0.2, 0.01, 0.1)
    # idle: nothing moved
    assert not productive(0.0, 0.0, 0.01, 0.01)
    # sub-noise movement is still idle
    assert not productive(0.005, -0.05, 0.01, 0.1)
    with pytest.raises(ValueError):
        productive(0.0, 0.0, 0.0, 0.01)  # δ must be positive


def test_trunc_idle_witness_runtime_analog_is_rejected():
    # the Lean idle witness: every delta is exactly zero
    verdict = gate_accept(sound=True, delta_capability=0.0, delta_loss=0.0,
                          cap_floor=0.01, loss_floor=0.05)
    assert verdict == "REJECT_IDLE"


def test_gate_accept_verdicts():
    assert gate_accept(True, 0.1, 0.0, 0.01, 0.05) == "ACCEPT"
    assert gate_accept(True, 0.0, -0.2, 0.01, 0.05) == "ACCEPT"
    assert gate_accept(True, 0.0, 0.0, 0.01, 0.05) == "REJECT_IDLE"
    assert gate_accept(False, 0.1, -0.2, 0.01, 0.05) == "REJECT_UNSOUND"


def test_noise_floor_discipline():
    v = productive_with_noise_floor(0.0, -0.04, cap_floor=0.01, loss_floor=0.05)
    assert v == "UNPRODUCTIVE"  # below the floor: noise, not progress
    with pytest.raises(ValueError):
        productive_with_noise_floor(float("nan"), 0.0, 0.01, 0.05)


def test_unified_transition_certificate_clean_and_idle():
    from hagi.train.productive_gate import TransitionCertificate

    good = TransitionCertificate.from_measurement(
        obj_before=5.0, obj_after=4.8, claimed_drop=0.2, risk=0.01,
        risk_limit=0.05, budget_before=10.0, budget_after=9.0, spend=1.0,
        measured=0.19, measurement_error=0.05, productive_ok=True,
    )
    assert good.verify() == []
    # the trunc_idle case: energy unchanged but the certificate was claimed
    idle = TransitionCertificate.from_measurement(
        obj_before=5.0, obj_after=5.0, claimed_drop=0.2, risk=0.0,
        risk_limit=0.05, budget_before=10.0, budget_after=10.0, spend=0.0,
        measured=0.2, measurement_error=0.05, productive_ok=False,
    )
    assert "objective_drop_pos" in idle.verify()
    assert "productive" in idle.verify()


def test_measurement_link_violation_is_caught():
    from hagi.train.productive_gate import TransitionCertificate

    bad = TransitionCertificate.from_measurement(
        obj_before=5.0, obj_after=4.8, claimed_drop=0.2, risk=0.0,
        risk_limit=0.05, budget_before=10.0, budget_after=9.0, spend=1.0,
        measured=0.1, measurement_error=0.05, productive_ok=True,
    )
    assert "measurement_link" in bad.verify()
