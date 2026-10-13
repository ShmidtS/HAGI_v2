"""§0 certificate telescope (System/GrowthState.lean
stage_certificate_telescope / termination_from_telescope)."""

import json

import pytest

from hagi.train.stage_telescope import (
    LEDGER_NAME,
    StageCertificate,
    append_certificate,
    certificate_from_ladder_entry,
    load_ledger,
    rebuild_from_ladder,
    telescope_bound,
    termination_bound,
)


def cert(gen, before, after, dec=None, risk=0.0):
    if dec is None:
        dec = max(0.0, before - after)
    return StageCertificate(gen, before, after, dec, risk, n_eval=100)


def test_telescope_bound_sum():
    chain = [cert(1, 5.0, 4.8), cert(2, 4.8, 4.6), cert(3, 4.6, 4.5)]
    # after(T-1) = 4.5 ≤ 5.0 - (0.2+0.2+0.1) + 0
    assert telescope_bound(chain) == pytest.approx(4.5)


def test_discontinuity_rejected_exact():
    chain = [cert(1, 5.0, 4.8), cert(2, 4.79, 4.6)]
    with pytest.raises(ValueError, match="discontinuity"):
        telescope_bound(chain)


def test_discontinuity_within_noise_tol_ok():
    chain = [cert(1, 5.0, 4.8), cert(2, 4.79, 4.6)]
    assert telescope_bound(chain, contiguity_tol=0.02) is not None


def test_unsound_stage_rejected():
    chain = [cert(1, 5.0, 4.8, dec=0.5)]  # 4.8 > 5.0-0.5+0
    with pytest.raises(ValueError, match="unsound"):
        telescope_bound(chain)


def test_termination_bound_math():
    chain = [cert(1, 5.0, 4.9), cert(2, 4.9, 4.8)]  # decrease ~0.1 ≥ eps
    bound = termination_bound(chain, phi_min=4.0, eps=0.09, r=0.0)
    assert bound == pytest.approx((5.0 - 4.0) / 0.09)


def test_termination_preconditions():
    chain = [cert(1, 5.0, 4.95)]  # decrease 0.05 < eps 0.1
    with pytest.raises(ValueError, match="decrease"):
        termination_bound(chain, phi_min=4.0, eps=0.1, r=0.0)
    with pytest.raises(ValueError, match="eps"):
        termination_bound([cert(1, 5.0, 4.9)], phi_min=4.0, eps=0.0, r=0.0)


def test_ladder_entry_mapping():
    accepted = {
        "gen": 42, "accepted": True, "incumbent_ce": 4.82,
        "cooldown_ce": 4.79, "ft_ce": 4.80, "accept_n": 1500,
    }
    c = certificate_from_ladder_entry(accepted)
    assert c is not None
    assert (c.before, c.after, c.decrease, c.risk_spend, c.n_eval) == (
        4.82, 4.79, pytest.approx(0.03), 0.0, 1500)
    rejected = dict(accepted, accepted=False)
    assert certificate_from_ladder_entry(rejected) is None


def test_ledger_roundtrip_and_rebuild(tmp_path):
    ladder = tmp_path / "growth_ladder.jsonl"
    entries = [
        {"gen": 1, "accepted": True, "incumbent_ce": 5.0,
         "cooldown_ce": 4.9, "accept_n": 100},
        {"gen": 2, "accepted": False, "incumbent_ce": 4.9,
         "ft_ce": 4.95, "accept_n": 100},
        {"gen": 3, "accepted": True, "incumbent_ce": 4.9,
         "ft_ce": 4.85, "accept_n": 100},
    ]
    ladder.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    n = rebuild_from_ladder(ladder, tmp_path)
    assert n == 2
    chain = load_ledger(tmp_path)
    assert [c.generation for c in chain] == [1, 3]
    # contiguity: gen3.before (4.9) == gen1.after (4.9) — rejected gen
    # kept the incumbent, chain over accepted stages is contiguous.
    assert telescope_bound(chain) is not None
    assert (tmp_path / LEDGER_NAME).exists()
    append_certificate(tmp_path, cert(4, 4.85, 4.8))
    assert len(load_ledger(tmp_path)) == 3
