"""Certificate-telescope wiring in orchestrator/real_cycle (2026-10-16).

THEORY_MAP open item "certificate-telescope in orchestrator/real_cycle.py":
every ACCEPTED synthetic-recursion generation appends a StageCertificate
(before = incumbent macro CE, after = candidate macro CE, risk_spend =
worst-source regression) to <parent>/stage_telescope.jsonl; rejected
generations emit nothing (no stage happened).
"""

from pathlib import Path

from hagi.orchestrator import real_cycle as rc


def test_generation_index_parser():
    assert rc._generation_index("generation-7") == 7
    assert rc._generation_index("depth-zero") == 0
    try:
        rc._generation_index("gen7")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_accepted_generation_appends_certificate(tmp_path: Path):
    class FakeResult:
        decision = "accepted"
        incumbent_macro_ce = 4.7
        candidate_macro_ce = 4.5
        worst_source_regression = 0.02

    class FakeRequest:
        generation_id = "generation-3"

    rc._record_telescope_certificate(tmp_path / "run", FakeRequest(), FakeResult())
    ledger = tmp_path / "stage_telescope.jsonl"
    line = ledger.read_text(encoding="utf-8").strip()
    assert '"generation": 3' in line
    assert '"risk_spend": 0.02' in line
    assert '"decrease": 0.19' in line or '"decrease": 0.2' in line


def test_rejected_generation_emits_nothing(tmp_path: Path):
    class FakeResult:
        decision = "rejected"
        incumbent_macro_ce = 4.7
        candidate_macro_ce = 4.5
        worst_source_regression = 0.0

    class FakeRequest:
        generation_id = "generation-4"

    rc._record_telescope_certificate(tmp_path / "run", FakeRequest(), FakeResult())
    assert not (tmp_path / "stage_telescope.jsonl").exists()
