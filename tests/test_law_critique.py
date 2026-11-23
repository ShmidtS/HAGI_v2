import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hagi.law_critique import run_law_self_critique


class IdentityLaw:
    def apply(self, ctx):
        return int(ctx[-1])

    def to_dict(self):
        return {"basis": "test-identity", "payload": {"kind": "identity"}}


class LastByteModel:
    def __init__(self, k=4, always=0):
        self.k = k
        self.device = "cpu"
        self.always = always
        self.last_inputs = None

    def logits(self, x):
        self.last_inputs = x.detach().cpu().tolist()
        out = torch.zeros((x.shape[0], 256), dtype=torch.float32)
        out[:, self.always] = 1.0
        return out


def _u(status="PARTIAL_LAW", accepted=True, domains=1):
    evidence = [
        {"context": [1, 2, 3, 7], "feature_context": [1, 2, 3, 7],
         "target": 7, "source_index": 10},
        {"context": [4, 5, 6, 8], "feature_context": [4, 5, 6, 8],
         "target": 8, "source_index": 11},
        {"context": [9, 10, 11, 9], "feature_context": [9, 10, 11, 9],
         "target": 9, "source_index": 12},
    ]
    entry = {
        "feature": "feature_from_data",
        "verify": status,
        "support": 0.62,
        "accepted_to_registry": accepted,
        "bp": IdentityLaw(),
        "evidence": evidence,
    }
    return SimpleNamespace(
        model=LastByteModel(k=4, always=0),
        law_registry={"domain-A": [entry]},
        domains={f"d{i}": {} for i in range(domains)},
    )


def test_generates_critique_from_registered_law_and_keeps_partial_status(tmp_path):
    u = _u()
    inbox = tmp_path / "inbox"
    result = run_law_self_critique(
        u, inbox_dir=inbox, state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    files = list(inbox.glob("*.chat"))
    assert result["evaluated"] == 1
    assert result["queued"] == 1
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "Статус проверки закона: PARTIAL_LAW" in text
    assert "teacher-forced replay accuracy" in text
    assert "00000000" not in text
    assert "feature_from_data" in text
    # It never upgrades the partial law.
    assert u.law_registry["domain-A"][0]["verify"] == "PARTIAL_LAW"


def test_model_context_is_passed_in_chronological_order(tmp_path):
    u = _u()
    model = u.model
    run_law_self_critique(
        u, inbox_dir=tmp_path / "inbox", state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    assert model.last_inputs[0] == [1, 2, 3, 7]
    assert model.last_inputs[1] == [4, 5, 6, 8]


def test_unaccepted_or_unknown_laws_are_ignored(tmp_path):
    u = _u(accepted=False)
    result = run_law_self_critique(
        u, inbox_dir=tmp_path / "inbox", state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    assert result["registry_laws"] == 0
    assert result["evaluated"] == 0
    assert list((tmp_path / "inbox").glob("*.chat")) == []

    u = _u(status="NO_LAW")
    result = run_law_self_critique(
        u, inbox_dir=tmp_path / "inbox2", state_path=tmp_path / "state2.json",
        ledger_path=tmp_path / "ledger2.jsonl",
    )
    assert result["registry_laws"] == 0


def test_repeated_identical_errors_are_not_queued_again(tmp_path):
    u = _u()
    args = dict(inbox_dir=tmp_path / "inbox",
                state_path=tmp_path / "state.json",
                ledger_path=tmp_path / "ledger.jsonl")
    first = run_law_self_critique(u, **args)
    second = run_law_self_critique(u, **args)
    assert first["queued"] == 1
    assert second["evaluated"] == 0
    assert second["queued"] == 0
    assert len(list((tmp_path / "inbox").glob("*.chat"))) == 1


def test_periodic_reevaluation_only_requeues_changed_error_signature(tmp_path):
    u = _u(domains=1)
    args = dict(inbox_dir=tmp_path / "inbox",
                state_path=tmp_path / "state.json",
                ledger_path=tmp_path / "ledger.jsonl",
                reevaluate_every=2)
    first = run_law_self_critique(u, **args)
    assert first["queued"] == 1
    u.domains["d1"] = {}
    skipped = run_law_self_critique(u, **args)
    assert skipped["evaluated"] == 0
    u.domains["d2"] = {}
    u.model.always = 7
    changed = run_law_self_critique(u, **args)
    assert changed["evaluated"] == 1
    assert changed["queued"] == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert list(state["laws"].values())[0]["accuracy"] == pytest.approx(1 / 3)


def test_evidence_must_match_the_symbolic_law(tmp_path):
    u = _u()
    # Corrupt one target: the committed blueprint no longer endorses that row.
    u.law_registry["domain-A"][0]["evidence"][0]["target"] = 99
    result = run_law_self_critique(
        u, inbox_dir=tmp_path / "inbox", state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    assert result["evaluated"] == 1
    report = result["reports"][0]
    assert report["n"] == 2


def test_accuracy_report_is_teacher_forced_not_rollout(tmp_path):
    u = _u()
    result = run_law_self_critique(
        u, inbox_dir=tmp_path / "inbox", state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    line = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
    report = json.loads(line)
    assert report["metric"] == "teacher_forced_replay_accuracy"
    assert report["n"] == 3


def test_shorter_unified_context_is_valid_below_attention_window(tmp_path):
    u = _u()
    u.model.k = 96  # attention max; evidence context remains Unified.k=4 in this fixture
    result = run_law_self_critique(
        u, inbox_dir=tmp_path / "inbox", state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    assert result["evaluated"] == 1
    assert result["reports"][0]["n"] == 3


def test_successful_law_still_writes_audit_but_does_not_queue_training(tmp_path):
    u = _u()
    result = run_law_self_critique(
        u,
        inbox_dir=tmp_path / "inbox",
        state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.jsonl",
        predict_fn=lambda _model, _contexts: [7, 8, 9],
    )
    assert result["evaluated"] == 1
    assert result["queued"] == 0
    assert not list((tmp_path / "inbox").glob("*.chat"))
    report = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[0])
    assert report["accuracy"] == 1.0
    assert report["errors"] == 0
