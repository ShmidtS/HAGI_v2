"""Regression tests for the two-phase recursive growth transaction gate."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hagi.orchestrator.state import (
    GrowthRunStore,
    PreparedDecision,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


def _decision(root: Path, generation_id: str = "g1", decision: str = "rejected"):
    manifest = root / f"{generation_id}-manifest.bin"
    candidate = root / f"{generation_id}-candidate.bin"
    holdout = root / f"{generation_id}-holdout.bin"
    manifest.write_bytes(b"manifest-v1")
    candidate.write_bytes(b"candidate-v1")
    holdout.write_bytes(b"holdout-v1")
    return PreparedDecision(
        generation_id,
        sha256_file(manifest),
        sha256_file(candidate),
        decision,
        sha256_file(holdout),
        str(manifest),
        str(candidate),
        str(holdout),
    )


def test_prepared_requires_and_persists_exact_evidence(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path)

    prepared = store.mark_prepared("g1", decision, owner_id="owner")
    run = tmp_path / "g1"
    assert prepared == run / "prepared-report.json"
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["state"] == "prepared"
    assert state["prepared_report_sha256"] == sha256_file(prepared)
    assert not (run / "report.json").exists()
    assert not (run / "failure.json").exists()
    assert not (run / "terminal-receipt.json").exists()
    assert (run / "evidence").is_dir()

    payload = json.loads(prepared.read_text(encoding="utf-8"))
    assert payload["generation_id"] == "g1"
    assert payload["decision"] == "rejected"
    assert payload["manifest_sha256"] == decision.manifest_sha256
    assert payload["candidate_checkpoint_sha256"] == decision.candidate_checkpoint_sha256
    assert payload["holdout_evidence_sha256"] == decision.holdout_evidence_sha256
    assert Path(payload["manifest_path"]).is_file()
    assert Path(payload["candidate_checkpoint_path"]).is_file()
    assert Path(payload["holdout_evidence_path"]).is_file()


def test_prepared_is_idempotent_but_conflicting_binding_fails(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path)
    first = store.mark_prepared("g1", decision, owner_id="owner")
    second = store.mark_prepared("g1", decision, owner_id="owner")
    assert first == second
    conflicting = replace(decision, manifest_sha256="f" * 64)
    with pytest.raises(ValueError, match="prepared binding"):
        store.mark_prepared("g1", conflicting, owner_id="owner")


def test_terminal_uses_persisted_prepared_binding(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path, decision="rejected")
    store.mark_prepared("g1", decision, owner_id="owner")

    terminal = store.publish_terminal("g1", owner_id="owner")
    run = tmp_path / "g1"
    assert terminal == run / "report.json"
    assert (run / "report.json").is_file()
    assert not (run / "terminal-receipt.json").exists()
    payload = json.loads(terminal.read_text(encoding="utf-8"))
    assert payload["prepared_report_sha256"] == sha256_file(run / "prepared-report.json")
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["state"] == "terminal_rejected"
    assert state["terminal_report_sha256"] == sha256_file(terminal)


def test_tampering_prepared_marker_blocks_terminal_and_parent(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path)
    prepared = store.mark_prepared("g1", decision, owner_id="owner")
    payload = json.loads(prepared.read_text(encoding="utf-8"))
    payload["decision"] = "accepted"
    prepared.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="accepted schema v1 unsupported"):
        store.publish_terminal("g1", owner_id="owner")
    assert not (tmp_path / "g1" / "report.json").exists()


def test_prepared_and_terminal_schema_versions_reject_boolean_one(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path)
    prepared = store.mark_prepared("g1", decision, owner_id="owner")
    prepared_payload = json.loads(prepared.read_text(encoding="utf-8"))
    prepared_payload["schema_version"] = True
    prepared.write_bytes(canonical_json_bytes(prepared_payload))
    with pytest.raises(ValueError, match="prepared report state"):
        store.publish_terminal("g1", owner_id="owner")

    store = GrowthRunStore(tmp_path / "terminal")
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path / "terminal", "g1", "rejected")
    store.mark_prepared("g1", decision, owner_id="owner")
    report = store.publish_terminal("g1", owner_id="owner")
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_payload["schema_version"] = True
    report.write_bytes(canonical_json_bytes(report_payload))
    with pytest.raises(ValueError, match="terminal report state"):
        store.recover("g1", owner_id="owner")


def test_prepared_marker_rejects_extra_or_noncanonical_fields(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path)
    prepared = store.mark_prepared("g1", decision, owner_id="owner")
    payload = json.loads(prepared.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    prepared.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="prepared"):
        store.publish_terminal("g1", owner_id="owner")


def test_mark_prepared_rejects_unbound_accepted_before_marker(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path, decision="accepted")
    state_path = tmp_path / "g1" / "state.json"
    before = state_path.read_bytes()

    with pytest.raises(ValueError, match="accepted schema v1 unsupported"):
        store.mark_prepared("g1", decision, owner_id="owner")

    assert state_path.read_bytes() == before
    assert not (tmp_path / "g1" / "prepared-report.json").exists()
    assert not (tmp_path / "g1" / "evidence").exists()


def test_terminal_and_failure_are_mutually_exclusive(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path, decision="rejected")
    store.mark_prepared("g1", decision, owner_id="owner")
    store.publish_terminal("g1", owner_id="owner")
    with pytest.raises(ValueError, match="terminal"):
        store.publish_failure("g1", "late failure", owner_id="owner")


def test_prepared_crash_window_recovers_without_pointer_change(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    old = __import__("hagi.orchestrator.state", fromlist=["ParentPointer"]).ParentPointer(
        "g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner"
    )
    store.bootstrap_parent(
        old,
        owner_id="owner",
        trusted_parent_token=__import__(
            "hagi.orchestrator.state", fromlist=["TrustedParentToken"]
        ).TrustedParentToken("owner", old.digest()),
    )
    before = (tmp_path / "current-parent.json").read_bytes()
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path)
    prepared = store.mark_prepared("g1", decision, owner_id="owner")
    # Simulate a crash after the immutable marker but before a client observes
    # the lifecycle result. Recovery must preserve the incumbent pointer.
    assert store.recover("g1", owner_id="owner").value == "prepared"
    assert (tmp_path / "g1" / "prepared-report.json") == prepared
    assert (tmp_path / "current-parent.json").read_bytes() == before


def test_state_and_markers_bind_prepared_digest(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path)
    prepared = store.mark_prepared("g1", decision, owner_id="owner")
    state_path = tmp_path / "g1" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["prepared_report_sha256"] = sha256_bytes(b"wrong")
    state_path.write_bytes(canonical_json_bytes(state))
    with pytest.raises(ValueError, match="prepared"):
        store.publish_terminal("g1", owner_id="owner")
    assert prepared.exists()


def test_prepared_marker_crash_window_is_recoverable(tmp_path: Path, monkeypatch):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path)
    original = state_module._atomic_write

    def fail_prepared_state(path, payload):
        if path.name == "state.json" and json.loads(payload)["state"] == "prepared":
            raise OSError("injected crash before PREPARED state")
        return original(path, payload)

    monkeypatch.setattr(state_module, "_atomic_write", fail_prepared_state)
    with pytest.raises(OSError, match="before PREPARED"):
        store.mark_prepared("g1", decision, owner_id="owner")

    run = tmp_path / "g1"
    assert json.loads((run / "state.json").read_text())["state"] == "running"
    assert (run / "prepared-report.json").is_file()
    monkeypatch.undo()
    assert store.recover("g1", owner_id="owner").value == "prepared"
    state = json.loads((run / "state.json").read_text())
    assert state["prepared_report_sha256"] == sha256_file(run / "prepared-report.json")


def test_failure_marker_crash_window_is_recoverable(tmp_path: Path, monkeypatch):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    original = state_module._atomic_write

    def fail_failed_state(path, payload):
        if path.name == "state.json" and json.loads(payload)["state"] == "failed":
            raise OSError("injected crash before FAILED state")
        return original(path, payload)

    monkeypatch.setattr(state_module, "_atomic_write", fail_failed_state)
    with pytest.raises(OSError, match="before FAILED"):
        store.publish_failure("g1", "injected failure", owner_id="owner")

    run = tmp_path / "g1"
    assert json.loads((run / "state.json").read_text())["state"] == "running"
    assert (run / "failure.json").is_file()
    monkeypatch.undo()
    assert store.recover("g1", owner_id="owner").value == "failed"
    assert store.publish_failure("g1", "injected failure", owner_id="owner").is_file()


def test_rejected_terminal_does_not_change_parent(tmp_path: Path):
    from hagi.orchestrator.state import ParentPointer, TrustedParentToken

    store = GrowthRunStore(tmp_path)
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner")
    store.bootstrap_parent(
        old,
        owner_id="owner",
        trusted_parent_token=TrustedParentToken("owner", old.digest()),
    )
    before = (tmp_path / "current-parent.json").read_bytes()
    store.reserve("g1", owner_id="owner")
    store.mark_running("g1", owner_id="owner")
    decision = _decision(tmp_path, decision="rejected")
    store.mark_prepared("g1", decision, owner_id="owner")
    store.publish_terminal("g1", owner_id="owner")
    assert (tmp_path / "current-parent.json").read_bytes() == before
