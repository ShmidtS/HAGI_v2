import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from hagi.orchestrator.recursive import CandidateContext, ChildContext
from hagi.orchestrator.state import (
    GrowthRunStore,
    ParentPointer,
    PreparedDecision,
    TrustedParentToken,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    state_key_digest,
)


def test_builder_contexts_expose_no_holdout_derived_digest():
    """§15.1-A: a builder must never receive the holdout data binding.

    The pinned dataset manifest enumerates both train and holdout shards, so its
    digest is a membership oracle. Owner-side validation and the artifact
    manifests keep the full binding; only the builder context is sealed.

    `dataclasses.fields()` alone is not a sufficient guard. Renaming the field to
    ``_data_manifest_sha256`` and re-exposing it through a property keeps the
    oracle fully readable while ``fields()`` reports only the private name, so the
    class attribute surface is checked as well.
    """
    sealed = ("data_manifest_sha256", "holdout_sha256", "holdout_path", "row_ids_sha256")
    for context_type in (ChildContext, CandidateContext):
        exposed = set(dir(context_type))
        for name in sealed:
            assert name not in {
                field.name for field in dataclasses.fields(context_type)
            }, f"{context_type.__name__} field {name}"
            assert not any(
                attribute == name or attribute == f"_{name}" for attribute in exposed
            ), f"{context_type.__name__} attribute {name}"


def test_banking77_preregistered_pin_matches_published_artifact():
    """The runtime pin must identify the repository's canonical Banking77 artifact."""
    from hagi.orchestrator.real_cycle import BANKING77_MANIFEST_SHA256

    repository = Path(__file__).resolve().parents[1]
    manifest = (
        repository
        / "artifacts"
        / "datasets"
        / "banking77"
        / "57ec275d8078af65b7731c2a98be812d844a6d6b"
        / "manifest.json"
    )
    assert manifest.is_file()
    assert sha256_file(manifest) == BANKING77_MANIFEST_SHA256


def test_build_callbacks_cannot_materialize_holdout_snapshot(tmp_path: Path):
    """The holdout snapshot must not exist while either builder callback runs."""
    from hagi.orchestrator.recursive import run_generation
    from tests.test_recursive_growth_owner import (
        OWNER,
        _candidate_builder,
        _child_builder,
        _evaluator,
        _fixtures,
    )

    store, request = _fixtures(tmp_path)
    run = (tmp_path / "g1").resolve()
    child_builder = _child_builder(tmp_path, [])
    candidate_builder = _candidate_builder(tmp_path, [])
    observations: dict[str, list[tuple[int, bytes | None]]] = {"child": [], "candidate": []}

    def observe(label: str, result):
        snapshots = sorted(run.rglob("holdout-*.bin"))
        observations[label].append(
            (len(snapshots), snapshots[0].read_bytes() if snapshots else None)
        )
        return result

    def child_probe(context):
        return observe("child", child_builder(context))

    def candidate_probe(context):
        return observe("candidate", candidate_builder(context))

    run_generation(
        request,
        store,
        child_probe,
        candidate_probe,
        _evaluator(),
        owner_id=OWNER,
    )

    for label in ("child", "candidate"):
        assert observations[label], label
        assert all(
            count == 0 and content is None for count, content in observations[label]
        ), label


def test_failure_marker_does_not_persist_exception_details(tmp_path: Path):
    """Durable failure state must not retain callback exception details."""
    from hagi.orchestrator.recursive import run_generation
    from tests.test_recursive_growth_owner import (
        OWNER,
        _candidate_builder,
        _child_builder,
        _evaluator,
        _fixtures,
    )

    store, request = _fixtures(tmp_path)
    parent_before = (tmp_path / "current-parent.json").read_bytes()
    baseline = _child_builder(tmp_path, [])
    sentinel = "secret-token=DO-NOT-PERSIST C:/private/customer/sample.txt"

    def failing_child(context):
        if context.name == "child_B":
            raise RuntimeError(sentinel)
        return baseline(context)

    with pytest.raises(RuntimeError) as exc_info:
        run_generation(
            request,
            store,
            failing_child,
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )
    assert sentinel in str(exc_info.value)

    failure_path = tmp_path / "g1" / "failure.json"
    failure_bytes = failure_path.read_bytes()
    assert sentinel.encode() not in failure_bytes
    assert json.loads(failure_bytes)["reason"] == "RuntimeError"
    assert (tmp_path / "current-parent.json").read_bytes() == parent_before


def test_digest_is_order_independent():
    import torch
    a = {"x": torch.ones(2), "y": torch.zeros(1)}
    b = {"y": torch.zeros(1), "x": torch.ones(2)}
    assert state_key_digest(a) == state_key_digest(b)


def _decision_for(
    tmp_path: Path,
    generation_id: str,
    decision: str = "accepted",
    manifest_digest: str | None = None,
):
    manifest = tmp_path / f"{generation_id}-manifest.bin"
    candidate = tmp_path / f"{generation_id}-candidate.bin"
    holdout = tmp_path / f"{generation_id}-holdout.bin"
    manifest.write_bytes(b"manifest")
    candidate.write_bytes(b"candidate")
    holdout.write_bytes(b"holdout")
    return PreparedDecision(
        generation_id,
        manifest_digest or sha256_file(manifest),
        sha256_file(candidate),
        decision,
        sha256_file(holdout),
        str(manifest), str(candidate), str(holdout),
    )


def _decision(tmp_path: Path, decision: str = "rejected", manifest_digest: str | None = None):
    return _decision_for(tmp_path, "g0", decision, manifest_digest)


def _prepare(store: GrowthRunStore, generation_id: str, decision: PreparedDecision, owner_id: str):
    return store.mark_prepared(generation_id, decision, owner_id=owner_id)


def test_terminal_markers_and_pointer_cas(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0")
    store.mark_running("g0")
    d = _decision(tmp_path, "rejected")
    _prepare(store, "g0", d, "default-owner")
    marker = store.publish_terminal("g0")
    assert marker.exists()
    assert store.publish_terminal("g0") == marker
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64)
    store.bootstrap_parent(
        old,
        trusted_parent_token=TrustedParentToken(old.owner_id, old.digest()),
    )
    with pytest.raises(ValueError, match="accepted prepared decision"):
        store.compare_and_swap_parent(
            old, ParentPointer("g1", "4" * 64, "5" * 64, "6" * 64)
        )
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g0"


def test_failure_rejected_after_report(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0")
    store.mark_running("g0")
    d = _decision(tmp_path, "rejected")
    _prepare(store, "g0", d, "default-owner")
    store.publish_terminal("g0")
    with pytest.raises(ValueError, match="terminal marker"):
        store.publish_failure("g0", "late failure")


def test_owner_mismatch_and_explicit_stale_lock_recovery(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a")
    store.mark_running("g0", owner_id="owner-a")
    d = _decision(tmp_path, "rejected")
    with pytest.raises(ValueError, match="owner mismatch"):
        _prepare(store, "g0", d, "owner-b")
    lock = tmp_path / "g0" / ".lifecycle.lock"
    lock.mkdir()
    (lock / "owner.json").write_bytes(
        canonical_json_bytes({"generation_id": "g0", "owner_id": "owner-a"})
    )
    lease = tmp_path / "g0" / ".owner.lease"
    lease_payload = json.loads(lease.read_text(encoding="utf-8"))
    lease_payload["expires_at"] = 0
    lease.write_bytes(canonical_json_bytes(lease_payload))
    store.recover_stale_lock("g0", "owner-a", "worker crashed")
    assert _prepare(store, "g0", d, "owner-a") is not None


def test_failure_crash_recovery_from_each_nonterminal_state(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("reserved", owner_id="owner")
    store.publish_failure("reserved", "crash", owner_id="owner")
    assert store.recover("reserved", owner_id="owner") is not None
    assert store.publish_failure("reserved", "crash", owner_id="owner").exists()

    store.reserve("running", owner_id="owner")
    store.mark_running("running", owner_id="owner")
    store.publish_failure("running", "crash", owner_id="owner")
    assert store.recover("running", owner_id="owner").value == "failed"

    store.reserve("prepared", owner_id="owner")
    store.mark_running("prepared", owner_id="owner")
    _prepare(
        store,
        "prepared",
        _decision_for(tmp_path, "prepared", decision="rejected"),
        "owner",
    )
    store.publish_failure("prepared", "crash", owner_id="owner")
    assert store.recover("prepared", owner_id="owner").value == "failed"


def test_evidence_mutation_after_snapshot_is_safe(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner")
    store.mark_running("g0", owner_id="owner")
    decision = _decision(tmp_path, "rejected")
    _prepare(store, "g0", decision, "owner")
    store.publish_terminal("g0", owner_id="owner")
    Path(decision.holdout_evidence_path).write_bytes(b"mutated-after-snapshot")
    report = store.publish_terminal("g0", owner_id="owner")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert "g0" in payload["manifest_path"]
    assert Path(payload["holdout_evidence_path"]).read_bytes() == b"holdout"


def test_malformed_state_and_report_are_rejected(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner")
    (tmp_path / "g0" / "state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid state"):
        store.mark_running("g0", owner_id="owner")
    store.reserve("g1", owner_id="owner")
    (tmp_path / "g1" / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal report|prepared report"):
        store.recover("g1", owner_id="owner")


def test_state_json_requires_canonical_unique_bytes(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner")
    state_path = tmp_path / "g0" / "state.json"
    canonical = state_path.read_bytes()
    duplicate_key = canonical.replace(
        b'"generation_id":"g0"',
        b'"generation_id":"g0","generation_id":"g0"',
        1,
    )
    assert duplicate_key != canonical

    state_path.write_bytes(duplicate_key)
    with pytest.raises(ValueError, match="malformed state.json"):
        store.mark_running("g0", owner_id="owner")

    state_path.write_bytes(canonical + b"\n")
    with pytest.raises(ValueError, match="malformed state.json"):
        store.mark_running("g0", owner_id="owner")

    state_path.write_bytes(canonical)
    store.mark_running("g0", owner_id="owner")
    assert json.loads(state_path.read_text(encoding="utf-8"))["state"] == "running"


def test_concurrent_reservation_has_one_winner(tmp_path: Path):
    store = GrowthRunStore(tmp_path)

    def attempt(owner: str):
        try:
            store.reserve("race", owner_id=owner)
            return "ok"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, ("owner-a", "owner-b"))) == ["exists", "ok"]
    assert (tmp_path / "race" / "state.json").is_file()


def test_expired_takeover_requires_recovery_lock_and_audit(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=1)
    run = tmp_path / "g0"
    lease = run / ".owner.lease"
    payload = json.loads(lease.read_text(encoding="utf-8"))
    payload["expires_at"] = 0
    lease.write_bytes(canonical_json_bytes(payload))
    (run / ".lifecycle.lock").mkdir()
    (run / ".lifecycle.lock" / "owner.json").write_bytes(
        canonical_json_bytes({"generation_id": "g0", "owner_id": "owner-a"})
    )
    store.recover_stale_lock("g0", "owner-b", "crashed", lease_seconds=60)
    assert list((run / "audit").glob("lease-takeover-*.json"))
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-b"
    assert json.loads(lease.read_text())["owner_id"] == "owner-b"


def test_active_lifecycle_lock_rejects_stale_takeover(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease_path = run / ".owner.lease"
    entered = threading.Event()
    release = threading.Event()

    def hold_lifecycle_lock():
        with store._lock("g0", owner_id="owner-a"):
            entered.set()
            assert release.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        held = pool.submit(hold_lifecycle_lock)
        assert entered.wait(timeout=5)
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
        lease["expires_at"] = 0
        lease_path.write_bytes(canonical_json_bytes(lease))
        try:
            with pytest.raises(ValueError, match="run lifecycle is locked"):
                store.recover_stale_lock("g0", "owner-b", "active holder")
        finally:
            release.set()
        held.result(timeout=5)
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-a"


def test_stale_takeover_retry_completes_partial_crash(tmp_path: Path, monkeypatch):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    original = state_module._atomic_write

    def fail_new_lease(path, payload):
        if path.name == ".owner.lease" and json.loads(payload)["owner_id"] == "owner-b":
            raise OSError("injected crash before new lease")
        return original(path, payload)

    monkeypatch.setattr(state_module, "_atomic_write", fail_new_lease)
    with pytest.raises(OSError, match="before new lease"):
        store.recover_stale_lock("g0", "owner-b", "worker crashed")
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-b"

    monkeypatch.undo()
    store.recover_stale_lock("g0", "owner-b", "worker crashed")
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-b"
    assert json.loads(lease_path.read_text())["owner_id"] == "owner-b"


def test_recovery_rejects_lease_bound_to_another_generation(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["generation_id"] = "g1"
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    audit_dir = run / "audit"
    audit_dir.mkdir()
    (audit_dir / "lease-takeover.pending.json").write_bytes(
        canonical_json_bytes(
            {
                "generation_id": "g0",
                "lease_seconds": 300,
                "new_owner_id": "owner-b",
                "previous_expires_at": 0,
                "previous_owner_id": "owner-a",
                "reason": "worker crashed",
                "schema_version": 1,
            }
        )
    )

    with pytest.raises(ValueError, match="run owner mismatch"):
        store.recover_stale_lock("g0", "owner-b", "worker crashed")
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-a"


def test_live_lease_replay_completes_missing_audit_and_cleanup(tmp_path: Path, monkeypatch):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    original = state_module.GrowthRunStore._publish_regular_no_replace

    def fail_audit_write(self, path, data, error_message):
        if path.name.startswith("lease-takeover-") and path.suffix == ".json":
            raise OSError("injected crash before takeover audit")
        return original(self, path, data, error_message)

    monkeypatch.setattr(
        state_module.GrowthRunStore,
        "_publish_regular_no_replace",
        fail_audit_write,
    )
    with pytest.raises(OSError, match="before takeover audit"):
        store.recover_stale_lock("g0", "owner-b", "worker crashed")
    pending = run / "audit" / "lease-takeover.pending.json"
    assert pending.is_file()
    assert not list((run / "audit").glob("lease-takeover-*.json"))
    lease_after_takeover = lease_path.read_bytes()
    assert lease_after_takeover != canonical_json_bytes(lease)

    with pytest.raises(ValueError, match="still active"):
        store.recover_stale_lock("g0", "owner-c", "third owner", lease_seconds=120)
    assert lease_path.read_bytes() == lease_after_takeover
    assert pending.is_file()

    monkeypatch.undo()
    store.recover_stale_lock("g0", "owner-b", "worker crashed")
    assert list((run / "audit").glob("lease-takeover-*.json"))
    assert not pending.exists()
    assert lease_path.read_bytes() == lease_after_takeover


def test_completed_intent_rotation_clears_stale_metadata_before_pending_rewrite(
    tmp_path: Path, monkeypatch
):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    lifecycle = run / ".lifecycle.lock"
    lifecycle.mkdir()
    (lifecycle / "owner.json").write_bytes(
        canonical_json_bytes({"generation_id": "g0", "owner_id": "owner-a"})
    )
    original_clear = state_module.GrowthRunStore._clear_lifecycle_lock
    clear_calls = 0

    def fail_first_clear(self, lock):
        nonlocal clear_calls
        clear_calls += 1
        if clear_calls == 1:
            raise OSError("injected first cleanup crash")
        return original_clear(self, lock)

    monkeypatch.setattr(
        state_module.GrowthRunStore,
        "_clear_lifecycle_lock",
        fail_first_clear,
    )
    with pytest.raises(OSError, match="first cleanup crash"):
        store.recover_stale_lock("g0", "owner-b", "owner-b took over", lease_seconds=60)
    assert lifecycle.is_dir()
    pending = run / "audit" / "lease-takeover.pending.json"
    assert json.loads(pending.read_text())["new_owner_id"] == "owner-b"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))

    original_atomic = state_module._atomic_write

    def fail_state_c(path, payload):
        if path.name == "state.json" and json.loads(payload)["owner_id"] == "owner-c":
            raise OSError("injected crash after pending rotation")
        return original_atomic(path, payload)

    monkeypatch.setattr(state_module, "_atomic_write", fail_state_c)
    with pytest.raises(OSError, match="after pending rotation"):
        store.recover_stale_lock(
            "g0", "owner-c", "owner-c takes over", lease_seconds=120
        )
    assert not lifecycle.exists()
    assert json.loads(pending.read_text())["new_owner_id"] == "owner-c"
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-b"

    monkeypatch.undo()
    store.recover_stale_lock("g0", "owner-c", "owner-c takes over", lease_seconds=120)
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-c"
    assert json.loads(lease_path.read_text())["owner_id"] == "owner-c"
    assert not pending.exists()
    store.mark_running("g0", owner_id="owner-c")


def test_unclaimed_takeover_can_be_superseded_after_crash_before_state(
    tmp_path: Path, monkeypatch
):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    original = state_module._atomic_write

    def fail_state_write(path, payload):
        if path.name == "state.json" and json.loads(payload)["owner_id"] == "owner-b":
            raise OSError("injected crash before takeover state")
        return original(path, payload)

    monkeypatch.setattr(state_module, "_atomic_write", fail_state_write)
    with pytest.raises(OSError, match="before takeover state"):
        store.recover_stale_lock("g0", "owner-b", "owner-b attempted")
    pending = run / "audit" / "lease-takeover.pending.json"
    assert pending.is_file()
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-a"

    monkeypatch.undo()
    store.recover_stale_lock("g0", "owner-c", "owner-c takes abandoned intent", lease_seconds=120)
    assert not pending.exists()
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-c"
    assert json.loads(lease_path.read_text())["owner_id"] == "owner-c"
    audits = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run / "audit").glob("lease-takeover-*.json")
    ]
    superseded = [audit for audit in audits if audit["status"] == "superseded"]
    complete = [audit for audit in audits if audit["status"] == "complete"]
    assert [(audit["previous_owner_id"], audit["new_owner_id"]) for audit in superseded] == [
        ("owner-a", "owner-b")
    ]
    assert superseded[0]["superseded_by"] == "owner-c"
    assert [(audit["previous_owner_id"], audit["new_owner_id"]) for audit in complete] == [
        ("owner-a", "owner-c")
    ]
    store.mark_running("g0", owner_id="owner-c")


def test_claimed_takeover_can_be_superseded_after_crash_before_lease(
    tmp_path: Path, monkeypatch
):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    original = state_module._atomic_write

    def fail_lease_write(path, payload):
        if path.name == ".owner.lease" and json.loads(payload)["owner_id"] == "owner-b":
            raise OSError("injected crash before takeover lease")
        return original(path, payload)

    monkeypatch.setattr(state_module, "_atomic_write", fail_lease_write)
    with pytest.raises(OSError, match="before takeover lease"):
        store.recover_stale_lock("g0", "owner-b", "owner-b attempted")
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-b"
    assert json.loads(lease_path.read_text())["owner_id"] == "owner-a"

    monkeypatch.undo()
    store.recover_stale_lock("g0", "owner-c", "owner-c supersedes partial claim", lease_seconds=120)
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-c"
    assert json.loads(lease_path.read_text())["owner_id"] == "owner-c"
    assert not (run / "audit" / "lease-takeover.pending.json").exists()
    store.mark_running("g0", owner_id="owner-c")


def test_recovery_rejects_intent_bound_to_another_generation(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    audit_dir = run / "audit"
    audit_dir.mkdir()
    (audit_dir / "lease-takeover.pending.json").write_bytes(
        canonical_json_bytes(
            {
                "generation_id": "g1",
                "lease_seconds": 300,
                "new_owner_id": "owner-b",
                "previous_expires_at": 0,
                "previous_owner_id": "owner-a",
                "reason": "worker crashed",
                "schema_version": 1,
            }
        )
    )

    with pytest.raises(ValueError, match="conflicting lease takeover intent"):
        store.recover_stale_lock("g0", "owner-b", "worker crashed")
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-a"
    assert not list(audit_dir.glob("lease-takeover-*.json"))


def test_same_owner_expired_lease_recovery_renews_without_lock(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))

    store.recover_stale_lock("g0", "owner-a", "expired lease renewal")

    renewed = json.loads(lease_path.read_text(encoding="utf-8"))
    assert renewed["owner_id"] == "owner-a"
    assert renewed["expires_at"] > time.time()
    assert list((run / "audit").glob("lease-takeover-*.json"))
    store.mark_running("g0", owner_id="owner-a")


def test_stale_takeover_retry_clears_lock_after_completed_intent(tmp_path: Path, monkeypatch):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease = json.loads((run / ".owner.lease").read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    (run / ".owner.lease").write_bytes(canonical_json_bytes(lease))
    lifecycle = run / ".lifecycle.lock"
    lifecycle.mkdir()
    (lifecycle / "owner.json").write_bytes(
        canonical_json_bytes({"generation_id": "g0", "owner_id": "owner-a"})
    )
    original = state_module.GrowthRunStore._clear_lifecycle_lock
    calls = {"count": 0}

    def fail_once(self, lock):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("injected crash before stale lock cleanup")
        return original(self, lock)

    monkeypatch.setattr(state_module.GrowthRunStore, "_clear_lifecycle_lock", fail_once)
    with pytest.raises(OSError, match="before stale lock cleanup"):
        store.recover_stale_lock("g0", "owner-b", "worker crashed")
    assert (run / "audit" / "lease-takeover.pending.json").is_file()
    lease_after_crash = (run / ".owner.lease").read_bytes()

    store.recover_stale_lock("g0", "owner-b", "worker crashed")
    assert not lifecycle.exists()
    assert not (run / "audit" / "lease-takeover.pending.json").exists()
    assert (run / ".owner.lease").read_bytes() == lease_after_crash
    assert json.loads((run / ".owner.lease").read_text())["owner_id"] == "owner-b"


def test_takeover_retry_rotates_completed_intent_after_compound_crash(
    tmp_path: Path, monkeypatch
):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    run = tmp_path / "g0"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    lifecycle = run / ".lifecycle.lock"
    lifecycle.mkdir()
    (lifecycle / "owner.json").write_bytes(
        canonical_json_bytes({"generation_id": "g0", "owner_id": "owner-a"})
    )
    original = state_module.GrowthRunStore._clear_lifecycle_lock
    calls = {"count": 0}

    def fail_first_cleanup(self, lock):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("injected cleanup crash")
        return original(self, lock)

    monkeypatch.setattr(state_module.GrowthRunStore, "_clear_lifecycle_lock", fail_first_cleanup)
    with pytest.raises(OSError, match="cleanup crash"):
        store.recover_stale_lock("g0", "owner-b", "worker crashed")
    pending = run / "audit" / "lease-takeover.pending.json"
    assert pending.is_file()

    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    (lifecycle / "owner.json").write_bytes(
        canonical_json_bytes({"generation_id": "g0", "owner_id": "owner-b"})
    )

    store.recover_stale_lock("g0", "owner-c", "owner-b crashed", lease_seconds=120)
    assert not lifecycle.exists()
    assert not pending.exists()
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-c"
    audits = [json.loads(path.read_text()) for path in (run / "audit").glob("lease-takeover-*.json")]
    assert {audit["new_owner_id"] for audit in audits} == {"owner-b", "owner-c"}
    store.mark_running("g0", owner_id="owner-c")


def test_lifecycle_owner_atomic_write_crash_recovers_from_orphan_temp(tmp_path: Path):
    crash_child = """
import os
import sys
from pathlib import Path

import hagi.orchestrator.state as state

root = Path(sys.argv[1])
store = state.GrowthRunStore(root)
store.reserve("g0", owner_id="owner-a", lease_seconds=60)
original_replace = os.replace

def crash_before_owner_replace(source, target, *args, **kwargs):
    if Path(target).name == "owner.json" and Path(target).parent.name == ".lifecycle.lock":
        os._exit(19)
    return original_replace(source, target, *args, **kwargs)

os.replace = crash_before_owner_replace
store.mark_running("g0", owner_id="owner-a")
"""
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    process = subprocess.run(
        [sys.executable, "-c", crash_child, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 19, process.stderr
    assert process.stderr == ""

    run = tmp_path / "g0"
    lifecycle = run / ".lifecycle.lock"
    assert lifecycle.is_dir()
    orphan_temps = list(lifecycle.glob(".owner.json.*"))
    assert len(orphan_temps) == 1
    assert orphan_temps[0].is_file() and not orphan_temps[0].is_symlink()

    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    store = GrowthRunStore(tmp_path)
    store.recover_stale_lock("g0", "owner-b", "owner metadata crash")
    assert not lifecycle.exists()
    store.mark_running("g0", owner_id="owner-b")


def test_active_lease_rejects_takeover(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a", lease_seconds=60)
    with pytest.raises(ValueError, match="still active"):
        store.recover_stale_lock("g0", "owner-b", "active")


def test_parent_cas_recovers_after_process_crash(tmp_path: Path):
    crash_child = """
import os
import sys
from pathlib import Path
from hagi.orchestrator.state import (
    GrowthRunStore, ParentPointer, TrustedParentToken, _atomic_write,
)

root = Path(sys.argv[1])
store = GrowthRunStore(root)
old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner")

def crash_before_pointer(path, payload):
    if path.name == "current-parent.json":
        print("locked", flush=True)
        sys.stdin.readline()
        os._exit(17)
    return _atomic_write(path, payload)

import hagi.orchestrator.state as state
state._atomic_write = crash_before_pointer
store.bootstrap_parent(
    old,
    owner_id="owner",
    trusted_parent_token=TrustedParentToken("owner", old.digest()),
)
"""
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    process = subprocess.Popen(
        [sys.executable, "-c", crash_child, str(tmp_path)],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "locked"

    store = GrowthRunStore(tmp_path)
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner")
    with pytest.raises(ValueError, match="locked"):
        store.bootstrap_parent(
            old,
            owner_id="owner",
            trusted_parent_token=TrustedParentToken("owner", old.digest()),
        )
    assert process.stdin is not None
    process.stdin.write("crash\n")
    process.stdin.close()
    assert process.wait(timeout=30) == 17
    assert process.stderr is not None
    assert process.stderr.read() == ""
    assert (tmp_path / ".parent.lock").is_file()
    assert not (tmp_path / "current-parent.json").exists()

    store.bootstrap_parent(
        old,
        owner_id="owner",
        trusted_parent_token=TrustedParentToken("owner", old.digest()),
    )
    assert json.loads((tmp_path / "current-parent.json").read_text()) == old.as_dict()
    assert (tmp_path / ".parent.lock").is_file()


def test_parent_cas_rejects_without_accepted_binding(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner")
    store.bootstrap_parent(
        old, owner_id="owner", trusted_parent_token=TrustedParentToken("owner", old.digest())
    )
    with pytest.raises(ValueError, match="accepted prepared decision"):
        store.compare_and_swap_parent(old, ParentPointer(
            "g1", "4" * 64, "5" * 64, "6" * 64, owner_id="owner"
        ), owner_id="owner")


def test_owner_lease_expiry_blocks_lifecycle_and_can_renew(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner", lease_seconds=1)
    lease = tmp_path / "g0" / ".owner.lease"
    payload = json.loads(lease.read_text(encoding="utf-8"))
    payload["expires_at"] = 0
    lease.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="lease expired"):
        store.mark_running("g0", owner_id="owner")


def test_owner_bound_parent_cas_and_pointer_owner(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner-a")
    token = TrustedParentToken("owner-a", old.digest())
    with pytest.raises(ValueError, match="replacement parent owner mismatch"):
        store.bootstrap_parent(old, owner_id="owner-b", trusted_parent_token=token)
    store.bootstrap_parent(old, owner_id="owner-a", trusted_parent_token=token)
    assert json.loads((tmp_path / "current-parent.json").read_text())["owner_id"] == "owner-a"

    store.reserve("g1", owner_id="owner-a")
    store.mark_running("g1", owner_id="owner-a")
    decision = _bind_parent(_decision_for(tmp_path, "g1", "accepted"), old)
    _prepare(store, "g1", decision, "owner-a")
    new = _replacement_for_prepared(tmp_path, store, decision, "owner-a")
    with pytest.raises(ValueError, match="accepted prepared decision"):
        store.compare_and_swap_parent(old, new, owner_id="owner-a")
    store.commit_accepted_terminal(old, new, decision, owner_id="owner-a")
    assert store.read_parent() == new


def test_reservation_failure_does_not_publish_invalid_run(tmp_path: Path, monkeypatch):
    store = GrowthRunStore(tmp_path)
    calls = {"n": 0}
    original = __import__("hagi.orchestrator.state", fromlist=["_atomic_write"])._atomic_write
    def fail_state(path, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("injected state write failure")
        return original(path, payload)
    monkeypatch.setattr("hagi.orchestrator.state._atomic_write", fail_state)
    with pytest.raises(OSError, match="injected state write failure"):
        store.reserve("g0", owner_id="owner")
    assert not (tmp_path / "g0").exists()
    assert not list(tmp_path.glob(".reserve-g0.*"))


def test_stale_lock_recovery_requires_matching_persisted_owner(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner-a")
    lock = tmp_path / "g0" / ".lifecycle.lock"
    lock.mkdir()
    (lock / "owner.json").write_bytes(
        canonical_json_bytes({"generation_id": "g0", "owner_id": "owner-a"})
    )
    lease = tmp_path / "g0" / ".owner.lease"
    payload = json.loads(lease.read_text(encoding="utf-8"))
    payload["expires_at"] = "not-a-number"
    lease.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="invalid owner lease"):
        store.mark_running("g0", owner_id="owner-a")


def test_competing_lifecycle_operation_is_rejected(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner")
    lock = tmp_path / "g0" / ".lifecycle.lock"
    lock.mkdir()
    (lock / "owner.json").write_bytes(
        canonical_json_bytes({"generation_id": "g0", "owner_id": "owner"})
    )
    with pytest.raises(ValueError, match="locked"):
        store.mark_running("g0", owner_id="owner")


def _bind_parent(decision: PreparedDecision, parent: ParentPointer) -> PreparedDecision:
    return replace(
        decision,
        parent_generation_id=parent.generation_id,
        parent_checkpoint_sha256=parent.checkpoint_sha256,
        parent_manifest_sha256=parent.manifest_sha256,
    )


def _replacement_for_prepared(
    tmp_path: Path,
    store: GrowthRunStore,
    decision: PreparedDecision,
    owner_id: str,
) -> ParentPointer:
    import hagi.orchestrator.state as state_module

    run = tmp_path / decision.generation_id
    _, prepared, snapshot, _, prepared_sha256 = store._read_prepared_report(
        run, decision.generation_id
    )
    report = state_module._terminal_payload(
        prepared, prepared_sha256, snapshot
    )
    return ParentPointer(
        decision.generation_id,
        decision.candidate_checkpoint_sha256,
        decision.manifest_sha256,
        state_module.sha256_bytes(state_module.canonical_json_bytes(report)),
        owner_id=owner_id,
    )


def test_bound_accepted_publish_terminal_requires_unified_commit(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    parent = ParentPointer(
        "g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner-a"
    )
    store.bootstrap_parent(
        parent,
        owner_id="owner-a",
        trusted_parent_token=TrustedParentToken("owner-a", parent.digest()),
    )
    before = (tmp_path / "current-parent.json").read_bytes()
    store.reserve("g1", owner_id="owner-a", lease_seconds=300)
    store.mark_running("g1", owner_id="owner-a")
    decision = _bind_parent(
        _decision_for(tmp_path, "g1", "accepted"), parent
    )
    _prepare(store, "g1", decision, "owner-a")

    with pytest.raises(ValueError, match="unified commit"):
        store.publish_terminal("g1", owner_id="owner-a")

    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert store.read_parent() == parent
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "prepared"
    assert not (tmp_path / "g1" / "report.json").exists()


def test_unbound_accepted_cannot_enter_prepared_or_parent_cas(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    parent = ParentPointer(
        "g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner-a"
    )
    store.bootstrap_parent(
        parent,
        owner_id="owner-a",
        trusted_parent_token=TrustedParentToken("owner-a", parent.digest()),
    )
    before = (tmp_path / "current-parent.json").read_bytes()
    store.reserve("g1", owner_id="owner-a", lease_seconds=300)
    store.mark_running("g1", owner_id="owner-a")
    decision = _decision_for(tmp_path, "g1", "accepted")
    state_path = tmp_path / "g1" / "state.json"
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match="accepted schema v1 unsupported"):
        _prepare(store, "g1", decision, "owner-a")

    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert store.read_parent() == parent
    assert state_path.read_bytes() == state_before
    assert not (tmp_path / "g1" / "prepared-report.json").exists()
    assert not (tmp_path / "g1" / "report.json").exists()


def test_unified_commit_rejects_expected_not_bound_to_decision(tmp_path: Path):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    actual = ParentPointer(
        "g-actual", "4" * 64, "5" * 64, "6" * 64, owner_id="owner-a"
    )
    store.bootstrap_parent(
        actual,
        owner_id="owner-a",
        trusted_parent_token=TrustedParentToken("owner-a", actual.digest()),
    )
    preflight = ParentPointer(
        "g-preflight", "1" * 64, "2" * 64, "3" * 64, owner_id="owner-a"
    )
    store.reserve("g1", owner_id="owner-a", lease_seconds=300)
    store.mark_running("g1", owner_id="owner-a")
    decision = _bind_parent(
        _decision_for(tmp_path, "g1", "accepted"), preflight
    )
    _prepare(store, "g1", decision, "owner-a")
    replacement = _replacement_for_prepared(
        tmp_path, store, decision, "owner-a"
    )

    with pytest.raises(ValueError, match="parent lineage"):
        store.commit_accepted_terminal(
            actual, replacement, decision, owner_id="owner-a"
        )

    assert store.read_parent() == actual
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "prepared"
    assert not (tmp_path / "g1" / "report.json").exists()
    assert state_module.sha256_file(tmp_path / "current-parent.json") == actual.digest()


def _prepared_accepted(tmp_path: Path):
    """Stage a bound accepted decision without committing it."""
    store = GrowthRunStore(tmp_path)
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner-a")
    store.bootstrap_parent(
        old,
        owner_id="owner-a",
        trusted_parent_token=TrustedParentToken("owner-a", old.digest()),
    )
    store.reserve("g1", owner_id="owner-a", lease_seconds=300)
    store.mark_running("g1", owner_id="owner-a")
    decision = _bind_parent(_decision_for(tmp_path, "g1", "accepted"), old)
    _prepare(store, "g1", decision, "owner-a")
    return store, old, decision, _replacement_for_prepared(
        tmp_path, store, decision, "owner-a"
    )


def test_recover_requires_committed_parent_for_accepted_terminal(
    tmp_path: Path,
):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner-a")
    store.bootstrap_parent(
        old,
        owner_id="owner-a",
        trusted_parent_token=TrustedParentToken("owner-a", old.digest()),
    )
    store.reserve("g1", owner_id="owner-a", lease_seconds=300)
    store.mark_running("g1", owner_id="owner-a")
    decision = _bind_parent(_decision_for(tmp_path, "g1", "accepted"), old)
    _prepare(store, "g1", decision, "owner-a")
    run = tmp_path / "g1"
    _, prepared, snapshot, _, prepared_sha256 = store._read_prepared_report(
        run, "g1"
    )
    report = state_module._terminal_payload(
        prepared, prepared_sha256, snapshot
    )
    report_bytes = canonical_json_bytes(report)
    report_path = run / "report.json"
    report_path.write_bytes(report_bytes)
    replacement = ParentPointer(
        "g1",
        decision.candidate_checkpoint_sha256,
        decision.manifest_sha256,
        state_module.sha256_bytes(report_bytes),
        owner_id="owner-a",
    )
    parent_bytes = (tmp_path / "current-parent.json").read_bytes()
    state_bytes = (run / "state.json").read_bytes()

    with pytest.raises(ValueError, match="accepted terminal recovery parent conflict"):
        store.recover("g1", owner_id="owner-a")

    assert (tmp_path / "current-parent.json").read_bytes() == parent_bytes
    assert (run / "state.json").read_bytes() == state_bytes
    assert json.loads(state_bytes)["state"] == "prepared"

    (tmp_path / "current-parent.json").unlink()
    with pytest.raises(
        ValueError,
        match="accepted terminal recovery parent conflict",
    ):
        store.recover("g1", owner_id="owner-a")
    assert not (tmp_path / "current-parent.json").exists()
    assert (run / "state.json").read_bytes() == state_bytes

    (tmp_path / "current-parent.json").write_bytes(
        canonical_json_bytes(replacement.as_dict())
    )
    assert store.recover("g1", owner_id="owner-a").value == "terminal_accepted"
    assert store.read_parent() == replacement


def test_uncommitted_prepared_can_be_taken_over_and_committed(tmp_path: Path):
    store, old, decision, _ = _prepared_accepted(tmp_path)
    run = tmp_path / "g1"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))

    store.recover_stale_lock("g1", "owner-b", "owner-a crashed before commit")
    replacement = _replacement_for_prepared(tmp_path, store, decision, "owner-b")
    report = store.commit_accepted_terminal(
        old, replacement, decision, owner_id="owner-b"
    )

    assert report.is_file()
    assert store.read_parent() == replacement
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["owner_id"] == "owner-b"
    assert state["state"] == "terminal_accepted"


def test_committed_prepared_can_be_taken_over_and_resumed(tmp_path: Path, monkeypatch):
    import hagi.orchestrator.state as state_module

    store, old, decision, replacement = _prepared_accepted(tmp_path)
    original = state_module._atomic_write
    fired = False

    def crash_after_pointer(path, payload):
        nonlocal fired
        result = original(path, payload)
        if path.name == "current-parent.json" and not fired:
            fired = True
            raise OSError("injected crash after pointer commit")
        return result

    monkeypatch.setattr(state_module, "_atomic_write", crash_after_pointer)
    with pytest.raises(OSError, match="after pointer commit"):
        store.commit_accepted_terminal(old, replacement, decision, owner_id="owner-a")
    monkeypatch.undo()

    run = tmp_path / "g1"
    committed = store.read_parent()
    assert committed == replacement
    assert committed.owner_id == "owner-a"
    assert json.loads((run / "state.json").read_text(encoding="utf-8"))["state"] == "prepared"
    assert not (run / "report.json").exists()

    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    store.recover_stale_lock("g1", "owner-b", "owner-a crashed after pointer commit")

    report = store.commit_accepted_terminal(
        old, committed, decision, owner_id="owner-b"
    )
    assert report.is_file()
    assert store.read_parent() == committed
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["owner_id"] == "owner-b"
    assert state["state"] == "terminal_accepted"


def test_completed_terminal_can_be_read_after_takeover_without_pointer_rewrite(
    tmp_path: Path,
):
    store, _old, _decision, replacement = _accepted_replacement(tmp_path)
    parent_before = (tmp_path / "current-parent.json").read_bytes()
    run = tmp_path / "g1"
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))

    store.recover_stale_lock("g1", "owner-b", "owner-a crashed after terminal")
    assert store.recover("g1", owner_id="owner-b").value == "terminal_accepted"
    assert (tmp_path / "current-parent.json").read_bytes() == parent_before
    assert store.read_parent() == replacement


def test_schema_v1_accepted_terminal_is_rejected_at_exact_committed_parent(
    tmp_path: Path,
):
    import hagi.orchestrator.state as state_module

    store = GrowthRunStore(tmp_path)
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner-a")
    store.bootstrap_parent(
        old,
        owner_id="owner-a",
        trusted_parent_token=TrustedParentToken("owner-a", old.digest()),
    )
    store.reserve("g1", owner_id="owner-a", lease_seconds=300)
    store.mark_running("g1", owner_id="owner-a")
    decision = _decision_for(tmp_path, "g1", "rejected")
    store.mark_prepared("g1", decision, owner_id="owner-a")
    run = tmp_path / "g1"
    prepared_path = run / "prepared-report.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    _, _, snapshot, _, _ = store._read_prepared_report(run, "g1")
    prepared["decision"] = "accepted"
    prepared_bytes = canonical_json_bytes(prepared)
    prepared_path.write_bytes(prepared_bytes)
    report = state_module._terminal_payload(
        prepared, sha256_bytes(prepared_bytes), replace(snapshot, decision="accepted")
    )
    report_bytes = canonical_json_bytes(report)
    (run / "report.json").write_bytes(report_bytes)
    replacement = ParentPointer(
        "g1",
        decision.candidate_checkpoint_sha256,
        decision.manifest_sha256,
        sha256_bytes(report_bytes),
        owner_id="owner-a",
    )
    (tmp_path / "current-parent.json").write_bytes(
        canonical_json_bytes(replacement.as_dict())
    )
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["prepared_report_sha256"] = sha256_bytes(prepared_bytes)
    state_path.write_bytes(canonical_json_bytes(state))

    parent_bytes = (tmp_path / "current-parent.json").read_bytes()
    state_bytes = state_path.read_bytes()

    with pytest.raises(ValueError, match="accepted schema v1 unsupported"):
        store.publish_terminal("g1", owner_id="owner-a")

    assert (tmp_path / "current-parent.json").read_bytes() == parent_bytes
    assert state_path.read_bytes() == state_bytes
    assert json.loads(state_path.read_text(encoding="utf-8"))["state"] == "prepared"

    with pytest.raises(ValueError, match="accepted schema v1 unsupported"):
        store.recover("g1", owner_id="owner-a")

    assert (tmp_path / "current-parent.json").read_bytes() == parent_bytes
    assert state_path.read_bytes() == state_bytes


def _accepted_replacement(tmp_path: Path):
    """Commit an accepted bound generation through the sole-commit path."""
    store, old, decision, replacement = _prepared_accepted(tmp_path)
    report = store.commit_accepted_terminal(
        old, replacement, decision, owner_id="owner-a"
    )
    assert report.is_file()
    return store, old, decision, replacement


def test_bootstrap_is_idempotent_after_successful_pointer_commit(tmp_path: Path):
    """A crash after the bootstrap pointer write leaves a committed parent.

    The trusted token binds owner and pointer digest, so an exact pointer match
    is provably the same commit and must be returned idempotently, exactly like
    the replacement CAS path. A different pointer must still fail closed.
    """
    store = GrowthRunStore(tmp_path)
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner")
    token = TrustedParentToken("owner", old.digest())
    store.bootstrap_parent(old, owner_id="owner", trusted_parent_token=token)
    after_first = (tmp_path / "current-parent.json").read_bytes()

    store.bootstrap_parent(old, owner_id="owner", trusted_parent_token=token)

    assert (tmp_path / "current-parent.json").read_bytes() == after_first
    assert store.read_parent() == old

    # Probe 1: same generation, different committed digests. A comparison that
    # only looked at generation_id would wrongly succeed here.
    same_gen = ParentPointer("g0", "9" * 64, "2" * 64, "3" * 64, owner_id="owner")
    with pytest.raises(ValueError, match="parent pointer already exists"):
        store.bootstrap_parent(
            same_gen,
            owner_id="owner",
            trusted_parent_token=TrustedParentToken("owner", same_gen.digest()),
        )

    # Probe 2: every field equal except the commit owner. A comparison that
    # dropped owner_id (for example by using _parent_identity) would wrongly
    # succeed here.
    other_owner = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner-b")
    with pytest.raises(ValueError, match="parent pointer already exists"):
        store.bootstrap_parent(
            other_owner,
            owner_id="owner-b",
            trusted_parent_token=TrustedParentToken("owner-b", other_owner.digest()),
        )

    # Probe 3: a mismatched token must be rejected even when it is replayed
    # against the pointer that is already committed. An implementation that
    # returned before validating the token would wrongly succeed here.
    with pytest.raises(ValueError, match="trusted parent token mismatch"):
        store.bootstrap_parent(
            old,
            owner_id="owner",
            trusted_parent_token=TrustedParentToken("owner", "f" * 64),
        )

    assert (tmp_path / "current-parent.json").read_bytes() == after_first
    assert store.read_parent() == old


def test_bootstrap_rejects_malformed_existing_pointer(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner")
    token = TrustedParentToken("owner", old.digest())
    store.bootstrap_parent(old, owner_id="owner", trusted_parent_token=token)
    (tmp_path / "current-parent.json").write_bytes(b"[]")

    with pytest.raises(ValueError, match="malformed current-parent.json"):
        store.bootstrap_parent(old, owner_id="owner", trusted_parent_token=token)


def test_parent_cas_is_idempotent_after_successful_commit(tmp_path: Path):
    store, old, decision, replacement = _accepted_replacement(tmp_path)
    store.compare_and_swap_parent(
        old, replacement, owner_id="owner-a", accepted_decision=decision
    )
    before = (tmp_path / "current-parent.json").read_bytes()

    store.compare_and_swap_parent(
        old, replacement, owner_id="owner-a", accepted_decision=decision
    )

    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert store.read_parent() == replacement


def test_unified_commit_rejects_legacy_terminal_receipt(tmp_path: Path):
    store, old, decision, replacement = _prepared_accepted(tmp_path)
    run = tmp_path / "g1"
    (run / "terminal-receipt.json").write_bytes(b"legacy receipt")
    parent_bytes = (tmp_path / "current-parent.json").read_bytes()
    state_bytes = (run / "state.json").read_bytes()

    with pytest.raises(ValueError, match="legacy terminal receipt|conflicting terminal markers"):
        store.commit_accepted_terminal(
            old, replacement, decision, owner_id="owner-a"
        )

    assert (tmp_path / "current-parent.json").read_bytes() == parent_bytes
    assert store.read_parent() == old
    assert (run / "state.json").read_bytes() == state_bytes
    assert json.loads(state_bytes)["state"] == "prepared"
    assert not (run / "report.json").exists()


def test_parent_cas_rejects_terminal_bound_to_different_preflight_parent(
    tmp_path: Path,
):
    store = GrowthRunStore(tmp_path)
    preflight = ParentPointer("g-preflight", "1" * 64, "2" * 64, "3" * 64, owner_id="owner-a")
    actual = ParentPointer("g-actual", "4" * 64, "5" * 64, "6" * 64, owner_id="owner-a")
    store.bootstrap_parent(
        actual,
        owner_id="owner-a",
        trusted_parent_token=TrustedParentToken("owner-a", actual.digest()),
    )
    store.reserve("g1", owner_id="owner-a", lease_seconds=300)
    store.mark_running("g1", owner_id="owner-a")
    decision = _bind_parent(_decision_for(tmp_path, "g1", "accepted"), preflight)
    _prepare(store, "g1", decision, "owner-a")
    replacement = _replacement_for_prepared(tmp_path, store, decision, "owner-a")

    with pytest.raises(ValueError, match="parent lineage"):
        store.commit_accepted_terminal(
            actual, replacement, decision, owner_id="owner-a"
        )
    assert store.read_parent() == actual
    assert not (tmp_path / "g1" / "report.json").exists()


def test_unbound_legacy_accepted_is_rejected_before_parent_cas(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    old = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id="owner-a")
    store.bootstrap_parent(
        old,
        owner_id="owner-a",
        trusted_parent_token=TrustedParentToken("owner-a", old.digest()),
    )
    store.reserve("g1", owner_id="owner-a", lease_seconds=300)
    store.mark_running("g1", owner_id="owner-a")
    decision = _decision_for(tmp_path, "g1", "accepted")
    state_path = tmp_path / "g1" / "state.json"
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match="accepted schema v1 unsupported"):
        _prepare(store, "g1", decision, "owner-a")

    assert store.read_parent() == old
    assert state_path.read_bytes() == state_before
    assert not (tmp_path / "g1" / "prepared-report.json").exists()


def test_atomic_publication_syncs_parent_directory(tmp_path: Path, monkeypatch):
    import hagi.orchestrator.state as state_module

    synced: list[Path] = []
    monkeypatch.setattr(
        state_module, "_fsync_directory", lambda path: synced.append(path)
    )
    replaced = tmp_path / "state.json"
    state_module._atomic_write(replaced, b"state")

    published = tmp_path / "report.json"
    store = state_module.GrowthRunStore(tmp_path / "store")
    store.root.mkdir()
    store._publish_regular_no_replace(published, b"report", "invalid marker")

    assert replaced.read_bytes() == b"state"
    assert published.read_bytes() == b"report"
    assert synced == [tmp_path, tmp_path]


def test_commit_revalidates_immutable_snapshot_evidence(tmp_path: Path):
    store, old, decision, replacement = _prepared_accepted(tmp_path)
    report = json.loads((tmp_path / "g1" / "report.json").read_text(encoding="utf-8")) if (
        tmp_path / "g1" / "report.json"
    ).exists() else json.loads(
        (tmp_path / "g1" / "prepared-report.json").read_text(encoding="utf-8")
    )
    Path(report["holdout_evidence_path"]).write_bytes(b"tampered-snapshot")

    with pytest.raises(ValueError, match="bound evidence digest mismatch"):
        store.commit_accepted_terminal(
            old, replacement, decision, owner_id="owner-a"
        )
    assert store.read_parent() == old


def test_commit_rejects_forged_accepted_decision_binding(tmp_path: Path):
    store, old, decision, replacement = _prepared_accepted(tmp_path)
    forged = replace(decision, holdout_evidence_sha256="f" * 64)

    with pytest.raises(
        ValueError,
        match="accepted decision receipt mismatch|prepared source binding",
    ):
        store.commit_accepted_terminal(
            old, replacement, forged, owner_id="owner-a"
        )
    assert store.read_parent() == old


def test_terminal_report_rejects_state_contradiction(tmp_path: Path):
    store, old, decision, replacement = _accepted_replacement(tmp_path)
    report_path = tmp_path / "g1" / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["decision"] = "rejected"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal report"):
        store.recover("g1", owner_id="owner-a")
    # The parent pointer is the commit point: a later tamper must fail closed
    # and must not roll the lineage back to the superseded parent.
    assert store.read_parent() == replacement


def test_commit_requires_live_accepted_generation_lease(tmp_path: Path):
    store, old, decision, replacement = _prepared_accepted(tmp_path)
    lease_path = tmp_path / "g1" / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))

    with pytest.raises(ValueError, match="lease expired"):
        store.commit_accepted_terminal(
            old, replacement, decision, owner_id="owner-a"
        )
    assert store.read_parent() == old


def test_commit_fences_lease_takeover_through_pointer_publication(
    tmp_path: Path, monkeypatch
):
    store, old, decision, replacement = _prepared_accepted(tmp_path)
    run = tmp_path / "g1"
    lease_path = run / ".owner.lease"
    original = store._accepted_payload_preflight
    takeover_succeeded = False

    def takeover_during_locked_preflight(expected, new_parent, bound):
        nonlocal takeover_succeeded
        result = original(expected, new_parent, bound)
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
        lease["expires_at"] = 0
        lease_path.write_bytes(canonical_json_bytes(lease))
        try:
            store.recover_stale_lock(
                "g1", "owner-b", "takeover raced accepted commit", lease_seconds=60
            )
        except ValueError as exc:
            assert "locked" in str(exc)
        else:
            takeover_succeeded = True
        return result

    monkeypatch.setattr(
        store, "_accepted_payload_preflight", takeover_during_locked_preflight
    )
    store.commit_accepted_terminal(old, replacement, decision, owner_id="owner-a")

    assert not takeover_succeeded
    assert json.loads((run / "state.json").read_text())["owner_id"] == "owner-a"
    assert store.read_parent() == replacement


def test_parent_cas_fences_lease_takeover_through_pointer_publication(
    tmp_path: Path, monkeypatch
):
    """The replacement-CAS fence must cover the whole pointer publication.

    ``compare_and_swap_parent`` carries its own generation fence, separate
    from the one in ``commit_accepted_terminal``. That fence had no test:
    deleting it left the entire orchestrator suite green, so it was a
    comment claiming a guarantee rather than a guarantee. The takeover is
    therefore injected inside the locked region, exactly as
    ``test_commit_fences_lease_takeover_through_pointer_publication`` does
    for the commit path.

    Removing ``locks.enter_context(self._recovery_locks(...))`` from
    ``compare_and_swap_parent`` makes ``takeover_succeeded`` true and this
    test fails.
    """
    # ``compare_and_swap_parent`` with an accepted decision is reached in
    # practice on the idempotent replay path: a crash after the terminal
    # commit re-enters the CAS with the same accepted decision, and the
    # binding revalidation still happens under the generation fence.
    store, old, decision, replacement = _accepted_replacement(tmp_path)
    run = tmp_path / "g1"
    lease_path = run / ".owner.lease"
    assert (run / "report.json").is_file()
    assert store.read_parent() == replacement

    # Now inject the takeover inside the locked region of the CAS.
    original = store._accepted_decision_binds
    takeover_succeeded = False
    calls = []

    def takeover_during_locked_validation(generation_id, pointer, bound):
        nonlocal takeover_succeeded
        calls.append(generation_id)
        result = original(generation_id, pointer, bound)
        # Only the call made under the fence is inside the CAS window; the
        # pre-lock call happens before the lock is taken.
        if len(calls) < 2:
            return result
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
        lease["expires_at"] = 0
        lease_path.write_bytes(canonical_json_bytes(lease))
        try:
            store.recover_stale_lock(
                "g1", "owner-b", "takeover raced parent CAS", lease_seconds=60
            )
        except ValueError as exc:
            assert "locked" in str(exc)
        else:
            takeover_succeeded = True
        return result

    monkeypatch.setattr(store, "_accepted_decision_binds", takeover_during_locked_validation)
    store.compare_and_swap_parent(
        old, replacement, owner_id="owner-a", accepted_decision=decision
    )

    assert not takeover_succeeded, (
        "lease takeover committed inside the parent CAS window; the generation "
        "fence did not cover pointer publication"
    )
    assert store.read_parent() == replacement


def test_report_crash_window_recovers_report_and_state(tmp_path: Path, monkeypatch):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner")
    store.mark_running("g0", owner_id="owner")
    d = _decision(tmp_path, "rejected")
    _prepare(store, "g0", d, "owner")

    original = store._publish_marker

    def crash_before_report(path, data, error_message):
        if path.name == "report.json":
            raise OSError("injected crash before report")
        return original(path, data, error_message)

    monkeypatch.setattr(store, "_publish_marker", crash_before_report)
    with pytest.raises(OSError, match="before report"):
        store.publish_terminal("g0", owner_id="owner")

    run = tmp_path / "g0"
    assert (run / "prepared-report.json").is_file()
    assert not (run / "report.json").exists()
    assert json.loads((run / "state.json").read_text())["state"] == "prepared"

    monkeypatch.undo()
    assert store.recover("g0", owner_id="owner").value == "prepared"
    assert not (run / "report.json").exists()
    assert store.publish_terminal("g0", owner_id="owner").is_file()
    state = json.loads((run / "state.json").read_text())
    assert state["terminal_report_sha256"] == sha256_file(run / "report.json")


def test_terminal_idempotency_uses_persisted_prepared_binding(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    store.reserve("g0", owner_id="owner")
    store.mark_running("g0", owner_id="owner")
    decision = _decision(tmp_path, "rejected")
    _prepare(store, "g0", decision, "owner")
    report = store.publish_terminal("g0", owner_id="owner")
    assert store.publish_terminal("g0", owner_id="owner") == report
    prepared = json.loads((tmp_path / "g0" / "prepared-report.json").read_text())
    assert prepared["decision"] == "rejected"


def test_tampered_terminal_report_is_never_accepted(tmp_path: Path):
    store, old, decision, replacement = _accepted_replacement(tmp_path)
    report_path = tmp_path / "g1" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["holdout_evidence_sha256"] = "f" * 64
    report_path.write_bytes(canonical_json_bytes(report))

    with pytest.raises(ValueError, match="bound evidence|terminal report|prepared"):
        store.recover("g1", owner_id="owner-a")
    assert store.read_parent() == replacement


def test_terminal_state_binds_exact_report_marker_bytes(tmp_path: Path):
    store, old, decision, replacement = _accepted_replacement(tmp_path)
    report_path = tmp_path / "g1" / "report.json"
    report_path.write_bytes(report_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="terminal report"):
        store.recover("g1", owner_id="owner-a")
    assert store.read_parent() == replacement


def test_terminal_state_binds_exact_report_digest(tmp_path: Path):
    store, old, decision, replacement = _accepted_replacement(tmp_path)
    state_path = tmp_path / "g1" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["terminal_report_sha256"] = "f" * 64
    state_path.write_bytes(canonical_json_bytes(state))

    with pytest.raises(ValueError, match="terminal report"):
        store.recover("g1", owner_id="owner-a")
    with pytest.raises(
        ValueError, match="terminal report|contradiction|parent lineage"
    ):
        store.commit_accepted_terminal(
            replacement, replacement, decision, owner_id="owner-a"
        )
    assert store.read_parent() == replacement


def test_symlink_leaf_and_ancestor_are_rejected(tmp_path: Path):
    import os
    real = tmp_path / "real"
    real.mkdir()
    store = GrowthRunStore(real)
    store.reserve("g0", owner_id="owner")
    state_path = tmp_path / "real" / "g0" / "state.json"
    state_path.unlink()
    state_path.symlink_to(tmp_path / "real" / "g0" / "state-link.json")
    with pytest.raises(ValueError, match="symlink or junction"):
        store.mark_running("g0", owner_id="owner")
    linked_parent = tmp_path / "linked-parent"
    try:
        os.symlink(real, linked_parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        return
    with pytest.raises(ValueError, match="symlink or junction"):
        GrowthRunStore(linked_parent)
