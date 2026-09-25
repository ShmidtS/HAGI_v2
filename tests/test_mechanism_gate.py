"""Verdict-independent mechanism checks over owner-committed artifacts.

The gate never reads the owner's verdict to decide whether a mechanism check
passed: every Tier 1 check is derived from digest-verified bytes. The negative
controls below mutate one artifact at a time and assert that the specific
check fails, so a check cannot silently degrade into a tautology.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

import hagi.orchestrator.mechanism_gate as gate
import hagi.orchestrator.real_cycle as real_cycle
from hagi.orchestrator.mechanism_gate import (
    CROSS_PARENT_TRANSFORM,
    MechanismGateError,
    evaluate_generation,
    evaluate_traversal,
    run_generations,
)
from hagi.orchestrator.real_cycle import SYNTHETIC_V3_SEED
from hagi.orchestrator.state import canonical_json_bytes

_DEVICE = "cpu"
_MAX_STEPS = 1


@pytest.fixture(scope="module")
def executed(tmp_path_factory: pytest.TempPathFactory):
    """Run one bounded two-call cycle and keep its artifacts for mutation."""
    root = tmp_path_factory.mktemp("mech-gate") / "cycle"
    observations, results, evaluator = run_generations(
        root, seed=SYNTHETIC_V3_SEED, max_steps=_MAX_STEPS, count=2, device=_DEVICE
    )
    return observations, results, evaluator, root


def _tier1(observations) -> dict[str, bool]:
    findings = evaluate_generation(observations[0])
    assert findings.failures == ()
    return dict(findings.checks)


def test_preregistered_transform_is_fixed_not_selected(executed) -> None:
    """The transform is pinned before execution, never chosen by outcome."""
    assert CROSS_PARENT_TRANSFORM == "f3_tree"
    plan = json.loads(
        (Path(__file__).resolve().parent.parent
         / ".omc" / "plans" / "two_generation_cpu_gate_v3.json").read_bytes()
    )
    assert plan["resolved_seed"] == SYNTHETIC_V3_SEED


def test_executor_never_accepts_a_caller_supplied_evaluator(executed) -> None:
    """There is no injection point that could supply optimistic metrics."""
    with pytest.raises(TypeError):
        gate.run_generations(
            Path("unused"),
            seed=SYNTHETIC_V3_SEED,
            max_steps=_MAX_STEPS,
            count=1,
            evaluator=lambda _ctx: None,
        )


def test_executor_requires_a_positive_count() -> None:
    for bad in (0, -1, True, 1.0):
        with pytest.raises(ValueError):
            gate.run_generations(Path("unused"), seed=SYNTHETIC_V3_SEED, count=bad)


def test_evaluator_delegate_is_the_real_bounded_evaluator(executed) -> None:
    _, _, evaluator, _ = executed
    assert evaluator.delegate is real_cycle.path_only_evaluator
    assert evaluator.requests, "the real evaluator never ran"


def test_tier1_passes_on_real_owner_artifacts(executed) -> None:
    checks = _tier1(executed[0])
    assert len(checks) >= 20
    assert all(checks.values())
    # Verdict independence: acceptance is not an input to the checks.
    assert evaluate_generation(executed[0][0]).passed is True


def test_tier1_ignores_a_rejected_owner_decision(executed) -> None:
    """A rejected generation still passes the mechanism checks."""
    observation = executed[0][0]
    assert observation.owner_decision == "rejected"
    findings = evaluate_generation(observation)
    assert findings.owner_decision == "rejected"
    assert findings.passed is True


def _mutate_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_bytes())
    mutated = mutate(copy.deepcopy(payload))
    path.write_bytes(canonical_json_bytes(mutated))


def _rebuild(observation, *, evidence=None, report=None):
    """Return a copy of the observation with substituted digest-verified payloads."""
    if evidence is None and report is None:
        return observation
    changes: dict[str, object] = {}
    if evidence is not None:
        changes["evidence"] = evidence
        changes["evidence_bytes"] = canonical_json_bytes(evidence)
    if report is not None:
        changes["report"] = report
        changes["report_bytes"] = canonical_json_bytes(report)
    return dataclasses.replace(observation, **changes)


def test_negative_control_detects_evidence_decision_drift(executed) -> None:
    observations = executed[0]
    tampered = _rebuild(observations[0], evidence={**observations[0].evidence, "decision": "accepted"})
    findings = evaluate_generation(tampered)
    assert "evidence_decision_matches_terminal" in findings.failures
    assert findings.passed is False


def test_negative_control_detects_generation_id_drift(executed) -> None:
    observations = executed[0]
    tampered = _rebuild(observations[0], evidence={**observations[0].evidence, "generation_id": "generation-9"})
    findings = evaluate_generation(tampered)
    assert "evidence_generation_matches_request" in findings.failures


def test_negative_control_detects_binding_drift(executed) -> None:
    observations = executed[0]
    evidence = copy.deepcopy(observations[0].evidence)
    evidence["bindings"]["base_seed"] += 1
    tampered = _rebuild(observations[0], evidence=evidence)
    findings = evaluate_generation(tampered)
    assert "bindings_base_seed_equals_request" in findings.failures


def test_negative_control_detects_tokenizer_drift(executed) -> None:
    observations = executed[0]
    evidence = copy.deepcopy(observations[0].evidence)
    evidence["bindings"]["tokenizer_name"] = "other/tokenizer"
    tampered = _rebuild(observations[0], evidence=evidence)
    findings = evaluate_generation(tampered)
    assert "bindings_tokenizer_equals_request" in findings.failures


def test_negative_control_detects_row_ids_drift(executed) -> None:
    observations = executed[0]
    evidence = copy.deepcopy(observations[0].evidence)
    evidence["bindings"]["row_ids_sha256"] = "0" * 64
    tampered = _rebuild(observations[0], evidence=evidence)
    findings = evaluate_generation(tampered)
    assert "bindings_row_ids_equal_request" in findings.failures


def test_negative_control_detects_holdout_digest_drift(executed) -> None:
    """A rebound holdout digest is the §15.1-A leak class and must fail."""
    observation = executed[0][0]
    evidence = copy.deepcopy(observation.evidence)
    evidence["bindings"]["holdout_sha256"] = "b" * 64
    findings = evaluate_generation(_rebuild(observation, evidence=evidence))
    assert "bindings_holdout_digest_equals_request" in findings.failures


def test_negative_control_detects_data_manifest_drift(executed) -> None:
    """A rebound data-manifest digest must fail."""
    observation = executed[0][0]
    evidence = copy.deepcopy(observation.evidence)
    evidence["bindings"]["data_manifest_sha256"] = "c" * 64
    findings = evaluate_generation(_rebuild(observation, evidence=evidence))
    assert "bindings_data_manifest_equals_request" in findings.failures


def test_negative_control_detects_parent_rebinding(executed) -> None:
    observations = executed[0]
    evidence = copy.deepcopy(observations[0].evidence)
    evidence["bindings"]["parent"]["checkpoint_sha256"] = "f" * 64
    tampered = _rebuild(observations[0], evidence=evidence)
    findings = evaluate_generation(tampered)
    assert "bindings_parent_equals_request_parent" in findings.failures


def test_negative_control_detects_child_plan_drift(executed) -> None:
    observations = executed[0]
    evidence = copy.deepcopy(observations[0].evidence)
    evidence["bindings"]["child_plans"] = "replaced"
    tampered = _rebuild(observations[0], evidence=evidence)
    findings = evaluate_generation(tampered)
    assert "bindings_child_plans_equal_request" in findings.failures


def test_negative_control_detects_unearned_quality_claim(executed) -> None:
    observations = executed[0]
    evidence = copy.deepcopy(observations[0].evidence)
    evidence["quality_supported"] = True
    tampered = _rebuild(observations[0], evidence=evidence)
    findings = evaluate_generation(tampered)
    assert "no_unearned_quality_claim" in findings.failures


def test_negative_control_detects_candidate_digest_aliasing_a_child(executed) -> None:
    """A candidate digest equal to a child digest must fail the gate."""
    observation = executed[0][0]
    f3 = copy.deepcopy(observation.evidence["candidate_f3"])
    f3["checkpoint_sha256"] = f3["child_checkpoint_sha256"][0]
    evidence = copy.deepcopy(observation.evidence)
    evidence["candidate_f3"] = f3
    findings = evaluate_generation(_rebuild(observation, evidence=evidence))
    assert "candidate_digest_differs_from_children" in findings.failures


def test_negative_control_detects_missing_lift(executed) -> None:
    """A child that is already lifted must fail the pre-lift check."""
    observation = executed[0][0]
    lifted = copy.deepcopy(observation.child_configs[0])
    lifted["merge"]["ternary_depth"] += 1
    tampered = dataclasses.replace(observation, child_configs=(lifted,) * 3)
    findings = evaluate_generation(tampered)
    assert "child_configs_merge_disabled_and_unlifted" in findings.failures


def test_negative_control_detects_unlifted_candidate(executed) -> None:
    """A candidate that never lifted the merge must fail."""
    observation = executed[0][0]
    unlifted = copy.deepcopy(observation.candidate_config)
    unlifted["merge"]["ternary_depth"] = 0
    unlifted["merge"]["enabled"] = False
    tampered = dataclasses.replace(observation, candidate_config=unlifted)
    findings = evaluate_generation(tampered)
    assert "candidate_lifts_merge_above_children" in findings.failures


def test_observation_reads_back_verified_bytes(executed) -> None:
    """The observation must come from digest-verified reads, not in-memory state."""
    observations, results, _, root = executed
    observation = observations[0]
    on_disk = Path(observation.report_path).read_bytes()
    assert on_disk == observation.report_bytes
    assert observation.report_sha256 == real_cycle.sha256_bytes(on_disk)
    # A second bounded call replays the same terminal generation.
    assert results[1].generation_id == results[0].generation_id


def test_depth_check_uses_parent_bytes_not_the_candidate(executed) -> None:
    """The depth check must compare against the parent, not self-reference."""
    observation = executed[0][0]
    assert observation.candidate_ternary_depth == observation.parent_ternary_depth + 1
    wrong_parent = dataclasses.replace(
        observation, parent_ternary_depth=observation.candidate_ternary_depth
    )
    assert "candidate_depth_equals_parent_plus_one" in evaluate_generation(wrong_parent).failures


_TAMPER_VECTORS = {
    "decision": ("evidence", lambda p: p.__setitem__("decision", "accepted")),
    "generation_id": ("evidence", lambda p: p.__setitem__("generation_id", "generation-2")),
    "base_seed": ("bindings", lambda p: p.__setitem__("base_seed", p["base_seed"] + 1)),
    "protocol": ("bindings", lambda p: p.__setitem__("protocol_sha256", "a" * 64)),
    "tokenizer": ("bindings", lambda p: p.__setitem__("tokenizer_name", "x/y")),
    "row_ids": ("bindings", lambda p: p.__setitem__(
        "row_ids_sha256", ["0" * 64] * len(p["row_ids_sha256"]))),
    "holdout": ("bindings", lambda p: p.__setitem__("holdout_sha256", "b" * 64)),
    "data_manifest": ("bindings", lambda p: p.__setitem__("data_manifest_sha256", "c" * 64)),
    "quality_claim": ("evidence", lambda p: p.__setitem__("quality_supported", True)),
    "child_plans": ("bindings", lambda p: p.__setitem__("child_plans", "replaced")),
    "parent_rebind": ("bindings", lambda p: p.__setitem__(
        "parent", {**p["parent"], "checkpoint_sha256": "f" * 64})),
    "candidate_alias": ("candidate_f3", lambda p: p.__setitem__(
        "checkpoint_sha256", p["child_checkpoint_sha256"][0])),
    "child_dup": ("candidate_f3", lambda p: p.__setitem__(
        "child_checkpoint_sha256", [p["child_checkpoint_sha256"][0]] * 3)),
}


def _apply_vector(observation, where: str, mutate):
    """Apply one tamper vector to a copy of the observation's evidence.

    ``where`` names a nested key when present, otherwise the root payload.
    """
    evidence = copy.deepcopy(observation.evidence)
    mutate(evidence[where] if where in evidence else evidence)
    return _rebuild(observation, evidence=evidence)


@pytest.mark.parametrize("vector", sorted(_TAMPER_VECTORS))
def test_tamper_matrix_detects_every_vector(executed, vector: str) -> None:
    """Quantified degradation check: no single-field tamper may pass the gate."""
    where, mutate = _TAMPER_VECTORS[vector]
    findings = evaluate_generation(_apply_vector(executed[0][0], where, mutate))
    assert findings.passed is False, f"tamper vector {vector!r} went undetected"


def test_tamper_matrix_is_total(executed) -> None:
    """Every enumerated vector must be individually detected, none silently skipped."""
    observation = executed[0][0]
    undetected = [
        name
        for name, (where, mutate) in sorted(_TAMPER_VECTORS.items())
        if evaluate_generation(_apply_vector(observation, where, mutate)).passed
    ]
    assert undetected == []
    assert len(_TAMPER_VECTORS) >= 13


def test_report_is_anchored_in_the_owner_ledger_not_itself(executed) -> None:
    """The report digest must come from state.json, never from the report.

    A self-anchored check is ``x == x``: it cannot fail. The probe rewrites a
    report field that no gate check inspects (``schema_version``). Its bytes
    change, so a ledger anchor rejects the whole report while a self-anchored
    read accepts the substitution — this is the only way the two differ.
    """
    observations, _, _, root = executed
    generation = observations[0].generation_id
    run = root / "store" / generation
    state = json.loads((run / "state.json").read_bytes())
    assert state["terminal_report_sha256"] == observations[0].report_sha256

    report = run / "report.json"
    original = report.read_bytes()
    try:
        payload = json.loads(original)
        assert "schema_version" in payload
        payload["schema_version"] = "tampered-but-uninspected"
        report.write_bytes(canonical_json_bytes(payload))
        assert report.read_bytes() != original, "probe did not change any byte"
        # Ledger anchor: the substitution is refused.
        with pytest.raises(MechanismGateError, match="owner-recorded digest"):
            gate._read_contained(
                root, report, state["terminal_report_sha256"], "terminal report"
            )
        # A self-anchored read would have accepted it, which is the bug.
        assert gate._read_bytes(report, gate._digest_of(report), "terminal report")
    finally:
        report.write_bytes(original)


def test_tampered_report_fails_the_ledger_anchor(executed) -> None:
    """Rewriting an inspected report field must make the gate refuse to observe."""
    observations, _, _, root = executed
    generation = observations[0].generation_id
    run = root / "store" / generation
    report = run / "report.json"
    original = report.read_bytes()
    try:
        payload = json.loads(original)
        payload["candidate_checkpoint_path"] = "C:/elsewhere/evil.bin"
        report.write_bytes(canonical_json_bytes(payload))
        with pytest.raises(MechanismGateError, match="owner-recorded digest"):
            gate._read_contained(
                root,
                report,
                json.loads((run / "state.json").read_bytes())["terminal_report_sha256"],
                "terminal report",
            )
    finally:
        report.write_bytes(original)


def test_gate_error_messages_never_carry_holdout_digests(executed) -> None:
    """A digest in a builder-visible error is a holdout-membership oracle."""
    observations, _, _, _ = executed
    observation = observations[0]
    path = Path(observation.evidence["candidate_f3"]["checkpoint_path"])
    original = path.read_bytes()
    secret = observation.evidence["candidate_f3"]["checkpoint_sha256"]
    try:
        path.write_bytes(original + b"tamper")
        with pytest.raises(MechanismGateError) as excinfo:
            gate._read_bytes(path, secret, "candidate")
        message = str(excinfo.value)
        assert secret not in message
        assert observation.evidence["bindings"]["holdout_sha256"] not in message
    finally:
        path.write_bytes(original)


def test_paths_outside_the_run_store_are_refused(executed) -> None:
    """Evidence naming an outside path must not be readable by the gate."""
    observations, _, _, root = executed
    with pytest.raises(MechanismGateError, match="outside the run store"):
        gate._read_contained(
            root,
            Path(__file__).resolve(),
            gate._digest_of(Path(__file__).resolve()),
            "candidate",
        )


def test_observe_refuses_a_report_the_ledger_does_not_anchor(executed) -> None:
    """Exercise the real observation path, not just the helper.

    Rewriting an uninspected report field changes the bytes without breaking
    any content check, so this is the only probe that distinguishes a ledger
    anchor (refuses) from a self-anchored digest (accepts).
    """
    observations, results, _, root = executed
    observation = observations[0]
    report = Path(observation.report_path)
    original = report.read_bytes()
    try:
        payload = json.loads(original)
        payload["schema_version"] = "tampered-but-uninspected"
        report.write_bytes(canonical_json_bytes(payload))
        with pytest.raises(MechanismGateError, match="owner-recorded digest"):
            gate._observe(root, results[0], observation.request)
    finally:
        report.write_bytes(original)
    # Sanity: the restored report is observed again without error.
    assert gate._observe(root, results[0], observation.request).report_bytes == original


def test_tier2_is_false_without_an_accepted_traversal(executed) -> None:
    """Tier 2 never claims a traversal that was never observed."""
    result = evaluate_traversal(executed[0])
    assert result["tier"] == 2
    assert result["passed"] is False
    assert result["reason"]


def test_tier2_is_false_for_an_empty_run() -> None:
    result = evaluate_traversal([])
    assert result["passed"] is False
    assert result["reason"]


def test_tier2_is_conditional_and_never_implied_by_tier1() -> None:
    result = evaluate_traversal([object()])
    assert result["conditional"] is True
    assert result["passed"] is False


def test_gate_rejects_a_tampered_checkpoint_on_read(executed) -> None:
    """A digest mismatch must fail closed rather than read unverified bytes."""
    observations, _, _, _ = executed
    observation = observations[0]
    path = Path(observation.evidence["candidate_f3"]["checkpoint_path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tamper")
        with pytest.raises(MechanismGateError):
            gate._read_bytes(str(path), observation.evidence["candidate_f3"]["checkpoint_sha256"], "candidate")
    finally:
        path.write_bytes(original)


def test_owner_artifacts_are_not_edited_by_the_gate(executed) -> None:
    """The gate is read-only over owner state."""
    observations, _, _, root = executed
    before = {
        path: path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    for observation in observations:
        evaluate_generation(observation)
    after = {
        path: path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert before == after


# --- state.json must be strict and must not anchor its own digest ---------
#
# The gate docstring claims the expected terminal-report digest "has to come
# from the store's own state record, which is written by the owner and is not
# reachable from a builder context". Before this fix the code did the
# opposite: it computed the state.json digest from state.json itself
# (``_digest_of``) and parsed it with a bare ``json.loads``. A builder that
# can write the run store could therefore edit ``state.json`` and
# ``report.json`` together and have the forged report validated.
#
# The probes below distinguish a real owner-side ledger anchor (refuses a
# tampered state) from a self-anchored digest (accepts anything).


def _run_state(root: Path, result) -> Path:
    return root / "store" / result.generation_id / "state.json"


def test_state_json_with_duplicate_keys_is_rejected(executed) -> None:
    """``json.loads`` silently keeps the last duplicate key; the gate must not."""
    observations, results, _, root = executed
    observation = observations[0]
    path = _run_state(root, results[0])
    original = path.read_bytes()
    try:
        # Prepend a duplicate-key member, so the last-wins json.loads would
        # silently pick the forged digest instead of the owner's.
        forged = b'{"terminal_report_sha256":"' + b"0" * 64 + b'",'
        forged += b'"state":"reserved",' + original.lstrip(b"{")
        path.write_bytes(forged)
        with pytest.raises(MechanismGateError):
            gate._observe(root, results[0], observation.request)
    finally:
        path.write_bytes(original)


def test_non_canonical_state_json_is_rejected(executed) -> None:
    """Valid-but-not-canonical state bytes must fail closed, not be coerced."""
    observations, results, _, root = executed
    observation = observations[0]
    path = _run_state(root, results[0])
    original = path.read_bytes()
    try:
        payload = json.loads(original)
        # Same content, different bytes: reversed key order, pretty-printed.
        path.write_bytes(json.dumps(payload, indent=2).encode("utf-8"))
        with pytest.raises(MechanismGateError):
            gate._observe(root, results[0], observation.request)
    finally:
        path.write_bytes(original)


def test_owner_state_is_read_through_the_strict_owner_reader(executed) -> None:
    """The self-anchored TOFU digest must be gone from the owner read.

    A vacuous "hash the file and compare it to itself" check can never fail,
    so it must not exist. This asserts the strict owner reader is the one
    doing the parsing, and that a state whose *content* violates the owner
    contract is refused even though its bytes are perfectly canonical.
    """
    observations, results, _, root = executed
    observation = observations[0]
    run = root / "store" / results[0].generation_id
    state_path = run / "state.json"
    original = state_path.read_bytes()
    try:
        state = json.loads(original)
        # Canonical bytes, but the owner contract is violated: a reserved run
        # may not carry a terminal report digest.
        state["state"] = "reserved"
        state["terminal_report_sha256"] = "0" * 64
        state_path.write_bytes(canonical_json_bytes(state))
        with pytest.raises(MechanismGateError):
            gate._observe(root, results[0], observation.request)
    finally:
        state_path.write_bytes(original)


def test_owner_state_read_is_not_self_verified() -> None:
    """Static guard: the owner read must not hash the file it is verifying.

    ``_digest_of`` may still exist as a plain utility (tests use it to
    compute "what the digest of these bytes is"), but the owner state read
    must not feed it back as the expected digest -- that comparison can never
    fail and is theater dressed as verification.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(gate._owner_terminal_digest)))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_digest_of" not in called, (
        "the owner state read must not verify state.json against itself"
    )
    assert "_read_state" in called, (
        "the owner state read must go through the owner's strict reader"
    )


def test_restored_state_is_observed_again_without_error(executed) -> None:
    """Guard the probes above: restoration must leave the store readable."""
    observations, results, _, root = executed
    observation = observations[0]
    assert gate._observe(root, results[0], observation.request).report_bytes
