"""CPU-only tests for the bounded recursive-generation production binding."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import hagi.orchestrator.real_cycle as real_cycle
from hagi.orchestrator.real_cycle import (
    BANKING77_MANIFEST_SHA256,
    SourceSpan,
    _compact_vocabulary,
    _sealed_span_budget,
    _sealed_token_slice,
    _synthetic_streams,
    _token_bytes_digest,
    _verify_token_stream,
    _write_exact,
    bounded_token_batches,
    build_deterministic_parent,
    path_only_evaluator,
    preflight_banking77,
    result_data_provenance,
    run_banking77_cycle,
    run_bounded_cycle,
)
from hagi.orchestrator.recursive import EvaluationResult, GenerationResult, SourceMetric
from hagi.orchestrator.state import GrowthRunStore, canonical_json_bytes, sha256_file

_ROOT = Path(__file__).resolve().parent.parent
_CLI = _ROOT / "scripts" / "recursive_growth.py"
_GATE_V2 = _ROOT / ".omc" / "plans" / "two_generation_cpu_gate_v2.json"
_GATE_V3 = _ROOT / ".omc" / "plans" / "two_generation_cpu_gate_v3.json"


def _derived_seed(path: Path) -> tuple[int, str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol.pop("resolved_seed")
    digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    return int(digest[:8], 16) % 1_000_000, digest


def test_gate_v2_is_invalidated_and_v3_preserves_its_protocol():
    """§15.1-B: a preregistration must satisfy its own seed derivation.

    v2 recorded a seed that its frozen protocol does not derive, so it was
    never executed. v3 changes that one derived field and nothing else: the
    protocol payload, acceptance criteria, and no-go list stay byte-identical.
    """
    v2_seed, v2_digest = _derived_seed(_GATE_V2)
    v2_recorded = json.loads(_GATE_V2.read_text(encoding="utf-8"))["resolved_seed"]
    assert v2_recorded != v2_seed

    v3_seed, v3_digest = _derived_seed(_GATE_V3)
    v3_recorded = json.loads(_GATE_V3.read_text(encoding="utf-8"))["resolved_seed"]
    assert v3_recorded == v3_seed
    assert v3_digest == v2_digest

    v2 = json.loads(_GATE_V2.read_text(encoding="utf-8"))
    v3 = json.loads(_GATE_V3.read_text(encoding="utf-8"))
    for field in (
        "device",
        "max_steps_per_child",
        "synthetic_domain",
        "seed_derivation",
        "acceptance",
        "no_go",
        "claim_boundary",
        "interpretation",
    ):
        assert canonical_json_bytes(v3[field]) == canonical_json_bytes(v2[field])


def test_executable_synthetic_seed_matches_frozen_preregistration():
    """The hard-pinned seed must equal the v3 derivation, not a literal."""
    derived, _ = _derived_seed(_GATE_V3)
    assert real_cycle.SYNTHETIC_V3_SEED == derived
    spec = importlib.util.spec_from_file_location("recursive_growth_seed_pin", _CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SYNTHETIC_V3_SEED == derived


def _rejecting_evaluator(context):
    row_ids = context.holdout.row_ids_sha256
    metrics = tuple(
        SourceMetric(source, 4, 1.0, row_ids[index])
        for index, source in enumerate(("A", "B", "C"))
    )
    return EvaluationResult(
        metrics,
        tuple(
            SourceMetric(item.source_id, 4, 2.0, item.row_ids_sha256)
            for item in metrics
        ),
        context.holdout.tokenizer_name,
        context.holdout.data_manifest_sha256,
        context.holdout.protocol_sha256,
        context.parent.checkpoint_sha256,
        context.candidate.checkpoint_sha256,
        context.candidate.manifest_sha256,
    )


def _accepting_evaluator(context):
    row_ids = context.holdout.row_ids_sha256
    incumbent = tuple(
        SourceMetric(source, 4, 1.0, row_ids[index])
        for index, source in enumerate(("A", "B", "C"))
    )
    candidate = tuple(
        SourceMetric(source, 4, 0.5, row_ids[index])
        for index, source in enumerate(("A", "B", "C"))
    )
    return EvaluationResult(
        incumbent,
        candidate,
        context.holdout.tokenizer_name,
        context.holdout.data_manifest_sha256,
        context.holdout.protocol_sha256,
        context.parent.checkpoint_sha256,
        context.candidate.checkpoint_sha256,
        context.candidate.manifest_sha256,
    )


def test_parent_checkpoint_bytes_are_deterministic(tmp_path):
    first, first_path = build_deterministic_parent(tmp_path / "one", seed=77)
    second, second_path = build_deterministic_parent(tmp_path / "two", seed=77)
    assert first.checkpoint_sha256 == second.checkpoint_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert sha256_file(first_path) == sha256_file(second_path)


def test_synthetic_seed_is_rejected_before_any_side_effect(tmp_path, monkeypatch):
    root = tmp_path / "wrong-seed"
    monkeypatch.setattr(
        real_cycle,
        "build_deterministic_parent",
        lambda *args, **kwargs: pytest.fail("parent built before seed validation"),
    )
    monkeypatch.setattr(
        real_cycle,
        "run_generation",
        lambda *args, **kwargs: pytest.fail("generation ran before seed validation"),
    )

    with pytest.raises(ValueError, match="preregistered seed 301097"):
        run_bounded_cycle(
            root,
            seed=1234,
            evaluator=lambda context: pytest.fail("evaluator ran before seed validation"),
        )

    assert not root.exists()


def test_real_synthetic_cycle_uses_real_owner_and_claims(tmp_path):
    result = run_bounded_cycle(
        tmp_path / "cycle",
        seed=301097,
        max_steps=1,
        evaluator=_rejecting_evaluator,
    )
    assert result.generation_id == "generation-1"
    # A rejected candidate must not claim mechanism support. This assertion is
    # the falsifier for the old hard-coded ``True``; reverting
    # ``_evidence_payload`` to the literal fails the run below too.
    assert result.decision == "rejected"
    assert result.mechanism_supported is False
    assert result.quality_supported is False
    assert result.security_supported is False
    assert result.production_promotion is False
    evidence = json.loads(Path(result.holdout_evidence_path).read_text(encoding="utf-8"))
    assert evidence["mechanism_supported"] is False
    assert evidence["quality_supported"] is False
    assert evidence["security_supported"] is False
    assert evidence["production_promotion"] is False


def test_reject_does_not_change_parent_and_rerun_is_idempotent(tmp_path):
    root = tmp_path / "reject"
    result = run_bounded_cycle(root, seed=301097, evaluator=_rejecting_evaluator)
    assert result.decision == "rejected"
    store = GrowthRunStore(root / "store")
    parent = store.read_parent()
    assert parent is not None and parent.generation_id == "depth-zero"
    again = run_bounded_cycle(
        root, seed=301097, evaluator=lambda context: pytest.fail("rerun evaluated")
    )
    assert again == result
    assert store.read_parent() == parent


def test_accept_commits_through_owner_only(tmp_path):
    root = tmp_path / "accept"
    before = GrowthRunStore(root / "store").read_parent()
    result = run_bounded_cycle(root, seed=301097, evaluator=_accepting_evaluator)
    assert before is None
    assert result.decision == "accepted"
    after = GrowthRunStore(root / "store").read_parent()
    assert after is not None
    assert after.checkpoint_sha256 == sha256_file(result.candidate_checkpoint_path)
    assert after.generation_id == "generation-1"


def test_sealed_span_budget_rejects_out_of_file_before_reading(tmp_path: Path):
    assert _sealed_span_budget(0, 4, 2, 16) == 9
    assert _sealed_span_budget(7, 4, 2, 16) == 9
    with pytest.raises(ValueError, match="escapes the"):
        _sealed_span_budget(8, 4, 2, 16)
    for bad in ((-1, 4, 2, 16), (0, 0, 2, 16), (0, 4, 0, 16)):
        with pytest.raises(ValueError):
            _sealed_span_budget(*bad)


def test_sealed_token_slice_reads_exactly_the_bounded_window(tmp_path: Path):
    shard = tmp_path / "tokens.bin"
    tokens = list(range(2, 26))
    shard.write_bytes(np.asarray(tokens, dtype="<u4").tobytes())
    sliced = _sealed_token_slice(shard, vocab_size=64, start=4, window=4, steps=2)
    assert sliced == tokens[4:13]
    with pytest.raises(ValueError, match="escapes the"):
        _sealed_token_slice(shard, vocab_size=64, start=18, window=4, steps=2)


def test_bounded_token_batches_default_to_one_microbatch():
    batches = bounded_token_batches(range(256), sequence_length=16)
    assert len(batches) == 1
    assert tuple(batches[0]["input_ids"].shape) == (2, 16)
    with pytest.raises(ValueError, match="max_batches"):
        bounded_token_batches(range(32), max_batches=0)


def test_compact_vocabulary_and_token_digest_are_byte_exact():
    train, test, vocab_size = _compact_vocabulary(
        [0, 1, 2, 10, 12], [11, 12, 99]
    )
    assert train == [0, 1, 2, 3, 4]
    assert test == [1, 4, 1]
    assert vocab_size == 5
    assert 1 in train
    assert _token_bytes_digest([1, 2]) == hashlib.sha256(
        b"\x01\x00\x00\x00\x02\x00\x00\x00"
    ).hexdigest()
    assert _token_bytes_digest([np.uint32(2**32 - 1)]) == hashlib.sha256(
        b"\xff\xff\xff\xff"
    ).hexdigest()
    for invalid in ([-1], [1.9], [True], [np.float64(1.0)], [[1]]):
        with pytest.raises(ValueError, match="uint32"):
            _token_bytes_digest(invalid)


def test_write_exact_accepts_identical_bytes_created_after_precheck(
    tmp_path, monkeypatch
):
    path = tmp_path / "immutable.bin"
    payload = b"same-bytes"
    path.write_bytes(payload)
    original_exists = Path.exists

    def hide_target_exists(self):
        if self == path:
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", hide_target_exists)
    _write_exact(path, payload)
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="immutable file conflict"):
        _write_exact(path, payload)


def test_token_stream_binding_rejects_length_and_digest_mismatch():
    stream = [3, 4, 5]
    span = SourceSpan("A", 0, len(stream), _token_bytes_digest(stream))
    _verify_token_stream(span, stream)
    with pytest.raises(ValueError, match="length"):
        _verify_token_stream(SourceSpan("A", 0, 4, span.source_manifest_sha256), stream)
    with pytest.raises(ValueError, match="digest"):
        _verify_token_stream(SourceSpan("A", 0, 3, "0" * 64), stream)


def test_path_evaluator_rejects_every_tampered_frozen_input(tmp_path):
    root = tmp_path / "cycle"
    captured_context = None

    def capture(context):
        nonlocal captured_context
        captured_context = context
        return _rejecting_evaluator(context)

    run_bounded_cycle(root, seed=301097, evaluator=capture)
    assert captured_context is not None
    protected = (
        (Path(captured_context.holdout.path), "holdout digest"),
        (
            Path(captured_context.request.parent_checkpoint_path),
            "parent checkpoint digest",
        ),
        (Path(captured_context.candidate.checkpoint_path), "candidate checkpoint digest"),
        (Path(captured_context.candidate.manifest_path), "candidate manifest digest"),
    )
    for path, message in protected:
        original = path.read_bytes()
        path.write_bytes(original + b"\x00")
        with pytest.raises(ValueError, match=message):
            path_only_evaluator(captured_context)
        path.write_bytes(original)


def test_path_evaluator_scores_the_same_verified_bytes_after_path_change(
    tmp_path, monkeypatch
):
    root = tmp_path / "cycle"
    captured_context = None

    def capture(context):
        nonlocal captured_context
        captured_context = context
        return _rejecting_evaluator(context)

    run_bounded_cycle(root, seed=301097, evaluator=capture)
    assert captured_context is not None
    read_paths: list[Path] = []
    original = real_cycle._verified_bytes

    def snapshot_then_mutate(path, expected, label):
        data = original(path, expected, label)
        read_paths.append(Path(path))
        if Path(path) == Path(captured_context.request.parent_checkpoint_path):
            Path(path).write_bytes(data + b"\x00")
        return data

    monkeypatch.setattr(real_cycle, "_verified_bytes", snapshot_then_mutate)
    result = path_only_evaluator(captured_context)
    assert result.parent_checkpoint_sha256 == captured_context.parent.checkpoint_sha256
    assert set(read_paths) == {
        Path(captured_context.holdout.path),
        Path(captured_context.request.parent_checkpoint_path),
        Path(captured_context.candidate.checkpoint_path),
        Path(captured_context.candidate.manifest_path),
    }


def test_path_evaluator_rejects_holdout_metadata_not_bound_by_context(
    tmp_path, monkeypatch
):
    root = tmp_path / "metadata"
    captured_context = None

    def capture(context):
        nonlocal captured_context
        captured_context = context
        return _rejecting_evaluator(context)

    run_bounded_cycle(root, seed=301097, evaluator=capture)
    assert captured_context is not None
    original_verified = real_cycle._verified_bytes
    original_payload = real_cycle._holdout_payload

    def matching_snapshot(path, expected, label):
        return original_verified(path, expected, label)

    for replacement, message in (
        ({**json.loads(Path(captured_context.holdout.path).read_bytes()), "tokenizer": "other-tokenizer"}, "tokenizer metadata"),
        (
            {
                **json.loads(Path(captured_context.holdout.path).read_bytes()),
                "sources": {
                    **json.loads(Path(captured_context.holdout.path).read_bytes())["sources"],
                    "A": [42, 43],
                },
                "row_ids_sha256": {
                    **json.loads(Path(captured_context.holdout.path).read_bytes())["row_ids_sha256"],
                    "A": _token_bytes_digest([42, 43]),
                },
            },
            "row identity metadata",
        ),
    ):
        def snapshot_with_mismatched_metadata(data):
            payload = original_payload(data)
            payload.update(replacement)
            return payload

        monkeypatch.setattr(real_cycle, "_verified_bytes", matching_snapshot)
        monkeypatch.setattr(
            real_cycle, "_holdout_payload", snapshot_with_mismatched_metadata
        )
        with pytest.raises(ValueError, match=message):
            path_only_evaluator(captured_context)


def test_tampered_terminal_fails_closed(tmp_path):
    root = tmp_path / "tamper"
    result = run_bounded_cycle(root, seed=301097, evaluator=_rejecting_evaluator)
    report = Path(result.report_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["decision"] = "accepted"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        run_bounded_cycle(root, seed=301097, evaluator=_rejecting_evaluator)


def test_cli_synthetic_output_declares_synthetic_provenance(
    tmp_path, monkeypatch, capsys
):
    spec = importlib.util.spec_from_file_location("recursive_growth_cli", _CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The synthetic path is preregistration-bound: a non-registered seed must
    # fail closed rather than silently run an uncontrolled experiment.
    assert module.main([
        "--output", str(tmp_path / "synthetic-run"),
        "--seed", "1234", "--max-steps", "1",
    ]) == 2
    assert "preregistered seed" in capsys.readouterr().err
    assert module.main([
        "--output", str(tmp_path / "synthetic-run"),
        "--seed", "301097", "--max-steps", "1",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data_kind"] == "synthetic"
    assert payload["tokenizer_name"] == "synthetic-packed-v1"
    assert "data_manifest_sha256" not in payload


def test_cli_synthetic_default_is_preregistered_seed(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("recursive_growth_seed_cli", _CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    def capture_seed(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after seed capture")

    monkeypatch.setattr(module, "run_bounded_cycle", capture_seed)
    # The probe raises on purpose; the CLI must surface it as a fail-closed
    # exit code and still hand the runner the preregistered default seed.
    assert module.main([
        "--output", str(tmp_path / "synthetic-run"),
        "--max-steps", "1",
    ]) == 2
    assert captured["seed"] == 301097
    assert "stop after seed capture" in capsys.readouterr().err


def test_synthetic_v2_domain_uses_shared_period_and_disjoint_holdout(
    tmp_path,
):
    root = tmp_path / "synthetic-v2"
    captured = None

    def capture(context):
        nonlocal captured
        captured = context
        return _accepting_evaluator(context)

    run_bounded_cycle(root, seed=301097, evaluator=capture)
    assert captured is not None
    streams = _synthetic_streams()
    for index, source in enumerate(("A", "B", "C")):
        expected = [
            2 + index * 11 + position % 23
            for position in range(48)
        ]
        assert streams[source] == expected
        assert captured.holdout.row_ids_sha256[index] == _token_bytes_digest(
            expected[24:48]
        )
        plan = captured.request.child_plans[index]
        assert plan.span.source_manifest_sha256 == _token_bytes_digest(
            expected[:24]
        )
        assert plan.span.end - plan.span.start == 24


def test_bounded_cycle_commits_two_sequential_generations_with_persisted_lineage(
    tmp_path,
):
    root = tmp_path / "two-generations"
    first = run_bounded_cycle(root, seed=301097, evaluator=_accepting_evaluator)
    store = GrowthRunStore(root / "store")
    first_parent = store.read_parent()
    assert first_parent is not None
    first_report = Path(first.report_path).read_bytes()
    first_evidence = json.loads(Path(first.holdout_evidence_path).read_bytes())
    first_config = torch.load(
        first.candidate_checkpoint_path, map_location="cpu", weights_only=True
    )["config"]

    second = run_bounded_cycle(root, seed=301097, evaluator=_accepting_evaluator)
    second_parent = store.read_parent()
    assert second_parent is not None
    second_report = Path(second.report_path).read_bytes()
    second_evidence = json.loads(Path(second.holdout_evidence_path).read_bytes())
    second_config = torch.load(
        second.candidate_checkpoint_path, map_location="cpu", weights_only=True
    )["config"]

    assert first.decision == second.decision == "accepted"
    assert first_parent.generation_id == "generation-1"
    assert second_parent.generation_id == "generation-2"
    assert first_report == Path(first.report_path).read_bytes()
    assert second.report_path != first.report_path
    assert first_parent.checkpoint_sha256 == sha256_file(first.candidate_checkpoint_path)
    assert second_parent.checkpoint_sha256 == sha256_file(second.candidate_checkpoint_path)
    assert hashlib.sha256(first_report).hexdigest() == first_parent.report_sha256
    assert hashlib.sha256(second_report).hexdigest() == second_parent.report_sha256

    binding = second_evidence["bindings"]
    assert binding["parent"] == first_parent.as_dict()
    assert binding["parent"]["generation_id"] == first_parent.generation_id
    assert binding["parent"]["checkpoint_sha256"] == first_parent.checkpoint_sha256
    assert binding["parent"]["manifest_sha256"] == first_parent.manifest_sha256
    assert binding["holdout_sha256"] == first_evidence["bindings"]["holdout_sha256"]
    assert first_config["merge"]["ternary_depth"] == 1
    assert second_config["merge"]["ternary_depth"] == 2
    assert first_config["model"]["hidden_size"] == 24
    assert second_config["model"]["hidden_size"] == 72
    assert not (root / "store" / "generation-3").exists()

    for child_path in second_evidence["candidate_f3"]["child_checkpoint_paths"]:
        child_payload = torch.load(child_path, map_location="cpu", weights_only=True)
        assert child_payload["config"]["merge"]["ternary_depth"] == 1
        assert child_payload["config"]["model"]["adapters"]["enabled"] is False
        assert not any(
            key.endswith(".adapters.pyramid.scale")
            for key in child_payload["model"]
        )


def test_explicit_gen2_replay_is_idempotent_and_gen1_is_stale(tmp_path):
    root = tmp_path / "replay"
    requests = []

    def capture_accepting(context):
        requests.append(context.request)
        return _accepting_evaluator(context)

    first = run_bounded_cycle(root, seed=301097, evaluator=capture_accepting)
    second = run_bounded_cycle(root, seed=301097, evaluator=capture_accepting)
    assert first.generation_id == "generation-1"
    assert second.generation_id == "generation-2"
    assert len(requests) == 2
    store = GrowthRunStore(root / "store")
    parent_before = store.read_parent()
    reports_before = {
        "generation-1": Path(first.report_path).read_bytes(),
        "generation-2": Path(second.report_path).read_bytes(),
    }

    replay = run_bounded_cycle(
        root,
        seed=301097,
        request=requests[1],
        evaluator=lambda context: pytest.fail("terminal replay evaluated"),
    )
    assert replay == second
    assert store.read_parent() == parent_before
    for generation_id, report_bytes in reports_before.items():
        assert (root / "store" / generation_id / "report.json").read_bytes() == report_bytes
    assert not (root / "store" / "generation-3").exists()

    with pytest.raises(ValueError, match="stale explicit request"):
        run_bounded_cycle(
            root,
            seed=301097,
            request=requests[0],
            evaluator=lambda context: pytest.fail("stale request evaluated"),
        )
    assert store.read_parent() == parent_before
    for generation_id, report_bytes in reports_before.items():
        assert (root / "store" / generation_id / "report.json").read_bytes() == report_bytes


def test_committed_parent_tamper_fails_before_generation_two_build(tmp_path):
    root = tmp_path / "tampered-parent"
    run_bounded_cycle(root, seed=301097, evaluator=_accepting_evaluator)
    report = root / "store" / "generation-1" / "report.json"
    payload = json.loads(report.read_bytes())
    payload["candidate_checkpoint_path"] = "substituted.bin"
    report.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="terminal report digest mismatch"):
        run_bounded_cycle(root, seed=301097, evaluator=_accepting_evaluator)
    assert not (root / "build" / "generation-2").exists()


def test_result_provenance_comes_from_persisted_evidence_not_hardcoded(
    tmp_path,
):
    root = tmp_path / "provenance"
    result = run_bounded_cycle(root, seed=301097, evaluator=_accepting_evaluator)
    evidence_path = Path(result.holdout_evidence_path)
    report_path = Path(result.report_path)
    evidence = json.loads(evidence_path.read_bytes())
    evidence["bindings"]["tokenizer_name"] = "manifest-derived-test-tokenizer"
    evidence_bytes = canonical_json_bytes(evidence)
    copied_evidence = tmp_path / "copied-evidence.json"
    copied_evidence.write_bytes(evidence_bytes)
    report = json.loads(report_path.read_bytes())
    report["holdout_evidence_path"] = str(copied_evidence)
    report["holdout_evidence_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    copied_report = tmp_path / "copied-report.json"
    copied_report.write_bytes(canonical_json_bytes(report))
    copied_result = dataclasses.replace(
        result,
        report_path=str(copied_report),
        holdout_evidence_path=str(copied_evidence),
    )
    data_sha, tokenizer = result_data_provenance(copied_result)
    assert data_sha == evidence["bindings"]["data_manifest_sha256"]
    assert tokenizer == "manifest-derived-test-tokenizer"


def test_cli_banking77_provenance_comes_from_result_evidence(
    tmp_path, monkeypatch, capsys
):
    spec = importlib.util.spec_from_file_location("recursive_growth_banking_cli", _CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake_result = GenerationResult(
        decision="accepted",
        generation_id="generation-1",
        report_path="report.json",
        manifest_path="manifest.json",
        candidate_checkpoint_path="candidate.pt",
        holdout_evidence_path="evidence.json",
        incumbent_macro_ce=1.0,
        candidate_macro_ce=0.5,
        ce_regression=-0.5,
        worst_source_regression=0.0,
        mechanism_supported=True,
        quality_supported=False,
        security_supported=False,
        production_promotion=False,
        pareto_improvement=True,
    )
    expected_sha = BANKING77_MANIFEST_SHA256
    monkeypatch.setattr(
        module, "run_banking77_cycle", lambda *args, **kwargs: fake_result
    )
    monkeypatch.setattr(
        module,
        "result_data_provenance",
        lambda result: (
            (expected_sha, "validated-manifest-tokenizer")
            if result is fake_result
            else pytest.fail("wrong result")
        ),
    )
    assert module.main([
        "--output", str(tmp_path / "banking-run"),
        "--artifact", str(tmp_path / "artifact"),
        "--manifest-sha256", expected_sha,
        "--seed", "17",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data_kind"] == "banking77"
    assert payload["tokenizer_name"] == "validated-manifest-tokenizer"
    assert payload["data_manifest_sha256"] == expected_sha


def test_banking77_preflight_is_exact_and_cli_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="frozen preregistration"):
        preflight_banking77(tmp_path, "0" * 64)
    artifact = tmp_path / "banking77"
    artifact.mkdir()
    with pytest.raises(ValueError, match="manifest.json is missing"):
        preflight_banking77(artifact, BANKING77_MANIFEST_SHA256)
    (artifact / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        preflight_banking77(artifact, BANKING77_MANIFEST_SHA256)
    completed = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--output",
            str(tmp_path / "banking-run"),
            "--artifact",
            str(artifact),
            "--manifest-sha256",
            BANKING77_MANIFEST_SHA256,
            "--seed",
            "17",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "digest does not match" in completed.stderr


def test_banking77_preflight_returns_the_fully_validated_manifest(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "banking77"
    artifact.mkdir()
    (artifact / "manifest.json").write_bytes(b"{}")
    expected = {"schema": "validated-by-loader"}
    calls: list[Path] = []

    monkeypatch.setattr(
        real_cycle, "sha256_file", lambda path: BANKING77_MANIFEST_SHA256
    )

    def load_validated(root):
        calls.append(Path(root))
        return expected

    monkeypatch.setattr(real_cycle, "load_published_artifact", load_validated)
    assert (
        preflight_banking77(artifact, BANKING77_MANIFEST_SHA256) is expected
    )
    assert calls == [artifact]


def test_banking77_adapter_uses_compact_parent_and_bounded_holdout(
    tmp_path, monkeypatch
):
    train_ids = [index % 11 for index in range(30)]
    test_ids = [index % 17 for index in range(900)]
    compact_vocab = 11
    manifest = {
        "vocab_size": 262_144,
        "tokenizer_name": "offline-test-tokenizer",
    }
    monkeypatch.setattr(
        real_cycle,
        "_banking_data",
        lambda artifact: (
            train_ids,
            test_ids,
            manifest,
            BANKING77_MANIFEST_SHA256,
            compact_vocab,
        ),
    )
    captured = {}

    def capture_run(request, store, *args, **kwargs):
        captured["request"] = request
        return "offline-sentinel"

    monkeypatch.setattr(real_cycle, "run_generation", capture_run)
    assert (
        run_banking77_cycle(
            tmp_path / "cycle",
            tmp_path / "unused-artifact",
            seed=17,
            max_steps=1,
        )
        == "offline-sentinel"
    )
    request = captured["request"]
    parent_payload = torch.load(
        request.parent_checkpoint_path, map_location="cpu", weights_only=True
    )
    assert parent_payload["config"]["model"]["vocab_size"] == compact_vocab
    holdout = json.loads(Path(request.holdout.path).read_text(encoding="utf-8"))
    assert {source: len(values) for source, values in holdout["sources"].items()} == {
        "A": 256,
        "B": 256,
        "C": 256,
    }
    parts = np.array_split(np.asarray(train_ids, dtype=np.int64), 3)
    for plan, part in zip(request.child_plans, parts, strict=True):
        assert plan.span.start <= plan.span.end
        assert plan.span.source_manifest_sha256 == _token_bytes_digest(part)
