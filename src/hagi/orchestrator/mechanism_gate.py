"""Per-generation mechanism gate for the bounded recursive-growth owner.

This harness answers a different question from the owner's quality verdict.

The owner decides ``accepted``/``rejected`` from exact-CE deltas
(``recursive._verdict``: ``regression <= 0.0 and worst <= 0.01``), so a
transactionally perfect candidate that does not improve CE is rejected and
the two-generation traversal can never be observed. That is the same failure
class that stopped the seed-4242 execution: macro CE improved by 0.0141862
nats while source B regressed 0.0307527, above the 0.01 budget.

So traversal and quality are separated:

Tier 1 (mechanism, per generation) is evaluated here, over owner-produced
artifacts only, and is independent of the owner's ``decision``. A rejected
generation can pass it.

Tier 2 (accepted gen1 -> gen2 traversal) is conditional and false by default.
A Tier 2 failure never invalidates Tier 1.

Nothing here mutates the owner, its CAS, ``_verdict``, schema-v2, or the claim
flags; the harness only reads durable artifacts the owner wrote.

Pattern reference (spec-as-data plus derived budgets, adapted, not imported):
https://github.com/fsdatalab/quail — quail/specs and quail/cost.
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hagi.orchestrator.real_cycle as real_cycle
from hagi.orchestrator.recursive import (
    EvaluationContext,
    EvaluationResult,
    GenerationRequest,
    GenerationResult,
)
from hagi.orchestrator.state import (
    GrowthRunStore,
    canonical_json_bytes,
    sha256_bytes,
)
from hagi.train.checkpoint import config_from_dict, config_to_dict, load_payload

MECHANISM_CLAIM = "recursive_f3_mechanism_evidence_v1"
CROSS_PARENT_TRANSFORM = "f3_tree"


class MechanismGateError(RuntimeError):
    """The gate cannot be evaluated on a sound snapshot."""


def _read_bytes(path: str | Path, expected_sha256: str, label: str) -> bytes:
    """Read a regular non-symlink file and require its exact digest.

    The digest must come from an independent owner-side record. A mismatch
    reports only the label: the full digests of holdout artifacts are
    membership oracles, so they must never reach a builder-visible message.
    """
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise MechanismGateError(f"{label} is not a regular file: {target}")
    data = target.read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise MechanismGateError(f"{label} does not match its owner-recorded digest")
    return data


def _read_contained(root: Path, path: str | Path, expected_sha256: str, label: str) -> bytes:
    """Read a digest-verified file and require it to live inside ``root``."""
    target = Path(path)
    resolved = target.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise MechanismGateError(f"{label} points outside the run store: {target}")
    return _read_bytes(resolved, expected_sha256, label)


def _read_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MechanismGateError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise MechanismGateError(f"{label} is not a JSON object")
    return payload


def _config_digest(config: Any) -> str:
    """Digest a checkpoint config dict exactly as the owner digests its own."""
    return sha256_bytes(
        canonical_json_bytes(config_to_dict(config_from_dict(config)))
    )


def _checkpoint_config(data: bytes, label: str) -> dict[str, Any]:
    """Return the config dict of a digest-verified checkpoint.

    The digest is checked against the owner-recorded value before the bytes
    are deserialized, so the gate never reads unverified content. The bytes go
    to a private temporary file because the owner loader takes a path.
    """
    with tempfile.TemporaryDirectory(prefix="hagi-bp-ckpt-") as tmp:
        staged = Path(tmp) / f"{label.replace(' ', '-')}.pt"
        staged.write_bytes(data)
        try:
            payload = load_payload(staged)
        except Exception as exc:  # noqa: BLE001 - surfaced as a gate failure
            raise MechanismGateError(f"{label} is not a loadable checkpoint") from exc
    if not isinstance(payload, dict) or "config" not in payload or "model" not in payload:
        raise MechanismGateError(f"{label} checkpoint lacks config or model")
    return payload["config"]


@dataclass(frozen=True)
class GenerationObservation:
    """Owner artifacts read for one executed generation."""

    generation_id: str
    owner_decision: str
    request: GenerationRequest
    report_bytes: bytes
    report: dict[str, Any]
    report_sha256: str
    evidence_bytes: bytes
    evidence: dict[str, Any]
    child_configs: tuple[dict[str, Any], ...]
    candidate_config: dict[str, Any]
    candidate_manifest: dict[str, Any]
    candidate_ternary_depth: int
    parent_ternary_depth: int
    report_path: Path


@dataclass(frozen=True)
class MechanismFindings:
    """Tier 1 result for one generation. ``owner_decision`` is not an input."""

    generation_id: str
    owner_decision: str
    checks: Mapping[str, bool]

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(name for name, ok in sorted(self.checks.items()) if not ok)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "owner_decision": self.owner_decision,
            "tier1_passed": self.passed,
            "checks": dict(sorted(self.checks.items())),
            "failures": list(self.failures),
        }


class _RecordingRealEvaluator:
    """Record each real request, then delegate to the real evaluator.

    This is not a fabricated evaluator: every metric comes from
    ``real_cycle.path_only_evaluator`` by object identity. The wrapper exists
    only so the gate can replay a real request and prove a stale one raises,
    without inventing a request. No caller can inject an evaluator.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.delegate = real_cycle.path_only_evaluator
        self.requests: list[GenerationRequest] = []

    def __call__(self, context: EvaluationContext) -> EvaluationResult:
        self.requests.append(context.request)
        return self.delegate(context, device=self.device)


def run_generations(
    output: str | Path,
    *,
    seed: int,
    max_steps: int = 1,
    device: str = "cpu",
    count: int = 2,
) -> tuple[list[GenerationObservation], list[GenerationResult], _RecordingRealEvaluator]:
    """Execute bounded generations, then read the artifacts actually evaluated.

    The evaluator is never caller-supplied: the real default runs through a
    recording delegate, so no gate path can inject optimistic metrics.

    A call whose generation was already terminal is a replay: the owner
    returns the stored result without invoking the evaluator. Replays are
    counted, not mistaken for new generations.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    root = Path(output).resolve()
    evaluator = _RecordingRealEvaluator(device=device)
    results = [
        real_cycle.run_bounded_cycle(
            root,
            seed=seed,
            max_steps=max_steps,
            device=device,
            evaluator=evaluator,
            cross_parent_transform=CROSS_PARENT_TRANSFORM,
        )
        for _ in range(count)
    ]
    if evaluator.delegate is not real_cycle.path_only_evaluator:
        raise MechanismGateError("gate evaluator is not the real bounded evaluator")
    if not evaluator.requests:
        raise MechanismGateError("gate observed no evaluated generation")
    observations = [
        _observe(root, result, request)
        for result, request in zip(results[: len(evaluator.requests)], evaluator.requests, strict=True)
    ]
    replayed = results[len(evaluator.requests):]
    if replayed and any(item.generation_id == observations[-1].generation_id for item in replayed):
        # A replayed call returned the same generation: the owner did not
        # advance. Recorded so Tier 2 can state why gen2 is absent.
        pass
    return observations, results, evaluator


def _digest_of(path: Path) -> str:
    """Digest a file the gate is about to read.

    NOT a trust anchor. This helper only computes "what the digest of these
    bytes is", for tests and for callers that already hold an independent
    expected digest. Passing its result back to ``_read_bytes`` verifies
    nothing (the file is compared against itself) and must never be used as
    the expected digest of a security decision. The owner state read used to
    do exactly that; it now goes through ``GrowthRunStore._read_state``.
    """
    return sha256_bytes(path.read_bytes())


def _owner_terminal_digest(run: Path, generation_id: str) -> str:
    """Return the terminal report digest recorded in the owner-side ledger.

    ``state.json`` is the owner's own ledger record, so it is read through the
    owner's strict reader (``GrowthRunStore._read_state``) rather than by a
    lenient local parse: duplicate keys and non-canonical bytes are rejected
    instead of silently coerced.

    TRUST BOUNDARY (stated, not overstated): this anchor is *not* independent
    of the run store. A context that can write both ``state.json`` and
    ``report.json`` can still re-point the digest at a forged report. An
    earlier version of this function hid that by hashing ``state.json`` with
    ``_digest_of`` and comparing the result against ``state.json`` itself --
    a trust-on-first-use check that can never fail, dressed up as
    verification. The vacuous comparison is gone rather than kept as theater.
    Defending against a same-store writer needs an out-of-band anchor (a
    signed receipt or an owner-held digest), which is a design change and is
    deliberately not faked here.
    """
    store_root = run.parent
    try:
        state = GrowthRunStore(store_root)._read_state(generation_id)
    except ValueError as exc:
        raise MechanismGateError(f"owner state is not owner-readable: {exc}") from exc
    if state.get("generation_id") != generation_id:
        raise MechanismGateError("owner state names a different generation")
    digest = state.get("terminal_report_sha256")
    if not isinstance(digest, str) or not digest:
        raise MechanismGateError("owner state has no terminal report digest")
    return digest


def _observe(
    root: Path, result: GenerationResult, request: GenerationRequest
) -> GenerationObservation:
    """Read one generation's durable owner artifacts strictly by digest."""
    run = root / "store" / result.generation_id
    report_path = run / "report.json"
    report_bytes = _read_contained(
        root, report_path, _owner_terminal_digest(run, result.generation_id), "terminal report"
    )
    report = _read_json(report_bytes, "terminal report")
    for field in ("generation_id", "decision", "holdout_evidence_path",
                  "holdout_evidence_sha256", "candidate_checkpoint_path",
                  "manifest_path"):
        if field not in report:
            raise MechanismGateError(f"terminal report lacks {field}")
    if report["generation_id"] != result.generation_id:
        raise MechanismGateError("terminal report generation mismatch")
    if report["decision"] != result.decision:
        raise MechanismGateError("terminal report decision mismatch")
    evidence_bytes = _read_contained(
        root,
        report["holdout_evidence_path"],
        report["holdout_evidence_sha256"],
        "holdout evidence",
    )
    evidence = _read_json(evidence_bytes, "holdout evidence")
    f3 = evidence.get("candidate_f3")
    if not isinstance(f3, dict):
        raise MechanismGateError("evidence lacks candidate_f3")
    child_paths = list(f3["child_checkpoint_paths"])
    child_digests = list(f3["child_checkpoint_sha256"])
    if len(child_paths) != 3 or len(child_digests) != 3:
        raise MechanismGateError("evidence does not record three children")
    children = tuple(
        _checkpoint_config(
            _read_contained(root, path, digest, "child checkpoint"), f"child {index}"
        )
        for index, (path, digest) in enumerate(
            zip(child_paths, child_digests, strict=True)
        )
    )
    if report["candidate_checkpoint_path"] == f3["checkpoint_path"]:
        raise MechanismGateError(
            "terminal report and evidence must not share one checkpoint path"
        )
    if not isinstance(report.get("source_candidate_checkpoint_path"), str):
        raise MechanismGateError("terminal report lacks the source checkpoint path")
    if report["candidate_checkpoint_sha256"] != f3["checkpoint_sha256"]:
        raise MechanismGateError("terminal report and evidence disagree on checkpoint digest")
    if report["manifest_sha256"] != f3["manifest_sha256"]:
        raise MechanismGateError("terminal report and evidence disagree on manifest digest")
    snapshot_bytes = _read_contained(
        root,
        report["candidate_checkpoint_path"],
        report["candidate_checkpoint_sha256"],
        "terminal checkpoint snapshot",
    )
    source_bytes = _read_contained(
        root, f3["checkpoint_path"], f3["checkpoint_sha256"], "candidate checkpoint source"
    )
    if snapshot_bytes != source_bytes:
        raise MechanismGateError("terminal snapshot differs from the evidence source bytes")
    if _read_contained(
        root, report["manifest_path"], report["manifest_sha256"], "terminal manifest snapshot"
    ) != _read_contained(
        root, f3["manifest_path"], f3["manifest_sha256"], "candidate manifest source"
    ):
        raise MechanismGateError("terminal manifest snapshot differs from the evidence source")
    candidate_config = _checkpoint_config(source_bytes, "candidate")
    candidate_manifest = _read_json(
        _read_contained(
            root, f3["manifest_path"], f3["manifest_sha256"], "candidate manifest"
        ),
        "candidate manifest",
    )
    parent_bytes = _read_contained(
        root,
        request.parent_checkpoint_path,
        request.parent.checkpoint_sha256,
        "parent checkpoint",
    )
    parent_config = _checkpoint_config(parent_bytes, "parent")
    return GenerationObservation(
        generation_id=result.generation_id,
        owner_decision=result.decision,
        request=request,
        report_bytes=report_bytes,
        report=report,
        report_sha256=sha256_bytes(report_bytes),
        evidence_bytes=evidence_bytes,
        evidence=evidence,
        child_configs=children,
        candidate_config=candidate_config,
        candidate_manifest=candidate_manifest,
        candidate_ternary_depth=int(candidate_config["merge"]["ternary_depth"]),
        parent_ternary_depth=int(parent_config["merge"]["ternary_depth"]),
        report_path=report_path,
    )


def evaluate_generation(observation: GenerationObservation) -> MechanismFindings:
    """Tier 1 predicates over owner-written bytes, verdict-independent."""
    checks: dict[str, bool] = {}

    def check(name: str, ok: bool) -> None:
        checks[name] = bool(ok)

    evidence = observation.evidence
    bindings = evidence.get("bindings")
    request = observation.request
    f3 = evidence["candidate_f3"]

    check("evidence_is_canonical_bytes",
          canonical_json_bytes(evidence) == observation.evidence_bytes)
    check("report_is_canonical_bytes",
          canonical_json_bytes(observation.report) == observation.report_bytes)
    check("evidence_decision_matches_terminal",
          evidence.get("decision") == observation.owner_decision)
    check("evidence_generation_matches_request",
          evidence.get("generation_id") == request.generation_id)
    check("bindings_parent_equals_request_parent",
          isinstance(bindings, dict) and bindings.get("parent") == request.parent.as_dict())
    check("bindings_base_seed_equals_request",
          isinstance(bindings, dict) and bindings.get("base_seed") == request.base_seed)
    check("bindings_protocol_equals_request",
          isinstance(bindings, dict)
          and bindings.get("protocol_sha256") == request.holdout.protocol_sha256)
    check("bindings_tokenizer_equals_request",
          isinstance(bindings, dict)
          and bindings.get("tokenizer_name") == request.holdout.tokenizer_name)
    check("bindings_holdout_digest_equals_request",
          isinstance(bindings, dict)
          and bindings.get("holdout_sha256") == request.holdout.sha256)
    check("bindings_data_manifest_equals_request",
          isinstance(bindings, dict)
          and bindings.get("data_manifest_sha256")
          == request.holdout.data_manifest_sha256)
    check("bindings_row_ids_equal_request",
          isinstance(bindings, dict)
          and bindings.get("row_ids_sha256") == list(request.holdout.row_ids_sha256))
    check("bindings_child_plans_equal_request",
          isinstance(bindings, dict)
          and bindings.get("child_plans") == [
              {
                  "name": name,
                  "source_id": plan.source_id,
                  "seed": seed,
                  "span": {
                      "source_id": plan.span.source_id,
                      "start": plan.span.start,
                      "end": plan.span.end,
                      "source_manifest_sha256": plan.span.source_manifest_sha256,
                  },
              }
              for name, plan, seed in zip(
                  ("child_A", "child_B", "child_C"),
                  request.child_plans,
                  request.child_seeds,
                  strict=True,
              )
          ])
    check("child_digests_distinct",
          len(set(f3["child_checkpoint_sha256"])) == 3)
    check("child_paths_distinct", len({str(Path(p)) for p in f3["child_checkpoint_paths"]}) == 3)
    check("child_checkpoint_paths_inside_run_inputs",
          all(
              Path(p).name.startswith("child_")
              and Path(p).is_file()
              and not Path(p).is_symlink()
              for p in f3["child_checkpoint_paths"]
          ))
    check("child_configs_identical",
          len({canonical_json_bytes(c) for c in observation.child_configs}) == 1)
    check("child_configs_match_recorded_digest",
          list(f3["child_config_sha256"]) == [_config_digest(c) for c in observation.child_configs])
    check("child_configs_merge_disabled_and_unlifted",
          all(
              c["merge"]["enabled"] is False
              and c["merge"]["ternary_depth"] == 0
              and c["merge"]["n_experts"] == 4
              and c["merge"]["expert_weight_source"] == "ternary_master"
              and c["merge"]["ternary_lift_mode"] == CROSS_PARENT_TRANSFORM
              for c in observation.child_configs
          ))
    check("candidate_lifts_merge_above_children",
          observation.candidate_config["merge"]["enabled"] is True
          and observation.candidate_config["merge"]["ternary_depth"]
          == observation.child_configs[0]["merge"]["ternary_depth"] + 1
          and observation.candidate_config["merge"]["ternary_lift_mode"]
          == CROSS_PARENT_TRANSFORM)
    check("candidate_grows_the_model",
          observation.candidate_config["model"]["hidden_size"]
          > observation.child_configs[0]["model"]["hidden_size"])
    check("candidate_digest_differs_from_children",
          f3["checkpoint_sha256"] not in f3["child_checkpoint_sha256"])
    check("candidate_config_digest_recorded",
          _config_digest(observation.candidate_config) == f3.get("candidate_config_sha256"))
    check("candidate_manifest_declares_checkpoint",
          observation.candidate_manifest.get("candidate_checkpoint_sha256")
          == f3["checkpoint_sha256"])
    check("candidate_manifest_declares_generation",
          observation.candidate_manifest.get("generation_id") == request.generation_id)
    check("candidate_manifest_declares_parent",
          observation.candidate_manifest.get("parent_checkpoint_sha256")
          == request.parent.checkpoint_sha256)
    check("candidate_depth_matches_manifest",
          observation.candidate_manifest.get("ternary_depth")
          == observation.candidate_ternary_depth)
    check("candidate_depth_equals_parent_plus_one",
          observation.candidate_ternary_depth == observation.parent_ternary_depth + 1)
    check("no_unearned_quality_claim",
          evidence.get("quality_supported") is False
          and evidence.get("security_supported") is False
          and evidence.get("production_promotion") is False)
    check("terminal_report_state_terminal",
          observation.report.get("state") == "terminal")
    check("terminal_report_binds_parent_generation",
          observation.report.get("parent_generation_id") == request.parent.generation_id)
    check("terminal_report_binds_parent_digests",
          observation.report.get("parent_checkpoint_sha256") == request.parent.checkpoint_sha256
          and observation.report.get("parent_manifest_sha256")
          == request.parent.manifest_sha256)
    return MechanismFindings(
        generation_id=observation.generation_id,
        owner_decision=observation.owner_decision,
        checks=checks,
    )


def evaluate_traversal(
    observations: Sequence[GenerationObservation], replay: bool = True
) -> dict[str, Any]:
    """Tier 2: conditional accepted gen1 -> gen2 traversal, false by default."""
    report: dict[str, Any] = {
        "tier": 2,
        "name": "accepted_gen1_gen2_traversal",
        "conditional": True,
        "passed": False,
    }
    if len(observations) < 2:
        report["reason"] = "fewer than two generations were executed"
        report["checks"] = {"two_generations_observed": False}
        return report
    first, second = observations[0], observations[1]
    checks = {
        "two_generations_observed": True,
        "gen1_accepted": first.owner_decision == "accepted",
        "gen2_accepted": second.owner_decision == "accepted",
        "gen2_parent_is_gen1": second.request.parent.generation_id == first.generation_id,
        "gen2_parent_checkpoint_equals_gen1_candidate":
            second.request.parent.checkpoint_sha256
            == first.evidence["candidate_f3"]["checkpoint_sha256"],
        "gen2_parent_manifest_equals_gen1_candidate_manifest":
            second.request.parent.manifest_sha256
            == first.evidence["candidate_f3"]["manifest_sha256"],
        "depth_increased": second.candidate_ternary_depth > first.candidate_ternary_depth,
        "gen1_report_bytes_unchanged_after_gen2":
            first.report_path.is_file()
            and first.report_path.read_bytes() == first.report_bytes,
    }
    report["checks"] = {name: bool(value) for name, value in sorted(checks.items())}
    report["passed"] = all(report["checks"].values())
    report["reason"] = (
        "accepted traversal observed"
        if report["passed"]
        else "accepted traversal not observed; Tier 1 is unaffected"
    )
    return report
