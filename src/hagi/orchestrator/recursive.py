"""Deterministic transaction owner for recursive ternary F3 growth.

Callback injection is an intentional trusted-local-code seam. The owner derives
acceptance from durable artifacts and recomputes the verdict, but it cannot
authenticate metrics returned by an evaluator or provide security against a
same-user adversary that controls callback code or the working filesystem.
The link/junction checks are trusted-directory defense only.

Production blocker: process-kill recovery for RESERVED/RUNNING is not solved;
those runs are deliberately rejected as non-resumable rather than guessed.
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hagi.model.merge import (
    KNOWN_COORDINATE_LAYOUTS as _KNOWN_COORDINATE_LAYOUTS,
)
from hagi.model.merge import (
    CrossParentPreservingTernaryTree as _CrossParentPreservingTernaryTree,
)
from hagi.model.merge import (
    RecursiveF3HAGI,
    TernaryF3Tree,
    merge_recursive_f3,
)
from hagi.orchestrator.state import (
    GrowthRunStore,
    ParentPointer,
    PreparedDecision,
    RunState,
    _prepared_contract,
    _regular_bytes,
    _reject_link_chain,
    _strict_json_bytes,
    _terminal_payload,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from hagi.train.checkpoint import config_from_dict, config_to_dict, load_payload

# Receiver gain per pinned cross-parent transform. The staged F3 tree
# aggregates the three children, so its declared receiver is sum(s_i)/sqrt(3);
# the parent-preserving lift fixes the repeated branch, giving sum(s_i)/3.
_TREE_RECEIVER_GAIN = 3.0
_F3_TREE_RECEIVER_GAIN = math.sqrt(3.0)

_F3_TREE_LAYOUT = TernaryF3Tree(1, 2).coordinate_layout
_PARENT_PRESERVING_LAYOUT = _CrossParentPreservingTernaryTree(1, 2).coordinate_layout


def _declared_transform_tree(candidate: Any) -> Any:
    """Rebuild the transform a candidate declares, and only that one.

    A candidate is always validated against the transform it declares, so a
    checkpoint written under one transform can never be replayed under the
    other, and an unknown layout stays rejected.
    """
    if candidate.coordinate_layout == _F3_TREE_LAYOUT:
        return TernaryF3Tree(candidate.ternary_depth, candidate.leaf_hidden)
    if candidate.coordinate_layout == _PARENT_PRESERVING_LAYOUT:
        return _CrossParentPreservingTernaryTree(
            candidate.ternary_depth, candidate.leaf_hidden
        )
    raise ValueError("invalid coordinate layout")


def _transform_receiver_gain(candidate: Any) -> float:
    if candidate.coordinate_layout == _PARENT_PRESERVING_LAYOUT:
        return _TREE_RECEIVER_GAIN
    return _F3_TREE_RECEIVER_GAIN

CHILD_NAMES = ("child_A", "child_B", "child_C")
SOURCE_NAMES = ("A", "B", "C")
_SEED_STRIDE = 1009
_SELF_IMPROVE_OFFSET = 9001
_MAX_BASE_SEED = 2**31 - 1 - _SELF_IMPROVE_OFFSET
_MAX_SELF_IMPROVE_UPDATES = 1
_SELF_IMPROVE_MODE = "gradient"
_SELF_IMPROVE_MAX_ITERATIONS = 1
_SELF_IMPROVE_N_NEW_TOKENS = 8
_SELF_IMPROVE_PATIENCE = 1
_SELF_IMPROVE_CE_MIN_IMPROVE = 0.0
_SELF_IMPROVE_KL_MAX = 1.0
_HEX64 = re.compile(r"[0-9a-f]{64}")
_KNOWN_WEIGHT_SOURCES = {"ternary_master", "effective_sparse"}
# Every coordinate layout a candidate may declare. The set lives in
# ``hagi.model.merge`` next to the transforms themselves (merge never imports
# the orchestrator, so the import direction stays acyclic) and is re-exported
# here under a public name for callers that already depend on the
# orchestrator. An unknown or misspelled layout is still rejected, fail-closed.
KNOWN_COORDINATE_LAYOUTS = _KNOWN_COORDINATE_LAYOUTS
_PYRAMID_CONTOUR_KEY = re.compile(r"^blocks\.[0-9]+\.adapters\.pyramid\.scale$")
_CHILD_MANIFEST_FIELDS = {
    "schema_version", "generation_id", "name", "source_id", "seed", "span",
    "checkpoint_sha256", "config_sha256", "parent_checkpoint_sha256",
    "protocol_sha256", "data_manifest_sha256",
}
_CANDIDATE_MANIFEST_FIELDS = {
    "schema_version", "generation_id", "parent_checkpoint_sha256",
    "parent_config_sha256", "protocol_sha256", "data_manifest_sha256",
    "candidate_checkpoint_sha256", "candidate_config_sha256",
    "candidate_state_key_digest", "ternary_depth", "leaf_hidden",
    "coordinate_layout", "transform_digest", "weight_source",
    "logit_scale_source", "logit_scale_target", "decision", "self_improve",
}
_CHILD_DRAFT_MANIFEST_FIELDS = _CHILD_MANIFEST_FIELDS - {
    "data_manifest_sha256"
}
_CANDIDATE_DRAFT_MANIFEST_FIELDS = _CANDIDATE_MANIFEST_FIELDS - {
    "data_manifest_sha256"
}
_EVIDENCE_FIELDS = {
    "schema", "schema_version", "generation_id", "decision",
    "mechanism_supported", "quality_supported", "security_supported",
    "production_promotion", "pareto_improvement", "incumbent_macro_ce",
    "candidate_macro_ce", "ce_regression", "worst_source_regression",
    "source_metrics", "bindings", "candidate_f3",
}
_CANDIDATE_EVIDENCE_FIELDS = {
    "checkpoint_path", "checkpoint_sha256", "manifest_path", "manifest_sha256",
    "candidate_config_sha256", "candidate_state_key_digest", "ternary_depth",
    "leaf_hidden", "coordinate_layout", "transform_digest", "weight_source",
    "logit_scale_source", "logit_scale_target", "self_improve",
    "child_checkpoint_paths", "child_checkpoint_sha256", "child_config_sha256",
}
_SELF_IMPROVE_FIELDS = {
    "seed", "source_id", "prompt_span", "prompt_token_sha256",
    "prompt_text_sha256", "prompt_token_count", "optimizer_parameter_ids",
    "accepted_updates", "derived_base_state_sha256", "contour_state_sha256",
    "mode", "max_iterations", "n_new_tokens", "patience", "ce_min_improve", "kl_max",
}
_CANDIDATE_PROVENANCE_FIELDS = frozenset({
    "candidate_config_sha256", "candidate_state_key_digest", "ternary_depth",
    "leaf_hidden", "coordinate_layout", "transform_digest", "weight_source",
    "logit_scale_source", "logit_scale_target", "self_improve",
})


def _exact_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact int")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hex characters")
    return value


def _path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    return value


def _protocol_payload_digest(payload: bytes, expected: str) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("protocol payload must be non-empty canonical JSON bytes")
    value = _strict_json_bytes(payload)
    if not isinstance(value, dict):
        raise ValueError("protocol payload must be a canonical JSON object")

    def require_nfc(item: Any) -> None:
        if isinstance(item, str):
            if unicodedata.normalize("NFC", item) != item:
                raise ValueError("protocol payload strings must be NFC-normalized")
        elif isinstance(item, dict):
            for key, nested in item.items():
                require_nfc(key)
                require_nfc(nested)
        elif isinstance(item, list):
            for nested in item:
                require_nfc(nested)

    require_nfc(value)
    if sha256_bytes(payload) != expected:
        raise ValueError("protocol payload digest mismatch")


def _config_digest(config: Any) -> str:
    return sha256_bytes(canonical_json_bytes(config_to_dict(config)))


def _state_key_digest(state: Mapping[str, Any]) -> str:
    import torch

    entries = []
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"state value must be a tensor: {key}")
        entries.append({"key": key, "shape": list(value.shape), "dtype": str(value.dtype)})
    return sha256_bytes(canonical_json_bytes(entries))


def _canonical_manifest(path: Path, digest: str, required: set[str], label: str) -> dict[str, Any]:
    payload = _strict_json_bytes(_regular_bytes(path, digest))
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"invalid {label} manifest schema")
    return payload


def _write_exclusive(path: Path, data: bytes, digest: str, label: str) -> None:
    """Durably publish deterministic bytes in a trusted run-owned directory."""
    _reject_link_chain(path.parent, allow_missing=False)
    if path.exists():
        if _regular_bytes(path) != data:
            raise ValueError(f"existing immutable {label} conflicts")
    else:
        try:
            with path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            if _regular_bytes(path) != data:
                raise ValueError(f"existing immutable {label} conflicts") from exc
    if _regular_bytes(path) != data or sha256_file(path) != digest:
        raise ValueError(f"{label} snapshot digest mismatch")


def _snapshot(
    path: Path,
    digest: str,
    run: Path,
    prefix: str,
    label: str,
    directory: str = "inputs",
) -> Path:
    data = _regular_bytes(path, digest)
    return _snapshot_from_bytes(run, data, digest, prefix, label, directory)


def _holdout_snapshot(source: Path, digest: str, run: Path) -> Path:
    """Materialize the late holdout snapshot exactly once per run attempt.

    The snapshot name is content-addressed, so any build callback that knows
    ``request.holdout.sha256`` can pre-create the exact target path. Contract
    section 15.2-A treats such an attempt as an adversarial failure, and only
    the owner may create this file, so a pre-existing target fails closed.
    """
    target = run / "evaluator-inputs" / f"holdout-{digest}.bin"
    if target.exists() or target.is_symlink():
        raise ValueError("late holdout snapshot already exists")
    _reject_link_chain(run / "evaluator-inputs", allow_missing=True)
    return _snapshot(
        source,
        digest,
        run,
        "holdout",
        "holdout",
        directory="evaluator-inputs",
    )


def _finalize_manifest(
    draft: Path,
    draft_digest: str,
    run: Path,
    prefix: str,
    data_manifest_sha256: str,
    draft_fields: set[str],
    label: str,
) -> tuple[Path, str]:
    """Publish the full data binding as an owner-created durable manifest."""
    payload = _canonical_manifest(draft, draft_digest, draft_fields, f"{label} draft")
    payload["data_manifest_sha256"] = data_manifest_sha256
    data = canonical_json_bytes(payload)
    digest = sha256_bytes(data)
    return _snapshot_from_bytes(run, data, digest, prefix, label), digest


def _snapshot_from_bytes(
    run: Path,
    data: bytes,
    digest: str,
    prefix: str,
    label: str,
    directory: str = "inputs",
) -> Path:
    target_directory = run / directory
    target_directory.mkdir(exist_ok=True)
    _reject_link_chain(target_directory, allow_missing=False)
    target = target_directory / f"{prefix}-{digest}.bin"
    _write_exclusive(target, data, digest, label)
    return target


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


@dataclass(frozen=True)
class SourceSpan:
    source_id: str
    start: int
    end: int
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.source_id not in SOURCE_NAMES:
            raise ValueError("source_id must be A, B, or C")
        if _exact_int(self.start, "start") < 0:
            raise ValueError("start must be non-negative")
        if _exact_int(self.end, "end") <= self.start:
            raise ValueError("end must be greater than start")
        _digest(self.source_manifest_sha256, "source_manifest_sha256")


@dataclass(frozen=True)
class ChildPlan:
    source_id: str
    span: SourceSpan

    def __post_init__(self) -> None:
        if self.source_id not in SOURCE_NAMES:
            raise ValueError("source_id must be A, B, or C")
        if not isinstance(self.span, SourceSpan) or self.span.source_id != self.source_id:
            raise ValueError("child plan source/span mismatch")


@dataclass(frozen=True)
class HoldoutContract:
    path: str
    sha256: str
    data_manifest_sha256: str
    protocol_sha256: str
    protocol_payload: bytes
    row_ids_sha256: tuple[str, str, str]
    tokenizer_name: str

    def __post_init__(self) -> None:
        _path(self.path, "holdout path")
        _digest(self.sha256, "holdout sha256")
        _digest(self.data_manifest_sha256, "data_manifest_sha256")
        _digest(self.protocol_sha256, "protocol_sha256")
        _protocol_payload_digest(self.protocol_payload, self.protocol_sha256)
        if not isinstance(self.row_ids_sha256, tuple) or len(self.row_ids_sha256) != 3:
            raise ValueError("row_ids_sha256 must contain A/B/C digests")
        for index, digest in enumerate(self.row_ids_sha256):
            _digest(digest, f"row_ids_sha256[{index}]")
        if not isinstance(self.tokenizer_name, str) or not self.tokenizer_name:
            raise ValueError("tokenizer_name is required")
        _regular_bytes(Path(self.path), self.sha256)


@dataclass(frozen=True)
class GenerationRequest:
    generation_id: str
    base_seed: int
    child_plans: tuple[ChildPlan, ...]
    parent: ParentPointer
    parent_checkpoint_path: str
    holdout: HoldoutContract

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, str) or not self.generation_id:
            raise ValueError("generation_id is required")
        if _exact_int(self.base_seed, "base_seed") < 0 or self.base_seed > _MAX_BASE_SEED:
            raise ValueError("base_seed is outside the deterministic range")
        if not isinstance(self.child_plans, tuple) or len(self.child_plans) != 3:
            raise ValueError("exactly three child plans are required")
        if any(not isinstance(plan, ChildPlan) for plan in self.child_plans):
            raise ValueError("child_plans must contain ChildPlan values")
        if tuple(plan.source_id for plan in self.child_plans) != SOURCE_NAMES:
            raise ValueError("child plans must be in A/B/C order")
        spans = sorted((plan.span.start, plan.span.end) for plan in self.child_plans)
        if any(spans[index][1] > spans[index + 1][0] for index in range(2)):
            raise ValueError("child spans must be pairwise disjoint")
        if not isinstance(self.parent, ParentPointer):
            raise ValueError("parent must be ParentPointer")
        _path(self.parent_checkpoint_path, "parent_checkpoint_path")
        _regular_bytes(Path(self.parent_checkpoint_path), self.parent.checkpoint_sha256)
        if not isinstance(self.holdout, HoldoutContract):
            raise ValueError("holdout must be HoldoutContract")
        if len({plan.span.source_manifest_sha256 for plan in self.child_plans}) != 3:
            raise ValueError("each child source must bind a distinct source manifest")

    @property
    def child_seeds(self) -> tuple[int, int, int]:
        return tuple(self.base_seed + _SEED_STRIDE * index for index in range(3))


@dataclass(frozen=True)
class ChildArtifact:
    name: str
    source_id: str
    seed: int
    span: SourceSpan
    checkpoint_path: str
    checkpoint_sha256: str
    manifest_path: str
    manifest_sha256: str
    config_sha256: str
    parent_checkpoint_sha256: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        if self.name not in CHILD_NAMES or self.source_id not in SOURCE_NAMES:
            raise ValueError("invalid child name/source")
        _exact_int(self.seed, "seed")
        if not isinstance(self.span, SourceSpan) or self.span.source_id != self.source_id:
            raise ValueError("child span mismatch")
        for role in ("checkpoint", "manifest"):
            _path(getattr(self, f"{role}_path"), f"{role}_path")
            _digest(getattr(self, f"{role}_sha256"), f"{role}_sha256")
        for name in ("config_sha256", "parent_checkpoint_sha256", "protocol_sha256"):
            _digest(getattr(self, name), name)


@dataclass(frozen=True)
class CandidateArtifact:
    generation_id: str
    parent_checkpoint_sha256: str
    parent_config_sha256: str
    protocol_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    manifest_path: str
    manifest_sha256: str
    candidate_config_sha256: str
    candidate_state_key_digest: str
    ternary_depth: int
    leaf_hidden: int
    coordinate_layout: str
    transform_digest: str
    weight_source: str
    logit_scale_source: float
    logit_scale_target: float
    child_checkpoint_sha256: tuple[str, str, str]
    child_config_sha256: tuple[str, str, str]
    self_improve: SelfImprovementLedger
    decision: str = "candidate"

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, str) or not self.generation_id:
            raise ValueError("generation_id is required")
        for name in (
            "parent_checkpoint_sha256", "parent_config_sha256", "protocol_sha256",
            "checkpoint_sha256", "manifest_sha256", "candidate_config_sha256",
            "candidate_state_key_digest", "transform_digest",
        ):
            _digest(getattr(self, name), name)
        if any(not isinstance(item, tuple) or len(item) != 3 for item in (
            self.child_checkpoint_sha256, self.child_config_sha256
        )):
            raise ValueError("candidate child digests must contain exactly three values")
        for item in (self.child_checkpoint_sha256, self.child_config_sha256):
            for index, digest in enumerate(item):
                _digest(digest, f"child digest[{index}]")
        if len(set(self.child_config_sha256)) != 1:
            raise ValueError("all child config digests must be identical")
        for role in ("checkpoint", "manifest"):
            _path(getattr(self, f"{role}_path"), f"{role}_path")
        if _exact_int(self.ternary_depth, "ternary_depth") < 1:
            raise ValueError("ternary_depth must be at least one")
        if _exact_int(self.leaf_hidden, "leaf_hidden") <= 0 or self.leaf_hidden % 2:
            raise ValueError("leaf_hidden must be positive and even")
        if self.coordinate_layout not in _KNOWN_COORDINATE_LAYOUTS:
            raise ValueError(
                f"invalid coordinate layout: {self.coordinate_layout!r}"
            )
        if self.weight_source not in _KNOWN_WEIGHT_SOURCES:
            raise ValueError("unknown weight source")
        if _finite(self.logit_scale_source, "logit_scale_source") <= 0 or _finite(self.logit_scale_target, "logit_scale_target") <= 0:
            raise ValueError("logit scales must be positive")
        if self.decision != "candidate":
            raise ValueError("candidate decision must be candidate")
        if not isinstance(self.self_improve, SelfImprovementLedger):
            raise ValueError("self_improve must be a SelfImprovementLedger")


@dataclass(frozen=True)
class SelfImprovementLedger:
    """Post-merge self-improvement provenance (V1 §5, §13.4).

    Post-merge self-improvement owns only the fresh pyramid contour. This record
    binds the deterministic seed, the child-owned prompt source, and the exact
    optimizer parameter IDs so the owner can prove that no other tensor was
    allowed to move.
    """

    seed: int
    source_id: str
    prompt_span: SourceSpan
    prompt_token_sha256: str
    prompt_text_sha256: str
    prompt_token_count: int
    optimizer_parameter_ids: tuple[str, ...]
    accepted_updates: int
    derived_base_state_sha256: str
    contour_state_sha256: str
    mode: str = _SELF_IMPROVE_MODE
    max_iterations: int = _SELF_IMPROVE_MAX_ITERATIONS
    n_new_tokens: int = _SELF_IMPROVE_N_NEW_TOKENS
    patience: int = _SELF_IMPROVE_PATIENCE
    ce_min_improve: float = _SELF_IMPROVE_CE_MIN_IMPROVE
    kl_max: float = _SELF_IMPROVE_KL_MAX

    def __post_init__(self) -> None:
        if _exact_int(self.seed, "self-improve seed") < 0:
            raise ValueError("self-improve seed must be non-negative")
        if self.source_id != "A":
            raise ValueError("self-improve prompt must come from source_A")
        if not isinstance(self.prompt_span, SourceSpan) or self.prompt_span.source_id != "A":
            raise ValueError("self-improve prompt span must belong to source_A")
        for name in (
            "prompt_token_sha256", "prompt_text_sha256",
            "derived_base_state_sha256", "contour_state_sha256"
        ):
            _digest(getattr(self, name), name)
        if _exact_int(self.prompt_token_count, "prompt_token_count") <= 0:
            raise ValueError("prompt_token_count must be positive")
        if _exact_int(self.accepted_updates, "accepted_updates") < 0:
            raise ValueError("accepted_updates must be non-negative")
        if not isinstance(self.optimizer_parameter_ids, tuple) or not self.optimizer_parameter_ids:
            raise ValueError("optimizer_parameter_ids must be a non-empty tuple")
        for index, name in enumerate(self.optimizer_parameter_ids):
            if not isinstance(name, str) or _PYRAMID_CONTOUR_KEY.fullmatch(name) is None:
                raise ValueError(
                    f"optimizer_parameter_ids[{index}] must be a fresh pyramid contour"
                )
        if len(set(self.optimizer_parameter_ids)) != len(self.optimizer_parameter_ids):
            raise ValueError("optimizer_parameter_ids must be distinct")
        if (
            type(self.mode) is not str
            or type(self.max_iterations) is not int
            or type(self.n_new_tokens) is not int
            or type(self.patience) is not int
            or type(self.ce_min_improve) not in (int, float)
            or type(self.kl_max) not in (int, float)
            or self.mode != _SELF_IMPROVE_MODE
            or self.max_iterations != _SELF_IMPROVE_MAX_ITERATIONS
            or self.n_new_tokens != _SELF_IMPROVE_N_NEW_TOKENS
            or self.patience != _SELF_IMPROVE_PATIENCE
            or self.ce_min_improve != _SELF_IMPROVE_CE_MIN_IMPROVE
            or self.kl_max != _SELF_IMPROVE_KL_MAX
            or not math.isfinite(float(self.ce_min_improve))
            or not math.isfinite(float(self.kl_max))
        ):
            raise ValueError("self-improve budget does not match the frozen V1 contract")


def _ledger_payload(ledger: SelfImprovementLedger) -> dict[str, Any]:
    return {
        "seed": ledger.seed,
        "source_id": ledger.source_id,
        "prompt_span": {
            "source_id": ledger.prompt_span.source_id,
            "start": ledger.prompt_span.start,
            "end": ledger.prompt_span.end,
            "source_manifest_sha256": ledger.prompt_span.source_manifest_sha256,
        },
        "prompt_token_sha256": ledger.prompt_token_sha256,
        "prompt_text_sha256": ledger.prompt_text_sha256,
        "prompt_token_count": ledger.prompt_token_count,
        "optimizer_parameter_ids": list(ledger.optimizer_parameter_ids),
        "accepted_updates": ledger.accepted_updates,
        "derived_base_state_sha256": ledger.derived_base_state_sha256,
        "contour_state_sha256": ledger.contour_state_sha256,
        "mode": ledger.mode,
        "max_iterations": ledger.max_iterations,
        "n_new_tokens": ledger.n_new_tokens,
        "patience": ledger.patience,
        "ce_min_improve": ledger.ce_min_improve,
        "kl_max": ledger.kl_max,
    }


def _ledger_from_payload(payload: Any) -> SelfImprovementLedger:
    if not isinstance(payload, dict) or set(payload) != _SELF_IMPROVE_FIELDS:
        raise ValueError("invalid persisted self-improvement ledger schema")
    if not isinstance(payload["prompt_span"], dict) or set(payload["prompt_span"]) != {
        "source_id", "start", "end", "source_manifest_sha256"
    }:
        raise ValueError("invalid persisted self-improvement prompt span")
    if not isinstance(payload["optimizer_parameter_ids"], list):
        raise ValueError("invalid persisted self-improvement optimizer IDs")
    return SelfImprovementLedger(
        seed=payload["seed"],
        source_id=payload["source_id"],
        prompt_span=SourceSpan(
            payload["prompt_span"]["source_id"],
            payload["prompt_span"]["start"],
            payload["prompt_span"]["end"],
            payload["prompt_span"]["source_manifest_sha256"],
        ),
        prompt_token_sha256=payload["prompt_token_sha256"],
        prompt_text_sha256=payload["prompt_text_sha256"],
        prompt_token_count=payload["prompt_token_count"],
        optimizer_parameter_ids=tuple(payload["optimizer_parameter_ids"]),
        accepted_updates=payload["accepted_updates"],
        derived_base_state_sha256=payload["derived_base_state_sha256"],
        contour_state_sha256=payload["contour_state_sha256"],
        mode=payload["mode"],
        max_iterations=payload["max_iterations"],
        n_new_tokens=payload["n_new_tokens"],
        patience=payload["patience"],
        ce_min_improve=payload["ce_min_improve"],
        kl_max=payload["kl_max"],
    )


@dataclass(frozen=True)
class SourceMetric:
    source_id: str
    scored_rows: int
    exact_ce: float
    row_ids_sha256: str

    def __post_init__(self) -> None:
        if self.source_id not in SOURCE_NAMES:
            raise ValueError("invalid source metric")
        if _exact_int(self.scored_rows, "scored_rows") <= 0:
            raise ValueError("scored_rows must be positive")
        if _finite(self.exact_ce, "exact_ce") < 0:
            raise ValueError("exact_ce must be nonnegative")
        _digest(self.row_ids_sha256, "row_ids_sha256")


@dataclass(frozen=True)
class EvaluationResult:
    incumbent_metrics: tuple[SourceMetric, ...]
    candidate_metrics: tuple[SourceMetric, ...]
    tokenizer_name: str
    data_manifest_sha256: str
    protocol_sha256: str
    parent_checkpoint_sha256: str
    candidate_checkpoint_sha256: str
    candidate_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in ("incumbent_metrics", "candidate_metrics"):
            metrics = getattr(self, name)
            if not isinstance(metrics, tuple) or any(not isinstance(item, SourceMetric) for item in metrics):
                raise ValueError(f"{name} must contain SourceMetric values")
        if not isinstance(self.tokenizer_name, str) or not self.tokenizer_name:
            raise ValueError("tokenizer_name is required")
        for name in (
            "data_manifest_sha256",
            "protocol_sha256",
            "parent_checkpoint_sha256",
            "candidate_checkpoint_sha256",
            "candidate_manifest_sha256",
        ):
            _digest(getattr(self, name), name)


@dataclass(frozen=True)
class GenerationResult:
    decision: str
    generation_id: str
    report_path: str
    manifest_path: str
    candidate_checkpoint_path: str
    holdout_evidence_path: str
    incumbent_macro_ce: float
    candidate_macro_ce: float
    ce_regression: float
    worst_source_regression: float
    mechanism_supported: bool
    quality_supported: bool
    security_supported: bool
    production_promotion: bool
    pareto_improvement: bool

    def __post_init__(self) -> None:
        if self.decision not in {"accepted", "rejected"}:
            raise ValueError("invalid mechanism decision")
        if self.mechanism_supported is not (self.decision == "accepted"):
            raise ValueError("mechanism_supported must equal (decision == 'accepted')")
        for name in ("incumbent_macro_ce", "candidate_macro_ce", "ce_regression", "worst_source_regression"):
            _finite(getattr(self, name), name)
        if self.quality_supported is not False or self.security_supported is not False or self.production_promotion is not False:
            raise ValueError("quality, security, and production claims are forbidden")
        if type(self.pareto_improvement) is not bool:
            raise ValueError("pareto_improvement must be bool")


@dataclass(frozen=True)
class ChildContext:
    generation_id: str
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    parent_manifest_sha256: str
    name: str
    source_id: str
    seed: int
    span: SourceSpan
    protocol_sha256: str


@dataclass(frozen=True)
class CandidateContext:
    generation_id: str
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    parent_manifest_sha256: str
    children: tuple[ChildArtifact, ...]
    protocol_sha256: str
    base_seed: int
    self_improve_seed: int

    def __post_init__(self) -> None:
        if _exact_int(self.base_seed, "candidate base seed") < 0 or self.base_seed > _MAX_BASE_SEED:
            raise ValueError("candidate base seed is outside the deterministic range")
        if self.self_improve_seed != self.base_seed + _SELF_IMPROVE_OFFSET:
            raise ValueError("candidate self-improve seed must be base_seed + 9001")


@dataclass(frozen=True)
class EvaluationContext:
    request: GenerationRequest
    parent: ParentPointer
    holdout: HoldoutContract
    children: tuple[ChildArtifact, ...]
    candidate: CandidateArtifact


def _payload(path: str | Path) -> tuple[dict[str, Any], Any]:
    payload = load_payload(path)
    return payload, config_from_dict(payload["config"])


def _forbidden_state_keys(state: Mapping[str, Any]) -> set[str]:
    """Reject every adaptive/legacy key in a child checkpoint."""
    return {
        key for key in state
        if key.startswith("mixers.") or key.startswith("cortex.")
        or key.startswith("decision_head.") or ".ttt_lora." in key
        or ".adapters." in key
    }


def _forbidden_candidate_state_keys(state: Mapping[str, Any]) -> set[str]:
    """Allow only the exact fresh-pyramid contour on a candidate."""
    return {
        key for key in state
        if key.startswith("mixers.") or key.startswith("cortex.")
        or key.startswith("decision_head.") or ".ttt_lora." in key
        or (".adapters." in key and not _PYRAMID_CONTOUR_KEY.fullmatch(key))
    }


def _same_geometry(config: Any, parent_config: Any) -> bool:
    return (
        config.model.num_layers == parent_config.model.num_layers
        and config.model.hidden_size == parent_config.model.hidden_size
        and config.model.attention.num_query_heads == parent_config.model.attention.num_query_heads
        and config.model.attention.num_kv_heads == parent_config.model.attention.num_kv_heads
        and config.model.attention.head_dim == parent_config.model.attention.head_dim
        and config.model.ffn.intermediate_size == parent_config.model.ffn.intermediate_size
        and config.merge.expert_hidden == parent_config.merge.expert_hidden
    )


def _validate_child_payload(child: ChildArtifact, parent_cfg: Any) -> None:
    payload, config = _payload(child.checkpoint_path)
    if _config_digest(config) != child.config_sha256:
        raise ValueError("child checkpoint config digest mismatch")
    state = payload["model"]
    forbidden = _forbidden_state_keys(state)
    if forbidden:
        raise ValueError(f"child contains forbidden adaptive keys: {sorted(forbidden)}")
    parent_depth = parent_cfg.merge.ternary_depth
    is_recursive = RecursiveF3HAGI.is_recursive_state(state)
    if parent_depth == 0:
        if is_recursive or config.merge.ternary_depth != 0:
            raise ValueError("depth-zero parent requires plain children")
        if config.model.adapters.enabled or config.model.cortex.enabled or config.model.decision.enabled:
            raise ValueError("plain child config must be adaptive-free")
    else:
        if not is_recursive or config.merge.ternary_depth != parent_depth:
            raise ValueError("recursive child depth must equal parent depth")
        if config.merge.mixer_type != "ternary_f3" or not _same_geometry(config, parent_cfg):
            raise ValueError("recursive child geometry mismatch")
    if config.model.hidden_size != parent_cfg.model.hidden_size:
        raise ValueError("child hidden size must equal parent hidden size")


def _validate_child(child: ChildArtifact, plan: ChildPlan, request: GenerationRequest, parent_cfg: Any, index: int) -> None:
    expected = (CHILD_NAMES[index], plan.source_id, request.child_seeds[index], plan.span)
    if (child.name, child.source_id, child.seed, child.span) != expected:
        raise ValueError("child artifact identity mismatch")
    if (child.parent_checkpoint_sha256, child.protocol_sha256) != (
        request.parent.checkpoint_sha256, request.holdout.protocol_sha256,
    ):
        raise ValueError("child provenance mismatch")
    _regular_bytes(Path(child.checkpoint_path), child.checkpoint_sha256)
    manifest = _canonical_manifest(
        Path(child.manifest_path),
        child.manifest_sha256,
        _CHILD_DRAFT_MANIFEST_FIELDS,
        "child",
    )
    expected_manifest = {
        "schema_version": 1, "generation_id": request.generation_id, "name": child.name,
        "source_id": child.source_id, "seed": child.seed,
        "span": {"source_id": child.source_id, "start": child.span.start, "end": child.span.end,
                 "source_manifest_sha256": child.span.source_manifest_sha256},
        "checkpoint_sha256": child.checkpoint_sha256, "config_sha256": child.config_sha256,
        "parent_checkpoint_sha256": child.parent_checkpoint_sha256,
        "protocol_sha256": child.protocol_sha256,
    }
    if canonical_json_bytes(manifest) != canonical_json_bytes(expected_manifest):
        raise ValueError("child manifest declaration mismatch")
    _validate_child_payload(child, parent_cfg)


def _decode_text_buffer(value: Any, name: str) -> str:
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.uint8:
        raise ValueError(f"{name} buffer is invalid")
    return bytes(value.tolist()).decode("utf-8")


def _derived_base_digest(state: Mapping[str, Any]) -> str:
    """Digest every non-contour tensor of a candidate, byte-for-byte."""
    import hashlib

    import torch

    digest = hashlib.sha256()
    for key in sorted(state):
        if _PYRAMID_CONTOUR_KEY.fullmatch(key):
            continue
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"candidate state value must be a tensor: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"candidate state tensor is non-finite: {key}")
        array = value.detach().cpu().contiguous()
        raw = array.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(canonical_json_bytes({
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes_sha256": sha256_bytes(raw),
        }))
    return digest.hexdigest()


def _contour_state_digest(state: Mapping[str, Any]) -> str:
    """Digest every fresh pyramid-contour tensor, byte-for-byte."""
    import hashlib

    import torch

    digest = hashlib.sha256()
    matched = False
    for key in sorted(state):
        if not _PYRAMID_CONTOUR_KEY.fullmatch(key):
            continue
        matched = True
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"candidate state value must be a tensor: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"candidate state tensor is non-finite: {key}")
        array = value.detach().cpu().contiguous()
        raw = array.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(canonical_json_bytes({
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes_sha256": sha256_bytes(raw),
        }))
    if not matched:
        raise ValueError("candidate state has no fresh pyramid contour")
    return digest.hexdigest()


def _validate_persisted_candidate_provenance(metadata: Any) -> None:
    """Cross-check candidate evidence against its durable manifest.

    This follows the in-toto/SLSA subject/material verification pattern: the
    evidence snapshot must bind the exact declared artifact, not merely a
    self-consistent set of digests. See https://slsa.dev/spec/v1.0/provenance.
    """
    if not isinstance(metadata, dict) or set(metadata) != _CANDIDATE_EVIDENCE_FIELDS:
        raise ValueError("invalid persisted candidate evidence schema")
    manifest = _canonical_manifest(
        Path(metadata["manifest_path"]),
        metadata["manifest_sha256"],
        _CANDIDATE_MANIFEST_FIELDS,
        "candidate",
    )
    if any(manifest[key] != metadata[key] for key in _CANDIDATE_PROVENANCE_FIELDS):
        raise ValueError("terminal/evidence candidate provenance mismatch")
    for name in ("child_checkpoint_sha256", "child_config_sha256"):
        values = metadata.get(name)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"persisted candidate {name} must contain three values")
        for index, value in enumerate(values):
            _digest(value, f"persisted {name}[{index}]")
    if len(set(metadata["child_checkpoint_sha256"])) != 3:
        raise ValueError("persisted child checkpoint digests must be distinct")
    if len(set(metadata["child_config_sha256"])) != 1:
        raise ValueError("persisted child config digests must be identical")
    paths = metadata.get("child_checkpoint_paths")
    if not isinstance(paths, list) or len(paths) != 3:
        raise ValueError("persisted candidate lacks child snapshot paths")
    for index, value in enumerate(paths):
        _path(value, f"persisted child checkpoint path[{index}]")


def _validate_self_improve_config(config: Any, ledger: SelfImprovementLedger) -> None:
    """Bind the frozen V1 self-improve seam to the candidate configuration."""
    if config.model.loop_depth != 2:
        raise ValueError("self-improve requires model.loop_depth=2")
    if config.model.head.sampled_softmax_k != 0:
        raise ValueError("self-improve requires exact full-alphabet CE")
    if config.train.ce_keep_rate != 1.0:
        raise ValueError("self-improve requires train.ce_keep_rate=1.0")
    if not config.train.adapt.freeze_base:
        raise ValueError("self-improve must freeze the base")
    if config.train.data.eos_token_id == config.train.data.pad_token_id:
        raise ValueError("self-improve EOS and PAD token IDs must differ")
    if ledger.prompt_token_count + ledger.n_new_tokens > config.model.attention.max_seq_len:
        raise ValueError("self-improve prompt and generated budget exceed max_seq_len")


def _validate_persisted_self_improve_ledger(
    metadata: Any, request: GenerationRequest
) -> SelfImprovementLedger:
    """Revalidate ledger semantics from the durable candidate snapshot."""
    _validate_persisted_candidate_provenance(metadata)
    ledger = _ledger_from_payload(metadata["self_improve"])
    if ledger.seed != request.base_seed + _SELF_IMPROVE_OFFSET:
        raise ValueError("self-improve seed must be base_seed + 9001")
    source_span = request.child_plans[0].span
    if (
        ledger.prompt_span.source_manifest_sha256 != source_span.source_manifest_sha256
        or ledger.prompt_span.start < source_span.start
        or ledger.prompt_span.end > source_span.end
    ):
        raise ValueError("self-improve prompt span escapes the child_A train range")
    if ledger.prompt_token_count != ledger.prompt_span.end - ledger.prompt_span.start:
        raise ValueError("self-improve prompt token count does not match its span")
    if ledger.accepted_updates > _MAX_SELF_IMPROVE_UPDATES:
        raise ValueError("self-improve accepted_updates exceeds the bounded contract")

    checkpoint_path = Path(_path(metadata["checkpoint_path"], "persisted checkpoint_path"))
    _regular_bytes(checkpoint_path, metadata["checkpoint_sha256"])
    payload, config = _payload(checkpoint_path)
    state = payload["model"]
    if _config_digest(config) != metadata["candidate_config_sha256"]:
        raise ValueError("persisted candidate config digest mismatch")
    if _state_key_digest(state) != metadata["candidate_state_key_digest"]:
        raise ValueError("persisted candidate state key digest mismatch")
    _validate_self_improve_config(config, ledger)
    contour_keys = _validate_candidate_contour_ownership(
        config, state, ledger.accepted_updates
    )
    if frozenset(ledger.optimizer_parameter_ids) != contour_keys:
        raise ValueError("self-improve optimizer must own exactly the fresh contour")
    if ledger.derived_base_state_sha256 != _derived_base_digest(state):
        raise ValueError("self-improve derived base digest mismatch")
    if ledger.contour_state_sha256 != _contour_state_digest(state):
        raise ValueError("self-improve contour state digest mismatch")
    if ledger.prompt_token_count > config.train.data.seq_len:
        raise ValueError("self-improve prompt exceeds train.data.seq_len")
    return ledger


def _validate_self_improve_ledger(
    candidate: CandidateArtifact,
    request: GenerationRequest,
    children: tuple[ChildArtifact, ...],
    candidate_config: Any,
    contour_keys: frozenset[str],
    derived_digest: str,
    state: Mapping[str, Any],
) -> None:
    """Prove the ledger matches the declared seed, span, and ownership."""
    ledger = candidate.self_improve
    if ledger is None:
        raise ValueError("candidate is missing the self-improvement ledger")
    if ledger.seed != request.base_seed + _SELF_IMPROVE_OFFSET:
        raise ValueError("self-improve seed must be base_seed + 9001")
    source_span = children[0].span
    if children[0].source_id != "A" or source_span.source_id != "A":
        raise ValueError("self-improve prompt must come from the child_A span")
    if (
        ledger.prompt_span.source_manifest_sha256 != source_span.source_manifest_sha256
        or ledger.prompt_span.start < source_span.start
        or ledger.prompt_span.end > source_span.end
    ):
        raise ValueError("self-improve prompt span escapes the child_A train range")
    if frozenset(ledger.optimizer_parameter_ids) != contour_keys:
        raise ValueError("self-improve optimizer must own exactly the fresh contour")
    if ledger.prompt_token_count != ledger.prompt_span.end - ledger.prompt_span.start:
        raise ValueError("self-improve prompt token count does not match its span")
    if ledger.prompt_token_count > candidate_config.train.data.seq_len:
        raise ValueError("self-improve prompt exceeds train.data.seq_len")
    if ledger.accepted_updates > _MAX_SELF_IMPROVE_UPDATES:
        raise ValueError("self-improve accepted_updates exceeds the bounded contract")
    _validate_self_improve_config(candidate_config, ledger)
    if ledger.derived_base_state_sha256 != derived_digest:
        raise ValueError("self-improve derived base digest mismatch")
    if ledger.contour_state_sha256 != _contour_state_digest(state):
        raise ValueError("self-improve contour state digest mismatch")


def _validate_candidate_contour_ownership(
    config: Any,
    state: Mapping[str, Any],
    accepted_updates: int,
) -> frozenset[str]:
    """Require the exact fresh-pyramid ownership split of V1 §14.2/§13.4.

    A child builds no adapter. The candidate owns only a fresh zero-initialized
    ``*.adapters.pyramid.scale`` contour. With zero accepted updates the contour
    must remain at exact zero init; a bounded positive update count may move
    only that contour, and the separately validated ledger must own it.
    """
    if _exact_int(accepted_updates, "accepted_updates") > _MAX_SELF_IMPROVE_UPDATES:
        raise ValueError("self-improve accepted_updates exceeds the bounded contract")
    adapters = config.model.adapters
    contour_keys = frozenset(key for key in state if _PYRAMID_CONTOUR_KEY.fullmatch(key))
    if config.model.cortex.enabled or config.model.decision.enabled:
        raise ValueError("candidate must disable cortex and decision heads")
    if config.model.adapters.ttt_lora.enabled:
        raise ValueError("candidate must disable TTT-LoRA")
    if not adapters.enabled or not adapters.pyramid.enabled:
        if contour_keys:
            raise ValueError("candidate contour state requires the fresh pyramid fragment")
        return contour_keys
    if not config.train.adapt.freeze_base:
        raise ValueError("candidate self-improvement must freeze the base")
    if not contour_keys:
        raise ValueError("candidate is missing the fresh pyramid contour state")
    expected = frozenset(
        f"blocks.{index}.adapters.pyramid.scale"
        for index in range(config.model.num_layers)
    )
    if contour_keys != expected:
        raise ValueError("candidate pyramid contour does not cover every block")
    import torch

    for key in sorted(contour_keys):
        value = state[key]
        if value.shape != (1,) or value.is_floating_point() is False:
            raise ValueError(f"candidate pyramid contour shape is invalid: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"candidate pyramid contour is non-finite: {key}")
        if accepted_updates == 0 and not torch.equal(value, torch.zeros_like(value)):
            raise ValueError(f"candidate pyramid contour is not fresh zero init: {key}")
    if accepted_updates > 0 and not any(
        not torch.equal(state[key], torch.zeros_like(state[key]))
        for key in sorted(contour_keys)
    ):
        raise ValueError("accepted self-improvement left every contour unchanged")
    return contour_keys


def _validate_candidate_payload(candidate: CandidateArtifact, request: GenerationRequest, parent_payload: dict[str, Any], parent_cfg: Any) -> tuple[dict[str, Any], Any]:
    payload, config = _payload(candidate.checkpoint_path)
    if _config_digest(config) != candidate.candidate_config_sha256:
        raise ValueError("candidate checkpoint config digest mismatch")
    if candidate.parent_config_sha256 != _config_digest(parent_cfg):
        raise ValueError("parent config digest mismatch")
    if candidate.parent_checkpoint_sha256 != request.parent.checkpoint_sha256:
        raise ValueError("candidate parent checkpoint mismatch")
    state = payload["model"]
    if not RecursiveF3HAGI.is_recursive_state(state):
        raise ValueError("candidate must be recursive F3 state")
    forbidden = _forbidden_candidate_state_keys(state)
    if forbidden:
        raise ValueError(f"candidate contains forbidden state keys: {sorted(forbidden)}")
    if candidate.self_improve.accepted_updates > _MAX_SELF_IMPROVE_UPDATES:
        raise ValueError("self-improve accepted_updates exceeds the bounded contract")
    contour_keys = _validate_candidate_contour_ownership(
        config, state, candidate.self_improve.accepted_updates
    )
    if _state_key_digest(state) != candidate.candidate_state_key_digest:
        raise ValueError("candidate state key digest mismatch")
    parent_depth = parent_cfg.merge.ternary_depth
    expected_depth = parent_depth + 1
    leaf_hidden = parent_cfg.model.hidden_size // (3**parent_depth)
    if candidate.ternary_depth != expected_depth or config.merge.ternary_depth != expected_depth:
        raise ValueError("candidate depth mismatch")
    if config.merge.expert_hidden != parent_cfg.model.hidden_size or config.model.hidden_size != 3 * parent_cfg.model.hidden_size:
        raise ValueError("candidate hidden geometry mismatch")
    if candidate.leaf_hidden != leaf_hidden or leaf_hidden <= 0 or leaf_hidden % 2:
        raise ValueError("candidate leaf geometry mismatch")
    if candidate.weight_source != config.merge.expert_weight_source:
        raise ValueError("candidate weight source mismatch")
    tree = _declared_transform_tree(candidate)
    if candidate.coordinate_layout != tree.coordinate_layout or candidate.transform_digest != tree.transform_digest:
        raise ValueError("candidate transform metadata mismatch")
    if _decode_text_buffer(state.get("recursive_f3_transform_digest"), "transform") != candidate.transform_digest:
        raise ValueError("candidate checkpoint transform digest mismatch")
    source_value = state.get("recursive_f3_logit_scale_source")
    target_value = state.get("recursive_f3_logit_scale_target")
    if source_value is None or target_value is None or source_value.numel() != 1 or target_value.numel() != 1:
        raise ValueError("candidate scale buffers are missing")
    source = _finite(source_value.item(), "checkpoint logit_scale_source")
    target = _finite(target_value.item(), "checkpoint logit_scale_target")
    # The receiver gain is a property of the declared transform, not a constant:
    # the staged F3 tree aggregates the three children (sqrt(3)), while the
    # parent-preserving lift fixes the repeated branch (3). Reading it from the
    # rebuilt tree keeps the invariant exact for both pinned transforms.
    expected_gain = _transform_receiver_gain(candidate)
    if source <= 0 or target <= 0 or not math.isclose(source, expected_gain * target, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("candidate scale relation is invalid")
    if not math.isclose(source, candidate.logit_scale_source, rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(target, candidate.logit_scale_target, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("candidate declared scales mismatch checkpoint")
    if state.get("recursive_f3_child_config_fingerprint") is None:
        raise ValueError("candidate child config fingerprint is missing")
    if _decode_text_buffer(state["recursive_f3_child_config_fingerprint"], "child config") != candidate.child_config_sha256[0]:
        raise ValueError("candidate child config fingerprint mismatch")
    return payload, config, contour_keys


def _validate_persisted_f3_derivation(
    metadata: Mapping[str, Any], run: Path
) -> None:
    """Rebuild the candidate from durable child snapshots during terminal replay."""
    import torch

    checkpoint_paths = metadata["child_checkpoint_paths"]
    if not isinstance(checkpoint_paths, list) or len(checkpoint_paths) != 3:
        raise ValueError("persisted candidate lacks child snapshot paths")
    child_states: list[Mapping[str, Any]] = []
    child_configs: list[Any] = []
    for index, name in enumerate(CHILD_NAMES):
        checkpoint_path = Path(_path(checkpoint_paths[index], "child checkpoint path"))
        digest = metadata["child_checkpoint_sha256"][index]
        expected_path = run / "inputs" / f"{name}-checkpoint-{digest}.bin"
        if _path_key(checkpoint_path) != _path_key(expected_path):
            raise ValueError("persisted child checkpoint path mismatch")
        _regular_bytes(checkpoint_path, digest)
        payload, config = _payload(checkpoint_path)
        if _config_digest(config) != metadata["child_config_sha256"][index]:
            raise ValueError("persisted child config digest mismatch")
        child_states.append(payload["model"])
        child_configs.append(config)

    candidate_path = Path(
        _path(metadata["checkpoint_path"], "persisted checkpoint_path")
    )
    _regular_bytes(candidate_path, metadata["checkpoint_sha256"])
    candidate_payload, candidate_config = _payload(candidate_path)
    if _ledger_from_payload(metadata["self_improve"]).contour_state_sha256 != _contour_state_digest(
        candidate_payload["model"]
    ):
        raise ValueError("self-improve contour state digest mismatch")
    depth = _exact_int(metadata["ternary_depth"], "persisted ternary_depth")
    if candidate_config.merge.ternary_depth != depth or depth < 1:
        raise ValueError("persisted candidate depth mismatch")
    expected_model = merge_recursive_f3(
        candidate_config,
        child_states,
        child_configs=child_configs,
        parent_depth=depth - 1,
        expert_weight_source=metadata["weight_source"],
    )
    expected_state = expected_model.state_dict()
    actual_state = candidate_payload["model"]
    if set(actual_state) != set(expected_state):
        raise ValueError("terminal candidate does not match derived F3 payload keys")
    for key in sorted(expected_state):
        expected = expected_state[key]
        actual = actual_state[key]
        if (
            not isinstance(actual, type(expected))
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
        ):
            raise ValueError(f"terminal candidate does not match derived F3 payload: {key}")
        if _PYRAMID_CONTOUR_KEY.fullmatch(key):
            if not torch.equal(expected, torch.zeros_like(expected)):
                raise ValueError(f"terminal fresh F3 contour is not zero initialized: {key}")
            if not torch.isfinite(actual).all():
                raise ValueError(f"terminal candidate F3 contour is non-finite: {key}")
            continue
        if not torch.equal(actual, expected):
            raise ValueError(f"terminal candidate does not match derived F3 payload: {key}")


def _validate_derived_f3_payload(
    candidate: CandidateArtifact,
    candidate_payload: Mapping[str, Any],
    candidate_config: Any,
    children: tuple[ChildArtifact, ...],
    parent_depth: int,
) -> str:
    """Require candidate tensors to be the exact F3 assembly of its children.

    Manifest and key digests prove binding, not derivation. Reconstructing with
    the same production constructor closes that gap. The only mutable contour
    is the exact post-merge pyramid scale; its provenance ledger is a separate
    contract and is not treated as an arbitrary payload exception here.
    """
    import torch

    child_states: list[Mapping[str, Any]] = []
    child_configs: list[Any] = []
    for child in children:
        child_payload, child_config = _payload(child.checkpoint_path)
        child_states.append(child_payload["model"])
        child_configs.append(child_config)

    expected_model = merge_recursive_f3(
        candidate_config,
        child_states,
        child_configs=child_configs,
        parent_depth=parent_depth,
        expert_weight_source=candidate.weight_source,
    )
    expected_state = expected_model.state_dict()
    actual_state = candidate_payload["model"]
    if set(actual_state) != set(expected_state):
        raise ValueError("candidate does not match derived F3 payload keys")

    for key in sorted(expected_state):
        expected = expected_state[key]
        actual = actual_state[key]
        if not isinstance(actual, type(expected)) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"candidate does not match derived F3 payload: {key}")
        if _PYRAMID_CONTOUR_KEY.fullmatch(key):
            if not torch.equal(expected, torch.zeros_like(expected)):
                raise ValueError(f"fresh F3 contour is not zero initialized: {key}")
            if actual.is_floating_point() and not torch.isfinite(actual).all():
                raise ValueError(f"candidate F3 contour is non-finite: {key}")
            continue
        if not torch.equal(actual, expected):
            raise ValueError(f"candidate does not match derived F3 payload: {key}")
    return _derived_base_digest(expected_state)


def _validate_candidate(candidate: CandidateArtifact, request: GenerationRequest, children: tuple[ChildArtifact, ...], parent_payload: dict[str, Any], parent_cfg: Any) -> None:
    if not isinstance(candidate, CandidateArtifact):
        raise ValueError("candidate must be CandidateArtifact")
    if candidate.generation_id != request.generation_id:
        raise ValueError("candidate generation mismatch")
    if candidate.child_checkpoint_sha256 != tuple(child.checkpoint_sha256 for child in children):
        raise ValueError("candidate child checkpoint binding mismatch")
    if candidate.child_config_sha256 != tuple(child.config_sha256 for child in children):
        raise ValueError("candidate child config binding mismatch")
    if candidate.protocol_sha256 != request.holdout.protocol_sha256:
        raise ValueError("candidate protocol identity mismatch")
    _regular_bytes(Path(candidate.checkpoint_path), candidate.checkpoint_sha256)
    manifest = _canonical_manifest(
        Path(candidate.manifest_path),
        candidate.manifest_sha256,
        _CANDIDATE_DRAFT_MANIFEST_FIELDS,
        "candidate",
    )
    expected = {
        "schema_version": 1, "generation_id": candidate.generation_id,
        "parent_checkpoint_sha256": candidate.parent_checkpoint_sha256,
        "parent_config_sha256": candidate.parent_config_sha256,
        "protocol_sha256": candidate.protocol_sha256,
        "candidate_checkpoint_sha256": candidate.checkpoint_sha256,
        "candidate_config_sha256": candidate.candidate_config_sha256,
        "candidate_state_key_digest": candidate.candidate_state_key_digest,
        "ternary_depth": candidate.ternary_depth, "leaf_hidden": candidate.leaf_hidden,
        "coordinate_layout": candidate.coordinate_layout,
        "transform_digest": candidate.transform_digest,
        "weight_source": candidate.weight_source,
        "logit_scale_source": candidate.logit_scale_source,
        "logit_scale_target": candidate.logit_scale_target,
        "decision": "candidate",
        "self_improve": _ledger_payload(candidate.self_improve),
    }
    if canonical_json_bytes(manifest) != canonical_json_bytes(expected):
        raise ValueError("candidate manifest declaration mismatch")
    candidate_payload, candidate_config, contour_keys = _validate_candidate_payload(
        candidate, request, parent_payload, parent_cfg
    )
    derived_digest = _validate_derived_f3_payload(
        candidate,
        candidate_payload,
        candidate_config,
        children,
        parent_cfg.merge.ternary_depth,
    )
    _validate_self_improve_ledger(
        candidate,
        request,
        children,
        candidate_config,
        contour_keys,
        derived_digest,
        candidate_payload["model"],
    )


def _metric(metric: SourceMetric) -> dict[str, Any]:
    return {
        "source_id": metric.source_id,
        "scored_rows": metric.scored_rows,
        "exact_ce": metric.exact_ce,
        "row_ids_sha256": metric.row_ids_sha256,
    }


def _ordered_metrics(metrics: tuple[SourceMetric, ...], label: str) -> dict[str, dict[str, Any]]:
    ids = tuple(item.source_id for item in metrics)
    if ids != SOURCE_NAMES:
        raise ValueError(f"{label} must contain each source exactly once in A/B/C order")
    return {item.source_id: _metric(item) for item in metrics}


def _verdict(
    evaluation: EvaluationResult,
    request: GenerationRequest,
    candidate: CandidateArtifact,
) -> dict[str, Any]:
    if not isinstance(evaluation, EvaluationResult):
        raise ValueError("evaluate must return EvaluationResult")
    if (
        evaluation.candidate_checkpoint_sha256,
        evaluation.candidate_manifest_sha256,
    ) != (candidate.checkpoint_sha256, candidate.manifest_sha256):
        raise ValueError("evaluation candidate identity mismatch")
    if (
        evaluation.tokenizer_name,
        evaluation.data_manifest_sha256,
        evaluation.protocol_sha256,
        evaluation.parent_checkpoint_sha256,
    ) != (
        request.holdout.tokenizer_name,
        request.holdout.data_manifest_sha256,
        request.holdout.protocol_sha256,
        request.parent.checkpoint_sha256,
    ):
        raise ValueError("evaluation holdout identity mismatch")
    incumbent = _ordered_metrics(evaluation.incumbent_metrics, "incumbent_metrics")
    candidate_metrics = _ordered_metrics(evaluation.candidate_metrics, "candidate_metrics")
    if any(
        candidate_metrics[source]["scored_rows"] != incumbent[source]["scored_rows"]
        for source in SOURCE_NAMES
    ):
        raise ValueError("candidate scored_rows must equal incumbent scored_rows")
    for index, source in enumerate(SOURCE_NAMES):
        expected_rows = request.holdout.row_ids_sha256[index]
        if (
            incumbent[source]["row_ids_sha256"] != expected_rows
            or candidate_metrics[source]["row_ids_sha256"] != expected_rows
        ):
            raise ValueError("evaluation row identity mismatch")
    incumbent_macro = sum(incumbent[source]["exact_ce"] for source in SOURCE_NAMES) / 3.0
    candidate_macro = sum(candidate_metrics[source]["exact_ce"] for source in SOURCE_NAMES) / 3.0
    deltas = {
        source: candidate_metrics[source]["exact_ce"] - incumbent[source]["exact_ce"]
        for source in SOURCE_NAMES
    }
    regression = candidate_macro - incumbent_macro
    worst = max(deltas.values())
    accepted = regression <= 0.0 and worst <= 0.01
    return {
        "decision": "accepted" if accepted else "rejected",
        "incumbent_macro_ce": incumbent_macro,
        "candidate_macro_ce": candidate_macro,
        "ce_regression": regression,
        "worst_source_regression": worst,
        "pareto_improvement": regression < 0.0,
        "mechanism_supported": accepted,
        "source_metrics": {
            "incumbent": incumbent,
            "candidate": candidate_metrics,
            "deltas": deltas,
        },
    }


def _candidate_metadata(
    candidate: CandidateArtifact, children: tuple[ChildArtifact, ...]
) -> dict[str, Any]:
    return {
        "checkpoint_path": candidate.checkpoint_path, "checkpoint_sha256": candidate.checkpoint_sha256,
        "manifest_path": candidate.manifest_path, "manifest_sha256": candidate.manifest_sha256,
        "candidate_config_sha256": candidate.candidate_config_sha256,
        "candidate_state_key_digest": candidate.candidate_state_key_digest,
        "ternary_depth": candidate.ternary_depth, "leaf_hidden": candidate.leaf_hidden,
        "coordinate_layout": candidate.coordinate_layout,
        "transform_digest": candidate.transform_digest, "weight_source": candidate.weight_source,
        "logit_scale_source": candidate.logit_scale_source,
        "logit_scale_target": candidate.logit_scale_target,
        "self_improve": _ledger_payload(candidate.self_improve),
        "child_checkpoint_paths": [child.checkpoint_path for child in children],
        "child_checkpoint_sha256": list(candidate.child_checkpoint_sha256),
        "child_config_sha256": list(candidate.child_config_sha256),
    }


def _request_bindings(request: GenerationRequest) -> dict[str, Any]:
    return {
        "generation_id": request.generation_id, "base_seed": request.base_seed,
        "parent": request.parent.as_dict(), "parent_checkpoint_path": request.parent_checkpoint_path,
        "parent_checkpoint_sha256": request.parent.checkpoint_sha256,
        "parent_manifest_sha256": request.parent.manifest_sha256,
        "protocol_sha256": request.holdout.protocol_sha256,
        "data_manifest_sha256": request.holdout.data_manifest_sha256,
        "tokenizer_name": request.holdout.tokenizer_name,
        "holdout_sha256": request.holdout.sha256,
        "row_ids_sha256": list(request.holdout.row_ids_sha256),
        "child_plans": [
            {"name": name, "source_id": plan.source_id, "seed": seed,
             "span": {"source_id": plan.span.source_id, "start": plan.span.start,
                      "end": plan.span.end, "source_manifest_sha256": plan.span.source_manifest_sha256}}
            for name, plan, seed in zip(CHILD_NAMES, request.child_plans, request.child_seeds)
        ],
    }


def _mechanism_supported(verdict: dict[str, Any]) -> bool:
    """Whether the held-out evidence actually supports the growth mechanism.

    This used to be the literal ``True``, written independently of the
    evaluator. A persisted evidence file therefore claimed mechanism support
    even for a rejected candidate, which is the same class of dishonesty that
    killed the DecisionPlane gate: an unconditional claim that no measurement
    can contradict.

    The claim is now derived from the verdict, and deliberately narrowly: the
    mechanism is supported only when the candidate was accepted on held-out
    evidence, i.e. macro CE did not regress and no source regressed beyond the
    declared tolerance. A rejected candidate supports nothing, and quality /
    security / production promotion stay ``False`` because this gate measures
    neither.
    """
    if verdict.get("decision") != "accepted":
        return False
    if verdict.get("ce_regression", 0.0) > 0.0:
        return False
    if verdict.get("worst_source_regression", 0.0) > 0.01:
        return False
    return True


def _evidence_payload(
    request: GenerationRequest,
    candidate: CandidateArtifact,
    verdict: dict[str, Any],
    children: tuple[ChildArtifact, ...],
) -> dict[str, Any]:
    return {
        "schema": "recursive_f3_holdout_evidence_v2", "schema_version": 2,
        "generation_id": request.generation_id, "decision": verdict["decision"],
        "mechanism_supported": _mechanism_supported(verdict),
        "quality_supported": False,
        "security_supported": False, "production_promotion": False,
        "pareto_improvement": verdict["pareto_improvement"],
        "incumbent_macro_ce": verdict["incumbent_macro_ce"],
        "candidate_macro_ce": verdict["candidate_macro_ce"],
        "ce_regression": verdict["ce_regression"],
        "worst_source_regression": verdict["worst_source_regression"],
        "source_metrics": verdict["source_metrics"],
        "bindings": _request_bindings(request),
        "candidate_f3": _candidate_metadata(candidate, children),
    }


def _write_evidence(path: Path, payload: dict[str, Any]) -> str:
    data = canonical_json_bytes(payload)
    _write_exclusive(path, data, sha256_bytes(data), "holdout evidence")
    return sha256_bytes(data)


def _legacy_mechanism_pinned(evidence: dict[str, Any], run: Path) -> bool:
    """Whether this evidence predates the derived mechanism claim.

    Schema v1 pinned ``mechanism_supported`` to the literal ``True`` for every
    run, accepted or rejected, so 39 historical ``.omc/runs`` artifacts carry
    ``rejected`` + ``True``. Those files carry no information in that field,
    but they are the record of runs that really happened; making them
    unresumable to enforce a cleaner invariant would discard history for a
    cosmetic gain ([Chesterton](omc-software-laws#chestertons-fence)).

    The escape is narrow and does not extend the claim: it applies only to
    v1 evidence that the writer of that era produced, and the flag is never
    surfaced from a legacy replay -- it is recomputed below, so the caller
    sees the honest value while the old bytes stay readable.
    """
    if evidence["schema"] != "recursive_f3_holdout_evidence_v1":
        return False
    return evidence["mechanism_supported"] is True


def _validate_evidence(
    evidence: Any, request: GenerationRequest, run: Path
) -> dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("invalid persisted holdout evidence schema")
    fixed = {
        "generation_id": request.generation_id,
        "quality_supported": False, "security_supported": False,
        "production_promotion": False,
    }
    # v1 pinned the mechanism claim to a literal; v2 derives it. Both schemas
    # stay readable, because 39 v1 runs on disk predate the derivation.
    if (evidence["schema"], evidence["schema_version"]) not in {
        ("recursive_f3_holdout_evidence_v1", 1),
        ("recursive_f3_holdout_evidence_v2", 2),
    }:
        raise ValueError("invalid persisted holdout evidence schema")
    boolean_claims = (
        "mechanism_supported",
        "quality_supported",
        "security_supported",
        "production_promotion",
        "pareto_improvement",
    )
    if type(evidence["schema_version"]) is not int or any(
        type(evidence[key]) is not bool for key in boolean_claims
    ) or any(evidence[key] != value for key, value in fixed.items()) or evidence["decision"] not in {"accepted", "rejected"}:
        raise ValueError("invalid persisted holdout evidence")
    # The mechanism claim must follow from the persisted metrics, not from a
    # literal. Recompute it the way the writer does, so a hand-edited evidence
    # file that flips the flag is rejected on load instead of being trusted.
    if evidence["mechanism_supported"] != _mechanism_supported(
        {
            "decision": evidence["decision"],
            "ce_regression": evidence["ce_regression"],
            "worst_source_regression": evidence["worst_source_regression"],
        }
    ) and not _legacy_mechanism_pinned(evidence, run):
        raise ValueError("persisted mechanism claim does not follow from the verdict")
    if evidence["bindings"] != _request_bindings(request):
        raise ValueError("persisted evidence request binding mismatch")
    _validate_persisted_self_improve_ledger(evidence["candidate_f3"], request)
    _validate_persisted_f3_derivation(evidence["candidate_f3"], run)
    metrics = evidence["source_metrics"]
    if not isinstance(metrics, dict) or set(metrics) != {"incumbent", "candidate", "deltas"}:
        raise ValueError("invalid persisted source metrics")
    macros: list[float] = []
    for role in ("incumbent", "candidate"):
        group = metrics[role]
        if not isinstance(group, dict) or set(group) != set(SOURCE_NAMES):
            raise ValueError("invalid persisted source metrics")
        for source in SOURCE_NAMES:
            item = group[source]
            if not isinstance(item, dict) or set(item) != {
                "source_id", "scored_rows", "exact_ce", "row_ids_sha256"
            }:
                raise ValueError("invalid persisted source metric")
            if item["source_id"] != source or type(item["scored_rows"]) is not int or item["scored_rows"] <= 0:
                raise ValueError("invalid persisted source metric")
            expected_rows = request.holdout.row_ids_sha256[SOURCE_NAMES.index(source)]
            _digest(item["row_ids_sha256"], "persisted row_ids_sha256")
            if item["row_ids_sha256"] != expected_rows:
                raise ValueError("persisted row identity mismatch")
            if _finite(item["exact_ce"], "persisted exact_ce") < 0:
                raise ValueError("persisted exact_ce must be nonnegative")
        macros.append(sum(group[source]["exact_ce"] for source in SOURCE_NAMES) / 3.0)
    if any(metrics["incumbent"][source]["scored_rows"] != metrics["candidate"][source]["scored_rows"] for source in SOURCE_NAMES):
        raise ValueError("persisted scored rows mismatch")
    deltas = {source: metrics["candidate"][source]["exact_ce"] - metrics["incumbent"][source]["exact_ce"] for source in SOURCE_NAMES}
    if metrics["deltas"] != deltas:
        raise ValueError("persisted source deltas mismatch")
    incumbent_macro, candidate_macro = macros
    regression = candidate_macro - incumbent_macro
    worst = max(deltas.values())
    decision = "accepted" if regression <= 0.0 and worst <= 0.01 else "rejected"
    expected = {
        "incumbent_macro_ce": incumbent_macro, "candidate_macro_ce": candidate_macro,
        "ce_regression": regression, "worst_source_regression": worst,
        "pareto_improvement": regression < 0.0, "decision": decision,
    }
    if any(evidence[key] != value for key, value in expected.items()):
        raise ValueError("persisted semantic verdict mismatch")
    return evidence


def _source_decision(generation_id: str, payload: dict[str, Any]) -> PreparedDecision:
    return PreparedDecision(
        generation_id=generation_id, manifest_sha256=payload["manifest_sha256"],
        candidate_checkpoint_sha256=payload["candidate_checkpoint_sha256"],
        decision=payload["decision"], holdout_evidence_sha256=payload["holdout_evidence_sha256"],
        manifest_path=payload["source_manifest_path"],
        candidate_checkpoint_path=payload["source_candidate_checkpoint_path"],
        holdout_evidence_path=payload["source_holdout_evidence_path"],
        parent_generation_id=payload.get("parent_generation_id"),
        parent_checkpoint_sha256=payload.get("parent_checkpoint_sha256"),
        parent_manifest_sha256=payload.get("parent_manifest_sha256"),
    )


def _read_terminal(run: Path, request: GenerationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _strict_json_bytes(_regular_bytes(run / "report.json"))
    if report.get("generation_id") != request.generation_id:
        raise ValueError("terminal generation mismatch")
    evidence = _strict_json_bytes(_regular_bytes(Path(report["holdout_evidence_path"]), report["holdout_evidence_sha256"]))
    evidence = _validate_evidence(evidence, request, run)
    candidate = evidence["candidate_f3"]
    report_binding = {
        "checkpoint_path": "source_candidate_checkpoint_path",
        "checkpoint_sha256": "candidate_checkpoint_sha256",
        "manifest_path": "source_manifest_path",
        "manifest_sha256": "manifest_sha256",
    }
    if not isinstance(candidate, dict) or set(candidate) != _CANDIDATE_EVIDENCE_FIELDS or any(
        candidate[key] != report[report_key] for key, report_key in report_binding.items()
    ):
        raise ValueError("terminal/evidence candidate binding mismatch")
    manifest = _canonical_manifest(
        Path(report["manifest_path"]),
        report["manifest_sha256"],
        _CANDIDATE_MANIFEST_FIELDS,
        "candidate",
    )
    if any(manifest[key] != candidate[key] for key in _CANDIDATE_PROVENANCE_FIELDS):
        raise ValueError("terminal/evidence candidate provenance mismatch")
    if evidence["decision"] != report["decision"]:
        raise ValueError("terminal/evidence decision mismatch")
    return report, evidence


def _result_from_evidence(request: GenerationRequest, report: dict[str, Any], evidence: dict[str, Any], report_path: str) -> GenerationResult:
    return GenerationResult(
        decision=report["decision"], generation_id=request.generation_id,
        report_path=report_path, manifest_path=report["manifest_path"],
        candidate_checkpoint_path=report["candidate_checkpoint_path"],
        holdout_evidence_path=report["holdout_evidence_path"],
        incumbent_macro_ce=evidence["incumbent_macro_ce"],
        candidate_macro_ce=evidence["candidate_macro_ce"],
        ce_regression=evidence["ce_regression"],
        worst_source_regression=evidence["worst_source_regression"],
        mechanism_supported=_mechanism_supported(
            {
                "decision": report["decision"],
                "ce_regression": evidence["ce_regression"],
                "worst_source_regression": evidence["worst_source_regression"],
            }
        ),
        quality_supported=False, security_supported=False,
        production_promotion=False, pareto_improvement=evidence["pareto_improvement"],
    )


def _resume_terminal(request: GenerationRequest, store: GrowthRunStore, owner_id: str) -> GenerationResult:
    run = store.root / request.generation_id
    state = store.recover(request.generation_id, owner_id=owner_id)
    if state is RunState.FAILED:
        failure = _strict_json_bytes(_regular_bytes(run / "failure.json"))
        raise RuntimeError(f"generation failed: {failure['reason']}")
    if state not in {RunState.TERMINAL_ACCEPTED, RunState.TERMINAL_REJECTED}:
        raise ValueError("terminal report is not published")
    report, evidence = _read_terminal(run, request)
    result = _result_from_evidence(request, report, evidence, str(run / "report.json"))
    if result.decision == "rejected":
        return result
    # ``recover`` above validates the committed parent identity, including
    # the immutable report digest. The pointer's owner_id is the commit-time
    # generation identity; the active lease operator may legitimately differ
    # after takeover, so do not compare it with the current owner.
    return result


def _result(report_path: Path, evidence: dict[str, Any], request: GenerationRequest, store: GrowthRunStore) -> GenerationResult:
    run = store.root / request.generation_id
    report = _strict_json_bytes(_regular_bytes(run / "report.json"))
    return _result_from_evidence(request, report, evidence, str(report_path))


def _prepared_commit(
    request: GenerationRequest, store: GrowthRunStore, owner_id: str
) -> tuple[PreparedDecision, ParentPointer]:
    """Return the persisted source decision and its exact future terminal digest."""
    prepared_path = store.root / request.generation_id / "prepared-report.json"
    prepared_bytes = _regular_bytes(prepared_path)
    prepared = _strict_json_bytes(prepared_bytes)
    _snapshot, source = _prepared_contract(prepared, request.generation_id)
    report = _terminal_payload(prepared, sha256_bytes(prepared_bytes), _snapshot)
    replacement = ParentPointer(
        request.generation_id,
        source.candidate_checkpoint_sha256,
        source.manifest_sha256,
        sha256_bytes(canonical_json_bytes(report)),
        owner_id=owner_id,
    )
    return source, replacement


def _finish_prepared(
    request: GenerationRequest, store: GrowthRunStore, owner_id: str
) -> GenerationResult:
    decision, replacement = _prepared_commit(request, store, owner_id)
    evidence = _strict_json_bytes(
        _regular_bytes(Path(decision.holdout_evidence_path), decision.holdout_evidence_sha256)
    )
    _validate_evidence(evidence, request, store.root / request.generation_id)
    if decision.decision == "accepted":
        store.commit_accepted_terminal(
            request.parent, replacement, decision, owner_id=owner_id
        )
    else:
        store.publish_terminal(request.generation_id, owner_id=owner_id)
    return _resume_terminal(request, store, owner_id)


def run_generation(
    request: GenerationRequest,
    store: GrowthRunStore,
    build_child: Callable[[ChildContext], ChildArtifact],
    build_candidate: Callable[[CandidateContext], CandidateArtifact],
    evaluate: Callable[[EvaluationContext], EvaluationResult],
    *,
    owner_id: str = "default-owner",
) -> GenerationResult:
    """Run one recursive generation using trusted local callbacks.

    The owner exclusively derives the acceptance verdict. It does not
    authenticate evaluator metrics and is not a same-user security boundary.
    """
    if not isinstance(request, GenerationRequest) or not isinstance(store, GrowthRunStore):
        raise TypeError("request and store have invalid types")
    run = store.root / request.generation_id
    if run.exists():
        state = store.recover(request.generation_id, owner_id=owner_id)
        if state in {RunState.TERMINAL_ACCEPTED, RunState.TERMINAL_REJECTED}:
            return _resume_terminal(request, store, owner_id)
        if state is RunState.PREPARED:
            return _finish_prepared(request, store, owner_id)
        if state is RunState.FAILED:
            failure = _strict_json_bytes(_regular_bytes(run / "failure.json"))
            raise RuntimeError(f"generation failed: {failure['reason']}")
        raise ValueError("generation run exists in a non-resumable state")
    current = store.read_parent()
    if current != request.parent:
        raise ValueError("request parent does not match current parent")
    store.reserve(request.generation_id, owner_id=owner_id)
    store.mark_running(request.generation_id, owner_id=owner_id)
    try:
        parent_payload, parent_cfg = _payload(request.parent_checkpoint_path)
        _regular_bytes(Path(request.parent_checkpoint_path), request.parent.checkpoint_sha256)
        # Publish the parent bytes the builders are allowed to read before any
        # callback runs. Callbacks must never consume the live pointer target,
        # otherwise a callback mutation would leave the owner attesting a
        # parent lineage that no builder actually consumed.
        parent_snapshot = _snapshot(
            Path(request.parent_checkpoint_path),
            request.parent.checkpoint_sha256,
            run,
            "parent-checkpoint",
            "parent checkpoint",
        )
        children: list[ChildArtifact] = []
        checkpoint_keys: set[str] = set()
        manifest_keys: set[str] = set()
        parent_key = _path_key(request.parent_checkpoint_path)
        for index, plan in enumerate(request.child_plans):
            context = ChildContext(
                generation_id=request.generation_id,
                parent_checkpoint_path=str(parent_snapshot),
                parent_checkpoint_sha256=request.parent.checkpoint_sha256,
                parent_manifest_sha256=request.parent.manifest_sha256,
                name=CHILD_NAMES[index],
                source_id=plan.source_id,
                seed=request.child_seeds[index],
                span=plan.span,
                protocol_sha256=request.holdout.protocol_sha256,
            )
            child = build_child(context)
            if not isinstance(child, ChildArtifact):
                raise ValueError("build_child must return ChildArtifact")
            _regular_bytes(Path(child.checkpoint_path), child.checkpoint_sha256)
            _regular_bytes(Path(child.manifest_path), child.manifest_sha256)
            child_checkpoint_key = _path_key(child.checkpoint_path)
            child_manifest_key = _path_key(child.manifest_path)
            if (
                child_checkpoint_key == parent_key
                or child_checkpoint_key in checkpoint_keys
                or child_manifest_key in manifest_keys
            ):
                raise ValueError("child artifact paths must be unique and cannot alias parent")
            _validate_child(child, plan, request, parent_cfg, index)
            checkpoint_keys.add(child_checkpoint_key)
            manifest_keys.add(child_manifest_key)
            child_checkpoint = _snapshot(
                Path(child.checkpoint_path),
                child.checkpoint_sha256,
                run,
                f"{child.name}-checkpoint",
                "child checkpoint",
            )
            child_manifest = _snapshot(
                Path(child.manifest_path),
                child.manifest_sha256,
                run,
                f"{child.name}-manifest-draft",
                "child draft manifest",
            )
            children.append(
                replace(
                    child,
                    checkpoint_path=str(child_checkpoint),
                    manifest_path=str(child_manifest),
                )
            )
        child_tuple = tuple(children)
        if len({child.config_sha256 for child in child_tuple}) != 1:
            raise ValueError("all child config digests must be identical")
        if len({child.checkpoint_sha256 for child in child_tuple}) != 3:
            raise ValueError("child checkpoint digests must be distinct")
        if len({child.manifest_sha256 for child in child_tuple}) != 3:
            raise ValueError("child manifest digests must be distinct")
        candidate = build_candidate(
            CandidateContext(
                generation_id=request.generation_id,
                parent_checkpoint_path=str(parent_snapshot),
                parent_checkpoint_sha256=request.parent.checkpoint_sha256,
                parent_manifest_sha256=request.parent.manifest_sha256,
                children=child_tuple,
                protocol_sha256=request.holdout.protocol_sha256,
                base_seed=request.base_seed,
                self_improve_seed=request.base_seed + _SELF_IMPROVE_OFFSET,
            )
        )
        # Re-verify after the last build callback. Both the live pointer target
        # and the owner snapshot the builders actually read must still hash to
        # the attested parent digest, otherwise a build-phase mutation would
        # reach PREPARED with a parent lineage that no builder consumed.
        _regular_bytes(Path(request.parent_checkpoint_path), request.parent.checkpoint_sha256)
        _regular_bytes(parent_snapshot, request.parent.checkpoint_sha256)
        _validate_candidate(candidate, request, child_tuple, parent_payload, parent_cfg)
        candidate_checkpoint = _snapshot(
            Path(candidate.checkpoint_path),
            candidate.checkpoint_sha256,
            run,
            "candidate-checkpoint",
            "candidate checkpoint",
        )
        final_children: list[ChildArtifact] = []
        for child in child_tuple:
            final_manifest, final_manifest_sha256 = _finalize_manifest(
                Path(child.manifest_path),
                child.manifest_sha256,
                run,
                f"{child.name}-manifest",
                request.holdout.data_manifest_sha256,
                _CHILD_DRAFT_MANIFEST_FIELDS,
                "child manifest",
            )
            final_children.append(
                replace(
                    child,
                    manifest_path=str(final_manifest),
                    manifest_sha256=final_manifest_sha256,
                )
            )
        final_child_tuple = tuple(final_children)
        final_candidate_manifest, final_candidate_manifest_sha256 = _finalize_manifest(
            Path(candidate.manifest_path),
            candidate.manifest_sha256,
            run,
            "candidate-manifest",
            request.holdout.data_manifest_sha256,
            _CANDIDATE_DRAFT_MANIFEST_FIELDS,
            "candidate manifest",
        )
        candidate = replace(
            candidate,
            checkpoint_path=str(candidate_checkpoint),
            manifest_path=str(final_candidate_manifest),
            manifest_sha256=final_candidate_manifest_sha256,
        )
        holdout_path = _holdout_snapshot(
            Path(request.holdout.path),
            request.holdout.sha256,
            run,
        )
        holdout = replace(request.holdout, path=str(holdout_path))
        evaluation = evaluate(
            EvaluationContext(
                request,
                request.parent,
                holdout,
                final_child_tuple,
                candidate,
            )
        )
        _regular_bytes(holdout_path, request.holdout.sha256)
        verdict = _verdict(evaluation, request, candidate)
        evidence_path = run / "holdout-evidence.json"
        evidence = _evidence_payload(request, candidate, verdict, final_child_tuple)
        # Validate the complete durable subject before PREPARED and before the
        # parent CAS. Terminal replay must never discover invalid owner-produced
        # evidence only after current-parent.json has already changed.
        _validate_evidence(evidence, request, run)
        _regular_bytes(holdout_path, request.holdout.sha256)
        evidence_sha256 = _write_evidence(evidence_path, evidence)
        decision = PreparedDecision(
            generation_id=request.generation_id, manifest_sha256=candidate.manifest_sha256,
            candidate_checkpoint_sha256=candidate.checkpoint_sha256,
            decision=verdict["decision"], holdout_evidence_sha256=evidence_sha256,
            manifest_path=candidate.manifest_path,
            candidate_checkpoint_path=candidate.checkpoint_path,
            holdout_evidence_path=str(evidence_path),
            parent_generation_id=request.parent.generation_id,
            parent_checkpoint_sha256=request.parent.checkpoint_sha256,
            parent_manifest_sha256=request.parent.manifest_sha256,
        )
        store.mark_prepared(request.generation_id, decision, owner_id=owner_id)
    except BaseException as exc:
        if (run / "prepared-report.json").exists():
            raise
        try:
            store.publish_failure(
                request.generation_id,
                type(exc).__name__,
                owner_id=owner_id,
            )
        except BaseException as failure_exc:
            raise failure_exc from exc
        raise
    if decision.decision == "rejected":
        report_path = store.publish_terminal(request.generation_id, owner_id=owner_id)
    else:
        _persisted_decision, replacement = _prepared_commit(request, store, owner_id)
        report_path = store.commit_accepted_terminal(
            request.parent, replacement, decision, owner_id=owner_id
        )
    report, evidence = _read_terminal(run, request)
    return _result(report_path, evidence, request, store)
