"""Fail-closed external task-suite evaluation and champion/challenger selection.

This module measures a checkpoint on a *pinned, never-trained-on* task suite
and accepts or rejects a challenger from that external evidence only. Every
identity in the contract (suite, split, source bytes, manifest, protocol,
scorer, per-row digests) is digest-bound, and any mismatch raises instead of
falling back to a passing default.

Claim boundary: this is a *mechanism* for falsifiable external measurement and
promotion gating. It proves nothing about model quality, security, or
autonomy, and the default scoring budget below is far above instrument
resolution on the current 256-row holdout (paired SE ~= 0.0251 nats), so a
margin must be supplied explicitly by the caller; there is no silent default.

References reused from the repository (no parallel hashing scheme invented):
- ``hagi.orchestrator.state.canonical_json_bytes`` / ``sha256_bytes`` /
  ``sha256_file`` — the single canonical digest scheme.
- ``hagi.orchestrator.recursive.SourceMetric`` / ``_metric`` / ``_verdict``
  (L579, L1260, L1276) — the live verdict layer whose structure and field
  naming this module mirrors.
- ``hagi.orchestrator.evaluation.evaluate_packed_tokens`` — the exact
  full-alphabet CE scorer.
- ``hagi.orchestrator.real_cycle._score_checkpoint_bytes`` (L436) — the
  checkpoint schema validation reused for external scoring.
- ``hagi.data.artifacts.validate_manifest`` — dataset manifest validation.
"""
from __future__ import annotations

import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from hagi.config import CHECKPOINT_FORMAT_VERSION
from hagi.data.artifacts import validate_manifest
from hagi.model.merge import build_model_from_payload
from hagi.orchestrator.evaluation import evaluate_packed_tokens
from hagi.orchestrator.state import canonical_json_bytes, sha256_bytes, sha256_file
from hagi.train.checkpoint import config_from_dict

_MATH_ISFINITE = math.isfinite

SUITE_SCHEMA = "hagi_external_suite_v1"
PROTOCOL_SCHEMA = "hagi_external_protocol_v1"
VERDICT_SCHEMA = "hagi_external_verdict_v1"
PROMOTION_SCHEMA = "hagi_external_promotion_v1"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def _digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256 hex digest")


def _finite_nonnegative(value: float, label: str) -> None:
    if type(value) is not float:
        raise ValueError(f"{label} must be a float")
    if not _MATH_ISFINITE(value) or value < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative float")


@dataclass(frozen=True)
class ExternalSourceMetric:
    """Per-source CE summary, field-for-field the shape of
    ``hagi.orchestrator.recursive.SourceMetric`` (L579). External suites name
    their own sources, so the A/B/C restriction of the internal gate does not
    apply here.
    """

    source_id: str
    scored_rows: int
    exact_ce: float
    row_ids_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("external source metric requires a non-empty source_id")
        if type(self.scored_rows) is not int or self.scored_rows <= 0:
            raise ValueError("scored_rows must be a positive integer")
        _finite_nonnegative(self.exact_ce, "exact_ce")
        _digest(self.row_ids_sha256, "row_ids_sha256")


def row_digest(token_ids: Sequence[int]) -> str:
    """Digest one packed token row (canonical-JSON of the id list)."""
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ValueError("token row must be a sequence of integers")
    ids = [int(value) for value in token_ids]
    if not ids:
        raise ValueError("token row must be non-empty")
    return sha256_bytes(canonical_json_bytes(ids))


def protocol_payload(scorer_id: str, *, vocab_size: int) -> bytes:
    """Return the canonical protocol bytes pinned by an external suite."""
    if not isinstance(scorer_id, str) or not scorer_id:
        raise ValueError("scorer_id is required")
    if type(vocab_size) is not int or vocab_size < 4 or vocab_size > 262_144:
        raise ValueError("vocab_size is outside the supported compact range")
    return canonical_json_bytes(
        {
            "schema": PROTOCOL_SCHEMA,
            "schema_version": 1,
            "scoring": "evaluate_packed_tokens/full_alphabet/exact_ce",
            "scorer_id": scorer_id,
            "vocab_size": vocab_size,
        }
    )


def suite_payload(
    *,
    suite_id: str,
    split: str,
    tokenizer_name: str,
    vocab_size: int,
    sources: Mapping[str, Sequence[Sequence[int]]],
) -> bytes:
    """Return canonical suite bytes: rows plus their pinned per-row digests."""
    for label, value in (("suite_id", suite_id), ("split", split),
                         ("tokenizer_name", tokenizer_name)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} is required")
    if type(vocab_size) is not int or vocab_size < 4 or vocab_size > 262_144:
        raise ValueError("vocab_size is outside the supported compact range")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("external suite must declare at least one source")
    rows: dict[str, list[dict[str, object]]] = {}
    digests: dict[str, list[str]] = {}
    for source in sorted(sources):
        source_rows = sources[source]
        if not isinstance(source_rows, Sequence) or not source_rows:
            raise ValueError(f"source {source!r} must be a non-empty sequence")
        rows[source] = []
        digests[source] = []
        for token_ids in source_rows:
            ids = [int(value) for value in token_ids]
            if any(value < 0 or value >= vocab_size for value in ids):
                raise ValueError(f"source {source!r} has a token outside vocab_size")
            rows[source].append({"token_ids": ids})
            digests[source].append(row_digest(ids))
    if len({d for values in digests.values() for d in values}) != sum(
        len(values) for values in digests.values()
    ):
        raise ValueError("external suite row digests must be globally distinct")
    return canonical_json_bytes(
        {
            "schema": SUITE_SCHEMA,
            "schema_version": 1,
            "suite_id": suite_id,
            "split": split,
            "tokenizer_name": tokenizer_name,
            "vocab_size": vocab_size,
            "rows": rows,
            "row_ids_sha256": digests,
        }
    )


@dataclass(frozen=True)
class ExternalSuite:
    """A digest-pinned task suite the model has never trained on."""

    suite_id: str
    split: str
    source_path: str
    source_sha256: str
    manifest_path: str
    manifest_sha256: str
    tokenizer_name: str
    protocol_sha256: str
    protocol_payload: bytes
    scorer_id: str
    row_ids_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("suite_id", self.suite_id),
            ("split", self.split),
            ("tokenizer_name", self.tokenizer_name),
            ("scorer_id", self.scorer_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} is required")
        for label, path in (("source_path", self.source_path),
                            ("manifest_path", self.manifest_path)):
            if not isinstance(path, str) or not path:
                raise ValueError(f"{label} is required")
        for label, value in (
            ("source_sha256", self.source_sha256),
            ("manifest_sha256", self.manifest_sha256),
            ("protocol_sha256", self.protocol_sha256),
        ):
            _digest(value, label)
        if not isinstance(self.protocol_payload, bytes):
            raise ValueError("protocol_payload must be bytes")
        if sha256_bytes(self.protocol_payload) != self.protocol_sha256:
            raise ValueError("protocol payload does not match protocol_sha256")
        if not isinstance(self.row_ids_sha256, tuple) or not self.row_ids_sha256:
            raise ValueError("row_ids_sha256 must be a non-empty tuple")
        for index, digest in enumerate(self.row_ids_sha256):
            _digest(digest, f"row_ids_sha256[{index}]")
        if len(set(self.row_ids_sha256)) != len(self.row_ids_sha256):
            raise ValueError("row_ids_sha256 must be distinct")

    @property
    def binding_digest(self) -> str:
        """Digest over every pinned identity of this suite."""
        return sha256_bytes(canonical_json_bytes(self.as_dict()))

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "hagi_external_suite_contract_v1",
            "suite_id": self.suite_id,
            "split": self.split,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "tokenizer_name": self.tokenizer_name,
            "protocol_sha256": self.protocol_sha256,
            "scorer_id": self.scorer_id,
            "row_ids_sha256": list(self.row_ids_sha256),
        }


def _write_exact(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    path.write_bytes(data)


def build_external_suite(
    root: str | Path,
    *,
    suite_id: str,
    split: str,
    tokenizer_name: str,
    vocab_size: int,
    sources: Mapping[str, Sequence[Sequence[int]]],
    manifest: Mapping[str, object],
    scorer_id: str,
) -> ExternalSuite:
    """Write the suite source, validate its manifest, and pin every digest.

    The manifest must itself be a canonical dataset manifest (see
    ``hagi.data.artifacts.validate_manifest``) and is stored next to the suite
    so the manifest digest is a second, independent pin on the same data.
    """
    base = Path(root)
    source_bytes = suite_payload(
        suite_id=suite_id,
        split=split,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        sources=sources,
    )
    manifest_bytes = canonical_json_bytes(dict(manifest))
    _write_exact(base / f"{suite_id}-{split}.suite.json", source_bytes)
    _write_exact(base / f"{suite_id}-{split}.manifest.json", manifest_bytes)
    protocol = protocol_payload(scorer_id, vocab_size=vocab_size)
    parsed = parse_suite_source(source_bytes)
    return ExternalSuite(
        suite_id=suite_id,
        split=split,
        source_path=str(base / f"{suite_id}-{split}.suite.json"),
        source_sha256=sha256_bytes(source_bytes),
        manifest_path=str(base / f"{suite_id}-{split}.manifest.json"),
        manifest_sha256=sha256_bytes(manifest_bytes),
        tokenizer_name=tokenizer_name,
        protocol_sha256=sha256_bytes(protocol),
        protocol_payload=protocol,
        scorer_id=scorer_id,
        row_ids_sha256=parsed.row_ids_sha256,
    )


def parse_suite_source(data: bytes) -> ExternalSuiteSource:
    """Parse and self-verify suite bytes, including every row digest."""
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("external suite source is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SUITE_SCHEMA:
        raise ValueError("invalid external suite schema")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported external suite schema version")
    vocab_size = payload.get("vocab_size")
    if type(vocab_size) is not int or not 4 <= vocab_size <= 262_144:
        raise ValueError("external suite vocab_size is invalid")
    for label in ("suite_id", "split", "tokenizer_name"):
        value = payload.get(label)
        if not isinstance(value, str) or not value:
            raise ValueError(f"external suite {label} is invalid")
    rows = payload.get("rows")
    digests = payload.get("row_ids_sha256")
    if (
        not isinstance(rows, dict)
        or not rows
        or not isinstance(digests, dict)
        or set(rows) != set(digests)
    ):
        raise ValueError("external suite rows are invalid")
    ordered: list[str] = []
    flat: list[list[int]] = []
    source_names: list[str] = []
    for source in sorted(rows):
        source_rows = rows[source]
        source_digests = digests[source]
        if (
            not isinstance(source_rows, list)
            or not source_rows
            or not isinstance(source_digests, list)
            or len(source_rows) != len(source_digests)
        ):
            raise ValueError("external suite source rows are invalid")
        for entry, digest in zip(source_rows, source_digests, strict=True):
            if not isinstance(entry, dict) or set(entry) != {"token_ids"}:
                raise ValueError("external suite row entry is invalid")
            ids = entry["token_ids"]
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(value) is not int for value in ids)
                or any(value < 0 or value >= vocab_size for value in ids)
            ):
                raise ValueError("external suite row tokens are invalid")
            if not isinstance(digest, str) or row_digest(ids) != digest:
                raise ValueError("external suite row digest mismatch")
            ordered.append(digest)
            flat.append(ids)
            source_names.append(source)
    if len(set(ordered)) != len(ordered):
        raise ValueError("external suite row digests must be distinct")
    return ExternalSuiteSource(
        suite_id=payload["suite_id"],
        split=payload["split"],
        tokenizer_name=payload["tokenizer_name"],
        vocab_size=vocab_size,
        row_ids_sha256=tuple(ordered),
        rows=tuple(tuple(ids) for ids in flat),
        row_sources=tuple(source_names),
    )


@dataclass(frozen=True)
class ExternalSuiteSource:
    """Verified in-memory view of a pinned suite file."""

    suite_id: str
    split: str
    tokenizer_name: str
    vocab_size: int
    row_ids_sha256: tuple[str, ...]
    rows: tuple[tuple[int, ...], ...]
    row_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.row_ids_sha256) != len(self.rows) or not self.rows:
            raise ValueError("external suite source rows and digests must align")
        if len(self.row_sources) != len(self.rows):
            raise ValueError("external suite source names and rows must align")


def verified_suite_source(suite: ExternalSuite) -> ExternalSuiteSource:
    """Read, digest-verify, and manifest-verify the pinned suite on disk."""
    if not isinstance(suite, ExternalSuite):
        raise TypeError("suite must be an ExternalSuite")
    data = _verified_bytes(suite.source_path, suite.source_sha256, "external suite")
    manifest_bytes = _verified_bytes(
        suite.manifest_path, suite.manifest_sha256, "external suite manifest"
    )
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("external suite manifest is unreadable") from exc
    validate_manifest(manifest)
    parsed = parse_suite_source(data)
    if (
        parsed.suite_id != suite.suite_id
        or parsed.split != suite.split
        or parsed.tokenizer_name != suite.tokenizer_name
        or parsed.row_ids_sha256 != suite.row_ids_sha256
    ):
        raise ValueError("external suite identity mismatch")
    protocol = protocol_payload(suite.scorer_id, vocab_size=parsed.vocab_size)
    if protocol != suite.protocol_payload:
        raise ValueError("external suite protocol mismatch")
    return parsed


def _verified_bytes(path: str | Path, expected: str, label: str) -> bytes:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if sha256_bytes(data) != expected:
        raise ValueError(f"{label} digest mismatch")
    return data


def _metric(metric: ExternalSourceMetric) -> dict[str, object]:
    """Mirror ``recursive._metric`` so verdict payloads compare directly."""
    return {
        "source_id": metric.source_id,
        "scored_rows": metric.scored_rows,
        "exact_ce": metric.exact_ce,
        "row_ids_sha256": metric.row_ids_sha256,
    }


def load_checkpoint(
    checkpoint_path: str | Path, checkpoint_sha256: str, *, device: str = "cpu"
) -> tuple[torch.nn.Module, object]:
    """Validate strict checkpoint bytes and rebuild the model (see real_cycle)."""
    _digest(checkpoint_sha256, "checkpoint_sha256")
    data = _verified_bytes(checkpoint_path, checkpoint_sha256, "checkpoint")
    try:
        payload = torch.load(io.BytesIO(data), map_location=device, weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint snapshot is unreadable") from exc
    required = {"format_version", "model", "config", "completed_steps"}
    optional = {"optimizer"}
    if (
        not isinstance(payload, dict)
        or set(payload) - (required | optional)
        or required - set(payload)
        or payload["format_version"] != CHECKPOINT_FORMAT_VERSION
        or not isinstance(payload["model"], dict)
        or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in payload["model"].items()
        )
        or not isinstance(payload["config"], dict)
        or type(payload["completed_steps"]) is not int
        or payload["completed_steps"] < 0
    ):
        raise ValueError("checkpoint snapshot is invalid")
    cfg = config_from_dict(payload["config"])
    model = build_model_from_payload(cfg, payload["model"], device=device)
    # NOTE: build_model_from_payload only instantiates the class; it does NOT
    # populate weights. Without this call every checkpoint scores as fresh
    # random init and the evaluation measures nothing about the checkpoint.
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, KeyError) as exc:
        raise ValueError("checkpoint weights do not match its declared config") from exc
    return model, cfg


def score_checkpoint(
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    suite: ExternalSuite,
    *,
    device: str = "cpu",
) -> dict[str, object]:
    """Score one checkpoint on the external suite; return a digested verdict.

    Raises on any missing, corrupt, or tampered input. The returned verdict is
    descriptive only: it reports measured CE and never asserts promotion.
    """
    parsed = verified_suite_source(suite)
    model, cfg = load_checkpoint(checkpoint_path, checkpoint_sha256, device=device)
    vocab_size = int(getattr(cfg.model, "vocab_size"))
    if vocab_size != parsed.vocab_size:
        raise ValueError("checkpoint vocabulary does not match the external suite")
    total_ce = 0.0
    total_rows = 0
    per_row: list[dict[str, object]] = []
    for index, ids in enumerate(parsed.rows):
        result = evaluate_packed_tokens(model, cfg, list(ids), device=device)
        exact_ce = float(result["exact_ce"])
        _finite_nonnegative(exact_ce, "exact_ce")
        per_row.append(
            {
                "source_id": parsed.row_sources[index],
                "row_id_sha256": parsed.row_ids_sha256[index],
                "exact_ce": exact_ce,
                "scored_tokens": int(result["scored_token_count"]),
            }
        )
        total_ce += exact_ce
        total_rows += 1
    macro_ce = total_ce / total_rows
    evidence = {
        "schema": "hagi_external_evidence_v1",
        "suite_binding_sha256": suite.binding_digest,
        "checkpoint_sha256": checkpoint_sha256,
        "scorer_id": suite.scorer_id,
        "protocol_sha256": suite.protocol_sha256,
        "macro_exact_ce": macro_ce,
        "scored_rows": total_rows,
        "rows": per_row,
    }
    return {
        "schema": VERDICT_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "suite_binding_sha256": suite.binding_digest,
        "macro_exact_ce": macro_ce,
        "scored_rows": total_rows,
        "source_metrics": tuple(_metric(m) for m in _source_metrics(per_row)),
        "evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
    }


def _source_metrics(per_row: list[dict[str, object]]) -> tuple[ExternalSourceMetric, ...]:
    """Group per-row scores into one metric per source, in source-name order."""
    groups: dict[str, list[dict[str, object]]] = {}
    for row in per_row:
        groups.setdefault(str(row["source_id"]), []).append(row)
    metrics: list[ExternalSourceMetric] = []
    for key in sorted(groups):
        rows = groups[key]
        mean = sum(float(row["exact_ce"]) for row in rows) / len(rows)
        metrics.append(
            ExternalSourceMetric(
                key,
                len(rows),
                mean,
                sha256_bytes(canonical_json_bytes([row["row_id_sha256"] for row in rows])),
            )
        )
    return tuple(metrics)


def select_champion(
    champion_path: str | Path,
    champion_sha256: str,
    challenger_path: str | Path,
    challenger_sha256: str,
    suite: ExternalSuite,
    *,
    margin: float | None,
    max_source_regression: float = 0.0,
    device: str = "cpu",
) -> dict[str, object]:
    """Promote the challenger only on external evidence; else keep champion.

    ``margin`` is mandatory: a missing margin fails closed, because the paired
    standard error of the current holdout (~0.0251 nats) exceeds any margin a
    caller might silently default to. ``max_source_regression`` is the
    per-source regression ceiling; the default of 0.0 forbids regressions.
    No bytes are written by this function.
    """
    if margin is None:
        raise ValueError("margin must be supplied explicitly; refusing to default")
    _finite_nonnegative(margin, "margin")
    _finite_nonnegative(max_source_regression, "max_source_regression")
    if champion_sha256 == challenger_sha256:
        raise ValueError("challenger must differ from the champion checkpoint")
    champion = score_checkpoint(champion_path, champion_sha256, suite, device=device)
    challenger = score_checkpoint(challenger_path, challenger_sha256, suite, device=device)
    if champion["suite_binding_sha256"] != challenger["suite_binding_sha256"]:
        raise ValueError("champion and challenger were scored on different suites")
    gain = float(champion["macro_exact_ce"]) - float(challenger["macro_exact_ce"])
    worst = _worst_source_regression(champion, challenger)
    accepted = gain > margin and worst <= max_source_regression
    payload = {
        "schema": PROMOTION_SCHEMA,
        "decision": "promote" if accepted else "retain_champion",
        "champion_checkpoint_sha256": champion_sha256,
        "challenger_checkpoint_sha256": challenger_sha256,
        "suite_binding_sha256": suite.binding_digest,
        "scorer_id": suite.scorer_id,
        "margin": margin,
        "max_source_regression": max_source_regression,
        "champion_macro_exact_ce": champion["macro_exact_ce"],
        "challenger_macro_exact_ce": challenger["macro_exact_ce"],
        "external_gain": gain,
        "worst_source_regression": worst,
        "champion_evidence_sha256": champion["evidence_sha256"],
        "challenger_evidence_sha256": challenger["evidence_sha256"],
    }
    return {**payload, "verdict_sha256": sha256_bytes(canonical_json_bytes(payload))}


def _worst_source_regression(
    champion: dict[str, object], challenger: dict[str, object]
) -> float:
    left = {str(m["source_id"]): float(m["exact_ce"]) for m in champion["source_metrics"]}
    right = {str(m["source_id"]): float(m["exact_ce"]) for m in challenger["source_metrics"]}
    if set(left) != set(right):
        raise ValueError("champion and challenger source metrics do not align")
    return max(right[key] - left[key] for key in sorted(left))


def apply_promotion(
    champion_path: str | Path,
    champion_sha256: str,
    challenger_path: str | Path,
    challenger_sha256: str,
    suite: ExternalSuite,
    *,
    margin: float | None,
    max_source_regression: float = 0.0,
    device: str = "cpu",
) -> dict[str, object]:
    """Promote only when the external verdict says so; champion bytes are
    rewritten exclusively on a ``promote`` decision, never on rejection."""
    verdict = select_champion(
        champion_path,
        champion_sha256,
        challenger_path,
        challenger_sha256,
        suite,
        margin=margin,
        max_source_regression=max_source_regression,
        device=device,
    )
    if verdict["decision"] == "promote":
        _write_exact(
            Path(champion_path), _verified_bytes(challenger_path, challenger_sha256, "challenger")
        )
        if sha256_file(champion_path) != challenger_sha256:
            raise ValueError("promoted champion bytes do not match the challenger")
    return verdict


__all__ = [
    "ExternalSourceMetric",
    "ExternalSuite",
    "ExternalSuiteSource",
    "apply_promotion",
    "build_external_suite",
    "load_checkpoint",
    "parse_suite_source",
    "protocol_payload",
    "row_digest",
    "score_checkpoint",
    "select_champion",
    "suite_payload",
    "verified_suite_source",
]
