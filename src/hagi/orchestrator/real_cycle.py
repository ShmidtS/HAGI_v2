"""Deterministic, CPU-first binding for one recursive F3 generation."""
from __future__ import annotations

import copy
import io
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hagi.config import CHECKPOINT_FORMAT_VERSION, Config, validate_config
from hagi.data.artifacts import load_published_artifact
from hagi.model.merge import (
    CROSS_PARENT_TRANSFORMS,
    build_model_from_payload,
    merge_recursive_f3,
    state_key_digest,
)
from hagi.model.model import HAGI
from hagi.orchestrator.evaluation import evaluate_packed_tokens
from hagi.orchestrator.recursive import (
    CandidateArtifact,
    CandidateContext,
    ChildArtifact,
    ChildContext,
    ChildPlan,
    EvaluationContext,
    EvaluationResult,
    GenerationRequest,
    GenerationResult,
    HoldoutContract,
    SelfImprovementLedger,
    SourceMetric,
    SourceSpan,
    _contour_state_digest,
    _derived_base_digest,
    _ledger_payload,
    run_generation,
)
from hagi.orchestrator.state import (
    GrowthRunStore,
    ParentPointer,
    TrustedParentToken,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from hagi.train.checkpoint import (
    config_from_dict,
    config_to_dict,
    load_payload,
    save_checkpoint,
)
from hagi.train.loop import Trainer
from hagi.train.self_improve import self_improve

OWNER_ID = "recursive-growth"
BANKING77_MANIFEST_SHA256 = "44d50edd994e7c32f30066a26052e15f902c74b79a7498ba60f475c76b453f3b"
SYNTHETIC_V3_SEED = 301097
# Retained only to bind the invalidated v2 preregistration artifact.
SYNTHETIC_V2_SEED = 416114
_SEED_STRIDE = 1009
_SELF_IMPROVE_OFFSET = 9001
_MAX_STEPS = 8


def canonical_digest(value: Any) -> str:
    """Return the digest of strict, sorted, compact JSON bytes."""
    return sha256_bytes(canonical_json_bytes(value))


def config_digest(config: Config) -> str:
    return canonical_digest(config_to_dict(config))


def bounded_token_batches(
    token_ids: Sequence[int],
    *,
    batch_size: int = 2,
    sequence_length: int = 16,
    max_batches: int = 1,
) -> list[dict[str, torch.Tensor]]:
    """Create a strictly bounded, deterministic Trainer micro-batch list."""
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError("batch_size must be in [1, 8]")
    if type(sequence_length) is not int or not 2 <= sequence_length <= 64:
        raise ValueError("sequence_length must be in [2, 64]")
    if type(max_batches) is not int or not 1 <= max_batches <= 8:
        raise ValueError("max_batches must be in [1, 8]")
    ids = [int(value) for value in token_ids]
    if len(ids) < 2 or any(value < 0 for value in ids):
        raise ValueError("token stream must contain at least two non-negative ids")
    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(ids) - 1, sequence_length - 1):
        chunk = ids[start : start + sequence_length]
        if len(chunk) < 2:
            break
        rows = [chunk for _ in range(batch_size)]
        batches.append(
            {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "targets": torch.tensor(rows, dtype=torch.long),
            }
        )
        if len(batches) == max_batches:
            break
    if not batches:
        raise ValueError("token stream produced no Trainer batches")
    return batches


def _base_config(max_steps: int, vocab_size: int = 128) -> Config:
    if type(max_steps) is not int or not 1 <= max_steps <= _MAX_STEPS:
        raise ValueError(f"max_steps must be in [1, {_MAX_STEPS}]")
    if type(vocab_size) is not int or vocab_size < 4 or vocab_size > 262_144:
        raise ValueError("vocab_size is outside the supported compact range")
    cfg = Config()
    cfg.model.vocab_size = vocab_size
    cfg.model.hidden_size = 8
    cfg.model.num_layers = 1
    cfg.model.attention.num_query_heads = 1
    cfg.model.attention.num_kv_heads = 1
    cfg.model.attention.head_dim = 8
    cfg.model.attention.max_seq_len = 32
    cfg.model.ffn.intermediate_size = 8
    cfg.model.ffn.multiple_of = 8
    cfg.model.embedding.conv_kernel = 1
    cfg.model.embedding.tie_lm_head = False
    cfg.model.ternary.enabled = False
    cfg.model.adapters.enabled = False
    cfg.model.cortex.enabled = False
    cfg.model.decision.enabled = False
    cfg.model.loop_depth = 2
    cfg.train.batch_size = 1
    cfg.train.grad_accum_steps = 1
    cfg.train.data.seq_len = 16
    cfg.train.max_steps = max_steps
    cfg.train.adapt.freeze_base = False
    cfg.train.precision = "fp32"
    cfg.train.compile_model = False
    cfg.train.logging.exact_ce_interval = 0
    validate_config(cfg)
    return cfg


def _child_config(parent: Config) -> Config:
    cfg = copy.deepcopy(parent)
    cfg.model.adapters.enabled = False
    cfg.model.adapters.pyramid.enabled = False
    cfg.model.adapters.ttt_lora.enabled = False
    cfg.model.cortex.enabled = False
    cfg.model.decision.enabled = False
    cfg.train.adapt.freeze_base = False
    validate_config(cfg)
    return cfg


#: Cross-parent transform used when a caller does not name one. The legacy
#: staged F3 tree stays the default: it is what an existing run produced, and
#: the parent-preserving Q(pi/2) lift must be selected explicitly so no
#: checkpoint can silently change meaning.
_DEFAULT_CROSS_PARENT_TRANSFORM = "f3_tree"


def _target_config(parent: Config) -> Config:
    cfg = copy.deepcopy(parent)
    cfg.merge.enabled = True
    cfg.merge.n_experts = 3
    cfg.merge.mixer_type = "ternary_f3"
    cfg.merge.ternary_depth = parent.merge.ternary_depth + 1
    cfg.merge.ternary_tree_schema_version = 1
    cfg.merge.expert_weight_source = "effective_sparse"
    cfg.merge.mixer_hadamard_groups = [3]
    cfg.merge.expert_hidden = parent.model.hidden_size
    cfg.model.hidden_size *= 3
    cfg.model.attention.num_query_heads *= 3
    cfg.model.attention.num_kv_heads *= 3
    cfg.model.ffn.intermediate_size *= 3
    cfg.model.adapters.enabled = True
    cfg.model.adapters.pyramid.enabled = True
    cfg.model.adapters.pyramid.levels = (1,)
    cfg.model.adapters.pyramid.residual_scale = 1.0
    cfg.model.adapters.ttt_lora.enabled = False
    cfg.model.cortex.enabled = False
    cfg.model.decision.enabled = False
    cfg.train.adapt.freeze_base = True
    cfg.model.head.sampled_softmax_k = 0
    validate_config(cfg)
    return cfg


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable file conflict: {path}")
        return
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"immutable file conflict: {path}")


def result_data_provenance(result: GenerationResult) -> tuple[str, str]:
    """Return data/tokenizer provenance from the owner's terminal evidence."""
    if not isinstance(result, GenerationResult):
        raise TypeError("result must be a GenerationResult")
    try:
        report = json.loads(Path(result.report_path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("terminal report is unreadable") from exc
    if (
        not isinstance(report, dict)
        or report.get("generation_id") != result.generation_id
        or report.get("decision") != result.decision
        or not isinstance(report.get("holdout_evidence_path"), str)
        or not isinstance(report.get("holdout_evidence_sha256"), str)
    ):
        raise ValueError("terminal result provenance binding is invalid")
    evidence_data = _verified_bytes(
        report["holdout_evidence_path"],
        report["holdout_evidence_sha256"],
        "holdout evidence",
    )
    try:
        evidence = json.loads(evidence_data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("holdout evidence is unreadable") from exc
    bindings = evidence.get("bindings") if isinstance(evidence, dict) else None
    data_sha = bindings.get("data_manifest_sha256") if isinstance(bindings, dict) else None
    tokenizer = bindings.get("tokenizer_name") if isinstance(bindings, dict) else None
    if (
        evidence.get("generation_id") != result.generation_id
        or evidence.get("decision") != result.decision
        or not isinstance(data_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", data_sha) is None
        or not isinstance(tokenizer, str)
        or not tokenizer
    ):
        raise ValueError("holdout evidence provenance is invalid")
    return data_sha, tokenizer


def _parent_manifest(path: Path, config_sha256: str, seed: int, checkpoint_sha256: str) -> Path:
    manifest = {
        "schema": "recursive_growth_parent_manifest_v1",
        "schema_version": 1,
        "config_sha256": config_sha256,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "depth": 0,
    }
    _write_exact(path, canonical_json_bytes(manifest))
    return path


def _deterministic_checkpoint(path: Path, model: HAGI, cfg: Config) -> str:
    """Write strict checkpoint bytes from an in-memory archive name."""
    stream = io.BytesIO()
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model": model.state_dict(),
            "config": config_to_dict(cfg),
            "completed_steps": 0,
        },
        stream,
    )
    payload = stream.getvalue()
    checkpoint = path / "step-0000000.pt"
    _write_exact(checkpoint, payload)
    return sha256_bytes(payload)


def build_deterministic_parent(
    output: str | Path, *, seed: int, max_steps: int = 1, vocab_size: int = 128
) -> tuple[ParentPointer, Path]:
    """Build and trust a real depth-zero checkpoint from config plus seed."""
    root = Path(output).resolve()
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1 - _SELF_IMPROVE_OFFSET:
        raise ValueError("seed is outside the deterministic range")
    root.mkdir(parents=True, exist_ok=True)
    cfg = _base_config(max_steps, vocab_size)
    torch.manual_seed(seed)
    model = HAGI(copy.deepcopy(cfg))
    checkpoint = root / "parent" / "step-0000000.pt"
    checkpoint_sha = _deterministic_checkpoint(root / "parent", model, cfg)
    cfg_sha = config_digest(cfg)
    manifest = _parent_manifest(root / "parent-manifest.json", cfg_sha, seed, checkpoint_sha)
    pointer = ParentPointer(
        "depth-zero",
        checkpoint_sha,
        sha256_file(manifest),
        sha256_bytes(b"recursive-growth-bootstrap-report"),
        owner_id=OWNER_ID,
    )
    return pointer, checkpoint


def _holdout_contract(
    path: Path,
    *,
    vocab_size: int,
    streams: Mapping[str, Sequence[int]],
    row_ids: Mapping[str, str],
    data_manifest_sha256: str,
    tokenizer_name: str,
) -> HoldoutContract:
    payload = {
        "schema": "recursive_f3_packed_holdout_v1",
        "schema_version": 1,
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "sources": {source: list(values) for source, values in streams.items()},
        "row_ids_sha256": dict(row_ids),
    }
    data = canonical_json_bytes(payload)
    _write_exact(path, data)
    protocol = canonical_json_bytes(
        {
            "kind": "recursive_f3_holdout_v1",
            "schema_version": 1,
            "scoring": "evaluate_packed_tokens/full_alphabet/exact_ce",
        }
    )
    return HoldoutContract(
        str(path),
        sha256_bytes(data),
        data_manifest_sha256,
        sha256_bytes(protocol),
        protocol,
        (row_ids["A"], row_ids["B"], row_ids["C"]),
        tokenizer_name,
    )


def _synthetic_streams() -> dict[str, list[int]]:
    """Return the preregistered shared-period synthetic A/B/C streams."""
    return {
        source: [
            2 + source_index * 11 + position % 23
            for position in range(48)
        ]
        for source_index, source in enumerate(("A", "B", "C"))
    }


def _synthetic_holdout(
    path: Path,
    vocab_size: int,
    training_tokens: Mapping[str, Sequence[int]],
) -> tuple[HoldoutContract, tuple[SourceSpan, ...]]:
    full_streams = _synthetic_streams()
    for source in ("A", "B", "C"):
        if list(training_tokens[source]) != full_streams[source][:24]:
            raise ValueError(
                f"synthetic training slice does not match preregistered source {source}"
            )
    streams = {
        source: full_streams[source][24:48] for source in ("A", "B", "C")
    }
    row_ids = {
        source: _token_bytes_digest(values) for source, values in streams.items()
    }
    contract = _holdout_contract(
        path,
        vocab_size=vocab_size,
        streams=streams,
        row_ids=row_ids,
        data_manifest_sha256=sha256_bytes(
            b"recursive-growth-synthetic-data-v2"
        ),
        tokenizer_name="synthetic-packed-v1",
    )
    offset = 0
    spans: list[SourceSpan] = []
    for source in ("A", "B", "C"):
        values = training_tokens[source]
        spans.append(
            SourceSpan(
                source,
                offset,
                offset + len(values),
                _token_bytes_digest(values),
            )
        )
        offset += len(values)
    return contract, tuple(spans)


def _holdout_payload(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("packed holdout is unreadable") from exc
    sources = payload.get("sources") if isinstance(payload, dict) else None
    row_ids = payload.get("row_ids_sha256") if isinstance(payload, dict) else None
    vocab_size = payload.get("vocab_size") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "recursive_f3_packed_holdout_v1"
        or payload.get("schema_version") != 1
        or type(vocab_size) is not int
        or not 4 <= vocab_size <= 262_144
        or not isinstance(sources, dict)
        or set(sources) != {"A", "B", "C"}
        or not isinstance(row_ids, dict)
        or set(row_ids) != {"A", "B", "C"}
    ):
        raise ValueError("invalid packed holdout schema")
    for source in ("A", "B", "C"):
        values = sources[source]
        row_id = row_ids[source]
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int for value in values)
            or any(value < 0 or value >= vocab_size for value in values)
            or not isinstance(row_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", row_id) is None
            or _token_bytes_digest(values) != row_id
        ):
            raise ValueError("invalid packed holdout contract")
    if len(set(row_ids.values())) != 3:
        raise ValueError("packed holdout source digests must be distinct")
    return payload


def _score_checkpoint_bytes(
    checkpoint_data: bytes, holdout: Mapping[str, Any], device: str
) -> dict[str, SourceMetric]:
    try:
        payload = torch.load(io.BytesIO(checkpoint_data), map_location=device, weights_only=True)
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
    metrics: list[SourceMetric] = []
    for source in ("A", "B", "C"):
        result = evaluate_packed_tokens(
            model, cfg, holdout["sources"][source], device=device
        )
        metrics.append(
            SourceMetric(
                source,
                int(result["scored_token_count"]),
                float(result["exact_ce"]),
                holdout["row_ids_sha256"][source],
            )
        )
    return {metric.source_id: metric for metric in metrics}


def _verified_bytes(path: str | Path, expected: str, label: str) -> bytes:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if sha256_bytes(data) != expected:
        raise ValueError(f"{label} digest mismatch")
    return data


def path_only_evaluator(
    context: EvaluationContext, *, device: str = "cpu"
) -> EvaluationResult:
    """Verify and score one immutable snapshot of every frozen input."""
    if not isinstance(context, EvaluationContext):
        raise TypeError("evaluator requires EvaluationContext")
    holdout_data = _verified_bytes(
        context.holdout.path, context.holdout.sha256, "holdout"
    )
    parent_data = _verified_bytes(
        context.request.parent_checkpoint_path,
        context.parent.checkpoint_sha256,
        "parent checkpoint",
    )
    candidate_data = _verified_bytes(
        context.candidate.checkpoint_path,
        context.candidate.checkpoint_sha256,
        "candidate checkpoint",
    )
    _verified_bytes(
        context.candidate.manifest_path,
        context.candidate.manifest_sha256,
        "candidate manifest",
    )
    holdout = _holdout_payload(holdout_data)
    if holdout.get("tokenizer") != context.holdout.tokenizer_name:
        raise ValueError("holdout tokenizer metadata mismatch")
    if tuple(
        holdout["row_ids_sha256"][source] for source in ("A", "B", "C")
    ) != context.holdout.row_ids_sha256:
        raise ValueError("holdout row identity metadata mismatch")
    parent = _score_checkpoint_bytes(parent_data, holdout, device)
    candidate = _score_checkpoint_bytes(candidate_data, holdout, device)
    return EvaluationResult(
        tuple(parent[source] for source in ("A", "B", "C")),
        tuple(candidate[source] for source in ("A", "B", "C")),
        context.holdout.tokenizer_name,
        context.holdout.data_manifest_sha256,
        context.holdout.protocol_sha256,
        context.parent.checkpoint_sha256,
        context.candidate.checkpoint_sha256,
        context.candidate.manifest_sha256,
    )


def _build_child(
    context: ChildContext,
    root: Path,
    tokens: Mapping[str, Sequence[int]],
    max_steps: int,
) -> ChildArtifact:
    parent_payload = load_payload(context.parent_checkpoint_path)
    parent_cfg = config_from_dict(parent_payload["config"])
    cfg = _child_config(parent_cfg)
    try:
        stream = tokens[context.source_id]
    except KeyError as exc:
        raise ValueError(f"child {context.name} has no bound token stream") from exc
    _verify_token_stream(context.span, stream)
    parent_state = {
        key: value
        for key, value in parent_payload["model"].items()
        if not key.endswith(".adapters.pyramid.scale")
    }
    model = build_model_from_payload(cfg, parent_state)
    # A bounded child run of a few steps cannot spend its whole budget inside the
    # scheduler's warmup, where the effective learning rate is a fraction of a
    # percent of the base rate. Left as configured (warmup 2000) every child
    # would keep the parent's weights almost exactly, and the merge test would
    # have no signal to resolve. This mirrors the same fix already applied to
    # the one-step self-improvement update.
    cfg.train.schedule.warmup_steps = 0
    trainer = Trainer(model, cfg)
    for step in range(max_steps):
        offset = (context.seed + step * 3) % max(1, len(stream) - 1)
        batch_stream = list(stream[offset : offset + 16])
        if len(batch_stream) < 2:
            batch_stream.extend(stream[: 2 - len(batch_stream)])
        result = trainer.train_step(bounded_token_batches(batch_stream))
        if not result.get("update_applied", False):
            raise RuntimeError(f"child {context.name} produced no applied update")
    checkpoint = save_checkpoint(
        trainer.model, cfg, trainer.step, root / context.name, keep_last=1, optimizer=trainer.optimizer
    )
    config_sha = config_digest(cfg)
    checkpoint_sha = sha256_file(checkpoint)
    manifest_payload = {
        "schema_version": 1,
        "generation_id": context.generation_id,
        "name": context.name,
        "source_id": context.source_id,
        "seed": context.seed,
        "span": {
            "source_id": context.span.source_id,
            "start": context.span.start,
            "end": context.span.end,
            "source_manifest_sha256": context.span.source_manifest_sha256,
        },
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "parent_checkpoint_sha256": context.parent_checkpoint_sha256,
        "protocol_sha256": context.protocol_sha256,
    }
    manifest = root / context.name / "manifest.json"
    _write_exact(manifest, canonical_json_bytes(manifest_payload))
    return ChildArtifact(
        context.name,
        context.source_id,
        context.seed,
        context.span,
        str(checkpoint),
        checkpoint_sha,
        str(manifest),
        sha256_file(manifest),
        config_sha,
        context.parent_checkpoint_sha256,
        context.protocol_sha256,
    )


def _build_candidate(
    context: CandidateContext,
    root: Path,
    prompt_ids: Sequence[int] | None = None,
    *,
    cross_parent_transform: str = _DEFAULT_CROSS_PARENT_TRANSFORM,
) -> CandidateArtifact:
    """Build a generation candidate from three children.

    ``cross_parent_transform`` names the transform that mixes the three parent
    streams of this recursive merge. It is passed by name, not inferred from
    the parent's depth, and defaults to the legacy staged F3 tree so an
    existing run reproduces byte-for-byte. The name is recorded in the
    candidate's provenance, so a checkpoint written under one transform can
    never be replayed under the other.
    """
    parent_payload = load_payload(context.parent_checkpoint_path)
    parent_cfg = config_from_dict(parent_payload["config"])
    child_payloads = [load_payload(child.checkpoint_path) for child in context.children]
    target_cfg = _target_config(parent_cfg)
    # The transform is selected explicitly, never inferred: an existing run
    # must reproduce byte-for-byte under the legacy default, and a checkpoint
    # written under one transform can never be replayed under the other.
    target_cfg.merge.ternary_lift_mode = cross_parent_transform
    validate_config(target_cfg)
    torch.manual_seed(context.base_seed)
    model = merge_recursive_f3(
        target_cfg,
        [payload["model"] for payload in child_payloads],
        child_configs=tuple(config_from_dict(payload["config"]) for payload in child_payloads),
        parent_depth=parent_cfg.merge.ternary_depth,
        expert_weight_source=target_cfg.merge.expert_weight_source,
        cross_parent_transform=cross_parent_transform,
    )
    contour = sorted(
        key for key in model.state_dict() if key.endswith(".adapters.pyramid.scale")
    )
    if not contour:
        raise ValueError("candidate has no fresh pyramid contour")
    with torch.no_grad():
        for key in contour:
            model.state_dict()[key].zero_()
    contour_before = {
        key: model.state_dict()[key].detach().clone() for key in contour
    }
    applied_updates = 0
    trainer = None
    if prompt_ids is not None:
        # A one-step bounded update cannot spend its only iteration at the
        # scheduler's warmup step, where AdamW's effective LR is exactly zero.
        target_cfg.train.schedule.warmup_steps = 0
        torch.manual_seed(context.self_improve_seed)
        trainer = Trainer(model, target_cfg)
        stats = self_improve(
            model,
            target_cfg,
            [int(value) for value in prompt_ids],
            mode="gradient",
            max_iterations=1,
            n_new_tokens=8,
            patience=1,
            kl_max=1.0,
            trainer=trainer,
        )
        contour_changed = any(
            not torch.equal(model.state_dict()[key], contour_before[key])
            for key in contour
        )
        applied_updates = (
            sum(int(iteration.update_applied) for iteration in stats.iterations)
            if contour_changed
            else 0
        )
    checkpoint = save_checkpoint(
        model,
        target_cfg,
        applied_updates,
        root / "candidate",
        keep_last=1,
        optimizer=trainer.optimizer if trainer is not None and applied_updates else None,
    )
    saved_state = load_payload(checkpoint)["model"]
    # The declared transform must be the one the model actually applied. Reading
    # it from the assembled model keeps the manifest honest under any
    # cross-parent transform instead of re-deriving the legacy default.
    tree = model.target_tree
    source_a = context.children[0].span
    prompt_span = SourceSpan("A", source_a.start, source_a.start + 1, source_a.source_manifest_sha256)
    ledger = SelfImprovementLedger(
        context.self_improve_seed,
        "A",
        prompt_span,
        sha256_bytes(b"synthetic-prompt-tokens"),
        sha256_bytes(b"synthetic-prompt-text"),
        1,
        tuple(contour),
        applied_updates,
        _derived_base_digest(saved_state),
        _contour_state_digest(saved_state),
    )
    metadata = {
        "schema_version": 1,
        "generation_id": context.generation_id,
        "parent_checkpoint_sha256": context.parent_checkpoint_sha256,
        "parent_config_sha256": config_digest(parent_cfg),
        "protocol_sha256": context.protocol_sha256,
        "candidate_checkpoint_sha256": sha256_file(checkpoint),
        "candidate_config_sha256": config_digest(target_cfg),
        "candidate_state_key_digest": state_key_digest(saved_state),
        "ternary_depth": target_cfg.merge.ternary_depth,
        "leaf_hidden": tree.leaf_hidden,
        "coordinate_layout": tree.coordinate_layout,
        "transform_digest": tree.transform_digest,
        "weight_source": target_cfg.merge.expert_weight_source,
        "logit_scale_source": float(model.logit_scale_source),
        "logit_scale_target": float(model.logit_scale_target),
        "decision": "candidate",
        "self_improve": _ledger_payload(ledger),
    }
    manifest = root / "candidate" / "manifest.json"
    _write_exact(manifest, canonical_json_bytes(metadata))
    return CandidateArtifact(
        context.generation_id,
        context.parent_checkpoint_sha256,
        metadata["parent_config_sha256"],
        context.protocol_sha256,
        str(checkpoint),
        metadata["candidate_checkpoint_sha256"],
        str(manifest),
        sha256_file(manifest),
        metadata["candidate_config_sha256"],
        metadata["candidate_state_key_digest"],
        metadata["ternary_depth"],
        metadata["leaf_hidden"],
        metadata["coordinate_layout"],
        metadata["transform_digest"],
        metadata["weight_source"],
        metadata["logit_scale_source"],
        metadata["logit_scale_target"],
        tuple(child.checkpoint_sha256 for child in context.children),
        tuple(child.config_sha256 for child in context.children),
        ledger,
    )


def _next_generation_id(pointer: ParentPointer) -> str:
    if not isinstance(pointer, ParentPointer):
        raise TypeError("parent must be a ParentPointer")
    if pointer.generation_id == "depth-zero":
        return "generation-1"
    match = re.fullmatch(r"generation-([1-9][0-9]*)", pointer.generation_id)
    if match is None:
        raise ValueError("current parent generation id is invalid")
    return f"generation-{int(match.group(1)) + 1}"


def _load_committed_parent(store: GrowthRunStore, pointer: ParentPointer) -> Path:
    """Resolve the exact parent checkpoint bound by the committed pointer."""
    if pointer.generation_id == "depth-zero":
        checkpoint = store.root.parent / "parent" / "step-0000000.pt"
        _verified_bytes(
            checkpoint, pointer.checkpoint_sha256, "depth-zero parent checkpoint"
        )
        return checkpoint
    report_path = store.root / pointer.generation_id / "report.json"
    report_data = _verified_bytes(
        report_path, pointer.report_sha256, "terminal report"
    )
    try:
        report = json.loads(report_data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("terminal report is unreadable") from exc
    required_paths = {"candidate_checkpoint_path", "manifest_path"}
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 2
        or report.get("state") != "terminal"
        or report.get("decision") != "accepted"
        or report.get("generation_id") != pointer.generation_id
        or report.get("candidate_checkpoint_sha256") != pointer.checkpoint_sha256
        or report.get("manifest_sha256") != pointer.manifest_sha256
        or not required_paths.issubset(report)
    ):
        raise ValueError("committed terminal report binding is invalid")
    store_root = store.root.resolve()
    artifacts = {
        "candidate": (
            Path(report["candidate_checkpoint_path"]),
            pointer.checkpoint_sha256,
        ),
        "manifest": (Path(report["manifest_path"]), pointer.manifest_sha256),
    }
    for label, (path, expected) in artifacts.items():
        if (
            not isinstance(path, Path)
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(store_root)
        ):
            raise ValueError(f"committed {label} path is invalid")
        _verified_bytes(path, expected, f"committed {label}")
    return artifacts["candidate"][0]


def _explicit_request_is_current(
    store: GrowthRunStore, pointer: ParentPointer, request: GenerationRequest
) -> bool:
    if (
        not isinstance(request, GenerationRequest)
        or pointer.generation_id == "depth-zero"
        or request.generation_id != pointer.generation_id
    ):
        return False
    report_data = _verified_bytes(
        store.root / pointer.generation_id / "report.json",
        pointer.report_sha256,
        "terminal report",
    )
    try:
        report = json.loads(report_data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("terminal report is unreadable") from exc
    return (
        isinstance(report, dict)
        and report.get("state") == "terminal"
        and report.get("decision") == "accepted"
        and report.get("generation_id") == pointer.generation_id
        and report.get("candidate_checkpoint_sha256") == pointer.checkpoint_sha256
        and report.get("manifest_sha256") == pointer.manifest_sha256
    )


def run_bounded_cycle(
    output: str | Path,
    *,
    seed: int,
    max_steps: int = 1,
    device: str = "cpu",
    evaluator: Callable[[EvaluationContext], EvaluationResult] | None = None,
    request: GenerationRequest | None = None,
    cross_parent_transform: str = _DEFAULT_CROSS_PARENT_TRANSFORM,
) -> GenerationResult:
    """Execute the next owner-authoritative synthetic recursive generation.

    ``cross_parent_transform`` is forwarded by name to the candidate builder so
    a preregistered experiment can compare merge transforms without editing the
    default. The default stays the legacy staged F3 tree, and the selected name
    is recorded in candidate provenance.
    """
    if type(seed) is not int or seed != SYNTHETIC_V3_SEED:
        raise ValueError(
            f"synthetic v3 requires preregistered seed {SYNTHETIC_V3_SEED}"
        )
    if device != "cpu":
        raise ValueError("this bounded adapter is CPU-first; device must be 'cpu'")
    if type(max_steps) is not int or not 1 <= max_steps <= _MAX_STEPS:
        raise ValueError(f"max_steps must be in [1, {_MAX_STEPS}]")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = GrowthRunStore(root / "store")
    pointer = store.read_parent()
    if pointer is None:
        pointer, parent_checkpoint = build_deterministic_parent(
            root, seed=seed, max_steps=max_steps
        )
        store.bootstrap_parent(
            pointer,
            owner_id=OWNER_ID,
            trusted_parent_token=TrustedParentToken(OWNER_ID, pointer.digest()),
        )
    else:
        parent_checkpoint = _load_committed_parent(store, pointer)
    training_tokens = {
        source: values[:24] for source, values in _synthetic_streams().items()
    }
    holdout, spans = _synthetic_holdout(
        root / "holdout.json", 128, training_tokens
    )
    derived_request = GenerationRequest(
        _next_generation_id(pointer),
        seed,
        tuple(
            ChildPlan(source, span)
            for source, span in zip(("A", "B", "C"), spans, strict=True)
        ),
        pointer,
        str(parent_checkpoint),
        holdout,
    )
    if request is None:
        selected_request = derived_request
    elif request == derived_request or _explicit_request_is_current(
        store, pointer, request
    ):
        selected_request = request
    else:
        raise ValueError("stale explicit request does not match committed lineage")
    build_root = root / "build" / selected_request.generation_id
    evaluate = evaluator or (lambda context: path_only_evaluator(context, device=device))
    return run_generation(
        selected_request,
        store,
        lambda context: _build_child(
            context, build_root, training_tokens, max_steps
        ),
        lambda context: _build_candidate(
            context, build_root, cross_parent_transform=cross_parent_transform
        ),
        evaluate,
        owner_id=OWNER_ID,
    )


def _packed_tokens(path: Path, vocab_size: int) -> list[int]:
    values = np.fromfile(path, dtype="<u4")
    if values.size < 2 or np.any(values >= vocab_size):
        raise ValueError("packed token shard is invalid")
    return [int(value) for value in values]


def _sealed_token_slice(
    path: Path, vocab_size: int, start: int, window: int, steps: int
) -> list[int]:
    """Read exactly the sealed ``[p, p + W*S + 1)`` slice of a token shard.

    The bound is checked against the real file length before any token is
    materialized, so an out-of-file span fails closed before it can reach a
    digest, an evaluator, or a training callback.
    """
    values = np.fromfile(path, dtype="<u4")
    if values.size < 2 or np.any(values >= vocab_size):
        raise ValueError("packed token shard is invalid")
    required = _sealed_span_budget(start, window, steps, int(values.size))
    return [int(value) for value in values[start : start + required]]


def _sealed_span_budget(start: int, window: int, steps: int, token_count: int) -> int:
    """Validate the sealed budget ``[p, p + W*S + 1)`` before any token is read.

    ``W`` is the training window length, ``S`` the bounded child training
    steps, and the trailing ``+1`` is the final target token. The bound is
    fail-closed: an out-of-file span is rejected before a slice, digest, or
    training callback observes the data.
    """
    for name, value in (("start", start), ("window", window), ("steps", steps)):
        if type(value) is not int or value < 0:
            raise ValueError(f"sealed span budget {name} must be a non-negative exact int")
    if type(token_count) is not int or token_count < 0:
        raise ValueError("sealed span budget token_count must be a non-negative exact int")
    if window == 0 or steps == 0:
        raise ValueError("sealed span budget requires a positive window and step count")
    required = window * steps + 1
    if start + required > token_count:
        raise ValueError(
            f"sealed span budget [{start}, {start + required}) escapes the "
            f"{token_count}-token source"
        )
    return required


def _token_bytes_digest(token_ids: Sequence[int]) -> str:
    """Hash the exact little-endian uint32 token-stream representation."""
    values = list(token_ids)
    if not values or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) > np.iinfo(np.uint32).max
        for value in values
    ):
        raise ValueError("token ids must be exact unsigned uint32 integers")
    try:
        packed = np.asarray([int(value) for value in values], dtype="<u4")
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("token ids must be exact unsigned uint32 integers") from exc
    return sha256_bytes(packed.tobytes())


def _verify_token_stream(span: SourceSpan, stream: Sequence[int]) -> None:
    if span.source_id not in ("A", "B", "C"):
        raise ValueError("source span id is invalid")
    if span.start < 0 or span.end <= span.start:
        raise ValueError("source span bounds are invalid")
    if len(stream) != span.end - span.start:
        raise ValueError(f"token stream length does not match source span {span.source_id}")
    if _token_bytes_digest(stream) != span.source_manifest_sha256:
        raise ValueError(f"token stream digest does not match source span {span.source_id}")


def _compact_vocabulary(
    train_ids: Sequence[int], test_ids: Sequence[int]
) -> tuple[list[int], list[int], int]:
    """Map native Banking77 tokens after the published PAD=0/EOS=1 prefix."""
    if not train_ids or not test_ids:
        raise ValueError("Banking77 train and test streams must be non-empty")
    if 1 not in train_ids:
        raise ValueError("Banking77 compact mapping requires token id 1")
    native_unique = sorted({int(value) for value in train_ids if value >= 2})
    old_to_new = {
        value: compact_id
        for compact_id, value in enumerate(native_unique, start=2)
    }

    def map_value(value: int) -> int:
        return value if value < 2 else old_to_new.get(value, -1)

    mapped_train = [map_value(int(value)) for value in train_ids]
    mapped_test: list[int] = []
    for value in test_ids:
        mapped = map_value(int(value))
        mapped_test.append(mapped if mapped >= 0 else 1)
    return mapped_train, mapped_test, len(native_unique) + 2


def _banking_data(
    artifact: str | Path,
) -> tuple[list[int], list[int], dict[str, Any], str, int]:
    root = Path(artifact).resolve()
    manifest = preflight_banking77(root, BANKING77_MANIFEST_SHA256)
    native_vocab = int(manifest["vocab_size"])
    train_paths = [
        root / item["path"]
        for item in manifest["files"]
        if item.get("kind") == "tokens"
        and str(item["path"]).startswith("train/")
    ]
    test_paths = [
        root / item["path"]
        for item in manifest["files"]
        if item.get("kind") == "tokens"
        and str(item["path"]).startswith("test/")
    ]
    train_ids: list[int] = []
    test_ids: list[int] = []
    for path in sorted(train_paths):
        train_ids.extend(_packed_tokens(path, native_vocab))
    for path in sorted(test_paths):
        test_ids.extend(_packed_tokens(path, native_vocab))
    if len(train_ids) < 12 or len(test_ids) < 12:
        raise ValueError("Banking77 packed split is too small for three bounded spans")
    mapped_train, mapped_test, compact_vocab = _compact_vocabulary(
        train_ids, test_ids
    )
    return (
        mapped_train,
        mapped_test,
        manifest,
        sha256_file(root / "manifest.json"),
        compact_vocab,
    )


def preflight_banking77(
    artifact: str | Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Validate and return the exact pinned published Banking77 manifest."""
    if expected_manifest_sha256 != BANKING77_MANIFEST_SHA256:
        raise ValueError(
            "Banking77 manifest SHA-256 does not match the frozen preregistration"
        )
    root = Path(artifact)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise ValueError("Banking77 manifest.json is missing")
    actual = sha256_file(manifest)
    if actual != BANKING77_MANIFEST_SHA256:
        raise ValueError(
            "Banking77 manifest.json digest does not match the frozen preregistration"
        )
    validated = load_published_artifact(root)
    if not isinstance(validated, dict):
        raise ValueError("Banking77 validated manifest is invalid")
    return validated


def run_banking77_cycle(
    output: str | Path,
    artifact: str | Path,
    *,
    seed: int,
    max_steps: int = 1,
    manifest_sha256: str = BANKING77_MANIFEST_SHA256,
    device: str = "cpu",
    cross_parent_transform: str = _DEFAULT_CROSS_PARENT_TRANSFORM,
) -> Any:
    """Run one bounded real-data cycle from validated packed Banking77 shards.

    ``cross_parent_transform`` selects the transform that mixes the three
    parent streams of the recursive merge. It defaults to the legacy staged F3
    tree so an existing run reproduces byte-for-byte; the alternative
    ``"parent_preserving"`` must be requested by name.
    """
    if cross_parent_transform not in CROSS_PARENT_TRANSFORMS:
        raise ValueError(
            "cross_parent_transform must be one of "
            f"{sorted(CROSS_PARENT_TRANSFORMS)}, got {cross_parent_transform!r}"
        )
    if device != "cpu":
        raise ValueError("this bounded adapter is CPU-first; device must be 'cpu'")
    if type(max_steps) is not int or not 1 <= max_steps <= _MAX_STEPS:
        raise ValueError(f"max_steps must be in [1, {_MAX_STEPS}]")
    (
        train_ids,
        test_ids,
        manifest,
        actual_manifest_sha,
        compact_vocab,
    ) = _banking_data(artifact)
    if (
        manifest_sha256 != actual_manifest_sha
        or actual_manifest_sha != BANKING77_MANIFEST_SHA256
    ):
        raise ValueError("Banking77 manifest digest mismatch")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = GrowthRunStore(root / "store")
    pointer = store.read_parent()
    if pointer is None:
        pointer, parent_checkpoint = build_deterministic_parent(
            root, seed=seed, max_steps=max_steps, vocab_size=compact_vocab
        )
        store.bootstrap_parent(
            pointer,
            owner_id=OWNER_ID,
            trusted_parent_token=TrustedParentToken(OWNER_ID, pointer.digest()),
        )
    else:
        parent_checkpoint = _load_committed_parent(store, pointer)
    train_parts = [
        [int(value) for value in part]
        for part in np.array_split(np.asarray(train_ids, dtype=np.int64), 3)
    ]
    if any(len(part) < 2 for part in train_parts):
        raise ValueError("Banking77 training partition is too small")
    holdout_parts = [
        [int(value) for value in part[:256]]
        for part in np.array_split(np.asarray(test_ids, dtype=np.int64), 3)
    ]
    if any(len(part) < 2 for part in holdout_parts):
        raise ValueError("Banking77 holdout partition is too small")
    row_ids = {
        source: _token_bytes_digest(part)
        for source, part in zip(("A", "B", "C"), holdout_parts, strict=True)
    }
    holdout = _holdout_contract(
        root / "banking77-holdout.json",
        vocab_size=compact_vocab,
        streams=dict(
            zip(("A", "B", "C"), holdout_parts, strict=True)
        ),
        row_ids=row_ids,
        data_manifest_sha256=actual_manifest_sha,
        tokenizer_name=str(manifest["tokenizer_name"]),
    )
    spans = tuple(
        SourceSpan(
            source,
            sum(len(previous) for previous in train_parts[:index]),
            sum(len(previous) for previous in train_parts[: index + 1]),
            _token_bytes_digest(part),
        )
        for index, (source, part) in enumerate(zip(("A", "B", "C"), train_parts, strict=True))
    )
    request = GenerationRequest(
        _next_generation_id(pointer),
        seed,
        tuple(ChildPlan(source, span) for source, span in zip(("A", "B", "C"), spans, strict=True)),
        pointer,
        str(parent_checkpoint),
        holdout,
    )
    tokens = dict(zip(("A", "B", "C"), train_parts, strict=True))
    build_root = root / "build" / request.generation_id
    return run_generation(
        request,
        store,
        lambda context: _build_child(context, build_root, tokens, max_steps),
        lambda context: _build_candidate(
            context,
            build_root,
            train_parts[0][:1],
            cross_parent_transform=cross_parent_transform,
        ),
        lambda context: path_only_evaluator(context, device=device),
        owner_id=OWNER_ID,
    )


__all__ = [
    "BANKING77_MANIFEST_SHA256",
    "SYNTHETIC_V2_SEED",
    "SYNTHETIC_V3_SEED",
    "bounded_token_batches",
    "build_deterministic_parent",
    "canonical_digest",
    "config_digest",
    "path_only_evaluator",
    "preflight_banking77",
    "result_data_provenance",
    "run_banking77_cycle",
    "run_bounded_cycle",
]
