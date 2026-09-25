"""Focused tests for the deterministic recursive generation transaction owner."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from hagi.config import Config, validate_config
from hagi.model.merge import RecursiveF3HAGI, TernaryF3Tree, merge_recursive_f3
from hagi.model.model import HAGI
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
    _path_key,
    _validate_persisted_f3_derivation,
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
from hagi.train.checkpoint import config_from_dict, config_to_dict, load_payload
from tests.conftest import tiny_config

OWNER = "owner"
PROTOCOL_PAYLOAD = canonical_json_bytes(
    {
        "kind": "recursive_f3_holdout_v1",
        "schema_version": 1,
        "thresholds": {"max_source_regression": 0.01},
    }
)
PROTOCOL = sha256_bytes(PROTOCOL_PAYLOAD)
DATA = "d" * 64
ROW_IDS = ("4" * 64, "5" * 64, "6" * 64)
PARENT_MANIFEST = "1" * 64
PARENT_REPORT = "2" * 64
SOURCE_A = "a" * 64
SOURCE_B = "b" * 64
SOURCE_C = "c" * 64
CANDIDATE_LEAF_HIDDEN = 8
CANDIDATE_DEPTH = 1


def _base_config() -> Config:
    cfg = tiny_config(
        **{
            "model.num_layers": 1,
            "model.hidden_size": 8,
            "model.attention.num_query_heads": 1,
            "model.attention.num_kv_heads": 1,
            "model.attention.head_dim": 8,
            "model.ffn.intermediate_size": 8,
            "model.embedding.conv_kernel": 1,
            "model.embedding.tie_lm_head": False,
            "model.ternary.enabled": False,
            "model.adapters.enabled": False,
            "model.cortex.enabled": False,
            "model.decision.enabled": False,
            "model.loop_depth": 2,
        }
    )
    validate_config(cfg)
    return cfg


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
    cfg.model.hidden_size = 3 * parent.model.hidden_size
    cfg.model.attention.num_query_heads *= 3
    cfg.model.attention.num_kv_heads *= 3
    cfg.model.ffn.intermediate_size *= 3
    cfg.model.adapters.enabled = True
    cfg.model.adapters.pyramid.enabled = True
    cfg.model.adapters.pyramid.levels = (1,)
    cfg.model.adapters.pyramid.residual_scale = 1.0
    cfg.model.adapters.ttt_lora.enabled = False
    cfg.train.adapt.freeze_base = True
    cfg.model.head.sampled_softmax_k = 0
    validate_config(cfg)
    return cfg


def _child_config(parent: Config) -> Config:
    """Child config: exact copy of the parent with no adaptive state (§14.2)."""
    cfg = copy.deepcopy(parent)
    cfg.model.adapters.enabled = False
    cfg.model.adapters.pyramid.enabled = False
    cfg.model.adapters.pyramid.levels = (1,)
    cfg.model.adapters.pyramid.residual_scale = 1.0
    cfg.model.adapters.ttt_lora.enabled = False
    cfg.model.cortex.enabled = False
    cfg.model.decision.enabled = False
    cfg.train.adapt.freeze_base = False
    validate_config(cfg)
    return cfg


def _config_digest(config: Config) -> str:
    return sha256_bytes(canonical_json_bytes(config_to_dict(config)))


def _state_key_digest(state: dict[str, torch.Tensor]) -> str:
    entries = [
        {"key": key, "shape": list(state[key].shape), "dtype": str(state[key].dtype)}
        for key in sorted(state)
    ]
    return sha256_bytes(canonical_json_bytes(entries))


def _save_checkpoint(path: Path, state: dict[str, torch.Tensor], config: Config) -> Path:
    torch.save(
        {
            "format_version": 12,
            "model": state,
            "config": config_to_dict(config),
            "completed_steps": 0,
        },
        path,
    )
    return path


def _sha(path: Path) -> str:
    return sha256_file(path)


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _parent_pointer(root: Path) -> ParentPointer:
    """The original request parent pointer, independent of later promotion."""
    return ParentPointer("g0", _sha(root / "parent.pt"), PARENT_MANIFEST, PARENT_REPORT, owner_id=OWNER)


def _request(root: Path, *, parent: ParentPointer | None = None) -> GenerationRequest:
    return GenerationRequest(
        "g1",
        1234,
        (
            ChildPlan("A", SourceSpan("A", 0, 10, SOURCE_A)),
            ChildPlan("B", SourceSpan("B", 10, 20, SOURCE_B)),
            ChildPlan("C", SourceSpan("C", 20, 30, SOURCE_C)),
        ),
        parent if parent is not None else _parent_pointer(root),
        str(root / "parent.pt"),
        HoldoutContract(
            str(root / "holdout.bin"),
            _sha(root / "holdout.bin"),
            DATA,
            PROTOCOL,
            PROTOCOL_PAYLOAD,
            ROW_IDS,
            "dummy-tokenizer",
        ),
    )


def _fixtures(root: Path) -> tuple[GrowthRunStore, GenerationRequest]:
    root.mkdir(parents=True, exist_ok=True)
    config = _base_config()
    torch.manual_seed(1)
    parent_model = HAGI(copy.deepcopy(config))
    _save_checkpoint(root / "parent.pt", parent_model.state_dict(), config)
    _write(root / "holdout.bin", b"holdout-rows")
    store = GrowthRunStore(root)
    pointer = _parent_pointer(root)
    store.bootstrap_parent(pointer, owner_id=OWNER, trusted_parent_token=TrustedParentToken(OWNER, pointer.digest()))
    return store, _request(root)


def _child_builder(root: Path, calls: list[ChildContext], *, subdir: str = ""):
    def build_child(context: ChildContext) -> ChildArtifact:
        calls.append(context)
        parent_payload = load_payload(context.parent_checkpoint_path)
        config = _child_config(config_from_dict(parent_payload["config"]))
        parent_state = parent_payload["model"]
        if RecursiveF3HAGI.is_recursive_state(parent_state):
            child_state = {
                key: value
                for key, value in parent_state.items()
                if not key.endswith(".adapters.pyramid.scale")
            }
            model = RecursiveF3HAGI.from_state_dict(config, child_state)
        else:
            model = HAGI(copy.deepcopy(config))
            model.load_state_dict(parent_state, strict=True)
        # Bounded deterministic specialization fixture. It touches base state
        # only; head/branch scalars and any candidate contour stay inherited.
        with torch.no_grad():
            source_offset = {"A": 1, "B": 2, "C": 3}[context.source_id]
            model.encoder.embedding.weight.add_(
                (context.seed % 97 + source_offset) * 1e-4
            )
        checkpoint = _save_checkpoint(root / f"{subdir}{context.name}.pt", model.state_dict(), config)
        config_sha = _config_digest(config)
        manifest = {
            "schema_version": 1,
            "generation_id": context.generation_id,
            "name": context.name,
            "source_id": context.source_id,
            "seed": context.seed,
            "span": dataclasses.asdict(context.span),
            "checkpoint_sha256": _sha(checkpoint),
            "config_sha256": config_sha,
            "parent_checkpoint_sha256": context.parent_checkpoint_sha256,
            "protocol_sha256": context.protocol_sha256,
        }
        manifest_path = _write(root / f"{subdir}{context.name}.json", canonical_json_bytes(manifest))
        return ChildArtifact(
            name=context.name,
            source_id=context.source_id,
            seed=context.seed,
            span=context.span,
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=_sha(checkpoint),
            manifest_path=str(manifest_path),
            manifest_sha256=_sha(manifest_path),
            config_sha256=config_sha,
            parent_checkpoint_sha256=context.parent_checkpoint_sha256,
            protocol_sha256=context.protocol_sha256,
        )

    return build_child


def _candidate_builder(root: Path, calls: list[CandidateContext], *, subdir: str = ""):
    def build_candidate(context: CandidateContext) -> CandidateArtifact:
        calls.append(context)
        parent_cfg = config_from_dict(load_payload(context.parent_checkpoint_path)["config"])
        child_cfg = _child_config(parent_cfg)
        child_states = [load_payload(child.checkpoint_path)["model"] for child in context.children]
        target_cfg = _target_config(parent_cfg)
        torch.manual_seed(999)
        model = merge_recursive_f3(
            target_cfg,
            child_states,
            child_configs=tuple(copy.deepcopy(child_cfg) for _ in range(3)),
        )
        checkpoint = _save_checkpoint(root / f"{subdir}candidate.pt", model.state_dict(), target_cfg)
        depth = target_cfg.merge.ternary_depth
        leaf_hidden = target_cfg.merge.expert_hidden // (3 ** (depth - 1))
        tree = TernaryF3Tree(depth, leaf_hidden)
        metadata = {
            "parent_config_sha256": _config_digest(parent_cfg),
            "candidate_config_sha256": _config_digest(target_cfg),
            "candidate_state_key_digest": _state_key_digest(model.state_dict()),
            "logit_scale_source": float(model.recursive_f3_logit_scale_source.item()),
            "logit_scale_target": float(model.recursive_f3_logit_scale_target.item()),
        }
        manifest = {
            "schema_version": 1,
            "generation_id": context.generation_id,
            "parent_checkpoint_sha256": context.parent_checkpoint_sha256,
            "parent_config_sha256": metadata["parent_config_sha256"],
            "protocol_sha256": context.protocol_sha256,
            "candidate_checkpoint_sha256": _sha(checkpoint),
            "candidate_config_sha256": metadata["candidate_config_sha256"],
            "candidate_state_key_digest": metadata["candidate_state_key_digest"],
            "ternary_depth": depth,
            "leaf_hidden": leaf_hidden,
            "coordinate_layout": tree.coordinate_layout,
            "transform_digest": tree.transform_digest,
            "weight_source": target_cfg.merge.expert_weight_source,
            "logit_scale_source": metadata["logit_scale_source"],
            "logit_scale_target": metadata["logit_scale_target"],
            "decision": "candidate",
        }
        child_span = context.children[0].span
        prompt_span = SourceSpan(
            "A",
            child_span.start,
            min(child_span.start + 4, child_span.end),
            child_span.source_manifest_sha256,
        )
        ledger = SelfImprovementLedger(
            seed=context.self_improve_seed,
            source_id="A",
            prompt_span=prompt_span,
            prompt_token_sha256=sha256_bytes(b"prompt-tokens"),
            prompt_text_sha256=sha256_bytes(b"prompt-text"),
            prompt_token_count=prompt_span.end - prompt_span.start,
            optimizer_parameter_ids=tuple(
                sorted(
                    key for key in model.state_dict()
                    if key.endswith(".adapters.pyramid.scale")
                )
            ),
            accepted_updates=0,
            derived_base_state_sha256=_derived_base_digest(model.state_dict()),
            contour_state_sha256=_contour_state_digest(model.state_dict()),
        )
        manifest["self_improve"] = _ledger_payload(ledger)
        manifest_path = _write(root / f"{subdir}candidate.json", canonical_json_bytes(manifest))
        return CandidateArtifact(
            generation_id=context.generation_id,
            parent_checkpoint_sha256=context.parent_checkpoint_sha256,
            parent_config_sha256=metadata["parent_config_sha256"],
            protocol_sha256=context.protocol_sha256,
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=_sha(checkpoint),
            manifest_path=str(manifest_path),
            manifest_sha256=_sha(manifest_path),
            candidate_config_sha256=metadata["candidate_config_sha256"],
            candidate_state_key_digest=metadata["candidate_state_key_digest"],
            ternary_depth=depth,
            leaf_hidden=leaf_hidden,
            coordinate_layout=tree.coordinate_layout,
            transform_digest=tree.transform_digest,
            weight_source=target_cfg.merge.expert_weight_source,
            logit_scale_source=metadata["logit_scale_source"],
            logit_scale_target=metadata["logit_scale_target"],
            child_checkpoint_sha256=tuple(child.checkpoint_sha256 for child in context.children),
            child_config_sha256=tuple(child.config_sha256 for child in context.children),
            self_improve=ledger,
        )

    return build_candidate


def _candidate_with_ledger(
    candidate: CandidateArtifact, ledger: SelfImprovementLedger
) -> CandidateArtifact:
    """Re-bind a candidate manifest to a forged-but-schema-valid ledger."""
    manifest_path = Path(candidate.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["self_improve"] = _ledger_payload(ledger)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return dataclasses.replace(
        candidate,
        manifest_sha256=_sha(manifest_path),
        self_improve=ledger,
    )


def _candidate_with_contour_update(
    candidate: CandidateArtifact,
    *,
    value: float,
    accepted_updates: int,
) -> CandidateArtifact:
    checkpoint_path = Path(candidate.checkpoint_path)
    payload = load_payload(checkpoint_path)
    state = payload["model"]
    state["blocks.0.adapters.pyramid.scale"].fill_(value)
    ledger = dataclasses.replace(
        candidate.self_improve,
        accepted_updates=accepted_updates,
        derived_base_state_sha256=_derived_base_digest(state),
        contour_state_sha256=_contour_state_digest(state),
    )
    torch.save(payload, checkpoint_path)
    checkpoint_sha256 = _sha(checkpoint_path)
    manifest_path = Path(candidate.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_checkpoint_sha256"] = checkpoint_sha256
    manifest["candidate_state_key_digest"] = _state_key_digest(state)
    manifest["self_improve"] = _ledger_payload(ledger)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return dataclasses.replace(
        candidate,
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=_sha(manifest_path),
        candidate_state_key_digest=_state_key_digest(state),
        self_improve=ledger,
    )


def _metrics(values: tuple[float, float, float]) -> tuple[SourceMetric, ...]:
    return tuple(
        SourceMetric(source, 4, value, row_digest)
        for source, value, row_digest in zip(("A", "B", "C"), values, ROW_IDS)
    )


def _evaluator(candidate=(0.9, 0.9, 0.9), incumbent=(1.0, 1.0, 1.0), calls=None, rows_candidate=(4, 4, 4)):
    def evaluate(context: EvaluationContext) -> EvaluationResult:
        if calls is not None:
            calls.append(context)
        return EvaluationResult(
            incumbent_metrics=_metrics(incumbent),
            candidate_metrics=tuple(
                SourceMetric(
                    metric.source_id,
                    rows_candidate[index],
                    value,
                    metric.row_ids_sha256,
                )
                for index, (metric, value) in enumerate(zip(_metrics(incumbent), candidate))
            ),
            tokenizer_name=context.holdout.tokenizer_name,
            data_manifest_sha256=context.holdout.data_manifest_sha256,
            protocol_sha256=context.holdout.protocol_sha256,
            parent_checkpoint_sha256=context.parent.checkpoint_sha256,
            candidate_checkpoint_sha256=context.candidate.checkpoint_sha256,
            candidate_manifest_sha256=context.candidate.manifest_sha256,
        )

    return evaluate


def _forbidden_callbacks():
    def forbidden(context):
        raise AssertionError("callbacks must not run")

    return forbidden, forbidden, forbidden


def _objects(value: Any, seen: set[int] | None = None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    yield value
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _objects(getattr(value, field.name), seen)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _objects(item, seen)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _objects(item, seen)


def test_self_improve_ledger_binds_seed_span_and_optimizer_ownership(tmp_path: Path):
    ledger = SelfImprovementLedger(
        seed=10235,
        source_id="A",
        prompt_span=SourceSpan("A", 0, 4, SOURCE_A),
        prompt_token_sha256=sha256_bytes(b"tokens"),
        prompt_text_sha256=sha256_bytes(b"tokens"),
        prompt_token_count=4,
        optimizer_parameter_ids=("blocks.0.adapters.pyramid.scale",),
        accepted_updates=0,
        derived_base_state_sha256="d" * 64,
        contour_state_sha256="e" * 64,
    )
    assert ledger.seed == 10235
    assert ledger.prompt_span.source_id == "A"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("seed", "self-improve seed"),
        ("span", "span escapes"),
        ("optimizer", "optimizer must own"),
        ("base_digest", "derived base digest"),
        ("prompt_count", "token count"),
        ("updates", "accepted_updates exceeds"),
    ],
)
def test_self_improve_ledger_tampering_fails_closed(
    tmp_path: Path, mutation: str, match: str
):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    baseline = _candidate_builder(tmp_path, [])

    def build(context: CandidateContext) -> CandidateArtifact:
        candidate = baseline(context)
        ledger = candidate.self_improve
        if mutation == "seed":
            ledger = dataclasses.replace(ledger, seed=ledger.seed + 1)
        elif mutation == "span":
            ledger = dataclasses.replace(
                ledger,
                prompt_span=SourceSpan(
                    "A",
                    ledger.prompt_span.end - 1,
                    context.children[0].span.end + 1,
                    ledger.prompt_span.source_manifest_sha256,
                ),
            )
        elif mutation == "optimizer":
            ledger = dataclasses.replace(
                ledger,
                optimizer_parameter_ids=(
                    *ledger.optimizer_parameter_ids,
                    "blocks.99.adapters.pyramid.scale",
                ),
            )
        elif mutation == "base_digest":
            ledger = dataclasses.replace(
                ledger, derived_base_state_sha256="d" * 64
            )
        elif mutation == "prompt_count":
            ledger = dataclasses.replace(
                ledger, prompt_token_count=ledger.prompt_token_count + 1
            )
        elif mutation == "updates":
            ledger = dataclasses.replace(ledger, accepted_updates=2)
        return _candidate_with_ledger(candidate, ledger)

    with pytest.raises(ValueError, match=match):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            build,
            _evaluator(),
            owner_id=OWNER,
        )
    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert (tmp_path / "g1" / "failure.json").is_file()


def test_owner_evidence_is_validated_before_prepared_and_parent_commit(
    tmp_path: Path, monkeypatch
):
    import hagi.orchestrator.recursive as recursive_module

    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    original = recursive_module._evidence_payload

    def invalid_evidence(request, candidate, verdict, children):
        evidence = original(request, candidate, verdict, children)
        evidence["candidate_f3"]["self_improve"]["seed"] += 1
        return evidence

    monkeypatch.setattr(recursive_module, "_evidence_payload", invalid_evidence)
    with pytest.raises(ValueError, match="candidate provenance"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )

    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "failed"
    assert not (tmp_path / "g1" / "prepared-report.json").exists()
    assert not (tmp_path / "g1" / "holdout-evidence.json").exists()
    assert not (tmp_path / "g1" / "report.json").exists()
    assert (tmp_path / "g1" / "failure.json").is_file()


@pytest.mark.parametrize(
    ("claim", "integer_value"),
    (
        ("mechanism_supported", 1),
        ("quality_supported", 0),
        ("security_supported", 0),
        ("production_promotion", 0),
        ("pareto_improvement", 1),
    ),
)
def test_persisted_holdout_claim_flags_reject_integer_coercion(
    tmp_path: Path, monkeypatch, claim: str, integer_value: int
):
    import hagi.orchestrator.recursive as recursive_module

    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    original = recursive_module._evidence_payload

    def integer_claim_evidence(request, candidate, verdict, children):
        evidence = original(request, candidate, verdict, children)
        evidence[claim] = integer_value
        return evidence

    monkeypatch.setattr(
        recursive_module, "_evidence_payload", integer_claim_evidence
    )
    with pytest.raises(ValueError, match="invalid persisted holdout evidence"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )

    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert not (tmp_path / "g1" / "prepared-report.json").exists()
    assert not (tmp_path / "g1" / "holdout-evidence.json").exists()
    assert not (tmp_path / "g1" / "report.json").exists()
    assert (tmp_path / "g1" / "failure.json").is_file()


def test_builder_contexts_are_sealed_from_full_holdout_bindings(
    tmp_path: Path,
):
    store, request = _fixtures(tmp_path)
    child_calls: list[ChildContext] = []
    candidate_calls: list[CandidateContext] = []

    run_generation(
        request,
        store,
        _child_builder(tmp_path, child_calls),
        _candidate_builder(tmp_path, candidate_calls),
        _evaluator(),
        owner_id=OWNER,
    )

    forbidden_values = {DATA}
    forbidden_fields = {
        "data_manifest_sha256",
        "holdout_sha256",
        "holdout_path",
        "row_ids_sha256",
    }
    for context in (*child_calls, *candidate_calls):
        names = {field.name for field in dataclasses.fields(context)}
        assert names.isdisjoint(forbidden_fields)
        for value in _objects(context):
            if isinstance(value, str):
                assert value not in forbidden_values

        for child in candidate_calls[0].children:
            assert "data_manifest_sha256" not in {
                field.name for field in dataclasses.fields(child)
            }
            draft = json.loads(Path(child.manifest_path).read_text(encoding="utf-8"))
            assert set(draft).isdisjoint(forbidden_fields)

    inputs = (tmp_path / "g1" / "inputs").resolve()
    strict_candidates = list(inputs.glob("candidate-manifest-*.bin"))
    assert len(strict_candidates) == 1
    strict = json.loads(strict_candidates[0].read_text(encoding="utf-8"))
    assert strict["data_manifest_sha256"] == DATA
    assert strict["protocol_sha256"] == PROTOCOL


def test_owner_materializes_holdout_and_strict_manifests_after_all_builds(
    tmp_path: Path,
):
    store, request = _fixtures(tmp_path)
    run = tmp_path / "g1"
    child = _child_builder(tmp_path, [])
    candidate = _candidate_builder(tmp_path, [])
    callback_leaks: list[tuple[str, ...]] = []

    def owner_bound_files() -> tuple[str, ...]:
        if not run.is_dir():
            return ()
        leaked = []
        for path in run.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if name.startswith("holdout-"):
                leaked.append(path.relative_to(run).as_posix())
            elif name.startswith("candidate-manifest-"):
                leaked.append(path.relative_to(run).as_posix())
            elif (
                name.startswith("child_")
                and "-manifest-" in name
                and "-manifest-draft-" not in name
            ):
                leaked.append(path.relative_to(run).as_posix())
        return tuple(sorted(leaked))

    def build_child(context: ChildContext) -> ChildArtifact:
        callback_leaks.append(owner_bound_files())
        return child(context)

    def build_candidate(context: CandidateContext) -> CandidateArtifact:
        callback_leaks.append(owner_bound_files())
        return candidate(context)

    result = run_generation(
        request,
        store,
        build_child,
        build_candidate,
        _evaluator(),
        owner_id=OWNER,
    )

    assert result.decision == "accepted"
    assert callback_leaks == [(), (), (), ()]
    evaluator_inputs = (run / "evaluator-inputs").resolve()
    assert any(path.name.startswith("holdout-") for path in evaluator_inputs.iterdir())
    assert any(
        path.name.startswith("child_A-manifest-")
        and "-manifest-draft-" not in path.name
        for path in (run / "inputs").iterdir()
    )
    assert any(
        path.name.startswith("candidate-manifest-")
        for path in (run / "inputs").iterdir()
    )


def test_builder_mutation_of_live_parent_fails_before_prepared(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    parent_path = Path(request.parent_checkpoint_path)
    baseline = _child_builder(tmp_path, [])
    mutated: list[str] = []

    def build_child(context: ChildContext) -> ChildArtifact:
        artifact = baseline(context)
        if not mutated:
            payload = load_payload(parent_path)
            payload["model"]["encoder.embedding.weight"] = (
                payload["model"]["encoder.embedding.weight"] + 1
            )
            torch.save(payload, parent_path)
            mutated.append("parent")
        return artifact

    with pytest.raises(ValueError, match="bound evidence digest mismatch"):
        run_generation(
            request,
            store,
            build_child,
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )

    assert mutated == ["parent"]
    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert not (tmp_path / "g1" / "prepared-report.json").exists()
    assert not (tmp_path / "g1" / "holdout-evidence.json").exists()
    assert not (tmp_path / "g1" / "report.json").exists()
    assert (tmp_path / "g1" / "failure.json").is_file()


def test_builder_mutation_of_parent_snapshot_fails_before_prepared(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    snapshot = (
        tmp_path / "g1" / "inputs"
        / f"parent-checkpoint-{request.parent.checkpoint_sha256}.bin"
    )
    baseline = _child_builder(tmp_path, [])
    mutated: list[str] = []

    def build_child(context: ChildContext) -> ChildArtifact:
        artifact = baseline(context)
        if not mutated:
            mutated.append(str(Path(context.parent_checkpoint_path)))
            payload = load_payload(snapshot)
            payload["model"]["encoder.embedding.weight"] = (
                payload["model"]["encoder.embedding.weight"] + 1
            )
            torch.save(payload, snapshot)
        return artifact

    with pytest.raises(ValueError, match="bound evidence digest mismatch"):
        run_generation(
            request,
            store,
            build_child,
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )

    assert mutated == [str(snapshot)]
    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert not (tmp_path / "g1" / "prepared-report.json").exists()
    assert not (tmp_path / "g1" / "holdout-evidence.json").exists()
    assert not (tmp_path / "g1" / "report.json").exists()
    assert (tmp_path / "g1" / "failure.json").is_file()


def test_builder_contexts_read_only_the_owner_parent_snapshot(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    parent_key = _path_key(request.parent_checkpoint_path)
    seen: list[tuple[bool, str]] = []
    child = _child_builder(tmp_path, [])
    candidate = _candidate_builder(tmp_path, [])

    def build_child(context: ChildContext) -> ChildArtifact:
        seen.append(
            (_path_key(context.parent_checkpoint_path) == parent_key, context.parent_checkpoint_path)
        )
        return child(context)

    def build_candidate(context: CandidateContext) -> CandidateArtifact:
        seen.append(
            (_path_key(context.parent_checkpoint_path) == parent_key, context.parent_checkpoint_path)
        )
        return candidate(context)

    run_generation(
        request,
        store,
        build_child,
        build_candidate,
        _evaluator(),
        owner_id=OWNER,
    )

    assert len(seen) == 4
    assert all(is_live is False for is_live, _ in seen)
    assert all(
        Path(path).parent == (tmp_path / "g1" / "inputs")
        for _, path in seen
    )
    assert {path for _, path in seen} == {
        str((tmp_path / "g1" / "inputs" / f"parent-checkpoint-{request.parent.checkpoint_sha256}.bin").resolve())
    }
    assert sha256_file(request.parent_checkpoint_path) == request.parent.checkpoint_sha256


def test_callback_preseceded_holdout_snapshot_fails_closed(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    baseline = _child_builder(tmp_path, [])
    holdout = Path(request.holdout.path).read_bytes()
    seeded: list[str] = []

    def build_child(context: ChildContext) -> ChildArtifact:
        artifact = baseline(context)
        if not seeded:
            seeded.append("pre-seeded")
            target = (
                tmp_path / "g1" / "evaluator-inputs"
                / f"holdout-{request.holdout.sha256}.bin"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(holdout)
        return artifact

    with pytest.raises(ValueError, match="late holdout snapshot already exists"):
        run_generation(
            request,
            store,
            build_child,
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )

    assert seeded == ["pre-seeded"]
    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert not (tmp_path / "g1" / "prepared-report.json").exists()
    assert not (tmp_path / "g1" / "report.json").exists()
    assert (tmp_path / "g1" / "failure.json").is_file()


def test_evaluator_holdout_mutation_fails_before_prepared(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    baseline = _evaluator()

    def evaluate(context: EvaluationContext) -> EvaluationResult:
        result = baseline(context)
        holdout_path = Path(context.holdout.path)
        holdout_path.write_bytes(holdout_path.read_bytes() + b"-mutated")
        return result

    with pytest.raises(ValueError, match="bound evidence digest mismatch"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            evaluate,
            owner_id=OWNER,
        )

    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert not (tmp_path / "g1" / "prepared-report.json").exists()
    assert not (tmp_path / "g1" / "holdout-evidence.json").exists()
    assert not (tmp_path / "g1" / "report.json").exists()
    assert (tmp_path / "g1" / "failure.json").is_file()


def test_late_owner_stage_holdout_mutation_fails_before_evidence(
    tmp_path: Path, monkeypatch
):
    import hagi.orchestrator.recursive as recursive_module

    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    original = recursive_module._evidence_payload

    def mutate_holdout(request, candidate, verdict, children):
        payload = original(request, candidate, verdict, children)
        holdout_path = next(
            (tmp_path / "g1" / "evaluator-inputs").glob("holdout-*.bin")
        )
        holdout_path.write_bytes(holdout_path.read_bytes() + b"-mutated")
        return payload

    monkeypatch.setattr(
        recursive_module,
        "_evidence_payload",
        mutate_holdout,
    )
    with pytest.raises(ValueError, match="bound evidence digest mismatch"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )

    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert not (tmp_path / "g1" / "prepared-report.json").exists()
    assert not (tmp_path / "g1" / "holdout-evidence.json").exists()
    assert not (tmp_path / "g1" / "report.json").exists()
    assert (tmp_path / "g1" / "failure.json").is_file()


def test_candidate_mutation_after_build_is_rejected_before_commit(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    baseline = _candidate_builder(tmp_path, [])

    def build(context: CandidateContext) -> CandidateArtifact:
        candidate = baseline(context)
        checkpoint = Path(candidate.checkpoint_path)
        payload = load_payload(checkpoint)
        payload["model"]["blocks.0.adapters.pyramid.scale"].fill_(0.25)
        torch.save(payload, checkpoint)
        return candidate

    with pytest.raises(ValueError, match="digest mismatch"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            build,
            _evaluator(),
            owner_id=OWNER,
        )
    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert (tmp_path / "g1" / "failure.json").is_file()


def test_post_merge_contour_requires_ledger_authority(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    baseline = _candidate_builder(tmp_path, [])

    def build(context: CandidateContext) -> CandidateArtifact:
        return _candidate_with_contour_update(
            baseline(context), value=0.25, accepted_updates=0
        )

    with pytest.raises(ValueError, match="not fresh zero init"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            build,
            _evaluator(),
            owner_id=OWNER,
        )
    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert (tmp_path / "g1" / "failure.json").is_file()


def test_accepted_update_cannot_leave_every_contour_unchanged(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    baseline = _candidate_builder(tmp_path, [])

    def build(context: CandidateContext) -> CandidateArtifact:
        return _candidate_with_contour_update(
            baseline(context), value=0.0, accepted_updates=1
        )

    with pytest.raises(ValueError, match="left every contour unchanged"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            build,
            _evaluator(),
            owner_id=OWNER,
        )


def test_single_post_merge_update_may_move_only_fresh_contour(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    baseline = _candidate_builder(tmp_path, [])

    def build(context: CandidateContext) -> CandidateArtifact:
        return _candidate_with_contour_update(
            baseline(context), value=0.25, accepted_updates=1
        )

    result = run_generation(
        request,
        store,
        _child_builder(tmp_path, []),
        build,
        _evaluator(),
        owner_id=OWNER,
    )
    contour = load_payload(result.candidate_checkpoint_path)["model"][
        "blocks.0.adapters.pyramid.scale"
    ]
    assert result.decision == "accepted"
    assert torch.equal(contour, torch.full_like(contour, 0.25))
    evidence = json.loads(
        Path(result.holdout_evidence_path).read_text(encoding="utf-8")
    )
    assert evidence["candidate_f3"]["self_improve"]["accepted_updates"] == 1


def test_accepted_non_regression_promotes_parent_with_real_payloads(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    child_calls: list[ChildContext] = []
    candidate_calls: list[CandidateContext] = []
    evaluation_calls: list[EvaluationContext] = []

    result = run_generation(
        request, store, _child_builder(tmp_path, child_calls), _candidate_builder(tmp_path, candidate_calls),
        _evaluator(calls=evaluation_calls), owner_id=OWNER,
    )

    assert result.decision == "accepted"
    assert result.quality_supported is False
    assert result.security_supported is False
    assert result.production_promotion is False
    assert result.mechanism_supported is True
    assert result.pareto_improvement is True
    assert result.ce_regression < 0
    assert json.loads((tmp_path / "current-parent.json").read_text(encoding="utf-8"))["generation_id"] == "g1"
    inputs = (tmp_path / "g1" / "inputs").resolve()
    assert [call.name for call in child_calls] == ["child_A", "child_B", "child_C"]
    assert all(Path(child.checkpoint_path).resolve().is_relative_to(inputs) for child in candidate_calls[0].children)
    assert all(Path(child.manifest_path).resolve().is_relative_to(inputs) for child in candidate_calls[0].children)
    assert Path(evaluation_calls[0].candidate.checkpoint_path).resolve().is_relative_to(inputs)
    evaluator_inputs = (tmp_path / "g1" / "evaluator-inputs").resolve()
    assert Path(evaluation_calls[0].holdout.path).resolve().is_relative_to(
        evaluator_inputs
    )
    assert Path(evaluation_calls[0].holdout.path) != Path(request.holdout.path)
    evidence = json.loads(Path(result.holdout_evidence_path).read_text(encoding="utf-8"))
    assert evidence["security_supported"] is False
    assert evidence["decision"] == "accepted"
    assert evidence["source_metrics"]["deltas"]["A"] == pytest.approx(-0.1)


def test_two_generations_share_accepted_parent_lineage(tmp_path: Path):
    store, first_request = _fixtures(tmp_path)
    first_children: list[ChildContext] = []
    first_candidates: list[CandidateContext] = []

    first = run_generation(
        first_request,
        store,
        _child_builder(tmp_path, first_children),
        _candidate_builder(tmp_path, first_candidates),
        _evaluator(),
        owner_id=OWNER,
    )
    first_pointer = store.read_parent()

    assert first.decision == "accepted"
    assert first_pointer.generation_id == "g1"
    assert first_pointer.checkpoint_sha256 == sha256_file(
        first.candidate_checkpoint_path
    )
    assert first_pointer.manifest_sha256 == sha256_file(first.manifest_path)
    assert all(
        config_from_dict(load_payload(child.checkpoint_path)["config"]).model.adapters.enabled
        is False
        for child in first_candidates[0].children
    )

    second_request = GenerationRequest(
        "g2",
        4242,
        (
            ChildPlan("A", SourceSpan("A", 30, 40, SOURCE_A)),
            ChildPlan("B", SourceSpan("B", 40, 50, SOURCE_B)),
            ChildPlan("C", SourceSpan("C", 50, 60, SOURCE_C)),
        ),
        first_pointer,
        first.candidate_checkpoint_path,
        first_request.holdout,
    )
    second_children: list[ChildContext] = []
    second_candidates: list[CandidateContext] = []
    second = run_generation(
        second_request,
        store,
        _child_builder(tmp_path, second_children, subdir="g2/"),
        _candidate_builder(tmp_path, second_candidates, subdir="g2/"),
        _evaluator(),
        owner_id=OWNER,
    )

    second_pointer = store.read_parent()
    second_report = json.loads(Path(second.report_path).read_text(encoding="utf-8"))
    second_payload = load_payload(second.candidate_checkpoint_path)
    second_cfg = config_from_dict(second_payload["config"])
    second_child_cfg = config_from_dict(
        load_payload(second_candidates[0].children[0].checkpoint_path)["config"]
    )

    assert second.decision == "accepted"
    assert second_pointer.generation_id == "g2"
    assert second_pointer.checkpoint_sha256 == sha256_file(second.candidate_checkpoint_path)
    assert second_report["parent_generation_id"] == "g1"
    assert second_report["parent_checkpoint_sha256"] == first_pointer.checkpoint_sha256
    assert second_report["parent_manifest_sha256"] == first_pointer.manifest_sha256
    assert [call.name for call in second_children] == ["child_A", "child_B", "child_C"]
    assert second_child_cfg.merge.ternary_depth == 1
    assert second_child_cfg.model.hidden_size == 24
    assert second_child_cfg.model.adapters.enabled is False
    assert second_cfg.merge.ternary_depth == 2
    assert second_cfg.model.hidden_size == 72
    assert second_cfg.model.adapters.enabled is True
    assert second_cfg.model.adapters.pyramid.enabled is True
    contour = second_payload["model"]["blocks.0.adapters.pyramid.scale"]
    assert torch.equal(contour, torch.zeros_like(contour))


def test_builder_context_graph_exposes_no_holdout_or_store(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    child_calls: list[ChildContext] = []
    candidate_calls: list[CandidateContext] = []
    run_generation(
        request, store, _child_builder(tmp_path, child_calls), _candidate_builder(tmp_path, candidate_calls),
        _evaluator(), owner_id=OWNER,
    )
    forbidden_types = (GenerationRequest, HoldoutContract, GrowthRunStore)
    for context in (*child_calls, *candidate_calls):
        names = {field.name for field in dataclasses.fields(context)}
        assert names == {
            "generation_id", "parent_checkpoint_path", "parent_checkpoint_sha256",
            "parent_manifest_sha256", "name", "source_id", "seed", "span",
            "protocol_sha256",
        } or names == {
            "generation_id", "parent_checkpoint_path", "parent_checkpoint_sha256",
            "parent_manifest_sha256", "children", "protocol_sha256", "base_seed",
            "self_improve_seed",
        }
        for value in _objects(context):
            assert not isinstance(value, forbidden_types)
            if isinstance(value, str):
                assert "holdout" not in value
        if isinstance(context, CandidateContext):
                assert all(
                    Path(child.checkpoint_path).resolve().is_relative_to(
                        (tmp_path / "g1" / "inputs").resolve()
                    )
                    for child in context.children
                )


def test_equality_is_accepted_without_pareto_improvement(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    result = run_generation(
        request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []),
        _evaluator(candidate=(1.0, 1.0, 1.0)), owner_id=OWNER,
    )
    assert result.decision == "accepted"
    assert result.pareto_improvement is False
    assert result.ce_regression == 0.0


def test_forced_regression_rejects_and_parent_bytes_unchanged(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    result = run_generation(
        request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []),
        _evaluator(candidate=(1.0, 1.05, 0.5)), owner_id=OWNER,
    )
    assert result.decision == "rejected"
    assert result.worst_source_regression > 0.01
    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert not (tmp_path / "g1" / "failure.json").exists()


def test_negative_and_nonfinite_metrics_fail_closed(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    with pytest.raises(ValueError, match="nonnegative"):
        run_generation(
            request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []),
            _evaluator(candidate=(0.9, -0.1, 0.9)), owner_id=OWNER,
        )
    assert (tmp_path / "g1" / "failure.json").is_file()
    assert (tmp_path / "current-parent.json").read_bytes() == before
    shutil.rmtree(tmp_path / "g1")
    with pytest.raises(ValueError, match="scored_rows"):
        run_generation(
            request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []),
            _evaluator(rows_candidate=(4, 4, 5)), owner_id=OWNER,
        )
    assert (tmp_path / "current-parent.json").read_bytes() == before


@pytest.mark.parametrize(
    "field", ["candidate_checkpoint_sha256", "candidate_manifest_sha256"]
)
def test_evaluation_candidate_identity_mismatch_fails_closed(
    tmp_path: Path, field: str
):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    baseline = _evaluator()

    def evaluate(context):
        return dataclasses.replace(baseline(context), **{field: "7" * 64})

    with pytest.raises(ValueError, match="candidate identity"):
        run_generation(
            request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []),
            evaluate, owner_id=OWNER,
        )
    assert (tmp_path / "g1" / "failure.json").is_file()
    assert (tmp_path / "current-parent.json").read_bytes() == before


def test_evaluation_row_identity_mismatch_fails_closed(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()
    baseline = _evaluator()

    def evaluate(context):
        result = baseline(context)
        first = dataclasses.replace(result.candidate_metrics[0], row_ids_sha256="8" * 64)
        return dataclasses.replace(
            result, candidate_metrics=(first, *result.candidate_metrics[1:])
        )

    with pytest.raises(ValueError, match="row identity"):
        run_generation(
            request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []),
            evaluate, owner_id=OWNER,
        )
    assert (tmp_path / "g1" / "failure.json").is_file()
    assert (tmp_path / "current-parent.json").read_bytes() == before


def test_holdout_protocol_payload_hash_is_checked(tmp_path: Path):
    _fixtures(tmp_path)
    with pytest.raises(ValueError, match="protocol payload digest mismatch"):
        HoldoutContract(
            str(tmp_path / "holdout.bin"),
            _sha(tmp_path / "holdout.bin"),
            DATA,
            "8" * 64,
            PROTOCOL_PAYLOAD,
            ROW_IDS,
            "dummy-tokenizer",
        )


def test_duplicate_child_checkpoint_digest_fails(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    baseline = _child_builder(tmp_path, [])

    def build(context):
        child = baseline(context)
        if context.name != "child_C":
            return child
        checkpoint = tmp_path / f"{context.name}.pt"
        checkpoint.write_bytes((tmp_path / "child_A.pt").read_bytes())
        manifest = json.loads(Path(child.manifest_path).read_text())
        manifest["checkpoint_sha256"] = _sha(checkpoint)
        manifest_path = Path(child.manifest_path)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        return dataclasses.replace(
            child,
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=_sha(checkpoint),
            manifest_sha256=_sha(manifest_path),
        )

    with pytest.raises(ValueError, match="checkpoint digests must be distinct"):
        run_generation(
            request, store, build, _candidate_builder(tmp_path, []), _evaluator(),
            owner_id=OWNER,
        )
    assert (tmp_path / "g1" / "failure.json").is_file()
    assert (tmp_path / "current-parent.json").read_text().find('"generation_id":"g0"') >= 0


def test_mismatched_protocol_identity_fails_closed(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()

    def evaluate(context):
        return EvaluationResult(
            incumbent_metrics=_metrics((1.0, 1.0, 1.0)),
            candidate_metrics=_metrics((0.9, 0.9, 0.9)),
            tokenizer_name=context.holdout.tokenizer_name,
            data_manifest_sha256=context.holdout.data_manifest_sha256,
            protocol_sha256="9" * 64,
            parent_checkpoint_sha256=context.parent.checkpoint_sha256,
            candidate_checkpoint_sha256=context.candidate.checkpoint_sha256,
            candidate_manifest_sha256=context.candidate.manifest_sha256,
        )

    with pytest.raises(ValueError, match="holdout identity"):
        run_generation(request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []), evaluate, owner_id=OWNER)
    assert (tmp_path / "g1" / "failure.json").is_file()
    assert (tmp_path / "current-parent.json").read_bytes() == before


def test_child_callback_failure_stops_pipeline(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    calls: list[ChildContext] = []
    candidate_calls: list[CandidateContext] = []
    evaluation_calls: list[EvaluationContext] = []
    build_child = _child_builder(tmp_path, calls)

    def failing(context):
        calls.append(context)
        if context.name == "child_A":
            raise RuntimeError("injected child A failure")
        return build_child(context)

    with pytest.raises(RuntimeError, match="injected child A failure"):
        run_generation(request, store, failing, _candidate_builder(tmp_path, candidate_calls), _evaluator(calls=evaluation_calls), owner_id=OWNER)
    assert [call.name for call in calls] == ["child_A"]
    assert candidate_calls == []
    assert evaluation_calls == []
    assert json.loads((tmp_path / "g1" / "state.json").read_text(encoding="utf-8"))["state"] == "failed"


def test_child_identity_and_provenance_mismatches_fail_closed(tmp_path: Path):
    for mutate, match in [
        (lambda child: dataclasses.replace(child, seed=child.seed + 1), "identity mismatch"),
        (lambda child: dataclasses.replace(child, span=SourceSpan("A", 5, 15, SOURCE_A)), "span"),
        (lambda child: dataclasses.replace(child, protocol_sha256="7" * 64), "provenance"),
        (lambda child: dataclasses.replace(child, checkpoint_sha256="8" * 64), "digest mismatch"),
    ]:
        root = tmp_path / f"case-{match[:8]}"
        store, request = _fixtures(root)
        baseline = _child_builder(root, [])

        def build(context, mutate=mutate):
            child = baseline(context)
            return mutate(child) if context.name == "child_B" else child

        with pytest.raises(ValueError, match=match):
            run_generation(request, store, build, _candidate_builder(root, []), _evaluator(), owner_id=OWNER)
        assert (root / "g1" / "failure.json").is_file()
        assert json.loads((root / "current-parent.json").read_text())["generation_id"] == "g0"


def test_overlapping_child_plans_fail_before_callbacks(tmp_path: Path):
    _, request = _fixtures(tmp_path)
    with pytest.raises(ValueError, match="pairwise disjoint"):
        dataclasses.replace(request, child_plans=(
            ChildPlan("A", SourceSpan("A", 0, 15, SOURCE_A)),
            ChildPlan("B", SourceSpan("B", 10, 20, SOURCE_B)),
            ChildPlan("C", SourceSpan("C", 20, 30, SOURCE_C)),
        ))
    with pytest.raises(ValueError, match="A/B/C order"):
        dataclasses.replace(request, child_plans=(
            ChildPlan("B", SourceSpan("B", 0, 10, SOURCE_B)),
            ChildPlan("A", SourceSpan("A", 10, 20, SOURCE_A)),
            ChildPlan("C", SourceSpan("C", 20, 30, SOURCE_C)),
        ))


def test_child_seeds_reach_builder(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    calls: list[ChildContext] = []
    run_generation(request, store, _child_builder(tmp_path, calls), _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)
    assert [(call.name, call.source_id, call.seed) for call in calls] == [
        ("child_A", "A", 1234), ("child_B", "B", 2243), ("child_C", "C", 3252),
    ]


def test_parent_aliased_child_checkpoint_fails(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    baseline = _child_builder(tmp_path, [])

    def build(context):
        child = baseline(context)
        if context.name == "child_B":
            return dataclasses.replace(
                child,
                checkpoint_path=request.parent_checkpoint_path,
                checkpoint_sha256=request.parent.checkpoint_sha256,
            )
        return child

    with pytest.raises(ValueError, match="alias parent"):
        run_generation(request, store, build, _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g0"


def test_duplicate_child_paths_fail(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    baseline = _child_builder(tmp_path, [])

    def build(context):
        child = baseline(context)
        if context.name == "child_C":
            return dataclasses.replace(
                child,
                checkpoint_path=str(tmp_path / "child_A.pt"),
                checkpoint_sha256=_sha(tmp_path / "child_A.pt"),
            )
        return child

    with pytest.raises(ValueError, match="unique"):
        run_generation(request, store, build, _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)


def test_duplicate_child_manifest_paths_fail(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    baseline = _child_builder(tmp_path, [])

    def build(context):
        child = baseline(context)
        if context.name == "child_C":
            return dataclasses.replace(
                child,
                manifest_path=str(tmp_path / "child_A.json"),
                manifest_sha256=_sha(tmp_path / "child_A.json"),
            )
        return child

    with pytest.raises(ValueError, match="unique"):
        run_generation(request, store, build, _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)


@pytest.mark.parametrize(
    "role, mode, match",
    [
        ("child", "extra", "child manifest schema"),
        ("child", "noncanonical", "non-canonical"),
        ("child", "dropped", "child manifest schema"),
        ("candidate", "extra", "candidate manifest schema"),
        ("candidate", "noncanonical", "non-canonical"),
        ("candidate", "swapped", "candidate manifest declaration"),
    ],
)
def test_noncanonical_manifests_fail(tmp_path: Path, role: str, mode: str, match: str):
    store, request = _fixtures(tmp_path)
    if role == "child":
        baseline = _child_builder(tmp_path, [])

        def build(context):
            child = baseline(context)
            if context.name != "child_A":
                return child
            payload = json.loads(Path(child.manifest_path).read_text())
            if mode == "extra":
                payload["extra"] = 1
            elif mode == "dropped":
                payload.pop("protocol_sha256")
            data = json.dumps(payload).encode() if mode == "noncanonical" else canonical_json_bytes(payload)
            Path(child.manifest_path).write_bytes(data)
            return dataclasses.replace(child, manifest_sha256=_sha(Path(child.manifest_path)))

    else:
        baseline = _candidate_builder(tmp_path, [])

        def build(context):
            candidate = baseline(context)
            payload = json.loads(Path(candidate.manifest_path).read_text())
            if mode == "extra":
                payload["extra"] = 1
            elif mode == "swapped":
                payload["leaf_hidden"] = 16
            data = json.dumps(payload).encode() if mode == "noncanonical" else canonical_json_bytes(payload)
            Path(candidate.manifest_path).write_bytes(data)
            return dataclasses.replace(candidate, manifest_sha256=_sha(Path(candidate.manifest_path)))

    with pytest.raises(ValueError, match=match):
        child_builder = build if role == "child" else _child_builder(tmp_path, [])
        run_generation(request, store, child_builder, build if role == "candidate" else _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)
    assert (tmp_path / "g1" / "failure.json").is_file()
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g0"


def test_child_config_digest_must_match_payload(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    baseline = _child_builder(tmp_path, [])

    def build(context):
        child = baseline(context)
        return dataclasses.replace(child, config_sha256="9" * 64) if context.name == "child_C" else child

    with pytest.raises(ValueError, match="manifest declaration"):
        run_generation(request, store, build, _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)


def test_child_config_hashes_must_be_identical(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    baseline = _child_builder(tmp_path, [])

    def build(context):
        child = baseline(context)
        if context.name == "child_C":
            other = _base_config()
            other.model.loop_depth = 2
            other.model.num_layers = 2
            validate_config(other)
            torch.manual_seed(7)
            checkpoint = _save_checkpoint(tmp_path / "child_C.pt", HAGI(other).state_dict(), other)
            manifest = json.loads(Path(child.manifest_path).read_text())
            manifest["checkpoint_sha256"] = _sha(checkpoint)
            manifest["config_sha256"] = _config_digest(other)
            Path(child.manifest_path).write_bytes(canonical_json_bytes(manifest))
            return dataclasses.replace(
                child,
                checkpoint_path=str(checkpoint),
                checkpoint_sha256=_sha(checkpoint),
                manifest_sha256=_sha(Path(child.manifest_path)),
                config_sha256=_config_digest(other),
            )
        return child

    with pytest.raises(ValueError, match="config digests must be identical"):
        run_generation(request, store, build, _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g0"


def test_forged_candidate_metadata_fails(tmp_path: Path):
    cases = {
        "transform": ("transform_digest", "9" * 64, "transform"),
        "scale": ("logit_scale_target", 0.25, "scale"),
        "state-digest": ("candidate_state_key_digest", "9" * 64, "state key digest"),
        "parent-config": ("parent_config_sha256", "9" * 64, "parent config digest"),
        "leaf": ("leaf_hidden", 16, "leaf geometry"),
    }
    for name, (field, value, match) in cases.items():
        root = tmp_path / name
        store, request = _fixtures(root)
        baseline = _candidate_builder(root, [])

        def build(context, field=field, value=value):
            candidate = baseline(context)
            replacement = value
            if field == "logit_scale_target":
                replacement = candidate.logit_scale_source / 2.0
            candidate = dataclasses.replace(candidate, **{field: replacement})
            manifest = json.loads(Path(candidate.manifest_path).read_text())
            manifest[field] = replacement
            Path(candidate.manifest_path).write_bytes(canonical_json_bytes(manifest))
            return dataclasses.replace(
                candidate,
                manifest_sha256=_sha(Path(candidate.manifest_path)),
            )

        with pytest.raises(ValueError, match=match):
            run_generation(request, store, _child_builder(root, []), build, _evaluator(), owner_id=OWNER)
        assert (root / "g1" / "failure.json").is_file()


def test_candidate_payload_must_equal_reconstructed_f3_assembly(tmp_path: Path):
    """A candidate is accepted only if its bytes are the F3 assembly of the
    three child snapshots. Agreeing metadata cannot stand in for derivation."""
    store, request = _fixtures(tmp_path)
    baseline = _candidate_builder(tmp_path, [])

    def tamper(context):
        candidate = baseline(context)
        checkpoint = Path(candidate.checkpoint_path)
        payload = load_payload(checkpoint)
        payload["model"]["head.projection.weight"].add_(1)
        torch.save(payload, checkpoint)
        checkpoint_sha256 = _sha(checkpoint)
        manifest_path = Path(candidate.manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["candidate_checkpoint_sha256"] = checkpoint_sha256
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        return dataclasses.replace(
            candidate,
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=_sha(manifest_path),
        )

    before = (tmp_path / "current-parent.json").read_bytes()
    with pytest.raises(ValueError, match="derived F3 payload"):
        run_generation(
            request, store, _child_builder(tmp_path, []), tamper, _evaluator(),
            owner_id=OWNER,
        )
    assert (tmp_path / "current-parent.json").read_bytes() == before
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g0"
    assert (tmp_path / "g1" / "failure.json").is_file()


def test_parent_cannot_be_promoted_as_candidate(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    baseline = _candidate_builder(tmp_path, [])

    def build(context):
        candidate = baseline(context)
        return dataclasses.replace(
            candidate,
            checkpoint_path=context.parent_checkpoint_path,
            checkpoint_sha256=context.parent_checkpoint_sha256,
        )

    with pytest.raises(ValueError, match="recursive F3|manifest declaration"):
        run_generation(request, store, _child_builder(tmp_path, []), build, _evaluator(), owner_id=OWNER)
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g0"


def test_terminal_rerun_is_idempotent_without_callbacks(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    first = run_generation(request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []), _evaluator(candidate=(1.0, 1.05, 0.5)), owner_id=OWNER)
    before = (tmp_path / "g1" / "report.json").read_bytes()
    assert run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER) == first
    assert (tmp_path / "g1" / "report.json").read_bytes() == before


def test_terminal_recovery_uses_snapshots_after_sources_change(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    first = run_generation(request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)
    (tmp_path / "candidate.pt").unlink()
    (tmp_path / "candidate.json").write_bytes(b"mutated")
    (tmp_path / "child_A.json").write_bytes(b"mutated")
    (tmp_path / "child_A.pt").unlink()
    (tmp_path / "holdout.bin").write_bytes(b"mutated holdout")
    (tmp_path / "g1" / "holdout-evidence.json").unlink()
    resumed = run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER)
    assert resumed == first
    assert Path(resumed.candidate_checkpoint_path).is_file()
    assert load_payload(resumed.candidate_checkpoint_path)["completed_steps"] == 0
    assert (tmp_path / "g1" / "inputs").is_dir()


def test_persisted_f3_derivation_rejects_coherently_resigned_child_snapshot(
    tmp_path: Path,
):
    store, request = _fixtures(tmp_path)
    run_generation(
        request,
        store,
        _child_builder(tmp_path, []),
        _candidate_builder(tmp_path, []),
        _evaluator(),
        owner_id=OWNER,
    )
    run = tmp_path / "g1"
    metadata = json.loads(
        (run / "holdout-evidence.json").read_text(encoding="utf-8")
    )["candidate_f3"]
    source_path = run / "inputs" / (
        f"child_A-checkpoint-{metadata['child_checkpoint_sha256'][0]}.bin"
    )
    payload = load_payload(source_path)
    payload["model"]["encoder.embedding.weight"].add_(1)
    forged_path = run / "inputs" / "forged-child.pt"
    torch.save(payload, forged_path)
    new_digest = _sha(forged_path)
    target_path = run / "inputs" / f"child_A-checkpoint-{new_digest}.bin"
    forged_path.replace(target_path)
    child_paths = [
        str(target_path),
        *(
            str(run / "inputs" / f"{name}-checkpoint-{digest}.bin")
            for name, digest in zip(
                ("child_B", "child_C"), metadata["child_checkpoint_sha256"][1:]
            )
        ),
    ]
    metadata["child_checkpoint_paths"] = child_paths
    metadata["child_checkpoint_sha256"][0] = new_digest

    with pytest.raises(
        ValueError, match="terminal candidate does not match derived F3 payload"
    ):
        _validate_persisted_f3_derivation(metadata, run)


def test_terminal_replay_invokes_durable_f3_reconstruction(
    tmp_path: Path, monkeypatch
):
    import hagi.orchestrator.recursive as recursive_module

    store, request = _fixtures(tmp_path)
    run_generation(
        request,
        store,
        _child_builder(tmp_path, []),
        _candidate_builder(tmp_path, []),
        _evaluator(),
        owner_id=OWNER,
    )
    calls: list[tuple[Path, frozenset[str]]] = []
    original = recursive_module._validate_persisted_f3_derivation

    def track(metadata, run):
        calls.append((Path(run), frozenset(metadata)))
        return original(metadata, run)

    monkeypatch.setattr(
        recursive_module, "_validate_persisted_f3_derivation", track
    )
    run_generation(
        request, store, *_forbidden_callbacks(), owner_id=OWNER
    )

    assert calls == [(tmp_path / "g1", frozenset(recursive_module._CANDIDATE_EVIDENCE_FIELDS))]


@pytest.mark.parametrize(
    "decision,flag,raises",
    [
        ("accepted", True, False),
        # The discriminating row: the old invariant accepted this, the new one
        # rejects it. Without it the test cannot distinguish the fix from a
        # relaxation of the check.
        ("accepted", False, True),
        ("rejected", False, False),
        ("rejected", True, True),
    ],
)
def test_generation_result_mechanism_flag_must_follow_decision(
    decision: str, flag: bool, raises: bool
) -> None:
    """The mechanism claim must be derivable, so it must be constrained.

    ``mechanism_supported`` used to be pinned to ``True`` by the writer, so a
    rejected generation persisted support it had not earned. The invariant is
    now bidirectional: neither ``rejected`` + ``True`` (the original defect)
    nor ``accepted`` + ``False`` (the silent-loss direction) can be built.
    """
    def build() -> GenerationResult:
        return GenerationResult(
            decision=decision,
            generation_id="g-invariant",
            report_path="r",
            manifest_path="m",
            candidate_checkpoint_path="c",
            holdout_evidence_path="e",
            incumbent_macro_ce=1.0,
            candidate_macro_ce=1.0,
            ce_regression=0.0,
            worst_source_regression=0.0,
            mechanism_supported=flag,
            quality_supported=False,
            security_supported=False,
            production_promotion=False,
            pareto_improvement=False,
        )

    if raises:
        with pytest.raises(ValueError):
            build()
    else:
        assert build().mechanism_supported is flag


@pytest.mark.parametrize(
    "claim",
    ("mechanism_supported", "pareto_improvement"),
)
def test_terminal_replay_rejects_integer_holdout_claim_tampering(
    tmp_path: Path, claim: str
):
    store, request = _fixtures(tmp_path)
    first = run_generation(
        request,
        store,
        _child_builder(tmp_path, []),
        _candidate_builder(tmp_path, []),
        _evaluator(),
        owner_id=OWNER,
    )
    evidence_path = Path(first.holdout_evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence[claim] = 1
    evidence_path.write_bytes(canonical_json_bytes(evidence))

    with pytest.raises(ValueError, match="bound evidence digest mismatch"):
        run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER)


def test_terminal_evidence_tampering_fails_closed(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    first = run_generation(request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)
    snapshot = Path(first.holdout_evidence_path)
    payload = json.loads(snapshot.read_text())
    payload["candidate_macro_ce"] += 0.1
    snapshot.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError):
        run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER)
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g1"


def test_terminal_evidence_candidate_binding_tampering_fails(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    first = run_generation(request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)
    snapshot = Path(first.holdout_evidence_path)
    payload = json.loads(snapshot.read_text())
    payload["candidate_f3"]["checkpoint_sha256"] = "5" * 64
    snapshot.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError):
        run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER)
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g1"


def test_request_mismatch_with_same_generation_fails_before_callbacks(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    run_generation(request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)
    other = _write(tmp_path / "other-holdout.bin", b"other")
    mismatched = dataclasses.replace(
        request, holdout=dataclasses.replace(request.holdout, path=str(other), sha256=_sha(other))
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        run_generation(mismatched, store, *_forbidden_callbacks(), owner_id=OWNER)


def test_concurrent_parent_transition_never_promotes_stale_branch(tmp_path: Path, monkeypatch):
    store, request = _fixtures(tmp_path)
    conflict = ParentPointer("other", "7" * 64, "8" * 64, "9" * 64, owner_id=OWNER)

    def concurrent(*args, **kwargs):
        (tmp_path / "current-parent.json").write_bytes(canonical_json_bytes(conflict.as_dict()))
        raise OSError("concurrent transition")

    monkeypatch.setattr(store, "commit_accepted_terminal", concurrent)
    with pytest.raises(OSError, match="concurrent"):
        run_generation(request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)
    state = json.loads((tmp_path / "g1" / "state.json").read_text())
    assert state["state"] == "prepared"
    assert not (tmp_path / "g1" / "report.json").exists()
    monkeypatch.undo()
    with pytest.raises(ValueError, match="parent pointer CAS conflict"):
        run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER)
    assert store.read_parent() == conflict


def test_accepted_crash_before_parent_commit_keeps_prepared_and_old_parent(
    tmp_path: Path, monkeypatch
):
    store, request = _fixtures(tmp_path)
    before = (tmp_path / "current-parent.json").read_bytes()

    import hagi.orchestrator.state as state_module

    original = state_module._atomic_write

    def crash_before_pointer(path, payload):
        if path.name == "current-parent.json":
            raise OSError("injected crash before parent commit")
        return original(path, payload)

    monkeypatch.setattr(state_module, "_atomic_write", crash_before_pointer)
    with pytest.raises(OSError, match="before parent commit"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )

    assert (tmp_path / "current-parent.json").read_bytes() == before
    state = json.loads((tmp_path / "g1" / "state.json").read_text())
    assert state["state"] == "prepared"
    assert (tmp_path / "g1" / "prepared-report.json").is_file()
    assert not (tmp_path / "g1" / "report.json").exists()

    monkeypatch.undo()
    resumed = run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER)
    assert resumed.decision == "accepted"
    assert store.read_parent().generation_id == "g1"
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "terminal_accepted"


def test_accepted_crash_before_parent_commit_resumes_after_lease_takeover(
    tmp_path: Path, monkeypatch
):
    store, request = _fixtures(tmp_path)
    import hagi.orchestrator.state as state_module

    original = state_module._atomic_write

    def crash_before_pointer(path, payload):
        if path.name == "current-parent.json":
            raise OSError("injected crash before parent commit")
        return original(path, payload)

    monkeypatch.setattr(state_module, "_atomic_write", crash_before_pointer)
    with pytest.raises(OSError, match="before parent commit"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )
    monkeypatch.undo()

    run = tmp_path / "g1"
    parent_before = (tmp_path / "current-parent.json").read_bytes()
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    store.recover_stale_lock(
        "g1", "owner-b", "owner crashed before parent commit"
    )

    assert (tmp_path / "current-parent.json").read_bytes() == parent_before
    resumed = run_generation(
        request, store, *_forbidden_callbacks(), owner_id="owner-b"
    )

    assert resumed.decision == "accepted"
    assert store.read_parent().generation_id == "g1"
    assert store.read_parent().owner_id == "owner-b"
    assert (run / "report.json").is_file()
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["owner_id"] == "owner-b"
    assert state["state"] == "terminal_accepted"


def test_accepted_crash_after_parent_commit_resumes_report_and_state(
    tmp_path: Path, monkeypatch
):
    store, request = _fixtures(tmp_path)

    import hagi.orchestrator.state as state_module

    original = state_module._atomic_write
    fired = False

    def crash_after_pointer(path, payload):
        nonlocal fired
        result = original(path, payload)
        if path.name == "current-parent.json" and not fired:
            fired = True
            raise OSError("injected crash after parent commit")
        return result

    monkeypatch.setattr(state_module, "_atomic_write", crash_after_pointer)
    with pytest.raises(OSError, match="after parent commit"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )

    assert store.read_parent().generation_id == "g1"
    state = json.loads((tmp_path / "g1" / "state.json").read_text())
    assert state["state"] == "prepared"
    assert not (tmp_path / "g1" / "report.json").exists()

    monkeypatch.undo()
    resumed = run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER)
    assert resumed.decision == "accepted"
    assert (tmp_path / "g1" / "report.json").is_file()
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "terminal_accepted"
    assert store.read_parent().generation_id == "g1"


def test_accepted_crash_after_parent_commit_resumes_after_lease_takeover(
    tmp_path: Path, monkeypatch
):
    store, request = _fixtures(tmp_path)
    import hagi.orchestrator.state as state_module

    original = state_module._atomic_write
    fired = False

    def crash_after_pointer(path, payload):
        nonlocal fired
        result = original(path, payload)
        if path.name == "current-parent.json" and not fired:
            fired = True
            raise OSError("injected crash after parent commit")
        return result

    monkeypatch.setattr(state_module, "_atomic_write", crash_after_pointer)
    with pytest.raises(OSError, match="after parent commit"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )
    monkeypatch.undo()

    run = tmp_path / "g1"
    parent_before = (tmp_path / "current-parent.json").read_bytes()
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    store.recover_stale_lock(
        "g1", "owner-b", "owner crashed after parent commit"
    )

    resumed = run_generation(
        request, store, *_forbidden_callbacks(), owner_id="owner-b"
    )

    assert resumed.decision == "accepted"
    assert (tmp_path / "current-parent.json").read_bytes() == parent_before
    assert store.read_parent().generation_id == "g1"
    assert store.read_parent().owner_id == OWNER
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["owner_id"] == "owner-b"
    assert state["state"] == "terminal_accepted"


def test_completed_terminal_resumes_after_lease_takeover(tmp_path: Path):
    store, request = _fixtures(tmp_path)
    first = run_generation(
        request,
        store,
        _child_builder(tmp_path, []),
        _candidate_builder(tmp_path, []),
        _evaluator(),
        owner_id=OWNER,
    )
    run = tmp_path / "g1"
    parent_before = (tmp_path / "current-parent.json").read_bytes()
    lease_path = run / ".owner.lease"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["expires_at"] = 0
    lease_path.write_bytes(canonical_json_bytes(lease))
    store.recover_stale_lock("g1", "owner-b", "owner crashed after terminal")

    resumed = run_generation(
        request, store, *_forbidden_callbacks(), owner_id="owner-b"
    )

    assert resumed == first
    assert (tmp_path / "current-parent.json").read_bytes() == parent_before
    assert store.read_parent().owner_id == OWNER
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert state["owner_id"] == "owner-b"
    assert state["state"] == "terminal_accepted"


def test_accepted_crash_after_report_commit_resumes_terminal_state(
    tmp_path: Path, monkeypatch
):
    store, request = _fixtures(tmp_path)

    import hagi.orchestrator.state as state_module

    original = state_module.GrowthRunStore._publish_regular_no_replace
    fired = False

    def crash_after_report(self, path, data, error_message):
        nonlocal fired
        result = original(self, path, data, error_message)
        if path.name == "report.json" and not fired:
            fired = True
            raise OSError("injected crash after report publication")
        return result

    monkeypatch.setattr(
        state_module.GrowthRunStore,
        "_publish_regular_no_replace",
        crash_after_report,
    )
    with pytest.raises(OSError, match="after report publication"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )

    assert store.read_parent().generation_id == "g1"
    assert (tmp_path / "g1" / "report.json").is_file()
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "prepared"

    monkeypatch.undo()
    resumed = run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER)
    assert resumed.decision == "accepted"
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "terminal_accepted"


def test_accepted_crash_after_terminal_state_is_idempotent(
    tmp_path: Path, monkeypatch
):
    store, request = _fixtures(tmp_path)

    import hagi.orchestrator.state as state_module

    original = state_module._atomic_write
    fired = False

    def crash_after_state(path, payload):
        nonlocal fired
        result = original(path, payload)
        if (
            path.name == "state.json"
            and json.loads(payload)["state"] == "terminal_accepted"
            and not fired
        ):
            fired = True
            raise OSError("injected crash after terminal state")
        return result

    monkeypatch.setattr(state_module, "_atomic_write", crash_after_state)
    with pytest.raises(OSError, match="after terminal state"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )

    report_before = (tmp_path / "g1" / "report.json").read_bytes()
    parent_before = (tmp_path / "current-parent.json").read_bytes()
    assert store.read_parent().generation_id == "g1"
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "terminal_accepted"

    monkeypatch.undo()
    resumed = run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER)
    assert resumed.decision == "accepted"
    assert (tmp_path / "g1" / "report.json").read_bytes() == report_before
    assert (tmp_path / "current-parent.json").read_bytes() == parent_before


def test_prepared_crash_resumes_terminal_and_cas(tmp_path: Path, monkeypatch):
    import hagi.orchestrator.state as state_module

    store, request = _fixtures(tmp_path)
    original = state_module._atomic_write

    def crash(path, payload):
        if path.name == "state.json" and json.loads(payload)["state"] == "prepared":
            raise OSError("injected crash before PREPARED state")
        return original(path, payload)

    monkeypatch.setattr(state_module, "_atomic_write", crash)
    with pytest.raises(OSError, match="before PREPARED"):
        run_generation(request, store, _child_builder(tmp_path, []), _candidate_builder(tmp_path, []), _evaluator(), owner_id=OWNER)
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "running"
    assert (tmp_path / "g1" / "prepared-report.json").is_file()
    assert not (tmp_path / "g1" / "failure.json").exists()
    assert not (tmp_path / "g1" / "report.json").exists()
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g0"
    monkeypatch.undo()
    resumed = run_generation(request, store, *_forbidden_callbacks(), owner_id=OWNER)
    assert resumed.decision == "accepted"
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g1"


def test_prepared_crash_with_mismatched_request_does_not_commit(
    tmp_path: Path, monkeypatch
):
    import hagi.orchestrator.state as state_module

    store, request = _fixtures(tmp_path)
    original = state_module._atomic_write

    def crash(path, payload):
        if path.name == "state.json" and json.loads(payload)["state"] == "prepared":
            raise OSError("injected crash before PREPARED state")
        return original(path, payload)

    monkeypatch.setattr(state_module, "_atomic_write", crash)
    with pytest.raises(OSError, match="before PREPARED"):
        run_generation(
            request,
            store,
            _child_builder(tmp_path, []),
            _candidate_builder(tmp_path, []),
            _evaluator(),
            owner_id=OWNER,
        )
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "running"
    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g0"
    assert not (tmp_path / "g1" / "report.json").exists()

    monkeypatch.undo()
    other = _write(tmp_path / "other-holdout.bin", b"other")
    mismatched = dataclasses.replace(
        request,
        holdout=dataclasses.replace(
            request.holdout, path=str(other), sha256=_sha(other)
        ),
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        run_generation(
            mismatched, store, *_forbidden_callbacks(), owner_id=OWNER
        )

    assert json.loads((tmp_path / "current-parent.json").read_text())["generation_id"] == "g0"
    assert json.loads((tmp_path / "g1" / "state.json").read_text())["state"] == "prepared"
    assert (tmp_path / "g1" / "prepared-report.json").is_file()
    assert not (tmp_path / "g1" / "report.json").exists()


def test_read_parent_is_fail_closed(tmp_path: Path):
    store = GrowthRunStore(tmp_path)
    assert store.read_parent() is None
    pointer = ParentPointer("g0", "1" * 64, "2" * 64, "3" * 64, owner_id=OWNER)
    store.bootstrap_parent(pointer, owner_id=OWNER, trusted_parent_token=TrustedParentToken(OWNER, pointer.digest()))
    assert store.read_parent() == pointer
    (tmp_path / "current-parent.json").write_bytes(canonical_json_bytes({"generation_id": "g0"}))
    with pytest.raises(ValueError, match="current-parent"):
        store.read_parent()


def test_source_metric_requires_finite_nonnegative_exact_ce():
    with pytest.raises(ValueError, match="nonnegative"):
        SourceMetric("A", 1, -0.1, ROW_IDS[0])
    with pytest.raises(ValueError, match="finite"):
        SourceMetric("A", 1, math.nan, ROW_IDS[0])
