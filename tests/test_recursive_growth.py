"""Focused mechanism and falsification tests for recursive ternary F3."""

from __future__ import annotations

import copy
import hashlib
import math
import struct

import pytest
import torch

from hagi.config import Config, MergeConfig, validate_config
from hagi.model.merge import (
    ParentPreservingTernaryLift,
    RecursiveF3HAGI,
    TernaryF3Tree,
    build_model_from_payload,
    f3_real_column_matrix,
    f3_real_row_matrix,
    parent_preserving_ternary_digest,
    parent_preserving_ternary_matrix,
)
from hagi.model.merge import (
    merge_recursive_f3 as _merge_recursive_f3,
)
from hagi.model.model import HAGI
from hagi.train.checkpoint import (
    IncompatibleCheckpointError,
    config_from_dict,
    load_model,
    load_payload,
    save_checkpoint,
)
from tests.conftest import tiny_config

C3_SHA256 = "07e2571f2bfd0ff907294851f9239e0bf0b858c91b47390e01b16838699b29ce"
R3_SHA256 = "7fcf44d923de72014765bff542cd12d124428b3d4cd3efa18baad5ce83dc9d60"
FP32_TOL = {"atol": 5e-5, "rtol": 5e-5}
CHILD_CONFIGS: tuple[Config, Config, Config] = ()


def merge_recursive_f3(*args, **kwargs):
    """Test wrapper: synthetic children share one canonical config."""
    kwargs.setdefault("child_configs", CHILD_CONFIGS)
    return _merge_recursive_f3(*args, **kwargs)


def _matrix_digest(matrix: torch.Tensor) -> str:
    raw = b"".join(struct.pack("<d", float(value)) for row in matrix.tolist() for value in row)
    return hashlib.sha256(raw).hexdigest()


def _base_config() -> Config:
    return tiny_config(
        **{
            "model.num_layers": 2,
            "model.hidden_size": 64,
            "model.attention.num_query_heads": 2,
            "model.attention.num_kv_heads": 2,
            "model.attention.head_dim": 32,
            "model.ffn.intermediate_size": 64,
            "model.embedding.conv_kernel": 1,
            "model.embedding.tie_lm_head": False,
            "model.ternary.enabled": False,
            "model.adapters.enabled": False,
            "model.cortex.enabled": False,
            "model.decision.enabled": False,
            "model.loop_depth": 2,
        }
    )


def _target_config(depth: int = 1, parent_depth: int | None = None) -> Config:
    cfg = _base_config()
    cfg.merge.enabled = True
    cfg.merge.n_experts = 3
    cfg.merge.expert_hidden = 64
    cfg.merge.mixer_type = "ternary_f3"
    cfg.merge.ternary_depth = depth
    cfg.merge.ternary_tree_schema_version = 1
    cfg.merge.expert_weight_source = "effective_sparse"
    cfg.merge.mixer_hadamard_groups = [3]
    cfg.model.hidden_size = 192
    cfg.model.attention.num_query_heads = 6
    cfg.model.attention.num_kv_heads = 6
    cfg.model.ffn.intermediate_size = 192
    validate_config(cfg)
    return cfg


def _children() -> list[dict[str, torch.Tensor]]:
    cfg = _base_config()
    models = []
    for index in range(3):
        torch.manual_seed(1234 + 1009 * index)
        models.append(HAGI(copy.deepcopy(cfg)))
    states = [dict(model.state_dict()) for model in models]
    assert not torch.equal(
        states[0]["blocks.0.attn.out_proj.weight"],
        states[1]["blocks.0.attn.out_proj.weight"],
    )
    assert not torch.equal(
        states[1]["blocks.0.attn.out_proj.weight"],
        states[2]["blocks.0.attn.out_proj.weight"],
    )
    return states


CHILD_CONFIGS = tuple(copy.deepcopy(_base_config()) for _ in range(3))


def _distinct_parent_copies(state: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Create distinct parent-derived child states without changing scalars."""
    copies: list[dict[str, torch.Tensor]] = []
    for index in range(3):
        child = {key: value.detach().clone() for key, value in state.items()}
        for key, value in child.items():
            if value.is_floating_point() and value.ndim == 2:
                child[key] = value + (index + 1) * 1e-4
        copies.append(child)
    return copies


def _target_from_parent(parent_cfg: Config, depth: int) -> Config:
    """Build the exact 3x-wide recursive config for a parent at depth-1."""
    cfg = copy.deepcopy(parent_cfg)
    cfg.merge.enabled = True
    cfg.merge.n_experts = 3
    cfg.merge.mixer_type = "ternary_f3"
    cfg.merge.ternary_depth = depth
    cfg.merge.ternary_tree_schema_version = 1
    cfg.merge.expert_weight_source = "effective_sparse"
    cfg.merge.mixer_hadamard_groups = [3]
    cfg.merge.expert_hidden = parent_cfg.model.hidden_size
    cfg.model.hidden_size = 3 * parent_cfg.model.hidden_size
    cfg.model.attention.num_query_heads = 3 * parent_cfg.model.attention.num_query_heads
    cfg.model.attention.num_kv_heads = 3 * parent_cfg.model.attention.num_kv_heads
    cfg.model.ffn.intermediate_size = 3 * parent_cfg.model.ffn.intermediate_size
    cfg.model.loop_depth = 2
    validate_config(cfg)
    return cfg


def _stage_streams(
    model: HAGI | RecursiveF3HAGI,
    input_ids: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay the production order while exposing each comparison boundary."""
    model.eval()
    positions = torch.arange(input_ids.shape[1], dtype=torch.long)
    hidden = model.encoder(input_ids, use_state=False)
    stages = [hidden]
    for block in model.blocks:
        hidden = block(hidden, positions, None)
        stages.append(hidden)
    normalized = model.out_norm(hidden)
    transformed = model._apply_mixers(normalized)
    return stages, normalized, transformed, model.head.logits(transformed)


def _independent_f3_row_matrix(depth: int, leaf_hidden: int) -> torch.Tensor:
    """Rebuild staged F3**depth without production transform helpers.

    ``permutation`` maps pair-major coordinates back to branch-major, so the
    row action in branch-major coordinates is ``P @ (I ⊗ R3) @ P.T``.
    """
    if depth < 1 or leaf_hidden < 1 or leaf_hidden % 2:
        raise ValueError("depth and leaf_hidden must be positive; leaf_hidden even")
    scale = 1.0 / math.sqrt(3.0)
    cosine = math.sqrt(3.0) / 2.0
    c3 = torch.tensor(
        [
            [scale, 0.0, scale, 0.0, scale, 0.0],
            [0.0, scale, 0.0, scale, 0.0, scale],
            [scale, 0.0, -0.5 * scale, -cosine * scale, -0.5 * scale, cosine * scale],
            [0.0, scale, cosine * scale, -0.5 * scale, -cosine * scale, -0.5 * scale],
            [scale, 0.0, -0.5 * scale, cosine * scale, -0.5 * scale, -cosine * scale],
            [0.0, scale, -cosine * scale, -0.5 * scale, cosine * scale, -0.5 * scale],
        ],
        dtype=torch.float64,
    )
    r3 = c3.transpose(0, 1).contiguous()
    assert _matrix_digest(c3) == C3_SHA256
    assert _matrix_digest(r3) == R3_SHA256

    def kron(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            (left.shape[0] * right.shape[0], left.shape[1] * right.shape[1]),
            dtype=torch.float64,
        )
        for row in range(left.shape[0]):
            for column in range(left.shape[1]):
                result[
                    row * right.shape[0] : (row + 1) * right.shape[0],
                    column * right.shape[1] : (column + 1) * right.shape[1],
                ] = left[row, column] * right
        return result

    def blockdiag3(matrix: torch.Tensor) -> torch.Tensor:
        size = matrix.shape[0]
        result = torch.zeros((3 * size, 3 * size), dtype=torch.float64)
        for branch in range(3):
            result[branch * size : (branch + 1) * size, branch * size : (branch + 1) * size] = matrix
        return result

    def permutation(block_dim: int) -> torch.Tensor:
        width = 3 * block_dim
        result = torch.zeros((width, width), dtype=torch.float64)
        for branch in range(3):
            for pair in range(block_dim // 2):
                for coordinate in range(2):
                    pair_major = pair * 6 + branch * 2 + coordinate
                    branch_major = branch * block_dim + pair * 2 + coordinate
                    result[pair_major, branch_major] = 1.0
        return result

    previous: torch.Tensor | None = None
    for level in range(depth):
        block_dim = leaf_hidden * 3**level
        pair_to_branch = permutation(block_dim)
        outer = (
            pair_to_branch.transpose(0, 1)
            @ kron(torch.eye(block_dim // 2, dtype=torch.float64), r3)
            @ pair_to_branch
        )
        previous = outer if previous is None else outer @ blockdiag3(previous)
    assert previous is not None
    return previous


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max())


def test_depth_one_f3_aggregate_invariant_end_to_end():
    """F3 preserves each child aggregate up to the declared 1/sqrt(3) gain."""
    ids = torch.tensor([[7, 19, 43, 91, 127]], dtype=torch.long)
    children = _children()
    child_models = []
    for child_state in children:
        child_model = HAGI(_base_config())
        child_model.load_state_dict(child_state, strict=True)
        child_models.append(child_model)
    target = merge_recursive_f3(
        _target_config(depth=1),
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
    )

    child_runs = [_stage_streams(model, ids) for model in child_models]
    target_stages, target_normalized, target_transformed, target_logits = _stage_streams(
        target, ids
    )

    for layer_index, target_stage in enumerate(target_stages):
        child_concat = torch.cat(
            [run[0][layer_index] for run in child_runs], dim=-1
        )
        stage_error = _max_abs(target_stage, child_concat)
        scale = float(child_concat.detach().abs().max())
        assert stage_error <= 5e-5 + 5e-5 * scale, (
            f"pre-F3 layer {layer_index} max_abs={stage_error:.9g}"
        )

    child_final_concat = torch.cat(
        [run[1] for run in child_runs], dim=-1
    )
    expected_transformed = target.target_tree.apply_row(child_final_concat)
    f3_error = _max_abs(target_transformed, expected_transformed)
    assert torch.allclose(target_transformed, expected_transformed, **FP32_TOL), (
        f"post-F3 hidden max_abs={f3_error:.9g}"
    )

    child_scale_sum = sum(float(child["head.logit_scale"].sum()) for child in children)
    expected_head = torch.cat(
        [
                child.head.projection.weight.detach()
                * (
                    float(child.head.logit_scale.detach().sum())
                    / child_scale_sum
                )
                for child in child_models

        ],
        dim=1,
    ) @ target.target_tree.row_matrix(dtype=target.head.projection.weight.dtype)
    head_weight_error = _max_abs(target.head.projection.weight, expected_head)
    assert torch.allclose(target.head.projection.weight, expected_head, **FP32_TOL), (
        f"head compensation max_abs={head_weight_error:.9g}"
    )
    assert torch.allclose(
        target.head.logit_scale,
        torch.tensor(
            sum(float(child.head.logit_scale.detach().sum()) for child in child_models)
            / math.sqrt(3.0)
        ),
        **FP32_TOL,
    )

    expected_logits = sum(run[3] for run in child_runs) / math.sqrt(3.0)
    logits_error = _max_abs(target_logits, expected_logits)
    assert torch.allclose(target_logits, expected_logits, **FP32_TOL), (
        f"aggregate logits max_abs={logits_error:.9g}"
    )
    assert target_normalized.shape == child_final_concat.shape


def test_depth_two_aggregate_invariant_after_accepted_depth_one_parent():
    """The accepted depth-1 parent composes with a new outer F3 exactly."""
    ids = torch.tensor([[3, 41, 77, 113, 173]], dtype=torch.long)
    gen1 = merge_recursive_f3(
        _target_config(depth=1),
        _children(),
        parent_depth=0,
        expert_weight_source="effective_sparse",
        child_configs=CHILD_CONFIGS,
    )
    gen1_cfg = copy.deepcopy(gen1.cfg)
    gen1_children = _distinct_parent_copies(dict(gen1.state_dict()))
    gen1_models = []
    for child_state in gen1_children:
        child_model = copy.deepcopy(gen1)
        child_model.load_state_dict(child_state, strict=True)
        gen1_models.append(child_model)
    gen1_runs = [_stage_streams(model, ids) for model in gen1_models]
    gen2 = merge_recursive_f3(
        _target_from_parent(gen1_cfg, depth=2),
        gen1_children,
        parent_depth=1,
        expert_weight_source="effective_sparse",
        child_configs=(copy.deepcopy(gen1_cfg),) * 3,
    )
    gen2_stages, gen2_normalized, gen2_transformed, gen2_logits = _stage_streams(
        gen2, ids
    )

    for layer_index, gen2_stage in enumerate(gen2_stages):
        parent_concat = torch.cat(
            [run[0][layer_index] for run in gen1_runs], dim=-1
        )
        stage_error = _max_abs(gen2_stage, parent_concat)
        scale = float(parent_concat.detach().abs().max())
        assert stage_error <= 5e-5 + 5e-5 * scale, (
            f"depth-2 pre-F3 layer {layer_index} max_abs={stage_error:.9g}"
        )

    gen1_normalized_concat = torch.cat(
        [run[1] for run in gen1_runs], dim=-1
    )
    expected_transformed = gen2.target_tree.apply_row(gen1_normalized_concat)
    f3_error = _max_abs(gen2_transformed, expected_transformed)
    assert torch.allclose(gen2_transformed, expected_transformed, **FP32_TOL), (
        f"depth-2 F3^2 hidden max_abs={f3_error:.9g}"
    )

    child_scale_sum = sum(
        float(child.head.logit_scale.detach().sum()) for child in gen1_models
    )
    expected_head = torch.cat(
        [
                child.head.projection.weight.detach()
                * (
                    float(child.head.logit_scale.detach().sum())
                    / child_scale_sum
                )
                for child in gen1_models

        ],
        dim=1,
    ) @ gen2.target_tree.row_matrix(dtype=gen2.head.projection.weight.dtype)
    head_weight_error = _max_abs(gen2.head.projection.weight, expected_head)
    assert torch.allclose(gen2.head.projection.weight, expected_head, **FP32_TOL), (
        f"depth-2 head provenance max_abs={head_weight_error:.9g}"
    )
    assert torch.allclose(
        gen2.head.logit_scale,
        torch.tensor(
            sum(float(child.head.logit_scale.detach().sum()) for child in gen1_models)
            / math.sqrt(3.0)
        ),
        **FP32_TOL,
    )

    expected_logits = sum(
        torch.einsum(
            "...h,vh->...v",
            run[1] * child.head.logit_scale,
            child.head.projection.weight,
        )
        for run, child in zip(gen1_runs, gen1_models, strict=True)
    ) / math.sqrt(3.0)
    logits_error = _max_abs(gen2_logits, expected_logits)
    assert torch.allclose(gen2_logits, expected_logits, **FP32_TOL), (
        f"depth-2 aggregate logits max_abs={logits_error:.9g}"
    )
    assert gen2_normalized.shape == gen1_normalized_concat.shape


def test_f3_lift_digests_orthogonality_and_independent_staged_matrix():
    column = f3_real_column_matrix()
    row = f3_real_row_matrix()
    assert _matrix_digest(column) == C3_SHA256
    assert _matrix_digest(row) == R3_SHA256
    assert torch.allclose(row, column.T)
    assert torch.allclose(column @ column.T, torch.eye(6, dtype=torch.float64))
    for depth in range(1, 4):
        expected = _independent_f3_row_matrix(depth, 4)
        assert torch.allclose(
            expected,
            TernaryF3Tree(depth, 4).row_matrix(dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


def test_parent_preserving_ternary_lift_is_orthogonal_identity_and_nontrivial():
    matrix = parent_preserving_ternary_matrix()
    assert matrix.dtype == torch.float64
    assert torch.allclose(matrix.T @ matrix, torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    assert torch.equal(matrix @ torch.ones(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64))
    assert matrix.shape == (3, 3)

    lift = ParentPreservingTernaryLift()
    base = torch.randn(2, 5, 1, 6, dtype=torch.float64)
    repeated = base.squeeze(2).unsqueeze(2).expand(2, 5, 3, 6).reshape(2, 5, 18)
    assert torch.allclose(lift.apply_row(repeated), repeated, atol=1e-12, rtol=1e-12)
    distinct = torch.randn(2, 5, 18, dtype=torch.float64)
    changed = lift.apply_row(distinct)
    assert changed.shape == distinct.shape
    assert not torch.equal(changed, distinct)
    assert torch.allclose(changed.norm(dim=(-2, -1)), distinct.norm(dim=(-2, -1)))
    assert torch.allclose(lift.apply_row_inverse(changed), distinct, atol=1e-12, rtol=1e-12)
    assert parent_preserving_ternary_digest() == "421b8bede29cdcade694a181d1ec104b2aa4aaacaa010b08782b1fa57cb5def4"


def test_f3_tree_staged_round_trip_and_matrix():
    for depth in range(1, 4):
        tree = TernaryF3Tree(depth, 2)
        x = torch.randn(3, 3**depth * 2, dtype=torch.float64)
        transformed = tree.apply_row(x)
        restored = tree.apply_row_inverse(transformed)
        assert torch.allclose(restored, x, atol=1e-10, rtol=1e-10)
        matrix = tree.row_matrix()
        identity = torch.eye(matrix.shape[0], dtype=torch.float64)
        assert torch.allclose(matrix @ matrix.T, identity, atol=1e-10, rtol=1e-10)
        assert torch.allclose(matrix.inverse(), tree.inverse_row_matrix(), atol=1e-10, rtol=1e-10)


def test_block_tree_norm_normalizes_each_leaf():
    from hagi.model.norms import BlockTreeNorm

    norm = BlockTreeNorm(2, 4)
    x = torch.ones(1, 3 * 3 * 4) * 3
    y = norm(x)
    assert tuple(y.shape) == tuple(x.shape)
    y = y.reshape(1, 3, 3, 4)
    for branch in range(3):
        for leaf in range(3):
            rms = y[0, branch, leaf].pow(2).mean().sqrt()
            assert torch.allclose(rms, torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_recursive_f3_assembly_exact_block_diagonals_and_forward():
    children = _children()
    target = _target_config()
    model = merge_recursive_f3(
        target,
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    assert isinstance(model, RecursiveF3HAGI)
    state = model.state_dict()
    for key, source_key in [
        ("blocks.0.attn.out_proj.weight", "blocks.0.attn.out_proj.weight"),
        ("blocks.0.mixer.mixer.gate.weight", "blocks.0.mixer.mixer.gate.weight"),
        ("blocks.0.mixer.mixer.up.weight", "blocks.0.mixer.mixer.up.weight"),
        ("blocks.0.mixer.mixer.down.weight", "blocks.0.mixer.mixer.down.weight"),
    ]:
        merged = state[key]
        for child_index, child_state in enumerate(children):
            source = child_state[source_key]
            row_slice = slice(child_index * 64, (child_index + 1) * 64)
            assert torch.equal(merged[row_slice, row_slice], source)

    assert model.logit_scale_source == pytest.approx(
        sum(float(child["head.logit_scale"].sum()) for child in children)
    )
    assert model.logit_scale_target == pytest.approx(
        model.logit_scale_source / math.sqrt(3.0)
    )
    assert torch.allclose(
        model.head.logit_scale,
        torch.tensor(model.logit_scale_source / math.sqrt(3.0)),
    )
    output = model(torch.randint(0, 512, (1, 4)), return_logits=True)
    assert tuple(output.hidden.shape) == (1, 4, 192)
    assert tuple(output.logits.shape) == (1, 4, 512)


def test_effective_sparse_does_not_reternarize_and_forbidden_drop_rejects(monkeypatch):
    children = _children()
    key = "blocks.0.attn.out_proj.weight"
    for child in children:
        with torch.no_grad():
            child[key].zero_()
            child[key][0, 0] = 0.37
            child[key][0, 1] = -1.91
    target = _target_config()
    with pytest.raises(ValueError, match="drop_expert_mixers"):
        merge_recursive_f3(target, children, drop_expert_mixers=True)

    monkeypatch.setattr(
        "hagi.model.merge._ternarize_block",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("effective_sparse must not ternarize")
        ),
    )
    model = merge_recursive_f3(
        target,
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    merged = model.state_dict()[key]
    assert torch.equal(merged[:64, :64], children[0][key])
    assert torch.count_nonzero(merged[:64, 64:]).item() == 0


def test_ternary_f3_keeps_loop_depth_technical_and_separate_from_growth_depth():
    cfg = _target_config()
    assert cfg.model.loop_depth == 2


def test_direct_hagi_rejects_ternary_f3_and_points_to_recursive_builders():
    with pytest.raises(
        ValueError,
        match=r"RecursiveF3HAGI or build_model_from_payload",
    ):
        HAGI(_target_config())



def test_ternary_f3_rejects_unigram_prior_until_receiver_semantics_are_implemented():
    cfg = _target_config()
    cfg.model.head.unigram_prior = True
    cfg.model.head.unigram_path = "prior.npy"
    with pytest.raises(ValueError, match="unigram_prior"):
        validate_config(cfg)


@pytest.mark.parametrize("child_index", [0, 1, 2])
@pytest.mark.parametrize(
    "forbidden_key",
    [
        "decision_head.weight",
        "blocks.0.adapters.pyramid.scale",
        "blocks.0.adapters.ttt_lora.lora_B",
    ],
)
def test_each_recursive_child_rejects_forbidden_adaptive_state(
    child_index: int, forbidden_key: str
):
    children = _children()
    children[child_index][forbidden_key] = torch.ones(1)
    with pytest.raises(ValueError, match="forbidden (state key|adaptive state)"):
        merge_recursive_f3(_target_config(), children, parent_depth=0)


@pytest.mark.parametrize("child_index", [0, 1, 2])
@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_each_recursive_child_rejects_nonfinite_tensors(
    child_index: int, nonfinite: float
):
    children = _children()
    children[child_index]["blocks.0.attn.out_proj.weight"][0, 0] = nonfinite
    with pytest.raises(ValueError, match="non-finite"):
        merge_recursive_f3(_target_config(), children, parent_depth=0)


def test_recursive_fail_closed_on_tied_head_count_and_legacy_state():
    children = _children()
    tied = _target_config()
    tied.model.embedding.tie_lm_head = True
    with pytest.raises(ValueError, match="tie_lm_head"):
        merge_recursive_f3(tied, children, parent_depth=0)

    with pytest.raises(ValueError, match="exactly three"):
        merge_recursive_f3(_target_config(), children[:2], parent_depth=0)

    forbidden = [dict(child) for child in children]
    for child in forbidden:
        child["mixers.0.gain"] = torch.zeros(())
    with pytest.raises(ValueError, match="forbidden state key"):
        merge_recursive_f3(_target_config(), forbidden, parent_depth=0)

    forbidden = [dict(child) for child in children]
    for child in forbidden:
        child["cortex.weights"] = torch.ones(1)
    with pytest.raises(ValueError, match="forbidden state key"):
        merge_recursive_f3(_target_config(), forbidden, parent_depth=0)


def test_recursive_f3_uses_sum_head_gain_and_per_leaf_branch_gain():
    children = _children()
    for index, child in enumerate(children):
        child["head.logit_scale"] = torch.tensor(
            [1.0 + 0.25 * index], dtype=children[0]["head.logit_scale"].dtype
        )
        child["blocks.0.attn.branch_scale.scale"] = torch.tensor(
            [0.75 + 0.5 * index], dtype=child["head.logit_scale"].dtype
        )

    merged = merge_recursive_f3(
        _target_config(), children, parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    expected_logit = sum(float(child["head.logit_scale"]) for child in children) / math.sqrt(3.0)
    expected_branch = torch.tensor(
        [float(child["blocks.0.attn.branch_scale.scale"]) for child in children]
    )
    assert merged.logit_scale_source == pytest.approx(3.75)
    assert merged.logit_scale_target == pytest.approx(expected_logit)
    assert merged.head.logit_scale.detach().item() == pytest.approx(expected_logit)
    assert torch.allclose(
        merged.blocks[0].attn.branch_scale.scale.detach(), expected_branch
    )


def test_recursive_f3_aggregate_logits_match_unequal_child_scales():
    children = _children()
    for index, child in enumerate(children):
        child["head.logit_scale"] = torch.tensor(
            [1.0 + 0.25 * index], dtype=children[0]["head.logit_scale"].dtype
        )
        child["blocks.0.attn.branch_scale.scale"] = torch.tensor(
            [0.75 + 0.5 * index], dtype=child["head.logit_scale"].dtype
        )
    ids = torch.tensor([[7, 19, 43, 91]], dtype=torch.long)
    child_models = []
    for child in children:
        model = HAGI(_base_config())
        model.load_state_dict(child, strict=True)
        child_models.append(model)
    merged = merge_recursive_f3(
        _target_config(), children, parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    actual = _stage_streams(merged, ids)[3]
    expected = sum(_stage_streams(model, ids)[3] for model in child_models) / math.sqrt(3.0)
    assert torch.allclose(actual, expected, **FP32_TOL)


@pytest.mark.parametrize("key", ["head.logit_scale", "blocks.0.attn.branch_scale.scale"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, -math.inf])
def test_recursive_f3_rejects_nonpositive_or_nonfinite_original_scalars(key, bad):
    children = _children()
    children[1][key] = children[0][key].clone()
    children[1][key].fill_(bad)
    with pytest.raises(ValueError, match="positive|non-finite"):
        merge_recursive_f3(_target_config(), children, parent_depth=0)


def test_recursive_f3_rejects_fingerprint_dtype_and_device_mismatch():
    children = _children()
    child_config = copy.deepcopy(_base_config())
    child_config.model.hidden_size = 32
    child_config.model.attention.num_query_heads = 1
    child_config.model.attention.num_kv_heads = 1
    child_config.model.attention.head_dim = 32
    child_config.model.ffn.intermediate_size = 64
    with pytest.raises(ValueError, match="canonical Config"):
        merge_recursive_f3(
            _target_config(), children, parent_depth=0,
            child_configs=(CHILD_CONFIGS[0], child_config, copy.deepcopy(CHILD_CONFIGS[0])),
        )

    mixed_dtype = [dict(child) for child in children]
    mixed_dtype[1]["blocks.0.attn.out_proj.weight"] = mixed_dtype[1][
        "blocks.0.attn.out_proj.weight"
    ].to(torch.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        merge_recursive_f3(_target_config(), mixed_dtype, parent_depth=0)

    if torch.cuda.is_available():
        mixed_device = [dict(child) for child in children]
        mixed_device[1]["blocks.0.attn.out_proj.weight"] = mixed_device[1][
            "blocks.0.attn.out_proj.weight"
        ].cuda()
        with pytest.raises(ValueError, match="device mismatch"):
            merge_recursive_f3(_target_config(), mixed_device, parent_depth=0)


def test_recursive_f3_runtime_config_describes_plain_effective_sparse_body():
    merged = merge_recursive_f3(
        _target_config(), _children(), parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    assert merged.cfg.model.ternary.enabled is False
    assert not any(
        type(module).__name__ == "BitLinear" for module in merged.modules()
    )


def test_ternarize_block_fails_closed_for_nonfinite_and_large_finite_values():
    from hagi.model.merge import _ternarize_block

    nonfinite = torch.zeros(2, 3)
    nonfinite[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        _ternarize_block(nonfinite, 1e-5)

    huge = torch.full((2, 3), torch.finfo(torch.float32).max)
    effective = _ternarize_block(huge, 1e-5)
    assert torch.isfinite(effective).all()


def test_two_generation_staged_handoff():
    gen0_children = _children()
    gen1 = merge_recursive_f3(
        _target_config(depth=1),
        gen0_children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    gen2_cfg = _target_config(depth=2)
    gen2_cfg.merge.expert_hidden = 192
    gen2_cfg.model.hidden_size = 576
    gen2_cfg.model.attention.num_query_heads = 18
    gen2_cfg.model.attention.num_kv_heads = 18
    gen2_cfg.model.ffn.intermediate_size = 576
    validate_config(gen2_cfg)
    gen1_children = _distinct_parent_copies(dict(gen1.state_dict()))
    gen2 = merge_recursive_f3(
        gen2_cfg,
        gen1_children,
        parent_depth=1,
        expert_weight_source="effective_sparse",
        child_configs=(copy.deepcopy(gen1.cfg),) * 3,
    )
    assert gen2.ternary_depth == 2
    assert gen2.leaf_hidden == 64
    assert gen2.out_norm.weight.shape == (3, 3, 64)
    assert gen2.logit_scale_target == pytest.approx(
        3.0 * gen1.logit_scale_target / math.sqrt(3.0)
    )
    output = gen2(torch.randint(0, 512, (1, 3)), return_logits=True)
    assert tuple(output.hidden.shape) == (1, 3, 576)
    assert tuple(output.logits.shape) == (1, 3, 512)


def test_recursive_f3_rejects_noncanonical_loop_depth_and_master_policy():
    target = _target_config(depth=1)
    target.model.loop_depth = 1
    with pytest.raises(ValueError, match="loop_depth=2"):
        validate_config(target)

    target = _target_config(depth=1)
    target.model.ternary.enabled = True
    target.train.ternary_fp32_master = True
    with pytest.raises(ValueError, match="FP32 masters"):
        validate_config(target)


def test_permuting_children_changes_branch_routing_but_not_aggregate_logits():
    children = _children()
    ids = torch.tensor([[7, 19, 43, 91]], dtype=torch.long)
    normal = merge_recursive_f3(
        _target_config(), children, parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    reordered = merge_recursive_f3(
        _target_config(), [children[1], children[2], children[0]],
        parent_depth=0, expert_weight_source="effective_sparse",
    )
    normal_run = _stage_streams(normal, ids)
    reordered_run = _stage_streams(reordered, ids)
    assert _max_abs(normal_run[2], reordered_run[2]) > 1e-3

    child_models = [
        HAGI(_base_config()) for child in children
    ]
    for model, child in zip(child_models, children, strict=True):
        model.load_state_dict(child, strict=True)
    child_runs = [_stage_streams(model, ids) for model in child_models]
    child_logits = sum(run[3] for run in child_runs)
    assert _max_abs(
        normal_run[3] - child_logits / math.sqrt(3.0),
        reordered_run[3] - child_logits / math.sqrt(3.0),
    ) < FP32_TOL["atol"]


def test_recursive_head_scale_rejects_sum_overflow_after_dtype_cast():
    children = _children()
    for child in children:
        child["head.logit_scale"] = torch.tensor(
            [2e38], dtype=torch.float32
        )
    with pytest.raises(ValueError, match="after child-dtype cast"):
        merge_recursive_f3(_target_config(), children, parent_depth=0)


def test_recursive_provenance_is_durable_and_round_trips_checkpoint(tmp_path):
    model = merge_recursive_f3(
        _target_config(), _children(), parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    state = model.state_dict()
    for key in RecursiveF3HAGI._PROVENANCE_KEYS:
        assert key in state
    assert RecursiveF3HAGI.is_recursive_state(state)

    path = save_checkpoint(model, model.cfg, 3, tmp_path)
    payload = load_payload(path)
    restored_cfg = config_from_dict(payload["config"])
    restored = RecursiveF3HAGI.from_state_dict(
        restored_cfg, payload["model"]
    )
    assert isinstance(restored, RecursiveF3HAGI)
    assert restored.child_config_fingerprint == model.child_config_fingerprint
    assert restored.transform_digest == model.transform_digest
    assert restored.logit_scale_source == pytest.approx(model.logit_scale_source)
    assert restored.logit_scale_target == pytest.approx(model.logit_scale_target)
    dispatched = build_model_from_payload(restored_cfg, payload["model"])
    assert isinstance(dispatched, RecursiveF3HAGI)
    ids = torch.tensor([[3, 17, 29]], dtype=torch.long)
    assert torch.equal(
        model(ids, return_logits=True).logits,
        restored(ids, return_logits=True).logits,
    )


def test_recursive_checkpoint_fails_closed_in_legacy_models(tmp_path):
    recursive = merge_recursive_f3(
        _target_config(), _children(), parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    path = save_checkpoint(recursive, recursive.cfg, 4, tmp_path)
    with pytest.raises(IncompatibleCheckpointError, match="does not match"):
        load_model(path, HAGI(_base_config()))

    child_config = _base_config()
    child_config.merge.enabled = True
    child_config.merge.n_experts = 3
    child_config.merge.expert_hidden = 64
    child_config.merge.mixer_type = "hadamard"
    child_config.merge.mixer_hadamard_groups = [3]
    child_config.model.hidden_size = 192
    child_config.model.attention.num_query_heads = 6
    child_config.model.attention.num_kv_heads = 6
    child_config.model.ffn.intermediate_size = 192
    child_config.validate = None
    legacy_merged = __import__(
        "hagi.model.merge", fromlist=["MergedHAGI"]
    ).MergedHAGI(child_config)
    with pytest.raises(IncompatibleCheckpointError, match="does not match"):
        load_model(path, legacy_merged)


def test_legacy_hagi_and_legacy_merge_remain_default():
    plain = _base_config()
    model = HAGI(plain)
    assert "mixers" not in dict(model.named_children())
    assert model.cfg.merge.mixer_type == "hadamard"
    assert model.cfg.merge.ternary_depth == 0


def _lift_target_config(mode: str, depth: int = 1) -> Config:
    cfg = _target_config(depth=depth)
    cfg.merge.ternary_lift_mode = mode
    validate_config(cfg)
    return cfg


def _identical_children() -> tuple[list[dict[str, torch.Tensor]], HAGI]:
    """Three byte-identical parent copies plus the parent model itself."""
    parent = HAGI(_base_config())
    state = {key: value.detach().clone() for key, value in parent.state_dict().items()}
    return [copy.deepcopy(state) for _ in range(3)], parent


def test_ternary_lift_mode_defaults_to_legacy_and_validates_explicitly():
    assert MergeConfig().ternary_lift_mode == "f3_tree"
    validate_config(_base_config())
    assert _base_config().merge.ternary_lift_mode == "f3_tree"

    bad = _base_config()
    bad.merge.ternary_lift_mode = "parent_preserving_typo"
    with pytest.raises(ValueError, match="ternary_lift_mode"):
        validate_config(bad)

    # The opt-in is only meaningful for the recursive ternary F3 body.
    legacy = _base_config()
    legacy.merge.ternary_lift_mode = "parent_preserving"
    with pytest.raises(ValueError, match="ternary_f3"):
        validate_config(legacy)

    assert _lift_target_config("parent_preserving").merge.ternary_lift_mode == (
        "parent_preserving"
    )


def test_parent_preserving_mode_preserves_parent_logits_for_three_identical_copies():
    ids = torch.tensor([[7, 19, 43, 91, 127]], dtype=torch.long)
    children, parent = _identical_children()
    parent.eval()
    parent_logits = parent(ids, return_logits=True).logits.detach()

    model = merge_recursive_f3(
        _lift_target_config("parent_preserving"),
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    model.eval()
    logits = model(ids, return_logits=True).logits.detach()
    assert tuple(logits.shape) == tuple(parent_logits.shape)
    error = _max_abs(logits, parent_logits)
    assert torch.allclose(logits, parent_logits, **FP32_TOL), (
        f"parent-preserving logits max_abs={error:.9g}"
    )
    # The gain-normalized head is the exact equivalent composition:
    # s_target = mean(s_i) and W = [w_1; w_2; w_3] @ Q.
    child_scale_sum = sum(float(child["head.logit_scale"].sum()) for child in children)
    assert model.logit_scale_target == pytest.approx(child_scale_sum / 3.0)
    assert model.logit_scale_source == pytest.approx(child_scale_sum)
    assert model.transform_digest == model.target_tree.transform_digest
    # coordinate_layout is a pure NAME: no geometry may leak into it, so it is
    # exactly the lift's layout rather than a geometry-qualified variant.
    assert (
        model.target_tree.coordinate_layout
        == ParentPreservingTernaryLift.coordinate_layout
        == "branch_major_outer_ternary"
    )
    assert ":" not in model.target_tree.coordinate_layout


def test_legacy_mode_keeps_its_aggregate_gain_for_three_identical_copies():
    ids = torch.tensor([[7, 19, 43, 91, 127]], dtype=torch.long)
    children, parent = _identical_children()
    parent.eval()
    parent_logits = parent(ids, return_logits=True).logits.detach()

    model = merge_recursive_f3(
        _lift_target_config("f3_tree"),
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    model.eval()
    logits = model(ids, return_logits=True).logits.detach()
    # Legacy F3 is byte-for-byte unchanged: identical parents aggregate to
    # sqrt(3) parent logits under the declared 1/sqrt(3) gain.
    assert torch.allclose(
        logits, parent_logits * math.sqrt(3.0), **FP32_TOL
    )
    assert not torch.allclose(logits, parent_logits, atol=1e-3, rtol=1e-3)
    legacy_tree_digest = model.target_tree.transform_digest
    assert model.target_tree.coordinate_layout == "branch_major_pair_interleaved"
    assert legacy_tree_digest != parent_preserving_ternary_digest()


def test_parent_preserving_checkpoint_rejects_replay_under_the_other_mode(tmp_path):
    children, parent = _identical_children()
    model = merge_recursive_f3(
        _lift_target_config("parent_preserving"),
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    path = save_checkpoint(model, model.cfg, 5, tmp_path)
    payload = load_payload(path)
    ids = torch.tensor([[3, 17, 29]], dtype=torch.long)
    restored = RecursiveF3HAGI.from_state_dict(
        config_from_dict(payload["config"]), payload["model"]
    )
    assert torch.equal(
        model(ids, return_logits=True).logits, restored(ids, return_logits=True).logits
    )
    assert build_model_from_payload(
        config_from_dict(payload["config"]), payload["model"]
    )(ids, return_logits=True).logits.shape[-1] == 512

    legacy_cfg = config_from_dict(payload["config"])
    legacy_cfg.merge.ternary_lift_mode = "f3_tree"
    with pytest.raises(ValueError, match="transform"):
        RecursiveF3HAGI.from_state_dict(legacy_cfg, payload["model"])

    parent_preserving_cfg = config_from_dict(payload["config"])
    parent_preserving_cfg.merge.ternary_lift_mode = "f3_tree"
    with pytest.raises(IncompatibleCheckpointError, match="does not match"):
        load_model(path, HAGI(_base_config()))
