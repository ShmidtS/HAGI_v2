"""Falsification tests for the cross-parent parent-preserving merge transform.

The mechanism under test is the *cross-parent* step of a recursive merge: the
transform that mixes the three parent streams. The legacy
:class:`~hagi.model.merge.TernaryF3Tree` maps a duplicated triple
``(x, x, x)`` to ``(sqrt(3) x, 0, 0)`` per complex coordinate, so merging
three identical copies of a model is not function-preserving. The
parent-preserving Q(pi/2) lift fixes the all-ones vector, so a duplicated
self-merge is the identity.

Each test here is a falsifier where it can be: they assert a property that
would FAIL for a near-identity or non-mixing transform, not merely that the
code runs. Mixing capacity is not claimed to be sufficient -- only preserved.

Claim boundary: this file asserts mechanism properties only. It sets no
quality, security, autonomy or promotion claim.
"""

from __future__ import annotations

import copy

import pytest
import torch

from hagi.config import Config, validate_config
from hagi.model.merge import (
    _CROSS_PARENT_TRANSFORMS,
    KNOWN_COORDINATE_LAYOUTS,
    CrossParentPreservingTernaryTree,
    RecursiveF3HAGI,
    TernaryF3Tree,
    parent_preserving_ternary_matrix,
)
from hagi.model.merge import (
    merge_recursive_f3 as _merge_recursive_f3,
)
from hagi.model.model import HAGI
from hagi.orchestrator.recursive import (
    CandidateArtifact,
    SelfImprovementLedger,
    SourceSpan,
)
from hagi.train.checkpoint import config_from_dict, load_payload, save_checkpoint
from tests.test_recursive_growth import (
    CHILD_CONFIGS,
    _base_config,
    _identical_children,
    _lift_target_config,
)

FP64 = {"atol": 1e-12, "rtol": 1e-12}
LEAF = 2


def merge_recursive_f3(*args, **kwargs):
    """Test wrapper: synthetic children share one canonical config."""
    kwargs.setdefault("child_configs", CHILD_CONFIGS)
    return _merge_recursive_f3(*args, **kwargs)


def _parent_stream(parent_depth: int, seed: int = 7) -> torch.Tensor:
    """One parent of width ``3**parent_depth * LEAF`` as a flat float64 row."""
    torch.manual_seed(seed)
    return torch.randn(1, 3**parent_depth * LEAF, dtype=torch.float64)


def _self_merge_input(parent_depth: int, seed: int = 7) -> torch.Tensor:
    """Three identical parent streams concatenated exactly as a merge sees them.

    At ``parent_depth > 0`` the parent itself is a recursive model, so the
    stream is first built by the inner child-to-parent transform
    (``TernaryF3Tree``), which is exactly the transform that must stay
    unchanged. Only the cross-parent step is under test here.
    """
    x = _parent_stream(parent_depth, seed)
    if parent_depth == 0:
        parent = x
    else:
        parent = TernaryF3Tree(parent_depth, LEAF).apply_row(x)
    return torch.cat([parent, parent, parent], dim=-1)


# F1 -- the property that fixes the measured bug.


@pytest.mark.parametrize("parent_depth", [0, 1, 2])
def test_f1_self_merge_of_three_identical_parents_is_unchanged(parent_depth: int) -> None:
    stream = _self_merge_input(parent_depth)
    tree = CrossParentPreservingTernaryTree(parent_depth + 1, LEAF)
    out = tree.apply_row(stream)
    assert out.shape == stream.shape
    assert torch.allclose(out, stream, **FP64), (
        f"F1 failed at parent_depth={parent_depth}: max_abs="
        f"{_max_abs(out, stream):.3e}"
    )
    # A near-identity transform would also pass this, so F5 below is the gate
    # that keeps this from being satisfied by a no-op. The control is a
    # DIFFERENT parent of the same width, in the same three-way shape.
    other = _self_merge_input(parent_depth, seed=8)
    assert not torch.allclose(out, other, **FP64)


# F2 -- orthogonality at the full width of the merge.


@pytest.mark.parametrize("parent_depth", [0, 1, 2])
def test_f2_full_width_operator_is_orthogonal(parent_depth: int) -> None:
    tree = CrossParentPreservingTernaryTree(parent_depth + 1, LEAF)
    matrix = tree.row_matrix(dtype=torch.float64)
    width = matrix.shape[0]
    assert width == 3 ** (parent_depth + 1) * LEAF
    identity = torch.eye(width, dtype=torch.float64)
    error = _max_abs(matrix.T @ matrix, identity)
    assert torch.allclose(matrix.T @ matrix, identity, **FP64), f"F2 failed: {error:.3e}"
    # The operator is the canonical lift, not a relabelling of it. As a row
    # action it is the Kronecker product ``kron(Q^T, I_{width//3})``: the flat
    # three-parent stream is split into three contiguous chunks of that width
    # and each chunk is mixed independently, which is exactly the layout
    # :class:`ParentPreservingTernaryLift` applies.
    expected = torch.kron(
        parent_preserving_ternary_matrix().t().contiguous(),
        torch.eye(width // 3, dtype=torch.float64),
    )
    assert torch.allclose(matrix, expected, **FP64), (
        f"F2 failed: row matrix is not kron(Q^T, I_{width // 3}); max_abs="
        f"{_max_abs(matrix, expected):.3e}"
    )


# F4 -- invertibility, so a merge stays reversible in principle.


@pytest.mark.parametrize("parent_depth", [0, 1, 2])
def test_f4_transform_is_invertible(parent_depth: int) -> None:
    tree = CrossParentPreservingTernaryTree(parent_depth + 1, LEAF)
    x = _parent_stream(parent_depth + 1, seed=11)
    restored = tree.apply_row_inverse(tree.apply_row(x))
    assert torch.allclose(restored, x, **FP64), (
        f"F4 failed at parent_depth={parent_depth}: max_abs={_max_abs(restored, x):.3e}"
    )
    matrix = tree.row_matrix(dtype=torch.float64)
    assert torch.allclose(
        matrix.inverse(), tree.inverse_row_matrix(), **FP64
    )


# F5 -- FALSIFIER. Guards against "fix the bug by doing nothing".


@pytest.mark.parametrize("parent_depth", [0, 1, 2])
def test_f5_transform_is_not_near_identity_on_distinct_parents(
    parent_depth: int,
) -> None:
    tree = CrossParentPreservingTernaryTree(parent_depth + 1, LEAF)
    parents = [
        _parent_stream(parent_depth, seed=seed) for seed in (21, 22, 23)
    ]
    distinct = torch.cat(parents, dim=-1)
    out = tree.apply_row(distinct)

    # (a) A near-identity transform would leave the stream alone.
    assert not torch.allclose(out, distinct, atol=1e-9), (
        f"F5 failed: transform is the identity at parent_depth={parent_depth}"
    )
    # (b) It must not merely permute or aggregate: every parent branch changes.
    per_parent = out.reshape(1, 3, -1)
    for branch in range(3):
        assert not torch.allclose(per_parent[0, branch], parents[branch], atol=1e-9)

    # (c) Mixing capacity: an impulse in ONE parent must reach all three
    # branches, and the total norm must be preserved. The split is 2/3, 2/3,
    # 1/3 in energy -- a genuine mix, not a copy and not a permutation.
    width = 3**parent_depth * LEAF
    impulse = torch.zeros(1, 3 * width, dtype=torch.float64)
    impulse[0, 0] = 1.0
    spread = tree.apply_row(impulse)
    branch_norms = spread.reshape(1, 3, width)[0].norm(dim=-1)
    assert torch.all(branch_norms > 1e-9), (
        f"F5 failed: impulse reached only {int((branch_norms > 1e-9).sum())}/3 "
        f"parent branches at parent_depth={parent_depth}"
    )
    assert torch.allclose(branch_norms, torch.tensor([2 / 3, 2 / 3, 1 / 3], dtype=torch.float64), **FP64), (
        f"F5 failed: unexpected impulse split: {branch_norms.tolist()}"
    )
    # Total energy across the three parent branches is preserved exactly.
    assert torch.allclose(spread.norm(), torch.tensor(1.0, dtype=torch.float64), **FP64), (
        f"F5 failed: impulse energy not conserved: {float(spread.norm()):.12f}"
    )

    # (d) Total norm of a general distinct stream is preserved (orthogonality).
    assert torch.allclose(out.norm(), distinct.norm(), **FP64)


def test_f5_distinct_parents_are_actually_mixed_pairwise() -> None:
    """Two parents equal, one different: the odd one must be spread."""
    tree = CrossParentPreservingTernaryTree(1, LEAF)
    a = _parent_stream(0, seed=31)
    stream = torch.cat([a, a, _parent_stream(0, seed=32)], dim=-1)
    out = tree.apply_row(stream).reshape(1, 3, -1)
    assert not torch.allclose(out[0, 0], a, atol=1e-9)
    assert not torch.allclose(out[0, 1], a, atol=1e-9)
    assert not torch.allclose(out[0, 2], a, atol=1e-9)


# The legacy transform is untouched and keeps its own (non-preserving) behaviour.


def test_ternary_f3_tree_is_unmodified_and_still_aggregates_duplicates() -> None:
    for parent_depth in (0, 1, 2):
        stream = _self_merge_input(parent_depth)
        legacy = TernaryF3Tree(parent_depth + 1, LEAF).apply_row(stream)
        assert not torch.allclose(legacy, stream, atol=1e-9), (
            "TernaryF3Tree must keep its aggregating behaviour; the fix belongs "
            f"to the cross-parent transform, not to TernaryF3Tree "
            f"(parent_depth={parent_depth})"
        )
    assert TernaryF3Tree(2, 4).transform_digest == TernaryF3Tree(2, 4).transform_digest


def test_new_transform_has_a_distinct_digest_from_the_legacy_tree() -> None:
    new_digest = CrossParentPreservingTernaryTree(2, 4).transform_digest
    legacy_digest = TernaryF3Tree(2, 4).transform_digest
    assert new_digest != legacy_digest
    # Digest is bound to geometry: a different depth or leaf width is a
    # different transform and must not share a digest.
    assert CrossParentPreservingTernaryTree(3, 4).transform_digest != new_digest
    assert CrossParentPreservingTernaryTree(2, 6).transform_digest != new_digest
    assert len(new_digest) == 64


def test_new_transform_validates_its_geometry() -> None:
    with pytest.raises(ValueError, match="depth must be a positive integer"):
        CrossParentPreservingTernaryTree(0, 4)
    with pytest.raises(ValueError, match="leaf_hidden must be a positive integer"):
        CrossParentPreservingTernaryTree(2, 0)
    tree = CrossParentPreservingTernaryTree(2, 4)
    with pytest.raises(ValueError, match="expected hidden width 36"):
        tree.apply_row(torch.zeros(1, 35, dtype=torch.float64))
    with pytest.raises(TypeError, match="float input"):
        tree.apply_row(torch.zeros(1, 36, dtype=torch.int64))


# The choice is explicit, not inferred, and is recorded in the provenance.


def test_cross_parent_transform_is_explicit_and_recorded() -> None:
    children, _parent = _identical_children()
    for mode in ("f3_tree", "parent_preserving"):
        cfg = _lift_target_config(mode)
        model = merge_recursive_f3(
            cfg,
            children,
            parent_depth=0,
            expert_weight_source="effective_sparse",
            cross_parent_transform=mode,
        )
        assert isinstance(model, RecursiveF3HAGI)
        assert model.cross_parent_transform == mode
        # The name is persisted, not just held in memory.
        assert "recursive_f3_cross_parent_transform" in model.state_dict()
        assert (
            bytes(model.recursive_f3_cross_parent_transform.detach().tolist()).decode(
                "utf-8"
            )
            == mode
        )
        assert model.transform_digest == model.target_tree.transform_digest
        if mode == "f3_tree":
            assert isinstance(model.target_tree, TernaryF3Tree)
        else:
            assert isinstance(model.target_tree, CrossParentPreservingTernaryTree)

    # The default is NOT the new transform: legacy behaviour stays reachable.
    legacy = merge_recursive_f3(
        _lift_target_config("f3_tree"),
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    assert legacy.cross_parent_transform == "f3_tree"


def test_unknown_cross_parent_transform_is_rejected() -> None:
    children, _parent = _identical_children()
    with pytest.raises(ValueError, match="cross_parent_transform must be one of"):
        merge_recursive_f3(
            _lift_target_config("f3_tree"),
            children,
            parent_depth=0,
            expert_weight_source="effective_sparse",
            cross_parent_transform="parent_preserving_typo",
        )


# F6 -- FALSIFIER. A checkpoint must not replay under the other transform.


def test_f6_checkpoint_round_trip_under_the_wrong_transform_raises(tmp_path) -> None:
    children, _parent = _identical_children()
    model = merge_recursive_f3(
        _lift_target_config("parent_preserving"),
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
        cross_parent_transform="parent_preserving",
    )
    path = save_checkpoint(model, model.cfg, 0, tmp_path)
    payload = load_payload(path)
    cfg = config_from_dict(payload["config"])

    # Same transform, named explicitly: must round-trip cleanly.
    restored = RecursiveF3HAGI.from_state_dict(
        cfg, payload["model"], cross_parent_transform="parent_preserving"
    )
    assert restored.cross_parent_transform == "parent_preserving"
    ids = torch.tensor([[5, 11, 23]], dtype=torch.long)
    assert torch.equal(
        model(ids, return_logits=True).logits, restored(ids, return_logits=True).logits
    )

    # F6: the OTHER transform, by name, must RAISE -- not silently succeed.
    with pytest.raises(ValueError, match="cross-parent transform"):
        RecursiveF3HAGI.from_state_dict(
            cfg, payload["model"], cross_parent_transform="f3_tree"
        )

    # F6 again, through the config boundary instead of the explicit name.
    legacy_cfg = config_from_dict(payload["config"])
    legacy_cfg.merge.ternary_lift_mode = "f3_tree"
    with pytest.raises(ValueError, match="cross-parent transform"):
        RecursiveF3HAGI.from_state_dict(legacy_cfg, payload["model"])

    # F6 both ways: a legacy checkpoint must not replay under the new transform.
    legacy_model = merge_recursive_f3(
        _lift_target_config("f3_tree"),
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
        cross_parent_transform="f3_tree",
    )
    legacy_path = save_checkpoint(legacy_model, legacy_model.cfg, 0, tmp_path)
    legacy_payload = load_payload(legacy_path)
    with pytest.raises(ValueError, match="cross-parent transform"):
        RecursiveF3HAGI.from_state_dict(
            config_from_dict(legacy_payload["config"]),
            legacy_payload["model"],
            cross_parent_transform="parent_preserving",
        )


def test_provenance_name_and_digest_both_gate_replay() -> None:
    """The recorded name and the recorded digest are both enforced.

    Each half forges a tampered state dict and requires ``from_state_dict`` to
    raise, so the test can fail: a state whose name is forged is rejected by
    name validation, and a state whose NAME is left intact but whose DIGEST
    buffer is mutated is rejected by the digest gate specifically.
    """
    children, _parent = _identical_children()
    model = merge_recursive_f3(
        _lift_target_config("parent_preserving"),
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
    )
    state = dict(model.state_dict())
    for key in RecursiveF3HAGI._PROVENANCE_KEYS:
        assert key in state, f"provenance key {key} is not persistent"

    # (a) Forge the NAME only. The transform must not exist, so replay fails
    # closed naming the recorded transform.
    name_buffer = state["recursive_f3_cross_parent_transform"]
    forged = torch.zeros_like(name_buffer)
    forged.view(-1)[: len(b"bogus")] = torch.frombuffer(
        bytearray(b"bogus"), dtype=torch.uint8
    )
    forged_name = dict(state)
    forged_name["recursive_f3_cross_parent_transform"] = forged
    with pytest.raises(ValueError, match="cross_parent_transform is unknown"):
        RecursiveF3HAGI.from_state_dict(model.cfg, forged_name)

    # (b) Forge the DIGEST only, leaving the name intact. The digest is a
    # 64-byte text buffer, so the forgery must stay valid UTF-8 of the same
    # length -- otherwise the failure would be a decode error rather than the
    # digest gate. The name gate must pass and the digest gate must be the
    # thing that raises.
    forged_digest = dict(state)
    digest_text = bytes(
        forged_digest["recursive_f3_transform_digest"].tolist()
    ).decode("utf-8")
    assert len(digest_text) == 64, digest_text
    flipped = ("1" if digest_text[0] != "1" else "2") + digest_text[1:]
    assert flipped != digest_text
    forged_digest["recursive_f3_transform_digest"] = torch.tensor(
        list(flipped.encode("utf-8")), dtype=torch.uint8
    )
    assert (
        forged_digest["recursive_f3_cross_parent_transform"]
        == state["recursive_f3_cross_parent_transform"]
    ).all(), "the digest forgery must leave the transform name untouched"
    with pytest.raises(ValueError, match="digest"):
        RecursiveF3HAGI.from_state_dict(model.cfg, forged_digest)

    # (c) The untampered state still replays, so (a) and (b) are not passing
    # because the state is rejected for some unrelated reason.
    RecursiveF3HAGI.from_state_dict(model.cfg, state)

    # The forged name is genuinely unknown, so the gate is not passing because
    # every name is rejected, nor because 'bogus' happens to be accepted.
    assert "bogus" not in _CROSS_PARENT_TRANSFORMS


# End-to-end: the mechanism survives a real model merge, not just matrices.


@pytest.mark.parametrize("mode", ["f3_tree", "parent_preserving"])
def test_merge_uses_the_selected_tree_for_the_receiver(mode: str) -> None:
    children, _parent = _identical_children()
    model = merge_recursive_f3(
        _lift_target_config(mode),
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
        cross_parent_transform=mode,
    )
    torch.manual_seed(5)
    stream = torch.randn(1, 3 * 64)
    assert torch.allclose(
        model.target_tree.apply_row(stream),
        model._apply_mixers(stream),
        **FP64,
    )


def test_new_transform_does_not_mutate_the_legacy_transform() -> None:
    """The two transforms must remain independent objects and digests."""
    legacy_before = TernaryF3Tree(3, 4).transform_digest
    _new = CrossParentPreservingTernaryTree(3, 4)
    assert TernaryF3Tree(3, 4).transform_digest == legacy_before
    # The pinned F3 constant is unchanged.
    from hagi.model.merge import _F3_C3_SHA256

    assert _F3_C3_SHA256 == (
        "07e2571f2bfd0ff907294851f9239e0bf0b858c91b47390e01b16838699b29ce"
    )


# STAGE -- the cross-parent mode is OUTER only. The inner child-to-parent
# step must keep using TernaryF3Tree, and must be bit-identical under both
# cross-parent modes, in every assembled part.


@pytest.mark.parametrize("mode", ["f3_tree", "parent_preserving"])
def test_parent_tree_is_the_legacy_inner_tree_in_both_modes(mode: str) -> None:
    """``parent_tree`` is the INNER child-to-parent step, never the cross one."""
    children, _parent = _identical_children()
    model = merge_recursive_f3(
        _lift_target_config(mode),
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
        cross_parent_transform=mode,
    )
    assert isinstance(model.parent_tree, TernaryF3Tree), (
        f"parent_tree must stay TernaryF3Tree in mode {mode}, got "
        f"{type(model.parent_tree).__name__}"
    )
    assert not isinstance(model.parent_tree, CrossParentPreservingTernaryTree), (
        f"the cross-parent tree must not be used for the inner step in mode {mode}"
    )
    # The cross-parent selection lives on target_tree, and it is the ONLY
    # place the mode is applied.
    if mode == "f3_tree":
        assert isinstance(model.target_tree, TernaryF3Tree)
    else:
        assert isinstance(model.target_tree, CrossParentPreservingTernaryTree)


def test_inner_assembly_is_identical_under_both_cross_parent_modes() -> None:
    """parent_depth=1: the outer mode must not perturb the inner step.

    With ``parent_depth=1`` the children are themselves recursive parents, so
    the inner ``TernaryF3Tree`` step is exercised. A regression that swapped
    ``parent_tree`` for the cross-parent tree would change every entry below.
    """
    children, _parent = _depth2_identical_children()
    child_configs = [_depth2_child_config() for _ in range(3)]
    merged = {}
    for mode in ("f3_tree", "parent_preserving"):
        model = _merge_recursive_f3(
            _depth2_target_config(mode),
            children,
            child_configs=child_configs,
            parent_depth=1,
            expert_weight_source="effective_sparse",
            cross_parent_transform=mode,
        )
        assert model.parent_depth == 1
        assert model.ternary_depth == 2
        # leaf_hidden = expert_hidden / 3**parent_depth = 192 / 3 = 64, and
        # the child width is exactly 3 * leaf_hidden, which is what makes this
        # a real inner step rather than the identity.
        assert model.leaf_hidden == 64
        assert model.leaf_hidden < 192, "leaf geometry collapsed -- inner step is identity"
        merged[mode] = model.state_dict()

    legacy, new = merged["f3_tree"], merged["parent_preserving"]
    assert set(legacy) == set(new)

    # Parts assembled by the INNER step / geometry, which the outer mode
    # must not touch.
    inner_keys = [key for key in sorted(legacy) if _is_inner_key(key)]
    assert inner_keys, "no inner-step keys were identified -- test is vacuous"
    for key in inner_keys:
        assert torch.equal(legacy[key], new[key]), (
            f"inner-step key {key!r} differs between cross-parent modes: "
            f"max_abs="
            f"{float((legacy[key] - new[key]).abs().max()):.3e}"
        )

    # The parts that legitimately DO differ: the receiver, the transform
    # provenance, and the head scale that compensates for it. Assert they
    # differ, otherwise the comparison above proves nothing.
    assert not torch.equal(
        legacy["head.projection.weight"], new["head.projection.weight"]
    ), "the two cross-parent modes produced the same receiver -- vacuous test"


def _depth2_identical_children() -> tuple[list[dict[str, torch.Tensor]], HAGI]:
    """Three identical DEPTH-1 recursive models used as the depth-2 children.

    A depth-2 target with ``parent_depth=1`` requires each child to be itself a
    recursive parent (its ``branch_scale`` has ``3**parent_depth = 3`` leaves
    and its widths are the target's ``expert_hidden``). A flat level-0 model
    is rejected, which is exactly why the earlier flat fixture could not
    exercise the inner step.
    """
    level0, _root = _identical_children()
    child_cfg = _lift_target_config("f3_tree")
    child = merge_recursive_f3(
        child_cfg,
        level0,
        parent_depth=0,
        expert_weight_source="effective_sparse",
        cross_parent_transform="f3_tree",
    )
    state = {key: value.detach().clone() for key, value in child.state_dict().items()}
    return [copy.deepcopy(state) for _ in range(3)], child


def _depth2_child_config() -> Config:
    """Config of the depth-1 recursive children (the depth-2 target's parents)."""
    return _lift_target_config("f3_tree")


def _depth2_target_config(mode: str) -> Config:
    cfg = _base_config()
    cfg.merge.enabled = True
    cfg.merge.n_experts = 3
    cfg.merge.expert_hidden = 192
    cfg.merge.mixer_type = "ternary_f3"
    cfg.merge.ternary_depth = 2
    cfg.merge.ternary_tree_schema_version = 1
    cfg.merge.expert_weight_source = "effective_sparse"
    cfg.merge.mixer_hadamard_groups = [3]
    cfg.merge.ternary_lift_mode = mode
    cfg.model.hidden_size = 3 * 192
    cfg.model.attention.num_query_heads = 18
    cfg.model.attention.num_kv_heads = 18
    cfg.model.attention.head_dim = 32
    cfg.model.ffn.intermediate_size = 3 * 192
    validate_config(cfg)
    return cfg


def _is_inner_key(key: str) -> bool:
    """Keys assembled by the inner child-to-parent step, not the receiver."""
    return key.endswith(
        (
            "out_norm.weight",
            ".attn.attn_norm.weight",
            ".mixer.norm.weight",
            ".attn.qkv_proj.weight",
            ".attn.out_proj.weight",
            ".mixer.gate.weight",
            ".mixer.up.weight",
            ".mixer.down.weight",
            ".branch_scale.scale",
            ".attn.q_norm.weight",
            ".attn.k_norm.weight",
            "encoder.embedding.weight",
        )
    )


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max())


# G-blocks -- ORCHESTRATOR ACCEPTANCE (the reported blocking defect).
#
# The tree's coordinate_layout used to be "branch_major_outer_ternary"
# concatenated with its geometry ("...:cross_parent:1:2"), while the
# orchestrator compared it against a single pinned string, so every
# parent-preserving candidate was rejected with
# ValueError("invalid coordinate layout") before it could ever be evaluated.
# These tests exercise the orchestrator's own validation dataclass, not a
# merge-level tensor unit, and are capable of failing.


def _artifact_fields(model: RecursiveF3HAGI, cfg: Config) -> dict[str, object]:
    """The fields a CandidateArtifact carries, from a real merged model."""
    depth = cfg.merge.ternary_depth
    leaf_hidden = cfg.merge.expert_hidden // (3 ** (depth - 1))
    return {
        "generation_id": "gen-0001",
        "parent_checkpoint_sha256": "a" * 64,
        "parent_config_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "checkpoint_path": "candidate.pt",
        "checkpoint_sha256": "d" * 64,
        "manifest_path": "candidate.json",
        "manifest_sha256": "e" * 64,
        "candidate_config_sha256": "f" * 64,
        "candidate_state_key_digest": "0" * 64,
        "ternary_depth": depth,
        "leaf_hidden": leaf_hidden,
        "coordinate_layout": model.target_tree.coordinate_layout,
        "transform_digest": model.target_tree.transform_digest,
        "weight_source": cfg.merge.expert_weight_source,
        "logit_scale_source": float(model.recursive_f3_logit_scale_source.item()),
        "logit_scale_target": float(model.recursive_f3_logit_scale_target.item()),
        "child_checkpoint_sha256": ("1" * 64, "2" * 64, "3" * 64),
        "child_config_sha256": ("4" * 64, "4" * 64, "4" * 64),
        "self_improve": _orchestrator_ledger(),
    }


def _orchestrator_ledger() -> SelfImprovementLedger:
    prompt_span = SourceSpan("A", 0, 4, "5" * 64)
    return SelfImprovementLedger(
        seed=1,
        source_id="A",
        prompt_span=prompt_span,
        prompt_token_sha256="6" * 64,
        prompt_text_sha256="7" * 64,
        prompt_token_count=4,
        optimizer_parameter_ids=("blocks.0.adapters.pyramid.scale",),
        accepted_updates=0,
        derived_base_state_sha256="8" * 64,
        contour_state_sha256="9" * 64,
    )


@pytest.mark.parametrize("mode", ["f3_tree", "parent_preserving"])
def test_orchestrator_accepts_a_candidate_under_either_cross_parent_transform(
    mode: str,
) -> None:
    children, _parent = _identical_children()
    cfg = _lift_target_config(mode)
    model = merge_recursive_f3(
        cfg,
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
        cross_parent_transform=mode,
    )
    # The candidate the real orchestrator would receive, built from the merge.
    artifact = CandidateArtifact(**_artifact_fields(model, cfg))
    assert artifact.coordinate_layout == model.target_tree.coordinate_layout
    assert artifact.transform_digest == model.transform_digest


def test_orchestrator_rejects_an_unknown_coordinate_layout() -> None:
    children, _parent = _identical_children()
    cfg = _lift_target_config("parent_preserving")
    model = merge_recursive_f3(
        cfg,
        children,
        parent_depth=0,
        expert_weight_source="effective_sparse",
        cross_parent_transform="parent_preserving",
    )
    fields = _artifact_fields(model, cfg)
    for bogus in (
        "branch_major_outer_ternary:cross_parent:1:2",  # the old geometry name
        "branch_major_ouster_ternary",  # typo
        "",
    ):
        with pytest.raises(ValueError, match="coordinate layout"):
            CandidateArtifact(**{**fields, "coordinate_layout": bogus})


def test_known_layouts_cover_both_transforms_and_nothing_else() -> None:
    assert KNOWN_COORDINATE_LAYOUTS == frozenset(
        {"branch_major_pair_interleaved", "branch_major_outer_ternary"}
    )
    assert KNOWN_COORDINATE_LAYOUTS is not None
    # The layout a candidate declares is a pure name: no geometry in it.
    for depth in (1, 2):
        tree = CrossParentPreservingTernaryTree(depth, 2)
        assert tree.coordinate_layout in KNOWN_COORDINATE_LAYOUTS
        assert ":" not in tree.coordinate_layout
        assert TernaryF3Tree(depth, 2).coordinate_layout in KNOWN_COORDINATE_LAYOUTS
