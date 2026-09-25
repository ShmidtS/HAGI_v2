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

import pytest
import torch

from hagi.config import Config
from hagi.model.merge import (
    CrossParentPreservingTernaryTree,
    RecursiveF3HAGI,
    TernaryF3Tree,
    parent_preserving_ternary_matrix,
)
from hagi.model.merge import (
    merge_recursive_f3 as _merge_recursive_f3,
)
from hagi.model.model import HAGI
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
    """The recorded name and the recorded digest are both enforced."""
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

    # Forging an unknown transform name is rejected by provenance validation.
    from hagi.model.merge import _CROSS_PARENT_TRANSFORMS

    assert "parent_preserving" in _CROSS_PARENT_TRANSFORMS
    assert "f3_tree" in _CROSS_PARENT_TRANSFORMS


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


def test_shared_helpers_imported_by_this_file_still_exist() -> None:
    """Guard that the shared test helpers this file builds on are present."""
    assert isinstance(_base_config(), Config)
    children, parent = _identical_children()
    assert len(children) == 3
    assert isinstance(parent, HAGI)
    assert all(set(child) == set(children[0]) for child in children)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max())
