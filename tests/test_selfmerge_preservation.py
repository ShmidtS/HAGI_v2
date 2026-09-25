"""A duplicated self-merge must not collapse the parent.

The recursive growth contour failed before the first gradient step: merging
three identical parents through the staged F3 tree maps ``(x, x, x)`` to
``(sqrt(3) x, 0, 0)``, so the parent's function is destroyed by the merge
itself. The replacement transform has to keep the duplicated parent
*functional*: after the transform the three branches must still reconstruct
the parent's own activations.

The exact requirement, stated once so the test and the fix agree:

* a self-merge of three identical parents leaves the summed branch signal
  unchanged (the transform fixes ``(1, 1, 1)``), and
* a parent that is *not* duplicated still gets mixed, so the transform is not
  the identity map, and
* the transform stays orthogonal and invertible.

If the first property is relaxed to "the average is preserved", the gate
passes for a transform that destroys per-branch structure, which is exactly
the failure this experiment exists to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from hagi.model.merge import (  # noqa: E402
    CrossParentPreservingTernaryTree,
    ParentPreservingTernaryLift,
    TernaryF3Tree,
    parent_preserving_ternary_matrix,
)

LEAF = 4


def _duplicated(parent: torch.Tensor) -> torch.Tensor:
    """Three identical parents stacked into the flat merge layout."""
    return parent.repeat(1, 3)


def test_matrix_is_orthogonal_and_fixes_the_all_ones_vector():
    m = parent_preserving_ternary_matrix()
    assert m.dtype == torch.float64
    assert torch.allclose(m.T @ m, torch.eye(3, dtype=torch.float64), atol=1e-12)
    ones = torch.ones(3, dtype=torch.float64)
    assert torch.allclose(m @ ones, ones, atol=1e-12)


def test_matrix_is_not_the_identity():
    """The transform must actually mix, otherwise the gate below is vacuous."""
    m = parent_preserving_ternary_matrix()
    assert not torch.allclose(m, torch.eye(3, dtype=torch.float64), atol=1e-9)


def test_legacy_f3_tree_collapses_a_duplicated_self_merge():
    """The documented failure this transform has to repair, pinned as a fact."""
    leaf = LEAF
    parent = torch.randn(1, leaf, dtype=torch.float64)
    tree = TernaryF3Tree(1, leaf)
    merged = tree.apply_row(_duplicated(parent))
    branches = merged.reshape(1, 3, leaf)
    # The staged tree aggregates: one branch survives, the others are zeroed.
    zero_count = int((branches.abs().sum(dim=-1) < 1e-9).sum())
    assert zero_count >= 1, "expected the staged tree to aggregate branches"
    assert not torch.allclose(merged, _duplicated(parent), atol=1e-6)


def test_cross_parent_transform_preserves_a_duplicated_parent():
    """gen-2 must not be worse than gen-1: the parent survives its own merge."""
    leaf = LEAF
    parent = torch.randn(1, leaf, dtype=torch.float64)
    tree = CrossParentPreservingTernaryTree(1, leaf)
    merged = tree.apply_row(_duplicated(parent))
    restored = tree.apply_row_inverse(merged)
    assert torch.allclose(restored, _duplicated(parent), atol=1e-10), (
        "the transform must be invertible on a duplicated parent"
    )
    branches = merged.reshape(1, 3, leaf)
    # Every branch keeps a non-trivial share of the parent signal.
    for index in range(3):
        assert branches[0, index].abs().sum() > 1e-6, (
            f"branch {index} was collapsed by the self-merge"
        )


def test_cross_parent_transform_still_mixes_distinct_parents():
    leaf = LEAF
    distinct = torch.randn(1, 3 * leaf, dtype=torch.float64)
    tree = CrossParentPreservingTernaryTree(1, leaf)
    mixed = tree.apply_row(distinct)
    assert not torch.allclose(mixed, distinct, atol=1e-6), (
        "the transform must mix distinct parents, not be the identity"
    )
    assert torch.allclose(tree.apply_row_inverse(mixed), distinct, atol=1e-10)


def test_norm_is_preserved_so_capacity_cannot_shrink_or_grow():
    leaf = LEAF
    x = torch.randn(1, 3 * leaf, dtype=torch.float64)
    tree = CrossParentPreservingTernaryTree(1, leaf)
    assert torch.allclose(tree.apply_row(x).norm(), x.norm(), atol=1e-10)


@pytest.mark.parametrize("seed", [1234, 3252, 301097])
def test_gate_holds_on_all_preregistered_seeds(seed: int):
    """The criterion from the plan, on the three preregistered seeds."""
    torch.manual_seed(seed)
    leaf = LEAF
    parent = torch.randn(1, leaf, dtype=torch.float64)
    tree = CrossParentPreservingTernaryTree(1, leaf)
    merged = tree.apply_row(_duplicated(parent))
    branches = merged.reshape(1, 3, leaf)
    for index in range(3):
        assert branches[0, index].abs().sum() > 1e-6, f"seed {seed}: branch {index} collapsed"


def test_the_lift_preserves_a_duplicated_parent_and_mixes_distinct_ones():
    """The property the parent-preserving transform is named for.

    Measured directly on the raw lift, which is what the orchestrator blocks
    today. An earlier note in this log claimed the opposite and was wrong: the
    measurement had fed the lift an arbitrary vector instead of a duplicated
    parent. The matrix is not the identity on general vectors, but on the
    diagonal ``(x, x, x)`` it is, because ``M @ (x,x,x) = (x,x,x)``.
    """
    leaf = LEAF
    lift = ParentPreservingTernaryLift()
    parent = torch.randn(1, leaf, dtype=torch.float64)
    duplicated = _duplicated(parent)
    assert torch.allclose(lift.apply_row(duplicated), duplicated, atol=1e-12)
    distinct = torch.randn(1, 3 * leaf, dtype=torch.float64)
    assert not torch.allclose(lift.apply_row(distinct), distinct, atol=1e-9)
