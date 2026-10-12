"""Tests for the §6 Hedge router (hagi.inference.hedge_router)."""
from __future__ import annotations

import math

import pytest
import torch

from hagi.inference.hedge_router import (
    SKIP_NO_PER_LEAF_LOGITS,
    hedge_route,
    leaf_disagreement_losses,
    maybe_hedge_route,
)
from hagi.train.hedge import minimal_topk


def _two_leaf_logits() -> torch.Tensor:
    torch.manual_seed(0)
    base = torch.randn(2, 4, 16)
    # leaf 0 tracks the consensus, leaf 1 is a noisy dissenter
    return torch.stack([base, base + 3.0 * torch.randn_like(base)])


def test_leaf_disagreement_losses_rank_the_dissenter_last() -> None:
    losses = leaf_disagreement_losses(_two_leaf_logits())
    assert losses.shape == (2,)
    assert (losses >= -1e-9).all()
    # the leaf that tracks the pooled ensemble disagrees less
    assert losses[0] < losses[1]


def test_hedge_route_mixes_with_normalized_weights_and_budget() -> None:
    per_leaf = _two_leaf_logits()
    res = hedge_route(per_leaf, eps_route=1e-9, round_index=50)
    assert res.skipped == ""
    assert res.weights is not None
    assert torch.allclose(res.weights.sum(), torch.tensor(1.0), atol=1e-5)
    # tight budget -> keep both leaves; the mix is a convex combination
    assert res.k == 2
    expected = (res.weights[0] * per_leaf[0] + res.weights[1] * per_leaf[1])
    assert torch.allclose(res.logits, expected.to(res.logits.dtype), atol=1e-4)
    # eta is the closed regret-optimal rate for K=2, T=50
    assert math.isclose(res.eta, math.sqrt(2.0 * math.log(2) / 50), rel_tol=1e-9)


def test_hedge_route_honours_tail_budget_topk() -> None:
    torch.manual_seed(0)
    base = torch.randn(1, 8)
    per_leaf = torch.stack([base, base.clone(), base + 5.0 * torch.randn(1, 8)])
    # importance = softmax(-losses): the two agreeing leaves carry the
    # mass; a tight budget keeps more of them, a loose one may drop the
    # tail (minimal_topk returns the SMALLEST compliant k).
    res_tight = hedge_route(per_leaf, eps_route=0.02, round_index=10)
    res_loose = hedge_route(per_leaf, eps_route=0.3, round_index=10)
    assert res_tight.k == 2
    assert res_loose.k == 1
    # the loose mix uses ONLY the kept leaf (the least-disagreeing one)
    assert torch.allclose(res_loose.logits, per_leaf[0], atol=1e-4)


def test_maybe_hedge_route_skips_with_documented_note() -> None:
    res = maybe_hedge_route(None, eps_route=0.01)
    assert res.skipped == SKIP_NO_PER_LEAF_LOGITS
    assert "per-leaf logits" in res.skipped
    assert res.logits.numel() == 0


def test_hedge_route_rejects_single_leaf() -> None:
    with pytest.raises(ValueError):
        hedge_route(torch.randn(1, 2, 8), eps_route=0.1)


def test_minimal_topk_budget_rule_matches_manual_sum() -> None:
    # §6: min k : sum_{i>k} c_i^2 <= eps_route -- the exact tail identity
    c = torch.tensor([0.5, 0.3, 0.1, 0.05])
    eps = (0.1 ** 2 + 0.05 ** 2) / 2.0      # half the 2-tail energy
    k = minimal_topk(c, eps)
    tail_after_k = float((c.sort(descending=True).values[k:] ** 2).sum())
    tail_before = float((c.sort(descending=True).values[k - 1:] ** 2).sum())
    assert tail_after_k <= eps < tail_before
