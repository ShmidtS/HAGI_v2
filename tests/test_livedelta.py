"""Property tests for src/hagi/model/livedelta.py — each mirrors a
Lean theorem (see module docstring there)."""

import math

import pytest
import torch

from hagi.model.livedelta import (
    LiveDelta, J3, add_block, blocks_cost, double_counting_identified,
    effective_weights, from_history, frob_sum_le,
    history_storage_cost, j3_nonincreasing, no_compression_below_entropy,
    quantized_safe_window, replace_blocks_delta, rollback, update_newest,
)


def test_effective_weights_and_rollback():
    # LiveDelta.effectiveWeights + rollback_addBlock
    torch.manual_seed(0)
    w0 = torch.randn(4, 5)
    a, b = torch.randn(2, 4), torch.randn(2, 5)
    s = LiveDelta(base=w0, blocks=[])
    s1 = add_block(a, b, s)
    eff = effective_weights(s1)
    assert torch.allclose(eff, w0 + a.t() @ b, atol=1e-6)
    assert torch.allclose(effective_weights(rollback(s1)), w0, atol=1e-6)
    # empty rollback is identity
    assert torch.allclose(effective_weights(rollback(s)), w0, atol=1e-6)


def test_update_newest_cost_and_length():
    # LiveDeltaSafe.updateNewest_base/_length/_cost
    torch.manual_seed(1)
    w0 = torch.randn(3, 3)
    s = LiveDelta(base=w0, blocks=[(torch.randn(2, 3), torch.randn(2, 3)),
                                   (torch.randn(2, 3), torch.randn(2, 3))])
    b_new = torch.randn(2, 3)
    s2 = update_newest(b_new, s)
    assert len(s2.blocks) == len(s.blocks)
    assert blocks_cost(s2) == blocks_cost(s)
    # newest B replaced, A and older block untouched
    assert torch.allclose(s2.blocks[0][0], s.blocks[0][0])
    assert torch.allclose(s2.blocks[0][1], b_new)
    assert torch.allclose(s2.blocks[1][0], s.blocks[1][0])


def test_exact_bridge_from_history():
    # DeltaHistory.descentFold_eq_effective — EXACT equality
    torch.manual_seed(2)
    w0 = torch.randn(6, 7)
    hist = [(torch.randn(6), torch.randn(7)) for _ in range(9)]
    dense = w0.clone()
    for v, x in hist:
        dense = dense + torch.outer(v, x)
    eff = effective_weights(from_history(w0, hist))
    assert torch.allclose(eff, dense, atol=1e-5)


def test_history_storage_cost():
    # DeltaHistory.history_storage_cost: k(m+n) livedelta vs k·m·n dense
    w0 = torch.zeros(64, 64)
    hist = [(torch.zeros(64), torch.zeros(64)) for _ in range(10)]
    ld, dense = history_storage_cost(w0, hist)
    assert ld == 10 * 128
    assert dense == 10 * 64 * 64
    # blocks_cost: rank-1 blocks, r=1
    assert blocks_cost(from_history(w0, hist)) == ld


def test_consolidation_budget():
    # DeltaConsolidate.frob_sum_le: ‖ΣΔ‖² ≤ k·Σ‖Δᵢ‖²
    torch.manual_seed(3)
    deltas = [torch.randn(5, 5) for _ in range(6)]
    lhs, rhs = frob_sum_le(deltas)
    assert lhs <= rhs + 1e-4
    # exact when deltas are collinear (same direction)
    d = torch.randn(4, 4)
    lhs1, rhs1 = frob_sum_le([d, d])
    assert math.isclose(lhs1, rhs1, rel_tol=1e-6)


def test_replace_blocks_delta():
    # DeltaConsolidate.effective_replaceBlocks: exact difference
    torch.manual_seed(4)
    old = [(torch.randn(3, 4), torch.randn(3, 5)) for _ in range(3)]
    new = [(torch.randn(3, 4), torch.randn(3, 5)) for _ in range(3)]
    eff_old = old[0][0].t() @ old[0][1] + old[1][0].t() @ old[1][1] \
        + old[2][0].t() @ old[2][1]
    eff_new = new[0][0].t() @ new[0][1] + new[1][0].t() @ new[1][1] \
        + new[2][0].t() @ new[2][1]
    delta = replace_blocks_delta(old, new)
    assert torch.allclose(eff_new - eff_old, delta, atol=1e-5)


def test_j3_monotone_and_double_counting():
    # J3_nonincreasing + double_counting_identified
    assert j3_nonincreasing((2.0, 100.0, 0.5), (1.9, 90.0, 0.4), 0.1, 0.2)
    assert not j3_nonincreasing((2.0, 100.0, 0.5), (2.1, 90.0, 0.4), 0.1, 0.2)
    assert double_counting_identified(2.0, 0.5, 2.0, 0.1, 0.2)
    assert not double_counting_identified(2.0, 0.5, 3.0, 0.1, 0.2)


def test_no_compression_below_entropy():
    # MDLThreeCurrency.no_compression_below_entropy
    p = torch.tensor([0.2, 0.3, 0.5])
    q = torch.tensor([0.4, 0.4, 0.2])
    h, ce = no_compression_below_entropy(p, q)
    assert h <= ce + 1e-6
    assert math.isclose(ce - h, float((p * (p / q).log()).sum()), rel_tol=1e-5)


def test_quantized_safe_window_ge_ideal():
    # QuantSafeUpdate.quantized_safe_window: bound = ideal + price ≥ ideal
    eps = [0.1, 0.2]
    g_norms = [1.0, 2.0]
    lips = [0.5, 1.0]
    out = quantized_safe_window(g_norms, lips, d_norm=1.0, eta=0.1,
                                q=0.05, eps=eps)
    for i in range(2):
        price = 0.1 * g_norms[i] * 0.05 \
            + lips[i] * 0.01 * (1.0 * 0.05 + 0.05 ** 2 / 2)
        assert math.isclose(out[i], eps[i] + price, rel_tol=1e-9)
        assert out[i] >= eps[i]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
