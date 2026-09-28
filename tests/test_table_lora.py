"""TableLoRA (Element.lean residualLeaf) invariants."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hagi.model.table_lora import TableLoRA, lora_compression, recenter  # noqa: E402


def test_residual_leaf_zero():
    # residualLeaf_zero: rank-0 delta == the base, bit-for-bit.
    W0 = torch.randn(64, 16)
    t = TableLoRA(W0, rank=8)
    # B zero-init -> effective == base exactly
    assert torch.equal(t.effective_weight(), W0)


def test_base_frozen():
    W0 = torch.randn(32, 16)
    t = TableLoRA(W0, rank=4)
    trainable = [n for n, p in t.named_parameters()]
    assert trainable == ["B"]
    assert not t.base.requires_grad and not t.A.requires_grad


def test_delta_moves_effective():
    W0 = torch.randn(64, 16)
    t = TableLoRA(W0, rank=4)
    with torch.no_grad():
        t.B.normal_()
    eff = t.effective_weight()
    assert not torch.equal(eff, W0)
    # delta is exactly A@B
    assert torch.allclose(eff - W0, t.A @ t.B, atol=1e-6)


def test_compression_ratios():
    # synthesis §6: N=3, r=8, V=32768, H=128 -> ~0.396 (2.52x)
    r = lora_compression(3, 32768, 128, 8)
    assert 0.39 < r < 0.40, r
    # N=9, r=8 -> ~0.174 (5.75x)
    r9 = lora_compression(9, 32768, 128, 8)
    assert 0.17 < r9 < 0.18, r9


def test_recenter_zero_sum():
    deltas = [torch.randn(8, 8) for _ in range(3)]
    mean, rec = recenter(deltas)
    total = torch.stack([d.reshape(-1) for d in rec]).sum(0)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-6)
    # conservation: parent + recentred child == original child
    assert torch.allclose(mean.view_as(deltas[0]) + rec[0], deltas[0], atol=1e-6)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
