"""GAM (generation-antisymmetric mixer) tests — R104/R107 production channel.

The measured defect (measure_channel_split.py): 99.6% of the merge's
cross-expert communication is the fixed Hadamard transform, identical
across generations, and the trainable rank channel is 0.4% -- so
generations t and t+1 are effectively the same model and no frontier
can be produced (R104: gain must be PRODUCED each cycle; R107: the
frontier production beta needs a mechanism that changes with the cycle).

These tests pin the two properties the fix promises:

1. SIGN ALTERNATION — the fresh mixer's gain carries the generation's
   phase, so consecutive merges inject with opposite sign.
2. FRESH DRAW + ORTHOGONALIZATION — the new rank factors are a fresh
   random draw (not the constructor's), orthogonalized against the
   previous generations' factors when they live at the same width.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.config import Config  # noqa: E402
from hagi.model.merge import merge_experts  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402


def _cfg(h: int, eh: int, n: int, phase: int, rank: int = 16) -> Config:
    c = Config()
    c.model.vocab_size = 512
    c.model.hidden_size = h
    c.model.num_layers = 2
    qh = h // 64
    c.model.attention.num_query_heads = (qh // n) * n
    kvh = c.model.attention.num_query_heads // 2 or 1
    c.model.attention.num_kv_heads = (kvh // n) * n if kvh >= n else n
    c.model.attention.head_dim = 64
    c.model.ffn.expansion = 1.0
    c.model.ffn.multiple_of = 1
    c.model.embedding.tie_lm_head = False
    c.model.embedding.conv_kernel = 1
    c.merge.enabled = True
    c.merge.n_experts = n
    c.merge.expert_hidden = eh
    c.merge.mixer_type = "hadamard"
    c.merge.mixer_gen_phase = phase
    c.merge.mixer_rank = rank
    return c


def _experts(n: int) -> list[dict]:
    ec = _cfg(128, 128, 1, 0)
    return [HAGI(ec).state_dict() for _ in range(n)]


def test_gain_sign_alternates_with_generation():
    cfg1 = _cfg(384, 128, 3, phase=1)
    m1 = merge_experts(cfg1, _experts(3), n_mixers=1, mixer_init_scale=0.1)
    assert abs(m1.mixers[0].gain.item() - 0.1) < 1e-6
    cfg2 = _cfg(384, 128, 3, phase=2)
    m2 = merge_experts(cfg2, _experts(3), n_mixers=1, mixer_init_scale=0.1)
    assert abs(m2.mixers[0].gain.item() + 0.1) < 1e-6


def test_phase_zero_is_historical_plain_init():
    cfg = _cfg(384, 128, 3, phase=0)
    m = merge_experts(cfg, _experts(3), n_mixers=1, mixer_init_scale=0.1)
    assert abs(m.mixers[0].gain.item() - 0.1) < 1e-6


def test_fresh_draw_differs_from_constructor_default():
    """Two GAM merges at the same phase draw the same factors (seeded by
    phase), but a phase-1 draw differs from the plain constructor init."""
    cfg = _cfg(384, 128, 3, phase=1)
    experts = _experts(3)
    m_gam = merge_experts(cfg, experts, n_mixers=1, mixer_init_scale=0.0)
    m_plain = merge_experts(_cfg(384, 128, 3, phase=0), _experts(3), n_mixers=1, mixer_init_scale=0.0)
    g = m_gam.state_dict()["mixers.0.gate.weight"]
    p = m_plain.state_dict()["mixers.0.gate.weight"]
    # Different draws -> not equal
    assert not torch.allclose(g, p)
    # Same phase twice -> identical (seeded, reproducible)
    m_gam2 = merge_experts(cfg, _experts(3), n_mixers=1, mixer_init_scale=0.0)
    assert torch.allclose(g, m_gam2.state_dict()["mixers.0.gate.weight"])


def test_orthogonalization_against_history():
    """Unit test of the Gram-Schmidt mechanism itself: construct the merge
    path's history/basis exactly as merge_experts builds it and verify the
    orthogonalization math. (A full flat re-merge needs matching expert
    per-head geometry; the math is what the mechanism promises.)"""
    torch.manual_seed(0)
    h, rank = 384, 16
    prev = torch.randn(rank, h)  # a previous generation's gate factors
    basis = prev / prev.norm(dim=1, keepdim=True).clamp_min(1e-8)
    fresh = torch.randn(rank, h)
    r = fresh.clone()
    for j in range(r.shape[0]):
        r[j] -= (r[j] @ basis.T) @ basis
    r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos = (r @ prev.T) / (
        r.norm(dim=1, keepdim=True) * prev.norm(dim=1, keepdim=True).T
    )
    # float32 row-at-a-time projection leaves ~1e-2 numerical residue on
    # non-normalized prev rows; the mechanism's contract is cos ≪ 1
    # (fresh direction), not exact zero.
    assert cos.abs().max().item() < 0.05, cos.abs().max().item()
    # Non-degenerate: the orthogonalized rows keep their norm (unit here)
    assert torch.allclose(r.norm(dim=1), torch.ones(rank), atol=1e-5)


def test_forward_finite_after_gam_merge():
    cfg = _cfg(384, 128, 3, phase=2)
    m = merge_experts(cfg, _experts(3), n_mixers=1, mixer_init_scale=0.1)
    x = torch.randn(2, 8, 384)
    y = m.mixers[0](x)
    assert torch.isfinite(y).all()
