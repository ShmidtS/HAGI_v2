"""Gradient-Gram scan tests (safeqp_controller, reviewer round-5 priority 2).

Two regimes on a tiny model, CPU-only:
- identical calibration windows → gradients align, no conflict;
- anti-aligned windows (targets flipped) → conflict detected and the
  SafeQP dual certificate is present.

Both must leave every parameter gradient as None: the scan is a
measurement channel and may not leak into the training step.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from hagi.config import load_config  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.safeqp_controller import corpus_grad_gram  # noqa: E402

VOCAB = 32


def _tiny_cfg(seed: int = 0):
    cfg = load_config("configs/parent_h128_flat.yaml")
    m = cfg.model
    m.vocab_size = VOCAB
    m.hidden_size = 16
    m.num_layers = 1
    m.attention.num_query_heads = 1
    m.attention.num_kv_heads = 1
    m.attention.head_dim = 16
    m.attention.max_seq_len = 64
    m.ffn.expansion = 1.0
    m.ffn.multiple_of = 8
    m.init_seed = seed
    cfg.merge.n_experts = 1
    m.attention.qk_norm = False
    return cfg


def _window(seed: int, t: int = 32):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (t + 1,), generator=g)
    return {"input_ids": ids[None, :-1], "targets": ids[None, 1:]}


def test_aligned_windows_no_conflict():
    torch.manual_seed(0)
    model = HAGI(_tiny_cfg())
    w = _window(7)
    out = corpus_grad_gram(model, [("a", w), ("b", w)])
    assert out["n_conflicts"] == 0
    assert out["min_cos"] > 0.99  # identical windows → cos ≈ 1
    assert "safe_qp_certificate" not in out
    # read-only: no residual gradients
    assert all(p.grad is None for p in model.parameters())


def test_anti_aligned_windows_conflict_with_certificate():
    torch.manual_seed(0)
    model = HAGI(_tiny_cfg())
    w1 = _window(7)
    # The Dominate.lean scenario: a dominant aligned pair (a, b) plus a
    # small anti-aligned corpus c (anti-targets V-1-y). The weighted mean
    # direction then harms c: <g_c, g> < 0 -- the exact condition the scan
    # flags and SafeQP projects away.
    w2 = {"input_ids": w1["input_ids"], "targets": (VOCAB - 1) - w1["targets"]}
    out = corpus_grad_gram(
        model,
        [("a", w1), ("b", w1), ("c", w2)],
        corpus_weights=[0.49, 0.49, 0.02],
    )
    assert out["min_cos"] < 0.0
    assert out["n_conflicts"] == 1
    assert "safe_qp_certificate" in out
    # certificate reports the active-constraint count as an int
    active = out["safe_qp_certificate"]["active"]
    assert isinstance(active, int) and active >= 1
    # the conflicted corpus is the small one, and its multiplier is positive
    assert out["safe_qp_lambda"][2] > 0.0
    assert all(p.grad is None for p in model.parameters())
    # T4 kappa telemetry: the certificate now carries kappa = ||d*||^2/||g||^2
    # (fraction of gradient kept by the projection); in [0, 1] here since
    # the projection only ever removes components
    kappa = out["safe_qp_certificate"]["kappa"]
    assert isinstance(kappa, float)
    assert 0.0 <= kappa <= 1.0 + 1e-9


def test_uniform_weights_domination_range():
    torch.manual_seed(0)
    model = HAGI(_tiny_cfg())
    out = corpus_grad_gram(
        model, [("a", _window(7)), ("b", _window(8)), ("c", _window(9))]
    )
    assert 0.0 <= out["domination_share"] <= 1.0
    assert out["n_conflicts"] == 0
