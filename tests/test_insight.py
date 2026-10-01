"""Insight-channel tests (hagi.train.insight, RLTL;DR-style internalization).

CPU-only tiny model, same fixture family as test_safeqp_controller.py.
Covers: extraction respects quantile/max_tokens/min_ce; the SFT loss is
finite and weighted; one cycle is guarded (KL bound + certified-improvement
gate) and transactional (rollback restores parameters, no residual grads);
R81 ``tldr_drift_null`` at code level — the base stays byte-identical and
a warm base is refused outright.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from hagi.config import InsightConfig, load_config  # noqa: E402
from hagi.model.adaptive import freeze_base_in_place  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.insight import (  # noqa: E402
    extract_insights,
    insight_sft_loss,
    run_insight_cycle,
)

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


def _adapter_cfg(seed: int = 0):
    cfg = _tiny_cfg(seed)
    cfg.model.adapters.enabled = True
    cfg.model.adapters.ttt_lora.enabled = True
    return cfg


def _window(seed: int, t: int = 32):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (t + 1,), generator=g)
    return {"input_ids": ids[None, :-1], "targets": ids[1:]}


def test_extract_respects_quantile_floor_and_length():
    torch.manual_seed(0)
    model = HAGI(_tiny_cfg())
    model.eval()
    w = _window(7)
    cfg = InsightConfig(fail_quantile=0.9, max_tokens=5, min_ce=0.0)
    insights = extract_insights(model, w["input_ids"], w["targets"], cfg)
    assert 0 < len(insights) <= 4  # ~10% of 31 hard positions, merged runs
    for ins in insights:
        assert ins.insight_ids.numel() <= 5
        assert ins.context_ids.numel() >= 1
    ces = [i.span_ce for i in insights]
    assert ces == sorted(ces, reverse=True)
    cfg_hi = InsightConfig(fail_quantile=0.9, max_tokens=5, min_ce=1e6)
    assert extract_insights(model, w["input_ids"], w["targets"], cfg_hi) == []


def test_extract_is_capped_and_keeps_the_hottest_spans():
    """Regression: an uncapped window OOMs the SFT step.

    insight_sft_loss materialises a [rows, len, V] logit tensor. On a
    fresh model ce = ln V everywhere, so the DEFAULT fail_quantile=0.9
    marks the top decile of every position as hard. Those are scattered,
    so run-merging yields hundreds of separate spans, and 32 rows x 1024
    positions of full-vocabulary logits is tens of GB -- the HIP launch
    fails unrecoverably. The cap keeps the HOTTEST spans, since the list
    is sorted by span CE descending.
    """
    torch.manual_seed(0)
    model = HAGI(_tiny_cfg())
    model.eval()
    w = _window(7, t=600)
    cfg = InsightConfig(fail_quantile=0.9, max_tokens=5, min_ce=0.0)

    uncapped = extract_insights(model, w["input_ids"], w["targets"], cfg, max_rows=10**9)
    assert len(uncapped) > 8, "fixture must produce more spans than the cap"

    capped = extract_insights(model, w["input_ids"], w["targets"], cfg, max_rows=8)
    assert len(capped) == 8
    # Sorted by span CE descending, and the same spans the uncapped call
    # would have produced first -- only the tail is dropped.
    assert [i.span_ce for i in capped] == sorted(
        (i.span_ce for i in uncapped[:8]), reverse=True
    )


def test_insight_loss_finite_and_weighted():
    torch.manual_seed(0)
    model = HAGI(_tiny_cfg())
    model.eval()
    w = _window(7)
    cfg = InsightConfig(fail_quantile=0.9, max_tokens=5, min_ce=0.0)
    insights = extract_insights(model, w["input_ids"], w["targets"], cfg)
    lam4, n4 = insight_sft_loss(model, insights, cfg)
    assert torch.isfinite(lam4)
    assert n4 > 0
    cfg1 = InsightConfig(fail_quantile=0.9, max_tokens=5, min_ce=0.0, lambda_insight=1.0)
    lam1, _ = insight_sft_loss(model, insights, cfg1)
    assert torch.isfinite(lam1) and float(lam1) > 0


def test_warm_base_refused():
    # R81 tldr_drift_null at code level: a trainable cortex is a hard
    # refusal — the hippocampus never writes to the cortex directly.
    torch.manual_seed(0)
    model = HAGI(_adapter_cfg())
    w = _window(7)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    cfg = InsightConfig(fail_quantile=0.9, max_tokens=5, min_ce=0.0)
    rep = run_insight_cycle(model, w["input_ids"], w["targets"], cfg)
    assert rep["applied"] is False
    assert rep["reason"] == "base_not_frozen"
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p.detach()), n


def test_cycle_certified_and_base_untouched():
    torch.manual_seed(0)
    model = HAGI(_adapter_cfg())
    freeze_base_in_place(model)  # only lora_B stays trainable
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("adapter" in n for n in trainable)
    base = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not p.requires_grad
    }
    w = _window(7)
    cfg = InsightConfig(fail_quantile=0.9, max_tokens=5, min_ce=0.0, kl_bound=1e9)
    rep = run_insight_cycle(model, w["input_ids"], w["targets"], cfg, lr=1e-2)
    for key in ("pre_ce", "post_ce", "kl", "n_insights", "n_scored", "applied", "reason"):
        assert key in rep, key
    assert rep["n_insights"] > 0
    # verifier rule: applied only with a certified CE improvement
    if rep["applied"]:
        assert rep["reason"] == "ok"
        assert rep["post_ce"] <= rep["pre_ce"] + 1e-6
        moved = any(
            not torch.equal(base[n], p.detach()) for n, p in model.named_parameters()
            if not p.requires_grad
        )
        assert not moved  # the base is byte-identical (tldr_drift_null)
    assert all(p.grad is None for p in model.parameters())


def test_kl_guard_rolls_back():
    torch.manual_seed(0)
    model = HAGI(_adapter_cfg())
    freeze_base_in_place(model)
    w = _window(7)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    cfg = InsightConfig(fail_quantile=0.9, max_tokens=5, min_ce=0.0, kl_bound=1e-12)
    rep = run_insight_cycle(model, w["input_ids"], w["targets"], cfg, lr=10.0)
    assert rep["applied"] is False
    assert rep["reason"] in ("kl_bound", "non_finite", "no_certified_gain")
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p.detach()), f"rollback missed {n}"
    assert all(p.grad is None for p in model.parameters())
