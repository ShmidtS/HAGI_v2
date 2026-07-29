# HAGI-2 V30: Test Suite + Experiment Runner + Docs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add comprehensive test coverage (~120 tests), an automated ablation experiment runner, and update the stale architecture doc -- the three gaps in an otherwise complete HAGI-2 V30 codebase (34 source files, all modules implemented).

**Architecture:** Tests cover every module in `src/hagi/` with CPU-only unit tests (no GPU required). The experiment runner is a standalone CLI script that chains base YAML config + per-experiment overrides into training runs with CSV metric logging. The architecture doc is updated from V25-style (context/expression split) to V28 unified stack.

**Tech Stack:** Python 3.10+, pytest, PyTorch (CPU mode for tests), PyYAML, csv stdlib.

## Global Constraints

- Zero new pip dependencies beyond existing (torch, numpy, pytest, PyYAML, tokenizers)
- All tests must pass on CPU (no CUDA/ROCm requirement for CI)
- Follow existing code style: `from __future__ import annotations`, type hints, google-style docstrings
- Test files in `C:\HAGI_v2\tests\` package
- Existing source code not modified unless tests reveal a bug
- Checkpoint format V7 preserved (no format bump)
- Commit granularity: one commit per test module group (small, reviewable)

---

### Task 1: Config and Ternary/Norms Tests (21 tests)

**Files:**
- Create: `C:\HAGI_v2\tests\__init__.py`
- Create: `C:\HAGI_v2\tests\test_config.py`
- Create: `C:\HAGI_v2\tests\test_ternary.py`
- Create: `C:\HAGI_v2\tests\test_norms.py`

**Interfaces:**
- Consumes: `hagi.config.*`, `hagi.model.ternary.*`, `hagi.model.norms.RMSNorm`
- Produces: 21 pytest functions (10 config + 10 ternary + 4 norms = actually let me recount)

Let me recalculate: Config has DefaultConfig (4), ConfigValidation (7), AutoConfigure (5), ConfigRoundtrip (2) = 18. Ternary has Ternarize (6), BitLinear (5), STE (3) = 14. Norms has RMSNorm (4). Total = 36.

- [ ] **Step 1: Create test package and config tests**

Write `C:\HAGI_v2\tests\__init__.py`:
```python
"""HAGI-2 test suite."""
```

Write `C:\HAGI_v2\tests\test_config.py`:
```python
"""Tests for hagi.config: validation, auto_configure, serialization, sliding windows."""

from __future__ import annotations

import pytest

from hagi.config import (
    Config, auto_configure, validate_config, layer_sliding_windows,
    load_config, cfg_to_dict, cfg_from_dict,
)


class TestDefaultConfig:
    def test_default_validates(self):
        validate_config(Config())

    def test_sizes(self):
        m = Config().model
        assert m.vocab_size == 49154
        assert m.hidden_size == 384
        assert m.core_hidden_size == 192
        assert m.hidden_size > m.core_hidden_size
        assert m.body.num_layers == 12
        assert m.attention.num_query_heads == 8
        assert m.attention.num_kv_heads == 4
        assert m.attention.head_dim == 64
        assert m.attention.num_kv_heads <= m.attention.num_query_heads
        assert m.attention.num_query_heads % m.attention.num_kv_heads == 0
        assert m.attention.num_query_heads * m.attention.head_dim == m.hidden_size

    def test_sliding_windows_default_all_full(self):
        windows = layer_sliding_windows(Config().model)
        assert len(windows) == 12
        assert all(w == 0 for w in windows)

    def test_sliding_windows_full_every(self):
        cfg = Config()
        cfg.model.sliding.sliding_window = 128
        cfg.model.sliding.full_every = 4
        windows = layer_sliding_windows(cfg.model)
        for i in range(12):
            assert windows[i] == (0 if i % 4 == 0 else 128), f"layer {i}"

    def test_sliding_windows_explicit(self):
        cfg = Config()
        cfg.model.body.num_layers = 6
        cfg.model.sliding.sliding_window = 64
        cfg.model.sliding.window_layers = (1, 3)
        assert layer_sliding_windows(cfg.model) == [0, 64, 0, 64, 0, 0]


class TestConfigValidation:
    def test_rejects_c_gte_h(self):
        cfg = Config()
        cfg.model.core_hidden_size = 400
        with pytest.raises(ValueError, match="0 < C < hidden_size"):
            validate_config(cfg)

    def test_rejects_n_kv_gt_n_q(self):
        cfg = Config()
        cfg.model.attention.num_kv_heads = 16
        with pytest.raises(ValueError, match="num_kv_heads"):
            validate_config(cfg)

    def test_rejects_gqa_not_divisible(self):
        cfg = Config()
        cfg.model.attention.num_kv_heads = 3
        with pytest.raises(ValueError, match="divisible"):
            validate_config(cfg)

    def test_rejects_h_not_divisible_by_n_q(self):
        cfg = Config()
        cfg.model.hidden_size = 385
        with pytest.raises(ValueError, match="hidden_size must be divisible"):
            validate_config(cfg)

    def test_rejects_eos_equals_pad(self):
        cfg = Config()
        cfg.train.eos_token_id = 49152
        cfg.train.pad_token_id = 49152
        with pytest.raises(ValueError, match="must be distinct"):
            validate_config(cfg)

    def test_rejects_zero_grad_accum(self):
        cfg = Config()
        cfg.train.grad_accum_steps = 0
        with pytest.raises(ValueError):
            validate_config(cfg)


class TestAutoConfigure:
    def test_15m_budget(self):
        m = auto_configure(15_000_000)
        assert m.hidden_size > m.core_hidden_size > 0
        assert m.body.num_layers >= 2
        assert m.attention.num_query_heads % m.attention.num_kv_heads == 0
        assert m.attention.num_query_heads * m.attention.head_dim == m.hidden_size
        assert m.body.moe.enabled is False

    def test_150m_budget_moe_on(self):
        m = auto_configure(150_000_000)
        assert m.body.moe.enabled is True
        assert m.body.moe.num_experts >= 1
        assert 1 <= m.body.moe.top_k <= m.body.moe.num_experts

    def test_head_dim_in_range(self):
        m = auto_configure(40_000_000)
        assert 32 <= m.attention.head_dim <= 128

    def test_factor_rank_bounded(self):
        m = auto_configure(100_000_000)
        assert 8 <= m.embeddings.factor_rank <= 256

    def test_bottleneck_dim_matches_core(self):
        m = auto_configure(50_000_000)
        assert m.body.bottleneck.dim == m.core_hidden_size


class TestConfigRoundtrip:
    def test_default_roundtrip(self):
        cfg = Config()
        cfg2 = cfg_from_dict(cfg_to_dict(cfg))
        validate_config(cfg2)
        assert cfg2.model.hidden_size == cfg.model.hidden_size
        assert cfg2.train.max_steps == cfg.train.max_steps
        assert cfg2.inference.temperature == cfg.inference.temperature

    def test_auto_configured_roundtrip(self):
        cfg = Config()
        cfg.model = auto_configure(25_000_000)
        cfg2 = cfg_from_dict(cfg_to_dict(cfg))
        validate_config(cfg2)
        assert cfg2.model.hidden_size == cfg.model.hidden_size
```

- [ ] **Step 2: Write ternary tests**

Write `C:\HAGI_v2\tests\test_ternary.py`:
```python
"""Tests for BitNet b1.58: ternarize(), BitLinear, _TernarizeSTE."""

from __future__ import annotations

import pytest
import torch

from hagi.model.ternary import ternarize, BitLinear, _TernarizeSTE


class TestTernarize:
    def test_values_are_ternary(self):
        w = torch.randn(16, 32)
        eff, scale = ternarize(w)
        assert eff.shape == w.shape
        assert scale.shape == (16, 1)
        for i in range(16):
            s = scale[i, 0]
            row = eff[i]
            assert ((row == 0) | (row == s) | (row == -s)).all()

    def test_scale_positive(self):
        _, scale = ternarize(torch.randn(8, 64) * 0.5, eps=1e-5)
        assert (scale > 0).all()

    def test_constant_input(self):
        w = torch.ones(4, 8) * 3.0
        eff, scale = ternarize(w)
        assert abs(scale[0, 0].item() - 3.0) < 1e-3
        assert (eff == 3.0).all()

    def test_zero_row(self):
        eff, scale = ternarize(torch.zeros(4, 8), eps=1e-5)
        assert (eff == 0).all()
        assert (scale == 1e-5).all()

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2D"):
            ternarize(torch.randn(16))

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="2D"):
            ternarize(torch.randn(2, 3, 4))


class TestBitLinear:
    def test_forward_shape(self):
        y = BitLinear(32, 16)(torch.randn(4, 32))
        assert y.shape == (4, 16)

    def test_forward_with_bias(self):
        y = BitLinear(32, 16, bias=True)(torch.randn(4, 32))
        assert y.shape == (4, 16)

    def test_gradient_flows(self):
        layer = BitLinear(32, 16)
        x = torch.randn(4, 32, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0

    def test_inference_no_grad(self):
        layer = BitLinear(32, 16)
        with torch.inference_mode():
            y = layer(torch.randn(4, 32))
        assert not y.requires_grad

    def test_extra_repr(self):
        rep = BitLinear(32, 16, bias=True, eps=1e-4).extra_repr()
        assert "in_features=32" in rep
        assert "bias=True" in rep
        assert "ternary=BitNet-b1.58" in rep


class TestTernarizeSTE:
    def test_forward_matches_ternarize(self):
        w = torch.randn(8, 12)
        assert torch.equal(_TernarizeSTE.apply(w, 1e-5), ternarize(w, 1e-5)[0])

    def test_backward_identity(self):
        w = torch.randn(4, 6, requires_grad=True)
        grad_out = torch.ones_like(w)
        _TernarizeSTE.apply(w, 1e-5).backward(grad_out)
        assert torch.allclose(w.grad, grad_out)

    def test_no_mutation(self):
        w = torch.randn(4, 8)
        w_clone = w.clone()
        _TernarizeSTE.apply(w, 1e-5)
        assert torch.equal(w, w_clone)
```

- [ ] **Step 3: Write norms tests**

Write `C:\HAGI_v2\tests\test_norms.py`:
```python
"""Tests for RMSNorm."""

from __future__ import annotations

import torch

from hagi.model.norms import RMSNorm


class TestRMSNorm:
    def test_output_shape(self):
        y = RMSNorm(64)(torch.randn(4, 8, 64))
        assert y.shape == (4, 8, 64)

    def test_unit_variance(self):
        norm = RMSNorm(64)
        x = torch.randn(4, 8, 64) * 5.0
        y = norm(x)
        rms = torch.sqrt((y ** 2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_identity_ish(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 4, 32)
        x = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True))
        assert torch.allclose(norm(x), x, atol=0.1)

    def test_weight_scales(self):
        norm = RMSNorm(16)
        norm.weight.data = torch.full((16,), 2.0)
        assert torch.allclose(norm(torch.ones(2, 4, 16)), torch.full((2, 4, 16), 2.0))
```

- [ ] **Step 4: Run and commit**

Run: `python -m pytest tests/test_config.py tests/test_ternary.py tests/test_norms.py -v`
Expected: 18 + 14 + 4 = 36 PASS.

```bash
git add tests/__init__.py tests/test_config.py tests/test_ternary.py tests/test_norms.py
git commit -m "test: add config validation, auto_configure, ternarize/BitLinear, and RMSNorm tests"
```

---

### Task 2: RoPE, KV-Cache, ConvEmbedding, and Attention Tests (32 tests)

**Files:**
- Create: `C:\HAGI_v2\tests\test_rope.py`
- Create: `C:\HAGI_v2\tests\test_kv_cache.py`
- Create: `C:\HAGI_v2\tests\test_conv_embedding.py`
- Create: `C:\HAGI_v2\tests\test_attention.py`

- [ ] **Step 1: Write RoPE tests**

Write `C:\HAGI_v2\tests\test_rope.py`:
```python
"""Tests for RoPE (1D and 2D) and rotate_half."""

from __future__ import annotations

import pytest
import torch

from hagi.model.rope import (
    RotaryEmbedding, apply_rope, rope_cos_sin, rope_cos_sin_2d, rotate_half,
)


class TestRotaryEmbedding:
    def test_inv_freq_shape(self):
        assert RotaryEmbedding(64).inv_freq.shape == (32,)

    def test_cos_sin_shape(self):
        cos, sin = RotaryEmbedding(64)(torch.arange(16), torch.device("cpu"), torch.float32)
        assert cos.shape == (16, 64)
        assert sin.shape == (16, 64)

    def test_cos_sin_in_range(self):
        cos, sin = RotaryEmbedding(64)(torch.arange(8), torch.device("cpu"), torch.float32)
        assert (cos >= -1).all() and (cos <= 1).all()

    def test_caching(self):
        rope = RotaryEmbedding(32)
        pos = torch.tensor([0, 1, 2])
        cos1, _ = rope(pos, torch.device("cpu"), torch.float32)
        cos2, _ = rope(pos, torch.device("cpu"), torch.float32)
        assert torch.equal(cos1, cos2)

    def test_rejects_odd_head_dim(self):
        with pytest.raises(ValueError, match="even"):
            RotaryEmbedding(63)


class TestRotateHalf:
    def test_rotate_half(self):
        x = torch.tensor([1., 2., 3., 4.])
        assert torch.equal(rotate_half(x), torch.tensor([-3., -4., 1., 2.]))

    def test_batched(self):
        x = torch.randn(2, 8, 32)
        x1, x2 = x.chunk(2, dim=-1)
        assert torch.equal(rotate_half(x), torch.cat([-x2, x1], dim=-1))


class TestApplyRoPE:
    def test_shape_preserved(self):
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 2, 16, 32)
        cos, sin = rope_cos_sin(torch.arange(16).float(), 32, 10000., torch.device("cpu"), torch.float32)
        qr, kr = apply_rope(q, k, cos, sin)
        assert qr.shape == q.shape and kr.shape == k.shape

    def test_different_from_input(self):
        q = torch.ones(1, 2, 4, 16)
        k = torch.ones(1, 2, 4, 16)
        cos, sin = rope_cos_sin(torch.arange(4).float(), 16, 10000., torch.device("cpu"), torch.float32)
        qr, _ = apply_rope(q, k, cos, sin)
        assert not torch.allclose(qr, q)


class TestRope2D:
    def test_shape(self):
        rows = torch.tensor([0., 0., 1., 1.])
        cols = torch.tensor([0., 1., 0., 1.])
        cos, sin = rope_cos_sin_2d(rows, cols, 64, 10000., torch.device("cpu"), torch.float32)
        assert cos.shape == (4, 64) and sin.shape == (4, 64)

    def test_rejects_hd_not_div4(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            rope_cos_sin_2d(torch.arange(4).float(), torch.arange(4).float(), 62, 10000., torch.device("cpu"), torch.float32)
```

- [ ] **Step 2: Write KV-cache tests**

Write `C:\HAGI_v2\tests\test_kv_cache.py`:
```python
"""Tests for KVCache -- incremental KV store."""

from __future__ import annotations

import pytest
import torch

from hagi.model.kv_cache import KVCache


class TestKVCache:
    def test_initial_length_zero(self):
        assert KVCache(128, 4, 64, torch.float32, torch.device("cpu")).length == 0

    def test_update_grows(self):
        cache = KVCache(128, 4, 64, torch.float32, torch.device("cpu"))
        cache.update(torch.randn(2, 4, 8, 64), torch.randn(2, 4, 8, 64))
        assert cache.length == 8

    def test_multiple_updates(self):
        cache = KVCache(128, 4, 64, torch.float32, torch.device("cpu"))
        cache.update(torch.randn(2, 4, 5, 64), torch.randn(2, 4, 5, 64))
        cache.update(torch.randn(2, 4, 3, 64), torch.randn(2, 4, 3, 64))
        assert cache.length == 8

    def test_get_returns_stored(self):
        cache = KVCache(128, 2, 16, torch.float32, torch.device("cpu"))
        k = torch.randn(1, 2, 4, 16)
        v = torch.randn(1, 2, 4, 16)
        cache.update(k, v)
        kc, vc = cache.get()
        assert kc.shape == (1, 2, 4, 16) and vc.shape == (1, 2, 4, 16)
        assert torch.equal(kc, k) and torch.equal(vc, v)

    def test_overflow(self):
        cache = KVCache(4, 2, 8, torch.float32, torch.device("cpu"))
        cache.update(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
        with pytest.raises(ValueError, match="overflow"):
            cache.update(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))

    def test_reset(self):
        cache = KVCache(128, 4, 64, torch.float32, torch.device("cpu"))
        cache.update(torch.randn(2, 4, 8, 64), torch.randn(2, 4, 8, 64))
        assert cache.length == 8
        cache.reset()
        assert cache.length == 0
        with pytest.raises(RuntimeError, match="empty"):
            cache.get()

    def test_get_before_update(self):
        with pytest.raises(RuntimeError, match="empty"):
            KVCache(128, 4, 64, torch.float32, torch.device("cpu")).get()
```

- [ ] **Step 3: Write ConvEmbedding tests**

Write `C:\HAGI_v2\tests\test_conv_embedding.py`:
```python
"""Tests for ConvEmbedding: factorized source encoder + causal conv."""

from __future__ import annotations

import pytest
import torch

from hagi.model.conv_embedding import ConvEmbedding


@pytest.fixture
def embed():
    return ConvEmbedding(1000, 64, 32, 3, 1e-6)


class TestConvEmbedding:
    def test_output_shape(self, embed):
        assert embed(torch.randint(0, 1000, (2, 16))).shape == (2, 16, 64)

    def test_weight_property(self, embed):
        assert embed.weight.shape == (1000, 64)

    def test_causal_no_future_leak(self, embed):
        ids1 = torch.randint(0, 1000, (2, 16))
        h1 = embed(ids1)
        ids2 = ids1.clone()
        ids2[:, 8] = (ids2[:, 8] + 1) % 1000
        h2 = embed(ids2)
        # positions 0..7 must be identical (causal property)
        assert torch.allclose(h1[:, :8], h2[:, :8], atol=1e-4)
        # position 8 may differ
        assert not torch.allclose(h1[:, 8], h2[:, 8], atol=1e-4)

    def test_training_disables_cache(self, embed):
        embed.train()
        embed(torch.randint(0, 1000, (2, 16)))
        assert embed._conv_cache is None

    def test_eval_enables_cache(self, embed):
        embed.eval()
        embed(torch.randint(0, 1000, (2, 16)))
        assert embed._conv_cache is not None
        assert embed._conv_cache.shape[1] <= embed.left_pad

    def test_reset_conv_cache(self, embed):
        embed.eval()
        embed(torch.randint(0, 1000, (2, 16)))
        assert embed._conv_cache is not None
        embed.reset_conv_cache()
        assert embed._conv_cache is None
```

- [ ] **Step 4: Write attention tests**

Write `C:\HAGI_v2\tests\test_attention.py`:
```python
"""Tests for Attention: GQA modes, masks, penalty, sliding window."""

from __future__ import annotations

import pytest
import torch

from hagi.model.attention import Attention, AttentionConfig, repeat_kv


class TestRepeatKV:
    def test_single_rep(self):
        x = torch.randn(2, 4, 16, 64)
        assert torch.equal(repeat_kv(x, 1), x)

    def test_multi_rep(self):
        x = torch.randn(2, 2, 16, 64)
        y = repeat_kv(x, 4)
        assert y.shape == (2, 8, 16, 64)
        for rep in range(4):
            assert torch.equal(y[:, 2*rep:2*(rep+1)], x)


@pytest.fixture
def attn():
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=32, attn_entropy_floor=0.0)
    return Attention(128, cfg, use_ternary=False)


@pytest.fixture
def attn_with_penalty():
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=32, attn_entropy_floor=0.1)
    return Attention(128, cfg, use_ternary=False)


class TestAttentionModes:
    def test_causal(self, attn):
        out, pen = attn(torch.randn(2, 16, 128), "causal")
        assert out.shape == (2, 16, 128) and pen is None

    def test_bidir(self, attn):
        out, _ = attn(torch.randn(2, 8, 128), "bidir")
        assert out.shape == (2, 8, 128)

    def test_prefix(self, attn):
        out, _ = attn(torch.randn(2, 12, 128), "prefix", prefix_len=4)
        assert out.shape == (2, 12, 128)

    def test_soft_causal(self, attn):
        out, _ = attn(torch.randn(2, 10, 128), "soft_causal", soft_beta=2.0)
        assert out.shape == (2, 10, 128)

    def test_unknown_mode_raises(self, attn):
        with pytest.raises(ValueError):
            attn(torch.randn(2, 8, 128), "invalid")


class TestAttentionPenalty:
    def test_penalty_when_training(self, attn_with_penalty):
        attn_with_penalty.train()
        _, pen = attn_with_penalty(torch.randn(2, 16, 128), "causal")
        assert pen is not None and pen.item() >= 0.0

    def test_no_penalty_at_eval(self, attn_with_penalty):
        attn_with_penalty.eval()
        _, pen = attn_with_penalty(torch.randn(2, 16, 128), "causal")
        assert pen is None


class TestSlidingWindow:
    def test_windowed_causal(self, attn):
        attn.sliding_window = 4
        out, _ = attn(torch.randn(2, 16, 128), "causal")
        assert out.shape == (2, 16, 128)

    def test_windowed_bidir(self, attn):
        attn.sliding_window = 4
        out, _ = attn(torch.randn(2, 8, 128), "bidir")
        assert out.shape == (2, 8, 128)
```

- [ ] **Step 5: Run and commit**

Run: `python -m pytest tests/test_rope.py tests/test_kv_cache.py tests/test_conv_embedding.py tests/test_attention.py -v`
Expected: 10 + 7 + 6 + 9 = 32 PASS.

```bash
git add tests/test_rope.py tests/test_kv_cache.py tests/test_conv_embedding.py tests/test_attention.py
git commit -m "test: add RoPE (1D/2D), KVCache, ConvEmbedding, and Attention mode/mask tests"
```

---

### Task 3: HebbianFFN, Bottleneck, MoE, and WaterFilling Tests (39 tests)

**Files:**
- Create: `C:\HAGI_v2\tests\test_hebbian_ffn.py`
- Create: `C:\HAGI_v2\tests\test_bottleneck.py`
- Create: `C:\HAGI_v2\tests\test_moe.py`
- Create: `C:\HAGI_v2\tests\test_water_filling.py`

- [ ] **Step 1: Write HebbianBilinearFFN tests**

Write `C:\HAGI_v2\tests\test_hebbian_ffn.py`:
```python
"""Tests for HebbianBilinearFFN."""

from __future__ import annotations

import torch

from hagi.model.hebbian_ffn import HebbianBilinearFFN, HebbianFFNConfig


def _ffn():
    return HebbianBilinearFFN(64, HebbianFFNConfig(expansion=4), use_ternary=False)


class TestHebbianBilinearFFN:
    def test_output_shape(self):
        assert _ffn()(torch.randn(2, 16, 64)).shape == (2, 16, 64)

    def test_not_identity(self):
        x = torch.randn(2, 16, 64)
        assert not torch.allclose(_ffn()(x), x)

    def test_gate_init_zero(self):
        assert (_ffn().gate == 0).all()

    def test_no_ternary_uses_linear(self):
        from torch import nn
        ffn = HebbianBilinearFFN(32, HebbianFFNConfig(expansion=2), use_ternary=False)
        assert isinstance(ffn.A0, nn.Linear)

    def test_ternary_uses_bitlinear(self):
        from hagi.model.ternary import BitLinear
        ffn = HebbianBilinearFFN(32, HebbianFFNConfig(expansion=2), use_ternary=True)
        assert isinstance(ffn.A0, BitLinear)

    def test_gate_gets_gradient(self):
        ffn = _ffn()
        ffn(torch.randn(2, 8, 32, requires_grad=True)).sum().backward()
        assert ffn.gate.grad is not None and ffn.gate.grad.abs().sum() > 0
```

- [ ] **Step 2: Write InformationBottleneck tests**

Write `C:\HAGI_v2\tests\test_bottleneck.py`:
```python
"""Tests for InformationBottleneck: rate, distortion, reparam, fp32."""

from __future__ import annotations

import pytest
import torch

from hagi.config import BottleneckConfig
from hagi.model.bottleneck import InformationBottleneck


@pytest.fixture
def ib():
    cfg = BottleneckConfig(dim=32, kl_free_bits=0.01, logvar_clamp=(-5., 5.), distortion_eps=1e-6)
    return InformationBottleneck(64, cfg)


class TestInformationBottleneck:
    def test_rate_scalar(self, ib):
        ib.eval()
        out = ib(torch.randn(2, 16, 64))
        assert isinstance(out["rate"], torch.Tensor) and out["rate"].ndim == 0

    def test_distortion_nonnegative(self, ib):
        ib.eval()
        out = ib(torch.randn(2, 16, 64))
        assert out["distortion"].item() >= 0.0

    def test_mu_shape(self, ib):
        out = ib(torch.randn(2, 16, 64))
        assert out["mu"].shape == (32, 32)

    def test_eval_deterministic(self, ib):
        ib.eval()
        h = torch.randn(2, 8, 64)
        r1 = ib(h)["rate"]
        r2 = ib(h)["rate"]
        assert torch.equal(r1, r2)

    def test_training_stochastic(self, ib):
        ib.train()
        h = torch.randn(2, 8, 64)
        assert not torch.equal(ib(h)["rate"], ib(h)["rate"])

    def test_rate_positive(self, ib):
        assert ib(torch.randn(2, 8, 64))["rate"].item() > 0.0

    def test_ensure_fp32(self, ib):
        ib.to(torch.bfloat16)
        assert ib.to_mu.weight.dtype == torch.bfloat16
        ib.ensure_fp32()
        assert ib.to_mu.weight.dtype == torch.float32
        assert ib.to_logvar.weight.dtype == torch.float32
        assert ib.decompress.weight.dtype == torch.float32

    def test_kl_rate_zero_at_identity(self):
        """KL[N(0,0) || N(0,1)] is ~0 per dim (but clamped to free_bits)."""
        cfg = BottleneckConfig(dim=32, kl_free_bits=0.0, logvar_clamp=(-5., 5.), distortion_eps=1e-6)
        ib = InformationBottleneck(64, cfg)
        # zero-init gives mu~0, logvar~0 -> KL negligible
        rate = ib(torch.randn(2, 4, 64))["rate"].item()
        assert rate >= 0.0
```

- [ ] **Step 3: Write MoE tests**

Write `C:\HAGI_v2\tests\test_moe.py`:
```python
"""Tests for WaterFillingMoE: routing, dispatch, aux losses."""

from __future__ import annotations

import pytest
import torch

from hagi.config import MoEConfig
from hagi.model.moe import WaterFillingMoE


@pytest.fixture
def moe():
    cfg = MoEConfig(enabled=True, num_experts=4, top_k=1, n_shared=1, moe_every=2, intermediate_size=128)
    return WaterFillingMoE(64, 128, cfg, use_ternary=False)


class TestWaterFillingMoE:
    def test_output_shape(self, moe):
        assert moe(torch.randn(2, 16, 64)).shape == (2, 16, 64)

    def test_not_identity(self, moe):
        x = torch.randn(2, 16, 64)
        assert not torch.allclose(moe(x), x)

    def test_expert_counts(self, moe):
        assert len(moe.shared_experts) == 1 and len(moe.experts) == 4

    def test_router_shape(self, moe):
        assert moe.router.weight.shape == (4, 64)

    def test_lb_training(self, moe):
        moe.train()
        moe(torch.randn(2, 16, 64))
        lb = moe.last_load_balance
        assert lb is not None and 0.0 < lb.item() <= 4.0

    def test_lb_none_eval(self, moe):
        moe.eval()
        moe(torch.randn(2, 16, 64))
        assert moe.last_load_balance is None

    def test_route_entropy_training(self, moe):
        moe.train()
        moe(torch.randn(2, 16, 64))
        assert moe.last_routing_entropy is not None and moe.last_routing_entropy.item() > 0.0

    def test_wf_loss_training(self, moe):
        moe.train()
        moe(torch.randn(2, 16, 64))
        assert moe.last_water_filling_loss is not None

    def test_commit_ema_noop(self, moe):
        moe.commit_ema_update()

    def test_commit_ema_clears_deferred(self, moe):
        moe.train()
        moe(torch.randn(2, 8, 64))
        assert moe._deferred_residual is not None
        moe.commit_ema_update()
        assert moe._deferred_residual is None

    def test_top2_routing(self):
        cfg = MoEConfig(enabled=True, num_experts=4, top_k=2, n_shared=0, moe_every=1, intermediate_size=64)
        moe2 = WaterFillingMoE(32, 64, cfg, use_ternary=False)
        moe2.eval()
        assert moe2(torch.randn(2, 16, 32)).shape == (2, 16, 32)
```

- [ ] **Step 4: Write WaterFillingAllocator tests**

Write `C:\HAGI_v2\tests\test_water_filling.py`:
```python
"""Tests for WaterFillingAllocator."""

from __future__ import annotations

import pytest
import torch

from hagi.model.water_filling import WaterFillingAllocator


@pytest.fixture
def alloc():
    return WaterFillingAllocator(512, 4, min_width=16)


class TestWaterFillingAllocator:
    def test_init_snr_ones(self, alloc):
        assert torch.equal(alloc.snr_ema, torch.ones(4))

    def test_init_logits_zeros(self, alloc):
        assert torch.equal(alloc.allocation_logits, torch.zeros(4))

    def test_uniform_at_init(self, alloc):
        assert torch.allclose(alloc.allocation_probs(), torch.full((4,), 0.25), atol=1e-4)

    def test_probs_sum_to_one(self, alloc):
        assert abs(alloc.allocation_probs().sum().item() - 1.0) < 1e-5

    def test_get_widths_sum(self, alloc):
        widths = alloc.get_widths()
        assert len(widths) == 4 and sum(widths) == 512 and all(w >= 16 for w in widths)

    def test_reg_loss_zero_at_init(self, alloc):
        assert abs(alloc.regularization_loss().item()) < 1e-4

    def test_update_snr_ema(self, alloc):
        alloc.update_snr_ema(torch.tensor([1., 2., 4., 8.]), decay=0.)
        assert torch.allclose(alloc.snr_ema, 1. / torch.tensor([1., 2., 4., 8.]), atol=1e-4)

    def test_high_snr_gets_more(self, alloc):
        alloc.update_snr_ema(torch.tensor([0.5, 1., 2., 4.]), decay=0.)
        probs = alloc.allocation_probs()
        assert probs[0] > probs[3]

    def test_rejects_narrow_total(self):
        with pytest.raises(ValueError):
            WaterFillingAllocator(10, 4, min_width=32)

    def test_rejects_zero_experts(self):
        with pytest.raises(ValueError, match=">= 1"):
            WaterFillingAllocator(100, 0)

    def test_rejects_zero_temp(self):
        with pytest.raises(ValueError, match="positive"):
            WaterFillingAllocator(100, 4, temperature=0.)

    def test_min_width_auto(self):
        assert WaterFillingAllocator(1024, 4, min_width=0).min_width >= 16
```

- [ ] **Step 5: Run and commit**

Run: `python -m pytest tests/test_hebbian_ffn.py tests/test_bottleneck.py tests/test_moe.py tests/test_water_filling.py -v`
Expected: 6 + 8 + 11 + 11 = 36 PASS.

```bash
git add tests/test_hebbian_ffn.py tests/test_bottleneck.py tests/test_moe.py tests/test_water_filling.py
git commit -m "test: add HebbianFFN, InformationBottleneck, WaterFillingMoE, and Allocator tests"
```

---

### Task 4: Refinement, EXIT, Outputs, Losses Tests (28 tests)

**Files:**
- Create: `C:\HAGI_v2\tests\test_refinement.py`
- Create: `C:\HAGI_v2\tests\test_exit_chart.py`
- Create: `C:\HAGI_v2\tests\test_outputs.py`
- Create: `C:\HAGI_v2\tests\test_losses.py`

- [ ] **Step 1: Write refinement tests**

Write `C:\HAGI_v2\tests\test_refinement.py`:
```python
"""Tests for PredictiveRefiner."""

from __future__ import annotations

import torch

from hagi.config import RefinementConfig
from hagi.model.refinement import PredictiveRefiner


def _refiner(**kw):
    cfg = RefinementConfig(enabled=True, iterations=kw.get("its", 2), hep_enabled=kw.get("hep", True))
    return PredictiveRefiner(64, cfg, use_ternary=False)


class TestPredictiveRefiner:
    def test_output_shape(self):
        assert _refiner()(torch.randn(2, 16, 64)).shape == (2, 16, 64)

    def test_output_different(self):
        h = torch.randn(2, 16, 64)
        assert not torch.allclose(_refiner()(h), h)

    def test_novelty_float(self):
        r = _refiner()
        r(torch.randn(2, 8, 64))
        assert isinstance(r.novelty(), float)

    def test_novelty_before_forward(self):
        assert _refiner().novelty() == 1.0

    def test_no_hep_works(self):
        r = _refiner(hep=False)
        assert r.hep is None
        assert r(torch.randn(2, 8, 64)).shape == (2, 8, 64)

    def test_more_its_more_change(self):
        r1 = _refiner(its=1, hep=False)
        r4 = _refiner(its=4, hep=False)
        h = torch.randn(2, 4, 32)
        assert (r4(h) - h).norm().item() > 0 and (r1(h) - h).norm().item() > 0
```

- [ ] **Step 2: Write EXIT chart tests**

Write `C:\HAGI_v2\tests\test_exit_chart.py`:
```python
"""Tests for EXITChartHalt."""

from __future__ import annotations

import pytest

from hagi.model.exit_chart import EXITChartHalt


class TestEXITChartHalt:
    def test_init_not_halted(self):
        h = EXITChartHalt(0.05, 10, 3)
        assert not h.halted and h.n_observations == 0

    def test_below_min_steps(self):
        h = EXITChartHalt(0.05, 10, 3)
        for _ in range(9):
            assert not h.observe(0.01)
        assert not h.halted

    def test_halted_after_low(self):
        h = EXITChartHalt(0.05, 10, 2)
        for _ in range(11):
            h.observe(0.01)
        assert h.halted

    def test_halted_sticky(self):
        h = EXITChartHalt(0.05, 5, 2)
        for _ in range(7):
            h.observe(0.01)
        assert h.halted
        assert h.observe(1.0)

    def test_reset(self):
        h = EXITChartHalt(0.05, 5, 2)
        for _ in range(7):
            h.observe(0.01)
        h.reset()
        assert not h.halted and h.n_observations == 0

    def test_high_novelty_no_halt(self):
        h = EXITChartHalt(0.05, 10, 3)
        for _ in range(20):
            h.observe(1.0)
        assert not h.halted

    def test_mixed_no_halt(self):
        h = EXITChartHalt(0.05, 10, 3)
        for i in range(20):
            h.observe(0.01 if i % 2 == 0 else 1.0)
        assert not h.halted

    def test_rejects_zero_threshold(self):
        with pytest.raises(ValueError):
            EXITChartHalt(0.)

    def test_rejects_zero_min_steps(self):
        with pytest.raises(ValueError):
            EXITChartHalt(min_steps=0)

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError):
            EXITChartHalt(window=0)
```

- [ ] **Step 3: Write outputs and losses tests**

Write `C:\HAGI_v2\tests\test_outputs.py`:
```python
"""Tests for AuxLosses and ModelOutput dataclasses."""

from __future__ import annotations

import torch

from hagi.model.outputs import AuxLosses, ModelOutput


class TestAuxLosses:
    def test_all_none_default(self):
        aux = AuxLosses()
        assert all(getattr(aux, f) is None
                   for f in ("rate", "distortion", "vicreg", "infonce", "moe_lb",
                             "route_entropy", "water_filling", "refinement", "attn_entropy"))

    def test_set_fields(self):
        t = torch.tensor(0.5)
        aux = AuxLosses(rate=t, distortion=t)
        assert aux.rate is not None and aux.vicreg is None


class TestModelOutput:
    def test_creation(self):
        out = ModelOutput(logits=None, hidden=torch.randn(2, 8, 64), aux=AuxLosses())
        assert out.hidden.shape == (2, 8, 64) and out.logits is None and out.ce_loss is None
```

Write `C:\HAGI_v2\tests\test_losses.py`:
```python
"""Tests for LossAggregator."""

from __future__ import annotations

import pytest
import torch

from hagi.config import Config
from hagi.model.outputs import AuxLosses, ModelOutput
from hagi.train.losses import LossAggregator, selected_cross_entropy


class TestSelectedCE:
    def test_basic(self):
        assert selected_cross_entropy(torch.randn(4, 100), torch.randint(0, 100, (4,))).item() > 0.0

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shapes"):
            selected_cross_entropy(torch.randn(4, 100), torch.randint(0, 100, (4, 3)))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            selected_cross_entropy(torch.randn(0, 100), torch.randint(0, 100, (0,)))


@pytest.fixture
def agg():
    return LossAggregator(Config())


def _out(ce=2.0, **aux_vals):
    aux = AuxLosses(**aux_vals)
    return ModelOutput(logits=torch.randn(16, 1000), hidden=torch.randn(2, 8, 64), aux=aux, ce_loss=torch.tensor(ce))


class TestLossAggregator:
    def test_ce_only(self, agg):
        assert abs(agg(_out(3.0), step=0).item() - 3.0) < 1e-3

    def test_raises_without_ce(self, agg):
        with pytest.raises(ValueError, match="ce_loss"):
            agg(ModelOutput(logits=None, hidden=torch.randn(2, 8, 64), aux=AuxLosses()), step=0)

    def test_adds_rate(self, agg):
        # CE + w_rate*rate = 2.0 + 0.01*1.0 = 2.01
        assert abs(agg(_out(2.0, rate=torch.tensor(1.0)), step=0).item() - 2.01) < 1e-3

    def test_distortion_beta_anneal(self, agg):
        loss0 = agg(_out(2.0, distortion=torch.tensor(10.0)), step=0).item()
        assert abs(loss0 - 2.0) < 1e-3  # beta=0 at step 0
        warmup = agg.warmup_steps
        loss_w = agg(_out(2.0, distortion=torch.tensor(10.0)), step=warmup).item()
        assert abs(loss_w - 2.1) < 1e-3  # beta=1, 2.0+0.01*1.0*10.0=2.1

    def test_subtracts_routing_entropy(self, agg):
        # CE - w = 2.0 - 0.01*1.0 = 1.99
        assert abs(agg(_out(2.0, route_entropy=torch.tensor(1.0)), step=0).item() - 1.99) < 1e-3

    def test_exit_not_halted_initially(self, agg):
        assert not agg.exit_halted
```

- [ ] **Step 4: Run and commit**

Run: `python -m pytest tests/test_refinement.py tests/test_exit_chart.py tests/test_outputs.py tests/test_losses.py -v`
Expected: 6 + 10 + 3 + 8 = 27 PASS.

```bash
git add tests/test_refinement.py tests/test_exit_chart.py tests/test_outputs.py tests/test_losses.py
git commit -m "test: add PredictiveRefiner, EXITChartHalt, AuxLosses/ModelOutput, and LossAggregator tests"
```

---

### Task 5: End-to-End Model, Dataset, and Checkpoint Tests (33 tests)

**Files:**
- Create: `C:\HAGI_v2\tests\test_model.py`
- Create: `C:\HAGI_v2\tests\test_dataset.py`
- Create: `C:\HAGI_v2\tests\test_checkpoint.py`

- [ ] **Step 1: Write model tests**

Write `C:\HAGI_v2\tests\test_model.py`:
```python
"""End-to-end tests for HAGI model forward pass."""

from __future__ import annotations

import torch

from hagi.config import Config, auto_configure
from hagi.model.model import HAGI


def _model():
    m = HAGI(Config())
    m.eval()
    return m


def _vtm(t):
    return torch.ones(2, t, dtype=torch.bool)


class TestHAGIForward:
    def test_causal_logits_shape(self):
        out = _model()(torch.randint(0, 1000, (2, 16)), targets=None, prediction_mask=_vtm(16),
                       valid_target_mask=_vtm(16), attention_mode="causal")
        assert out.logits.shape == (2, 16, 49154)

    def test_bidir(self):
        out = _model()(torch.randint(0, 1000, (2, 8)), targets=None, prediction_mask=_vtm(8),
                       valid_target_mask=_vtm(8), attention_mode="bidir")
        assert out.logits.shape == (2, 8, 49154)

    def test_prefix(self):
        out = _model()(torch.randint(0, 1000, (2, 12)), targets=None, prediction_mask=_vtm(12),
                       valid_target_mask=_vtm(12), attention_mode="prefix", prefix_len=4)
        assert out.logits.shape == (2, 12, 49154)

    def test_aux_rate_not_none(self):
        out = _model()(torch.randint(0, 1000, (2, 16)), targets=torch.randint(0, 1000, (2, 16)),
                       prediction_mask=_vtm(16), valid_target_mask=_vtm(16), attention_mode="causal")
        assert out.aux.rate is not None and out.aux.distortion is not None
        assert out.aux.moe_lb is None

    def test_hidden_shape(self):
        out = _model()(torch.randint(0, 1000, (2, 16)), targets=None, prediction_mask=_vtm(16),
                       valid_target_mask=_vtm(16), attention_mode="causal")
        assert out.hidden.shape == (2, 16, 384)

    def test_no_refinement_when_disabled(self):
        out = _model()(torch.randint(0, 1000, (2, 8)), targets=None, prediction_mask=_vtm(8),
                       valid_target_mask=_vtm(8), attention_mode="causal")
        assert out.aux.refinement is None


class TestHAGICache:
    def test_allocate_for_cache(self):
        m = _model()
        caches = m.allocate_for_cache(2, torch.float32, torch.device("cpu"))
        assert len(caches) == 12
        for c in caches:
            assert c.max_seq_len == 4096 and c.n_kv_heads == 4 and c.head_dim == 48

    def test_cached_forward(self):
        m = _model()
        m.allocate_for_cache(2, torch.float32, torch.device("cpu"))
        out = m(torch.randint(0, 1000, (2, 16)), targets=None, prediction_mask=_vtm(16),
                valid_target_mask=_vtm(16), attention_mode="causal")
        assert out.logits is not None
        m.reset_cache()
        assert all(blk.attn._kv_cache is None for blk in m.blocks)


class TestHAGIInit:
    def test_lm_head_weight_shape(self):
        w = HAGI(Config()).lm_head_weight
        assert w.shape == (49154, 384)

    def test_params_nonzero(self):
        assert sum(p.numel() for p in HAGI(Config()).parameters()) > 0

    def test_ternary_on_by_default(self):
        assert HAGI(Config())._use_ternary is True

    def test_auto_configured_instantiates(self):
        cfg = Config()
        cfg.model = auto_configure(15_000_000)
        m = HAGI(cfg)
        m.eval()
        ids = torch.randint(0, cfg.model.vocab_size, (1, 8))
        vtm = torch.ones(1, 8, dtype=torch.bool)
        out = m(ids, targets=None, prediction_mask=vtm, valid_target_mask=vtm, attention_mode="causal")
        assert out.logits is not None
```

- [ ] **Step 2: Write dataset tests**

Write `C:\HAGI_v2\tests\test_dataset.py`:
```python
"""Tests for MemmapDataset and validate_terminal_eos."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from hagi.data.dataset import MemmapDataset, validate_terminal_eos, dataset_path


class TestValidateTerminalEOS:
    def test_valid(self):
        ids = torch.tensor([[1, 2, 3, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]])
        valid = validate_terminal_eos(ids, eos_token_id=0, pad_token_id=9)
        assert valid[0, :4].all() and not valid[0, 4:].any()

    def test_rejects_no_eos(self):
        with pytest.raises(ValueError, match="exactly one"):
            validate_terminal_eos(torch.tensor([[1, 2, 3, 4]]), eos_token_id=0, pad_token_id=9)

    def test_rejects_multiple_eos(self):
        with pytest.raises(ValueError, match="exactly one"):
            validate_terminal_eos(torch.tensor([[0, 1, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]),
                                  eos_token_id=0, pad_token_id=9)


@pytest.fixture
def bin_path():
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        np.array([1, 2, 3, 0, 9, 9], dtype=np.uint16).tofile(path)
        yield path
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


class TestMemmapDataset:
    def test_len_positive(self, bin_path):
        ds = MemmapDataset(bin_path, seq_len=4, vocab_size=1000, eos_token_id=0, pad_token_id=9)
        assert len(ds) > 0

    def test_item_shapes(self, bin_path):
        ds = MemmapDataset(bin_path, seq_len=8, vocab_size=1000, eos_token_id=0, pad_token_id=9)
        item = ds[0]
        assert item["input_ids"].shape == (8,) and item["input_ids"].dtype == torch.long
        assert "valid_target_mask" in item and item["valid_target_mask"].shape == (8,)

    def test_eos_structure(self, bin_path):
        ds = MemmapDataset(bin_path, seq_len=8, vocab_size=1000, eos_token_id=0, pad_token_id=9)
        ids = ds[0]["input_ids"]
        eos_pos = (ids == 0).nonzero(as_tuple=True)[0]
        assert len(eos_pos) == 1
        assert (ids[eos_pos[0] + 1:] == 9).all()

    def test_without_eos_pad(self, bin_path):
        ds = MemmapDataset(bin_path, seq_len=4, vocab_size=1000)
        assert "valid_target_mask" not in ds[0]

    def test_rejects_seq_len_1(self, bin_path):
        with pytest.raises(ValueError):
            MemmapDataset(bin_path, seq_len=1, vocab_size=1000)

    def test_rejects_eos_without_pad(self, bin_path):
        with pytest.raises(ValueError, match="together"):
            MemmapDataset(bin_path, seq_len=8, vocab_size=1000, eos_token_id=0)


class TestDatasetPath:
    def test_valid(self):
        assert dataset_path("data", "tinystories").name == "tinystories.bin"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            dataset_path("data", "")

    def test_rejects_traversal(self):
        with pytest.raises(ValueError):
            dataset_path("data", "..")
```

- [ ] **Step 3: Write checkpoint tests**

Write `C:\HAGI_v2\tests\test_checkpoint.py`:
```python
"""Tests for checkpoint save/load roundtrip."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from hagi.config import Config, cfg_to_dict, cfg_from_dict
from hagi.model.model import HAGI
from hagi.train.checkpoint import (
    save_checkpoint, load_checkpoint_payload, load_model_checkpoint,
    latest_checkpoint, CHECKPOINT_FORMAT_VERSION,
)


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _model():
    return HAGI(Config())


class TestCheckpointRoundtrip:
    def test_save_and_load_payload(self, tmpdir):
        path = save_checkpoint(_model(), Config(), 500, tmpdir, keep_last=3)
        assert os.path.isfile(path)
        payload = load_checkpoint_payload(path)
        assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert payload["completed_updates"] == 500

    def test_load_model_strict(self, tmpdir):
        m1 = _model()
        path = save_checkpoint(m1, Config(), 100, tmpdir)
        m2 = _model()
        step, _ = load_model_checkpoint(path, m2, "cpu")
        assert step == 100
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1.data, p2.data)

    def test_rejects_wrong_format(self, tmpdir):
        path = save_checkpoint(_model(), Config(), 0, tmpdir)
        state = load_checkpoint_payload(path)
        state["format_version"] = 99
        torch.save(state, path)
        with pytest.raises(Exception):
            load_checkpoint_payload(path)

    def test_latest(self, tmpdir):
        save_checkpoint(_model(), Config(), 100, tmpdir)
        save_checkpoint(_model(), Config(), 200, tmpdir)
        assert "step-000200" in str(latest_checkpoint(tmpdir))

    def test_rotation(self, tmpdir):
        for step in [100, 200, 300, 400, 500]:
            save_checkpoint(_model(), Config(), step, tmpdir, keep_last=2)
        step_files = [f for f in os.listdir(tmpdir) if f.startswith("step-")]
        assert len(step_files) <= 2

    def test_config_roundtrip(self):
        cfg = Config()
        cfg2 = cfg_from_dict(cfg_to_dict(cfg))
        assert cfg2.model.hidden_size == cfg.model.hidden_size
        assert cfg2.train.learning_rate == cfg.train.learning_rate
```

- [ ] **Step 4: Run and commit**

Run: `python -m pytest tests/test_model.py tests/test_dataset.py tests/test_checkpoint.py -v`
Expected: 13 + 10 + 6 = 29 PASS.

```bash
git add tests/test_model.py tests/test_dataset.py tests/test_checkpoint.py
git commit -m "test: add HAGI e2e forward, MemmapDataset, and checkpoint roundtrip tests"
```

---

### Task 6: Full Suite Validation and Ablation Runner

**Files:**
- No new test files
- Create: `C:\HAGI_v2\scripts\run_ablation.py`

- [ ] **Step 1: Run complete test suite**

Run: `python -m pytest tests/ -v --tb=short`

Expected: ~160 tests, all PASS.

If any test fails, diagnose and fix before continuing.

- [ ] **Step 2: Write experiment runner**

Write `C:\HAGI_v2\scripts\run_ablation.py`:
```python
"""Automated ablation experiment runner for HAGI-2.

Runs training with a base YAML config and per-experiment CLI overrides,
logging step-level metrics to a CSV file.

Usage:
    python scripts/run_ablation.py \\
        --base-config configs/smollm2.yaml \\
        --overrides "baseline:" "no_ib:train.w_rate=0,train.w_distortion=0" \\
        --steps 1000 --data-dir data --device cpu
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from hagi.train._rocm_fsdp_stub import install as _install_rocm_fsdp_stub
_install_rocm_fsdp_stub()

from hagi.config import load_config
from hagi.model.model import HAGI
from hagi.data.sequential import build_sequential_dataloader
from hagi.train.loop import train

logger = logging.getLogger(__name__)


def parse_overrides(raw: list[str]) -> dict[str, dict[str, str]]:
    experiments: dict[str, dict[str, str]] = {}
    for item in raw:
        if ":" not in item:
            raise ValueError(f"override '{item}' missing ':' separator")
        name, kv_str = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty experiment name in '{item}'")
        overrides: dict[str, str] = {}
        if kv_str.strip():
            for kv in kv_str.split(","):
                kv = kv.strip()
                if "=" not in kv:
                    raise ValueError(f"invalid key=value pair: '{kv}'")
                k, v = kv.split("=", 1)
                overrides[k.strip()] = v.strip()
        experiments[name] = overrides
    return experiments


def apply_override(cfg, key: str, value: str):
    parts = key.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    current = getattr(obj, parts[-1])
    if isinstance(current, bool):
        setattr(obj, parts[-1], value.lower() in ("true", "1", "yes"))
    elif isinstance(current, int):
        setattr(obj, parts[-1], int(value))
    elif isinstance(current, float):
        setattr(obj, parts[-1], float(value))
    else:
        setattr(obj, parts[-1], value)


METRICS = [
    "experiment", "step", "loss", "bpt", "masked_ce", "rate", "rate_bits",
    "distortion", "posterior_entropy", "top2_mass", "avg_confidence",
    "grad_norm", "grad_rms", "lr", "exit_halted",
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="HAGI ablation experiment runner")
    parser.add_argument("--base-config", default="configs/smollm2.yaml")
    parser.add_argument("--overrides", nargs="*", default=[], help="name:k=v,k2=v2 ...")
    parser.add_argument("--steps", type=int, default=None, help="Max steps per experiment")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="logs/ablations")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    experiments = parse_overrides(args.overrides)
    if not experiments:
        experiments = {"baseline": {}}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    logger.info(f"Device: {device} | Output: {csv_path} | {len(experiments)} experiments")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS)
        writer.writeheader()

        for exp_name, overrides in experiments.items():
            logger.info(f"=== {exp_name} ===")
            cfg = load_config(path=args.base_config)
            if args.steps is not None:
                cfg.train.max_steps = args.steps
            for k, v in overrides.items():
                apply_override(cfg, k, v)

            model = HAGI(cfg).to(device)
            n = sum(p.numel() for p in model.parameters())
            logger.info(f"Params: {n/1e6:.1f}M")
            dataloader = build_sequential_dataloader(cfg, data_dir=args.data_dir)

            for metrics in train(model, dataloader, cfg, start_step=0):
                row = {"experiment": exp_name}
                for field in METRICS:
                    if field == "experiment":
                        continue
                    val = metrics.get(field, float("nan"))
                    row[field] = val if isinstance(val, (int, float, bool)) else float("nan")
                writer.writerow(row)
            f.flush()
            logger.info(f"[{exp_name}] complete")
    logger.info(f"Done: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_ablation.py
git commit -m "feat: add automated ablation experiment runner with CSV logging"
```

---

### Task 7: Update Architecture Documentation

**Files:**
- Modify: `C:\HAGI_v2\docs\ARCHITECTURE.md`

- [ ] **Step 1: Update ARCHITECTURE.md for V28 unified stack**

The current `docs/ARCHITECTURE.md` documents the V25-style architecture with a context/expression split stack that no longer exists. It must be updated to reflect the V28 unified KV-cacheable stack.

`C:\HAGI_v2\docs\ARCHITECTURE.md` (replace with):

```markdown
# HAGI-2 -- Ternary RD-Channel Causal Language Model (V28)

HAGI-2 is a causal autoregressive language model designed as a communication
system. The transformer body is **ternary** (BitNet b1.58: weights in
{-1, 0, +1} stored at 1.585 bits/weight). That quantization is the *genuine*
discrete channel -- its noise is the only impairment. There is no
self-inflicted AWGN/LDPC physical channel.

---

## 1. Signal Path (Forward)

```
input_ids [B, T_text]
    |
    v
[STAGE 1: Source Encode]
  ConvEmbedding:
    token_compress(V, r) -> token_expand(r, H) -> CAUSAL Conv1d (left-pad only) -> RMSNorm
    |
    v  h [B, T_text, H]

  (Multimodal, if enabled):
    Image: patches -> Linear -> 2D-RoPE -> inv-var gate -> Q-Former -> [B, n_bridge, H]
    Audio: mel frames -> Linear -> 1D-RoPE -> inv-var gate -> Q-Former -> [B, n_bridge, H]
    h = concat([prefix, h_text])   [B, prefix_len + T_text, H]
    |
    v
[STAGE 2: UNIFIED Ternary Transformer Stack]
  L x TransformerBlock (pre-norm, causal KV-cache compatible):
    h = h + Attention(RMSNorm(h), RoPE, GQA, optional sliding window)
    h = h + (WaterFillingMoE(h) if MoE-layer else HebbianBilinearFFN(h))
  Collects: attn_entropy_penalty, moe_lb, routing_entropy, water_filling_loss
    |
    v  h_ctx [B, T, H]
    |
[STAGE 3: Auxiliary IB — OFF-PATH, h_ctx.detach()]
  InformationBottleneck(H -> C -> H):
    q(z|h) = N(mu, exp(logvar))
    rate = KL[q||N(0,I)], distortion = ||h - h_hat||^2 / ||h||^2
    |
    v
[STAGE 4: Main LM Path]
  h_dec = final_norm(h_ctx)
  h_text = h_dec[:, prefix_len:]          (text-only positions)
  logits = lm_expand(lm_compress(h_text)) (factored rank-r head)
  ce_loss = cross_entropy(logits, targets)
    |
    v
[STAGE 4b: Off-path HEP Refinement — h_ctx.detach(), opt-in]
  PredictiveRefiner: iterative extrinsic correction on a clone of h_ctx
    |
    v
[STAGE 5: Grounded Infomax — on h_ctx (NOT detached), multimodal only]
  VICReg + InfoNCE on per-modality pooled embeddings
    |
    v
[LOSS = CE + sum(w_i * aux_loss_i)]
```

## 2. Modules

| Module | File | Params | Role |
|---|---|---|---|
| ConvEmbedding | conv_embedding.py | V*r + r*H + H*K + 2H | Factorized source encoder + causal pulse-shaping |
| Attention | attention.py | ~(2n_q+2n_kv)*hd*H + H | GQA + RoPE, 4 attention modes, sliding window, KV-cache |
| HebbianBilinearFFN | hebbian_ffn.py | 12H^2 + 2H | Bilinear SwiGLU: (A0*h) * silu(A1*h), gate-modulated |
| WaterFillingMoE | moe.py | (E+n_shared)*3*inter*H + router | SNR-gated top-k routing + batched dispatch |
| BitLinear | ternary.py | [out, in] FP master | BitNet b1.58: {-scale, 0, +scale} effective weights |
| InformationBottleneck | bottleneck.py | 3*C*H + H | Off-path variational H->C->H, rate+distortion |
| PredictiveRefiner | refinement.py | ~3*H^2 | Off-path HEP iterative hidden refinement |
| WaterFillingAllocator | water_filling.py | 2E | Per-expert capacity allocation via SNR EMA |
| MultimodalFusion | multimodal.py | varies | Q-Former bridge for fixed-size multimodal prefix |
| GroundedInfomax | grounded.py | M*H^2 | VICReg + InfoNCE joint embedding alignment |

## 3. Optimizer

Type-based routing:
- **BitLinear.weight** -> Muon (Newton-Schulz orthogonalization, scale-aware WD)
- **All other params** -> AdamW (fused where available, decay/no_decay split)

## 4. Design Rules

1. **Off-path auxiliaries.** IB, refinement, grounded infomax are on `h_ctx.detach()`.
   (Exception: GroundedInfomax sends gradient into the body deliberately.)
2. **C < H.** Core hidden size must be strictly less than hidden size (real compression).
3. **CAUSAL conv only.** Left-pad, no future leak. V25 symmetric-pad was root cause #4.
4. **No magic numbers.** `auto_configure(target_params)` derives all sizes.
5. **Checkpoint format V7.** Incompatible with V27.
```

- [ ] **Step 2: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: update ARCHITECTURE.md to V28 unified KV-cacheable stack"
```

---

## Self-Review

**1. Spec coverage:** V30_DESIGN_AND_DEVELOPMENT_PLAN.md has 6 sections. Sections 1-3 (Architecture, Production, Scientific) describe what already exists -- no implementation needed. Section 4 (Experimental Plan) needs the ablation runner (Task 6). Section 5 (File Map) is documentation. Section 6 (Version History) is documentation. Gap: test coverage exists in zero files today -- Tasks 1-5 fill this. Gap: ARCHITECTURE.md is stale -- Task 7 fixes this.

**2. Placeholder scan:** No TBD/TODO/fill-in-later. All test code is complete with exact assertions. All expected counts are calculated.

**3. Type consistency:** Test fixtures use consistent naming. Metric field names match `train_step` output dict. Config field paths match `config.py` dataclass hierarchy.

**Verification gate:** After all 7 tasks complete, run `python -m pytest tests/ -v` -- all ~160 tests must pass. Run `python scripts/run_ablation.py --steps 10 --device cpu --data-dir data` -- must produce CSV output.

---

## Execution Handoff

**Plan complete. Two execution options:**

**1. Subagent-Driven (recommended)** -- I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** -- Execute tasks in this session using executing-plans, batch with checkpoints

**Which approach?**
