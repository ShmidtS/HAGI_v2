"""Attention: mask composition, GQA expansion, KV-cache decode exactness.

The mask is where causality lives. A leak here does not raise and does not look
wrong in the loss — it makes next-token prediction partly trivial at training
time and impossible at inference, which is the "training works, generation is
garbage" failure. So every constraint the mask composes is asserted against an
explicit reference built by nested loops.
"""

from __future__ import annotations

import pytest
import torch

from hagi.model.attention import Attention, AttentionConfig, build_attention_mask, repeat_kv
from hagi.model.kv_cache import KVCache
from tests.conftest import assert_finite


def reference_allowed(
    t_q: int, t_total: int, *, window: int, doc_ids: torch.Tensor | None, prefix_len: int
) -> torch.Tensor:
    """Nested-loop truth for the mask, one boolean per (query, key) pair."""
    batch = 1 if doc_ids is None else doc_ids.shape[0]
    out = torch.zeros(batch, t_q, t_total, dtype=torch.bool)
    offset = t_total - t_q
    for b in range(batch):
        for i in range(t_q):
            q_pos = offset + i
            for j in range(t_total):
                ok = j <= q_pos
                if window > 0:
                    ok = ok and j > q_pos - window
                if doc_ids is not None:
                    ok = ok and int(doc_ids[b, q_pos]) == int(doc_ids[b, j])
                if prefix_len > 0 and j < prefix_len:
                    ok = True
                out[b, i, j] = ok
    return out


class TestMask:
    def test_plain_causal_returns_none(self):
        assert (
            build_attention_mask(4, 4, device=torch.device("cpu"), dtype=torch.float32) is None
        ), "an unconstrained causal mask must fall through to SDPA's fused kernel"

    @pytest.mark.parametrize(
        "window,use_docs,prefix",
        [(3, False, 0), (0, True, 0), (3, True, 0), (0, False, 2), (2, True, 2)],
        ids=["window", "docs", "window_docs", "prefix", "all"],
    )
    def test_matches_reference(self, window, use_docs, prefix):
        t = 8
        doc_ids = None
        if use_docs:
            doc_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 1, 1, 2, 2, 3, 3]])
        mask = build_attention_mask(
            t, t, window=window, doc_ids=doc_ids, prefix_len=prefix,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        allowed = mask == 0.0
        expected = reference_allowed(t, t, window=window, doc_ids=doc_ids, prefix_len=prefix)
        assert torch.equal(allowed.squeeze(1), expected)

    def test_no_future_key_is_ever_allowed(self):
        """The one property whose violation is silent and fatal."""
        t = 16
        doc_ids = torch.zeros(1, t, dtype=torch.long)
        mask = build_attention_mask(
            t, t, window=5, doc_ids=doc_ids, device=torch.device("cpu"), dtype=torch.float32
        )
        allowed = (mask == 0.0)[0, 0]
        for i in range(t):
            assert not allowed[i, i + 1 :].any(), f"query {i} can see the future"

    def test_decode_offset(self):
        """A single query against a cached prefix takes the last query position."""
        mask = build_attention_mask(
            1, 10, window=4, device=torch.device("cpu"), dtype=torch.float32
        )
        allowed = (mask == 0.0)[0, 0, 0]
        assert allowed[6:].all() and not allowed[:6].any()

    def test_doc_ids_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_attention_mask(
                4, 4, doc_ids=torch.zeros(1, 5, dtype=torch.long),
                device=torch.device("cpu"), dtype=torch.float32,
            )


class TestGQA:
    def test_single_rep_is_identity(self):
        x = torch.randn(2, 3, 4, 8)
        assert repeat_kv(x, 1) is x

    def test_repeat_groups_are_contiguous(self):
        x = torch.randn(1, 2, 3, 4)
        out = repeat_kv(x, 3)
        assert out.shape == (1, 6, 3, 4)
        for group in range(2):
            for rep in range(3):
                assert torch.equal(out[0, group * 3 + rep], x[0, group])


class TestAttention:
    def make(self, **kw) -> Attention:
        cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16, **kw)
        return Attention(64, cfg, use_ternary=False)

    def test_shape_and_finiteness(self):
        attn = self.make()
        out = attn(torch.randn(2, 6, 64))
        assert out.shape == (2, 6, 64)
        assert_finite(out, "attention output")

    def test_hidden_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            Attention(48, AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16))

    def test_kv_heads_must_divide(self):
        with pytest.raises(ValueError):
            Attention(64, AttentionConfig(num_heads=4, num_kv_heads=3, head_dim=16))

    def test_causality_by_perturbation(self):
        """Changing a future input must not change a past output."""
        attn = self.make().eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            base = attn(x)
            x2 = x.clone()
            x2[:, 5:] += 10.0
            perturbed = attn(x2)
        assert torch.allclose(base[:, :5], perturbed[:, :5], atol=1e-5)

    def test_qk_norm_bounds_the_logit_range(self):
        """With QK-norm the score scale must not follow the projection norms.

        This is the V30 divergence in miniature: Muon removes the ``1/||W||``
        brake, so ``||q|| ||k||`` drifts outward and the softmax saturates. The
        test inflates the projections by 8x and asserts the normed variant's
        logit range does not follow.
        """
        def logit_range(module: Attention) -> float:
            h = module.attn_norm(torch.randn(1, 8, 64))
            qkv = module.qkv_proj(h)
            n_q = module.n_heads * module.head_dim
            q = qkv[..., :n_q].view(1, 8, 4, 16).transpose(1, 2)
            kv = qkv[..., n_q:].view(1, 8, 2, 2, 16)
            k = kv.unbind(dim=2)[0].transpose(1, 2)
            if module.q_norm is not None:
                q, k = module.q_norm(q), module.k_norm(k)
            return float((q @ repeat_kv(k, module.n_rep).transpose(-1, -2)).abs().max())

        torch.manual_seed(0)
        normed = self.make(qk_norm=True).eval()
        torch.manual_seed(0)
        plain = self.make(qk_norm=False).eval()

        with torch.no_grad():
            torch.manual_seed(7)
            base_normed = logit_range(normed)
            torch.manual_seed(7)
            base_plain = logit_range(plain)
            for module in (normed, plain):
                module.qkv_proj.weight.mul_(8.0)
            torch.manual_seed(7)
            grown_normed = logit_range(normed)
            torch.manual_seed(7)
            grown_plain = logit_range(plain)

        assert grown_plain / base_plain > 50.0, "unnormalized logits should grow ~64x"
        assert grown_normed / base_normed < 1.5, (
            f"qk_norm let the logit range grow {grown_normed / base_normed:.1f}x"
        )


class TestKVCacheDecode:
    def test_incremental_matches_full_forward(self):
        """Decode with a cache must reproduce the full forward, position by position."""
        attn = Attention(
            64, AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16), use_ternary=False
        ).eval()
        x = torch.randn(1, 6, 64)
        with torch.no_grad():
            full = attn(x)

            cache = KVCache(16, 2, 16, torch.float32, torch.device("cpu"))
            attn.attach_cache(cache)
            steps = []
            for t in range(6):
                steps.append(
                    attn(x[:, t : t + 1], positions=torch.arange(t, t + 1))
                )
            attn.detach_cache()
        incremental = torch.cat(steps, dim=1)
        assert float((full - incremental).abs().max()) < 1e-5


class TestKVCache:
    def make(self) -> KVCache:
        return KVCache(8, 2, 4, torch.float32, torch.device("cpu"))

    def test_length_tracks_updates(self):
        cache = self.make()
        assert cache.length == 0
        cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
        assert cache.length == 3
        cache.update(torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))
        assert cache.length == 5

    def test_get_returns_only_the_filled_prefix(self):
        cache = self.make()
        k = torch.randn(1, 2, 3, 4)
        cache.update(k, k)
        got_k, _ = cache.get()
        assert got_k.shape == (1, 2, 3, 4)
        assert torch.equal(got_k, k)

    def test_overflow_raises(self):
        cache = self.make()
        with pytest.raises(ValueError, match="overflow"):
            cache.update(torch.randn(1, 2, 9, 4), torch.randn(1, 2, 9, 4))

    def test_shape_mismatch_raises(self):
        cache = self.make()
        with pytest.raises(ValueError):
            cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 2, 4))
        with pytest.raises(ValueError):
            cache.update(torch.randn(1, 5, 3, 4), torch.randn(1, 5, 3, 4))

    def test_empty_get_raises(self):
        with pytest.raises(RuntimeError):
            self.make().get()

    def test_reset_keeps_capacity(self):
        cache = self.make()
        cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
        cache.reset()
        assert cache.length == 0
        cache.update(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4))
        assert cache.length == 8

    def test_batch_change_reallocates(self):
        cache = self.make()
        cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
        cache.update(torch.randn(4, 2, 2, 4), torch.randn(4, 2, 2, 4))
        assert cache.length == 2
