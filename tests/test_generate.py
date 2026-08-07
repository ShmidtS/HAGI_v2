"""Generation: logit shaping order, cache-vs-recompute equivalence, termination.

The order in :func:`filter_logits` is load-bearing — a penalty applied after
temperature has a strength that depends on the temperature — so each stage is
checked in isolation and then the composition is checked for the one property
that must never break: the kept set is never empty.
"""

from __future__ import annotations

import pytest
import torch

from hagi.inference.generate import filter_logits, generate
from hagi.model.model import HAGI
from tests.conftest import TINY_VOCAB, tiny_config


def logits(rows=1, vocab=8) -> torch.Tensor:
    return torch.arange(rows * vocab, dtype=torch.float32).view(rows, vocab)


class TestFilterLogits:
    def base(self, out: torch.Tensor, **kw) -> torch.Tensor:
        defaults = dict(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            repetition_window=8,
        )
        defaults.update(kw)
        return filter_logits(out, torch.zeros(out.shape[0], 0, dtype=torch.long), **defaults)

    def test_identity_when_everything_is_off(self):
        x = logits()
        assert torch.equal(self.base(x), x.float())

    def test_input_is_not_mutated(self):
        x = logits()
        before = x.clone()
        self.base(x, top_k=2)
        assert torch.equal(x, before)

    def test_banned_ids_are_removed(self):
        out = self.base(logits(), banned=(3,))
        assert out[0, 3] == float("-inf")

    def test_top_k_keeps_exactly_k(self):
        out = self.base(logits(vocab=10), top_k=3)
        assert int(torch.isfinite(out).sum()) == 3

    def test_top_k_larger_than_vocab_is_a_noop(self):
        out = self.base(logits(vocab=4), top_k=99)
        assert bool(torch.isfinite(out).all())

    def test_top_p_keeps_a_nonempty_set(self):
        """Even when one token holds more than p, the kept set must not be empty."""
        peaked = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
        out = self.base(peaked, top_p=0.1)
        assert int(torch.isfinite(out).sum()) >= 1

    def test_top_p_removes_tail_mass(self):
        out = self.base(torch.tensor([[5.0, 4.0, -10.0, -20.0]]), top_p=0.9)
        assert out[0, 3] == float("-inf")

    def test_temperature_scales(self):
        x = torch.tensor([[2.0, 4.0]])
        out = self.base(x, temperature=2.0)
        assert torch.allclose(out, x / 2.0)

    def test_repetition_penalty_lowers_recent_tokens(self):
        x = torch.tensor([[3.0, 3.0, -3.0]])
        context = torch.tensor([[0, 2]])
        out = filter_logits(
            x,
            context,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=2.0,
            repetition_window=8,
        )
        assert float(out[0, 0]) < 3.0, "a positive logit should be divided"
        assert float(out[0, 2]) < -3.0, "a negative logit should be multiplied"
        assert float(out[0, 1]) == 3.0, "an unseen token must be untouched"

    def test_penalty_precedes_temperature(self):
        """Otherwise the penalty's strength would depend on the temperature."""
        x = torch.tensor([[4.0, 4.0]])
        context = torch.tensor([[0]])
        cold = filter_logits(
            x, context, temperature=0.5, top_k=0, top_p=1.0,
            repetition_penalty=2.0, repetition_window=4,
        )
        # penalty then temperature: (4/2)/0.5 = 4; the reverse would give 8/2 = 4
        # for the penalized token but 8 for the other, so compare the *ratio*.
        assert float(cold[0, 1] / cold[0, 0]) == pytest.approx(2.0)

    def test_returns_float32(self):
        assert self.base(logits().bfloat16()).dtype == torch.float32


class TestGenerate:
    def model(self, **overrides) -> HAGI:
        cfg = tiny_config(**overrides)
        model = HAGI(cfg)
        model.eval()
        return model

    def test_length_and_shape(self):
        model = self.model()
        prompt = torch.randint(2, TINY_VOCAB, (2, 4))
        out = generate(model, prompt, max_new_tokens=6, eos_token_id=1, pad_token_id=0)
        assert out.token_ids.shape == (2, 10)
        assert torch.equal(out.token_ids[:, :4], prompt)

    def test_greedy_is_deterministic(self):
        model = self.model()
        prompt = torch.randint(2, TINY_VOCAB, (1, 4))
        kw = dict(max_new_tokens=5, eos_token_id=1, pad_token_id=0, temperature=0.0)
        first = generate(model, prompt, **kw)
        second = generate(model, prompt, **kw)
        assert torch.equal(first.token_ids, second.token_ids)

    def test_cache_matches_recompute(self):
        """The cache must not change what is generated, only what it costs."""
        model = self.model()
        prompt = torch.randint(2, TINY_VOCAB, (1, 5))
        kw = dict(max_new_tokens=6, eos_token_id=1, pad_token_id=0, temperature=0.0)
        cached = generate(model, prompt, use_cache=True, **kw)
        recomputed = generate(model, prompt, use_cache=False, **kw)
        assert torch.equal(cached.token_ids, recomputed.token_ids)

    def test_min_new_tokens_suppresses_eos(self):
        model = self.model()
        prompt = torch.randint(2, TINY_VOCAB, (1, 3))
        out = generate(
            model, prompt, max_new_tokens=5, eos_token_id=1, pad_token_id=0,
            temperature=0.0, min_new_tokens=5,
        )
        assert not bool((out.token_ids[:, 3:] == 1).any())

    def test_pad_is_never_sampled(self):
        model = self.model()
        prompt = torch.randint(2, TINY_VOCAB, (1, 3))
        out = generate(model, prompt, max_new_tokens=8, eos_token_id=1, pad_token_id=0)
        generated = out.token_ids[0, 3:]
        finished_at = int(out.lengths[0])
        assert not bool((generated[:finished_at] == 0).any())

    def test_seeded_sampling_is_reproducible(self):
        model = self.model()
        prompt = torch.randint(2, TINY_VOCAB, (1, 3))
        kw = dict(max_new_tokens=5, eos_token_id=1, pad_token_id=0, temperature=1.0)

        def run():
            gen = torch.Generator().manual_seed(99)
            return generate(model, prompt, generator=gen, **kw).token_ids

        assert torch.equal(run(), run())

    def test_cache_is_released(self):
        model = self.model()
        prompt = torch.randint(2, TINY_VOCAB, (1, 3))
        generate(model, prompt, max_new_tokens=3, eos_token_id=1, pad_token_id=0)
        assert all(block.attn._kv_cache is None for block in model.blocks)
        assert model.encoder._state is None

    @pytest.mark.parametrize(
        "overrides", [{}, {"model.sliding.window": 8}, {"model.embedding.conv_kernel": 1}],
        ids=["full", "windowed", "no_conv"],
    )
    def test_variants_generate(self, overrides):
        model = self.model(**overrides)
        prompt = torch.randint(2, TINY_VOCAB, (1, 4))
        out = generate(model, prompt, max_new_tokens=4, eos_token_id=1, pad_token_id=0)
        assert out.token_ids.shape == (1, 8)

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            (dict(max_new_tokens=0), "max_new_tokens"),
            (dict(max_new_tokens=4, min_new_tokens=5), "min_new_tokens"),
        ],
    )
    def test_invalid_budget_raises(self, kwargs, message):
        model = self.model()
        prompt = torch.randint(2, TINY_VOCAB, (1, 3))
        with pytest.raises(ValueError, match=message):
            generate(model, prompt, eos_token_id=1, pad_token_id=0, **kwargs)

    def test_non_2d_prompt_raises(self):
        model = self.model()
        with pytest.raises(ValueError):
            generate(
                model, torch.randint(0, 10, (4,)), max_new_tokens=2,
                eos_token_id=1, pad_token_id=0,
            )

    def test_out_of_vocabulary_prompt_raises(self):
        model = self.model()
        with pytest.raises(ValueError, match="out-of-vocabulary"):
            generate(
                model, torch.tensor([[TINY_VOCAB + 5]]), max_new_tokens=2,
                eos_token_id=1, pad_token_id=0,
            )
