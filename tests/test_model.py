"""End-to-end model: loss path, packed masking, KV-cache decode, diagnostics.

The two assertions worth the runtime: every trainable parameter receives a
gradient (a parameter with none trains at rate zero forever and nothing reports
it), and incremental decoding reproduces the full forward (a cache that drifts
produces a model whose generation quality has nothing to do with its training
loss).
"""

from __future__ import annotations

import math

import pytest
import torch

from hagi.model.model import HAGI
from tests.conftest import TINY_VOCAB, assert_finite, tiny_config


def batch(b=2, t=16, vocab=TINY_VOCAB):
    ids = torch.randint(0, vocab, (b, t))
    targets = torch.randint(0, vocab, (b, t))
    return ids, targets


class TestForward:
    def test_hidden_only_without_targets(self):
        cfg = tiny_config()
        out = HAGI(cfg)(batch()[0])
        assert out.hidden.shape == (2, 16, cfg.model.hidden_size)
        assert out.loss is None and out.ce is None

    def test_loss_path(self):
        cfg = tiny_config()
        ids, targets = batch()
        out = HAGI(cfg)(ids, targets)
        assert out.n_tokens == 32
        assert_finite(out.loss, "loss")
        assert float(out.ce.detach()) > 0

    def test_untrained_ce_is_near_ln_v(self):
        """Without a prior, an untrained model must sit at ``ln V``."""
        cfg = tiny_config()
        ids, targets = batch(b=4, t=32)
        out = HAGI(cfg).eval()(ids, targets)
        assert abs(float(out.ce.detach()) - math.log(TINY_VOCAB)) < 0.5

    def test_return_logits(self):
        cfg = tiny_config()
        out = HAGI(cfg)(batch()[0], return_logits=True)
        assert out.logits.shape == (2, 16, TINY_VOCAB)

    def test_target_shape_mismatch_raises(self):
        model = HAGI(tiny_config())
        with pytest.raises(ValueError):
            model(torch.randint(0, TINY_VOCAB, (2, 16)), torch.randint(0, TINY_VOCAB, (2, 8)))

    def test_loss_mask_selects_positions(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        ids, targets = batch()
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[:, :4] = True
        out = model(ids, targets, loss_mask=mask)
        assert out.n_tokens == 8

    def test_all_parameters_receive_gradient(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        ids, targets = batch()
        model(ids, targets).loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"no gradient reached: {missing}"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"model.sliding.window": 8},
            {"model.moe.enabled": True, "model.moe.num_experts": 4, "model.moe.moe_every": 2},
            {"model.embedding.conv_kernel": 1},
            {"model.embedding.tie_lm_head": False},
            {"model.ternary.enabled": False},
            {"train.grad_checkpointing": True},
        ],
        ids=["windowed", "moe", "no_conv", "untied", "no_ternary", "checkpointing"],
    )
    def test_variants_train_end_to_end(self, overrides):
        cfg = tiny_config(**overrides)
        model = HAGI(cfg)
        model.train()
        ids, targets = batch()
        out = model(ids, targets)
        out.loss.backward()
        assert_finite(out.loss, "loss")
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"no gradient reached: {missing}"


class TestPackedDocuments:
    def test_doc_boundary_blocks_leakage(self):
        """A token in document 1 must not be affected by document 0's content."""
        cfg = tiny_config()
        model = HAGI(cfg).eval()
        ids = torch.randint(0, TINY_VOCAB, (1, 12))
        doc_ids = torch.tensor([[0] * 6 + [1] * 6])

        with torch.no_grad():
            base = model(ids, doc_ids=doc_ids).hidden
            changed = ids.clone()
            changed[0, 0] = (int(ids[0, 0]) + 7) % TINY_VOCAB
            perturbed = model(changed, doc_ids=doc_ids).hidden

        # The causal filter has a k-1 reach, so only positions at or beyond
        # conv_kernel-1 past the boundary are fully isolated.
        reach = cfg.model.embedding.conv_kernel - 1
        assert float((base[:, 6 + reach :] - perturbed[:, 6 + reach :]).abs().max()) < 1e-5

    def test_cross_doc_attention_does_leak(self):
        """The control: without doc_ids the same perturbation must propagate."""
        cfg = tiny_config()
        model = HAGI(cfg).eval()
        ids = torch.randint(0, TINY_VOCAB, (1, 12))
        with torch.no_grad():
            base = model(ids).hidden
            changed = ids.clone()
            changed[0, 0] = (int(ids[0, 0]) + 7) % TINY_VOCAB
            perturbed = model(changed).hidden
        assert float((base[:, 8:] - perturbed[:, 8:]).abs().max()) > 1e-6


class TestDecode:
    @pytest.mark.parametrize(
        "overrides",
        [{}, {"model.sliding.window": 8}, {"model.embedding.conv_kernel": 1}],
        ids=["full", "windowed", "no_conv"],
    )
    def test_incremental_matches_full(self, overrides):
        cfg = tiny_config(**overrides)
        model = HAGI(cfg).eval()
        ids = torch.randint(0, TINY_VOCAB, (1, 10))

        with torch.no_grad():
            full = model(ids, return_logits=True).logits

            model.reset_cache()
            model.allocate_cache(torch.float32, torch.device("cpu"))
            steps = []
            for t in range(10):
                out = model(
                    ids[:, t : t + 1],
                    positions=torch.arange(t, t + 1),
                    use_cache=True,
                    return_logits=True,
                )
                steps.append(out.logits)
            model.reset_cache()

        incremental = torch.cat(steps, dim=1)
        assert float((full - incremental).abs().max()) < 1e-4
        agreement = float((full.argmax(-1) == incremental.argmax(-1)).float().mean())
        assert agreement == 1.0, f"argmax agreement {agreement:.3f}"

    def test_prefill_then_decode(self):
        cfg = tiny_config()
        model = HAGI(cfg).eval()
        ids = torch.randint(0, TINY_VOCAB, (1, 8))
        with torch.no_grad():
            full = model(ids, return_logits=True).logits
            model.reset_cache()
            model.allocate_cache(torch.float32, torch.device("cpu"))
            prefill = model(ids[:, :5], use_cache=True, return_logits=True).logits
            rest = [
                model(
                    ids[:, t : t + 1],
                    positions=torch.arange(t, t + 1),
                    use_cache=True,
                    return_logits=True,
                ).logits
                for t in range(5, 8)
            ]
            model.reset_cache()
        combined = torch.cat([prefill, *rest], dim=1)
        assert float((full - combined).abs().max()) < 1e-4


class TestStructure:
    def test_residual_scale_follows_depth(self):
        shallow = HAGI(tiny_config(**{"model.num_layers": 2}))
        deep = HAGI(tiny_config(**{"model.num_layers": 8}))
        shallow_std = float(shallow.blocks[0].attn.out_proj.weight.detach().std())
        deep_std = float(deep.blocks[0].attn.out_proj.weight.detach().std())
        assert deep_std < shallow_std

    def test_loop_depth_scales_residual_like_deeper_stack(self):
        plain = HAGI(tiny_config(**{"model.num_layers": 2, "model.loop_depth": 1}))
        looped = HAGI(tiny_config(**{"model.num_layers": 2, "model.loop_depth": 4}))
        plain_std = float(plain.blocks[0].attn.out_proj.weight.detach().std())
        loop_std = float(looped.blocks[0].attn.out_proj.weight.detach().std())
        assert loop_std < plain_std
        assert looped._loop_depth == 4

    def test_loop_depth_forward_shape(self):
        cfg = tiny_config(**{"model.loop_depth": 2})
        model = HAGI(cfg)
        ids, targets = batch()
        out = model(ids, targets)
        assert out.hidden.shape == (2, 16, cfg.model.hidden_size)
        assert_finite(out.loss, "looped loss")

    def test_head_is_tied_to_the_codebook(self):
        model = HAGI(tiny_config())
        assert model.head.weight is model.encoder.weight

    def test_untied_head_owns_its_projection(self):
        model = HAGI(tiny_config(**{"model.embedding.tie_lm_head": False}))
        assert model.head.weight is not model.encoder.weight

    def test_param_summary_matches_reality(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        assert model.param_summary()["total"] == sum(p.numel() for p in model.parameters())


class TestDiagnostics:
    def test_dense_keys(self):
        stats = HAGI(tiny_config()).diagnostics()
        assert set(stats) == {"qk_gain", "residual_gain", "logit_scale"}
        assert stats["logit_scale"] == pytest.approx(tiny_config().model.hidden_size**-0.5)

    def test_moe_keys_present(self):
        cfg = tiny_config(
            **{"model.moe.enabled": True, "model.moe.num_experts": 4, "model.moe.moe_every": 2}
        )
        stats = HAGI(cfg).diagnostics()
        assert "moe/entropy_ratio" in stats and "moe/bias_span" in stats

    def test_no_qk_gain_without_qk_norm(self):
        stats = HAGI(tiny_config(**{"model.attention.qk_norm": False})).diagnostics()
        assert "qk_gain" not in stats

    def test_controller_commit_is_a_noop_when_dense(self):
        HAGI(tiny_config()).commit_controller_updates()


class TestObjective:
    def test_z_loss_is_included_when_weighted(self):
        cfg = tiny_config()
        cfg.train.z_loss_weight = 1.0
        ids, targets = batch()
        out = HAGI(cfg)(ids, targets)
        assert float(out.loss.detach()) > float(out.ce.detach())

    def test_router_z_loss_summed_over_moe_layers(self):
        cfg = tiny_config(
            **{"model.moe.enabled": True, "model.moe.num_experts": 4, "model.moe.moe_every": 1}
        )
        cfg.train.moe_z_loss_weight = 1.0
        model = HAGI(cfg)
        model.train()
        out = model(*batch())
        assert out.router_z_loss is not None
        assert float(out.router_z_loss.detach()) > 0

    def test_no_router_z_loss_at_eval(self):
        cfg = tiny_config(
            **{"model.moe.enabled": True, "model.moe.num_experts": 4, "model.moe.moe_every": 1}
        )
        model = HAGI(cfg).eval()
        with torch.no_grad():
            out = model(*batch())
        assert out.router_z_loss is None
