"""Training loop: microbatch weighting, non-finite skip, fp32 gain preservation.

Two properties that are invisible when wrong:

* Token-count weighting. Averaging microbatches uniformly over-weights sparse
  ones, which silently changes the objective by a few percent — enough to matter
  over 200k steps and impossible to notice in a log.
* fp32 normalization gains under bf16. A gain at 1.0 receives gradients around
  1e-4; the smallest bf16 step above 1.0 is ~0.0078, so in bf16 every
  normalization layer in the model is frozen at initialization for the whole run
  and the loss curve merely looks slightly worse than it should.
"""

from __future__ import annotations

import math

import pytest
import torch

from hagi.model.model import HAGI
from hagi.model.norms import HeadNorm, RMSNorm
from hagi.train.loop import Trainer, cast_model, clip_gradients, format_metrics, puncture_loss_mask
from tests.conftest import TINY_VOCAB, tiny_config


def microbatch(b=2, t=8, mask=None):
    item = {
        "input_ids": torch.randint(0, TINY_VOCAB, (b, t)),
        "targets": torch.randint(0, TINY_VOCAB, (b, t)),
    }
    if mask is not None:
        item["loss_mask"] = mask
    return item


class TestCastModel:
    def test_bf16_body_with_fp32_gains(self):
        model = HAGI(tiny_config())
        cast_model(model, "bf16")
        assert model.blocks[0].attn.qkv_proj.weight.dtype == torch.bfloat16
        for module in model.modules():
            if isinstance(module, (RMSNorm, HeadNorm)):
                assert module.weight.dtype == torch.float32, "a gain was left in bf16"

    def test_receiver_gain_stays_fp32(self):
        model = HAGI(tiny_config())
        cast_model(model, "bf16")
        assert model.head.logit_scale.dtype == torch.float32

    def test_fp32_is_a_noop(self):
        model = HAGI(tiny_config())
        cast_model(model, "fp32")
        assert model.blocks[0].attn.qkv_proj.weight.dtype == torch.float32

    def test_bf16_forward_still_runs(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        cast_model(model, "bf16")
        out = model(*[microbatch()[k] for k in ("input_ids", "targets")])
        assert math.isfinite(float(out.loss.detach()))


class TestClipGradients:
    def test_returns_the_preclip_norm(self):
        param = torch.nn.Parameter(torch.zeros(4))
        param.grad = torch.full((4,), 3.0)  # norm 6.0
        model = torch.nn.Module()
        model.p = param
        assert clip_gradients(model, 1.0) == pytest.approx(6.0)
        assert float(param.grad.norm()) == pytest.approx(1.0, abs=1e-5)

    def test_below_threshold_is_untouched(self):
        param = torch.nn.Parameter(torch.zeros(4))
        param.grad = torch.full((4,), 0.1)
        model = torch.nn.Module()
        model.p = param
        before = param.grad.clone()
        clip_gradients(model, 10.0)
        assert torch.allclose(param.grad, before)


class TestPunctureLossMask:
    def test_full_rate_is_noop(self):
        assert puncture_loss_mask((2, 8), rate=1.0, mode="bernoulli", step=0, device=torch.device("cpu")) is None

    def test_bernoulli_rate(self):
        torch.manual_seed(0)
        m = puncture_loss_mask((4, 64), rate=0.5, mode="bernoulli", step=0, device=torch.device("cpu"))
        assert m is not None and m.dtype == torch.bool and m.shape == (4, 64)
        frac = float(m.float().mean())
        assert 0.3 < frac < 0.7

    def test_stride_phase_rotates(self):
        a = puncture_loss_mask((1, 12), rate=0.5, mode="stride", step=0, device=torch.device("cpu"))
        b = puncture_loss_mask((1, 12), rate=0.5, mode="stride", step=1, device=torch.device("cpu"))
        assert a is not None and b is not None
        assert int(a.sum()) == 6 and int(b.sum()) == 6
        assert not torch.equal(a, b)
        # phase 0 keeps even indices for k=2
        assert bool(a[0, 0].item()) and not bool(a[0, 1].item())

    def test_and_with_base_mask(self):
        base = torch.zeros(1, 8, dtype=torch.bool)
        base[0, :4] = True
        m = puncture_loss_mask(
            (1, 8), rate=1.0, mode="bernoulli", step=0, device=torch.device("cpu"), base=base
        )
        assert torch.equal(m, base)


class TestTrainStep:
    def test_metrics_shape(self):
        cfg = tiny_config()
        trainer = Trainer(HAGI(cfg), cfg)
        metrics = trainer.train_step([microbatch(), microbatch()])
        assert metrics["update_applied"] is True
        assert metrics["step"] == 0
        assert trainer.step == 1
        for key in ("loss", "ce", "bpt", "ppl", "grad_norm", "lr", "muon_lr", "tokens"):
            assert key in metrics
        assert metrics["tokens"] == 32
        assert metrics["bpt"] == pytest.approx(metrics["ce"] / math.log(2.0))

    def test_ce_keep_rate_thins_scored_tokens(self):
        cfg = tiny_config(**{"train.ce_keep_rate": 0.5, "train.ce_keep_mode": "stride"})
        trainer = Trainer(HAGI(cfg), cfg)
        # T=8, stride k=2, phase 0 → 4 scored per microbatch of 2×8 = 8 tokens
        metrics = trainer.train_step([microbatch()])
        assert metrics["tokens"] == 8
        assert metrics["ce_keep_rate"] == 0.5
        assert math.isfinite(metrics["ce"])

    def test_empty_microbatch_list_raises(self):
        cfg = tiny_config()
        trainer = Trainer(HAGI(cfg), cfg)
        with pytest.raises(ValueError):
            trainer.train_step([])

    def test_parameters_move(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        trainer = Trainer(model, cfg)
        trainer.step = cfg.train.schedule.warmup_steps  # past warmup so lr > 0
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        trainer.train_step([microbatch()])
        moved = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])]
        assert len(moved) > 10

    def test_loss_mask_is_reflected_in_the_token_count(self):
        cfg = tiny_config()
        trainer = Trainer(HAGI(cfg), cfg)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, :3] = True
        metrics = trainer.train_step([microbatch(mask=mask)])
        assert metrics["tokens"] == 6

    def test_unequal_microbatches_are_token_weighted(self):
        """The scored-token count, not the microbatch count, sets the weights."""
        cfg = tiny_config()
        trainer = Trainer(HAGI(cfg), cfg)
        dense = torch.ones(2, 8, dtype=torch.bool)
        sparse = torch.zeros(2, 8, dtype=torch.bool)
        sparse[:, 0] = True
        metrics = trainer.train_step([microbatch(mask=dense), microbatch(mask=sparse)])
        assert metrics["tokens"] == 16 + 2

    def test_nonfinite_gradient_skips_the_update(self, monkeypatch):
        cfg = tiny_config()
        model = HAGI(cfg)
        trainer = Trainer(model, cfg)
        monkeypatch.setattr(
            "hagi.train.loop.clip_gradients_by_group", lambda *_: (float("nan"), 0.1)
        )
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        metrics = trainer.train_step([microbatch()])
        assert metrics["update_applied"] is False
        for name, p in model.named_parameters():
            assert torch.equal(p.detach(), before[name]), f"{name} moved on a skipped step"

    def test_skip_does_not_advance_the_step_counter(self, monkeypatch):
        cfg = tiny_config()
        trainer = Trainer(HAGI(cfg), cfg)
        monkeypatch.setattr(
            "hagi.train.loop.clip_gradients_by_group", lambda *_: (float("inf"), 0.1)
        )
        trainer.train_step([microbatch()])
        assert trainer.step == 0

    def test_diagnostics_appear_on_the_interval(self):
        cfg = tiny_config()
        cfg.train.logging.diag_interval = 1
        trainer = Trainer(HAGI(cfg), cfg)
        metrics = trainer.train_step([microbatch()])
        assert "qk_gain" in metrics and "logit_scale" in metrics

    def test_exact_ce_diagnostic_is_periodic_and_full_alphabet(self):
        cfg = tiny_config(
            **{
                "model.head.sampled_softmax_k": 8,
                "train.ce_keep_rate": 0.5,
                "train.ce_keep_mode": "stride",
                "train.logging.exact_ce_interval": 2,
                "train.logging.exact_ce_rows": 4,
            }
        )
        trainer = Trainer(HAGI(cfg), cfg)
        first = trainer.train_step([microbatch()])
        second = trainer.train_step([microbatch()])
        assert math.isfinite(first["exact_ce"])
        assert first["exact_bpt"] == pytest.approx(first["exact_ce"] / math.log(2.0))
        assert first["exact_ppl"] == pytest.approx(math.exp(first["exact_ce"]))
        assert first["nce"] == pytest.approx(first["ce"])
        assert first["nce_bits"] == pytest.approx(first["bpt"])
        assert "exact_ce" not in second

    def test_checkpointing_path_produces_the_same_token_count(self):
        cfg = tiny_config(**{"train.grad_checkpointing": True})
        trainer = Trainer(HAGI(cfg), cfg)
        metrics = trainer.train_step([microbatch()])
        assert metrics["tokens"] == 16
        assert metrics["update_applied"] is True


class TestFormatMetrics:
    def test_reports_ce_first(self):
        line = format_metrics(
            {
                "step": 7,
                "ce": 4.5,
                "bpt": 6.49,
                "ppl": 90.0,
                "grad_norm": 0.8,
                "lr": 3e-4,
                "update_applied": True,
            }
        )
        assert line.startswith("step 7 | ce=4.5000")

    def test_reports_conditional_nce_honestly(self):
        line = format_metrics(
            {
                "step": 8,
                "ce": 4.2,
                "bpt": 6.1,
                "ppl": 66.0,
                "grad_norm": 0.7,
                "lr": 2e-4,
                "receiver": "conditional_nce",
                "update_applied": True,
            }
        )
        assert line.startswith("step 8 | nce=4.2000")
        assert "ppl=" in line

    def test_skipped_step_is_marked(self):
        line = format_metrics({"step": 3, "grad_norm": float("nan"), "update_applied": False})
        assert "skipped" in line

    def test_diagnostics_are_appended(self):
        line = format_metrics(
            {
                "step": 1,
                "ce": 4.0,
                "bpt": 5.8,
                "ppl": 55.0,
                "grad_norm": 0.5,
                "lr": 1e-4,
                "update_applied": True,
                "qk_gain": 1.02,
                "logit_scale": 0.03,
                "exact_ce": 7.2,
            }
        )
        assert "qk_gain=1.020" in line
        assert "logit_scale=0.030" in line
        assert "exact_ce=7.2000" in line
        assert "grad=" not in line
        assert "gb=" not in line
        assert "ce_keep_rate=" not in line
        assert "kl=" in line
