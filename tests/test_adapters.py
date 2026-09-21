"""Characterization tests for the opt-in HAGI residual adapters.

These tests pin down the adapter contract documented in
``src/hagi/model/adapters.py`` and ``.pi/agent/notes/2026-09-01-e1-pyramid-contract.md``:

* **Default-path invariant** — ``adapters.enabled=False`` (the default) creates
  no adapter modules, adds no parameters, and the forward is bit-for-bit the
  original model output (no extra computation, no dtype/device drift).
* **Zero-init enabled is a no-op** — an enabled-but-untrained adapter (pyramid
  ``scale=0``, LoRA ``lora_B=0``) produces exactly zero delta.
* **Pyramid topology** — ``levels=(1,)`` is the flat identity gate; ``(1,2,4)``
  runs 3 levels (1+2+4 branches), each mean-reduced, summed into the delta.
* **LoRA layout** — ``A:[H,r]`` frozen buffer (orthonormal), ``B:[H,r]``
  zero-init trainable, ``delta = scaling * (x @ A) @ B.T`` (PEFT layout).
* **Optimizer routing** — adapter gains/LoRA parameters are 1D and must land in
  AdamW, never Muon (no ``is_channel_weight`` marker).
* **Checkpoint round-trip** — adapter state survives save/load when enabled.
* **KV-cache / generation contract** — enabled adapters do not corrupt KV-cache
  incremental decode equality (the adapter delta depends only on `mixer_input`).
"""

from __future__ import annotations

import pytest
import torch

from hagi.config import Config
from hagi.model.adapters import (
    PyramidAdapter,
    TttLoraAdapter,
    _qr_orthonormal,
)
from hagi.model.model import HAGI
from hagi.train.optim import build_optimizer
from tests.conftest import tiny_config

TINY_HIDDEN = 64


def _adapter_cfg(
    cfg: Config | None = None,
    *,
    pyramid: bool = False,
    ttt: bool = False,
    pyramid_levels=(1,),
    ttt_rank: int = 4,
) -> Config:
    cfg = cfg or tiny_config()
    cfg.model.adapters.enabled = True
    cfg.model.adapters.pyramid.enabled = pyramid
    cfg.model.adapters.pyramid.levels = pyramid_levels
    cfg.model.adapters.ttt_lora.enabled = ttt
    cfg.model.adapters.ttt_lora.rank = ttt_rank
    from hagi.config import validate_config

    validate_config(cfg)
    return cfg


class TestDefaultPathInvariant:
    def test_disabled_creates_no_adapter_modules(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        assert all(b.adapters is None for b in model.blocks)
        assert sum(p.numel() for p in model.parameters()) == model.param_summary()["total"]

    def test_disabled_forward_matches_bit_for_bit(self):
        cfg = tiny_config()
        model = HAGI(cfg).eval()
        ids = torch.randint(0, cfg.model.vocab_size, (2, 16))
        first = model(ids, return_logits=True)
        second = model(ids, return_logits=True)
        assert torch.equal(first.hidden, second.hidden)
        assert torch.equal(first.logits, second.logits)

    def test_disabled_no_extra_parameters(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        assert model.param_summary()["adapter"] == 0


class TestZeroInitEnabledIsNoop:
    def test_pyramid_scale_zero_is_zero_delta(self):
        cfg = tiny_config()
        cfg.model.adapters.enabled = True
        cfg.model.adapters.pyramid.enabled = True
        from hagi.config import validate_config

        validate_config(cfg)
        model = HAGI(cfg).eval()
        ids = torch.randint(0, cfg.model.vocab_size, (2, 16))
        with torch.no_grad():
            model(ids, return_logits=True)
        # Every block's adapter scale must be exactly zero.
        for block in model.blocks:
            assert block.adapters is not None
            assert float(block.adapters.pyramid.scale.detach()) == 0.0

    def test_lora_B_zero_is_zero_delta(self):
        cfg = tiny_config()
        cfg.model.adapters.enabled = True
        cfg.model.adapters.ttt_lora.enabled = True
        cfg.model.adapters.ttt_lora.rank = 4
        from hagi.config import validate_config

        validate_config(cfg)
        model = HAGI(cfg).eval()
        for block in model.blocks:
            assert block.adapters is not None
            assert float(block.adapters.ttt_lora.lora_B.detach().abs().sum()) == 0.0

    def test_enabled_untrained_delta_is_exactly_zero(self):
        cfg = _adapter_cfg(cfg=tiny_config(), pyramid=False, ttt=True, ttt_rank=4)
        model = HAGI(cfg).eval()

        # Build a plain (disabled) reference with the SAME base weights.
        ref_cfg = tiny_config()
        model_ref = HAGI(ref_cfg).eval()
        model_ref.load_state_dict(model.state_dict(), strict=False)

        ids = torch.randint(0, cfg.model.vocab_size, (2, 16))
        with torch.no_grad():
            enabled = model(ids, return_logits=True)
            disabled = model_ref(ids, return_logits=True)
        assert torch.equal(enabled.hidden, disabled.hidden)


class TestPyramidAdapter:
    def test_single_level_one_branch_is_identity_gate_shape(self):
        cfg = _adapter_cfg(pyramid=True, pyramid_levels=(1,), ttt=False)
        adapter = PyramidAdapter(cfg.model.adapters.pyramid)
        assert adapter.n_branches == 1
        assert adapter.levels == (1,)

    def test_multi_level_branch_count(self):
        cfg = _adapter_cfg(pyramid=True, pyramid_levels=(1, 2, 4), ttt=False)
        adapter = PyramidAdapter(cfg.model.adapters.pyramid)
        assert adapter.n_branches == 7
        assert adapter.levels == (1, 2, 4)

    def test_scale_is_not_a_channel_weight(self):
        adapter = PyramidAdapter(tiny_config().model.adapters.pyramid)
        assert getattr(adapter.scale, "is_channel_weight", False) is False

    def test_forward_zero_delta_when_scale_zero(self):
        adapter = PyramidAdapter(tiny_config().model.adapters.pyramid)
        mixer = torch.nn.Linear(8, 8, bias=False)
        adapter.attach(mixer)
        x = torch.randn(2, 4, 8)
        out = adapter(x, x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_forward_nonzero_when_scale_nonzero(self):
        cfg = tiny_config()
        cfg.model.adapters.pyramid.enabled = True
        cfg.model.adapters.pyramid.levels = (1,)
        adapter = PyramidAdapter(cfg.model.adapters.pyramid)
        mixer = torch.nn.Linear(8, 8, bias=False)
        adapter.attach(mixer)
        with torch.no_grad():
            adapter.scale.fill_(1.0)
        x = torch.randn(2, 4, 8)
        out = adapter(x, x)
        expected = mixer(x)
        assert torch.allclose(out, expected)


class TestTttLoraAdapter:
    def test_shapes(self):
        adapter = TttLoraAdapter(TINY_HIDDEN, tiny_config().model.adapters.ttt_lora, init_seed=0)
        assert adapter.lora_A.shape == (TINY_HIDDEN, tiny_config().model.adapters.ttt_lora.rank)
        assert adapter.lora_B.shape == (TINY_HIDDEN, tiny_config().model.adapters.ttt_lora.rank)

    def test_A_is_orthonormal(self):
        r = tiny_config().model.adapters.ttt_lora.rank
        adapter = TttLoraAdapter(TINY_HIDDEN, tiny_config().model.adapters.ttt_lora, init_seed=0)
        a = adapter.lora_A
        assert torch.allclose(a.T @ a, torch.eye(r), atol=1e-5)

    def test_B_is_zero_init(self):
        adapter = TttLoraAdapter(TINY_HIDDEN, tiny_config().model.adapters.ttt_lora)
        assert float(adapter.lora_B.detach().abs().sum()) == 0.0

    def test_B_is_not_a_channel_weight(self):
        adapter = TttLoraAdapter(TINY_HIDDEN, tiny_config().model.adapters.ttt_lora)
        assert getattr(adapter.lora_B, "is_channel_weight", False) is False

    def test_scaling_is_alpha_over_r(self):
        rank = 4
        cfg = tiny_config()
        cfg.model.adapters.ttt_lora.rank = rank
        cfg.model.adapters.ttt_lora.alpha = 3.0
        adapter = TttLoraAdapter(TINY_HIDDEN, cfg.model.adapters.ttt_lora)
        assert adapter.scaling == pytest.approx(3.0 / rank)

    def test_delta_shape(self):
        rank = 4
        cfg = tiny_config()
        cfg.model.adapters.ttt_lora.rank = rank
        adapter = TttLoraAdapter(TINY_HIDDEN, cfg.model.adapters.ttt_lora)
        x = torch.randn(3, 7, TINY_HIDDEN)
        delta = adapter(x, x)
        assert delta.shape == (3, 7, TINY_HIDDEN)

    def test_frozen_A_is_not_a_parameter(self):
        adapter = TttLoraAdapter(TINY_HIDDEN, tiny_config().model.adapters.ttt_lora)
        params = dict(adapter.named_parameters())
        assert "lora_A" not in params
        assert "lora_B" in params
        buffers = dict(adapter.named_buffers())
        assert "lora_A" in buffers

    def test_qr_orthonormal_shapes(self):
        tall = _qr_orthonormal(16, 4)
        assert tall.shape == (16, 4)
        assert torch.allclose(tall.T @ tall, torch.eye(4), atol=1e-5)
        wide = _qr_orthonormal(4, 16)
        assert wide.shape == (4, 16)
        assert torch.allclose(wide @ wide.T, torch.eye(4), atol=1e-5)


class TestBlockAdapterValidation:
    def test_no_contour_when_enabled_raises(self):
        cfg = tiny_config()
        cfg.model.adapters.enabled = True
        from hagi.config import validate_config

        with pytest.raises(ValueError, match="no contour"):
            validate_config(cfg)

    def test_master_off_with_contour_raises(self):
        cfg = tiny_config()
        cfg.model.adapters.pyramid.enabled = True
        from hagi.config import validate_config

        with pytest.raises(ValueError, match="must be True"):
            validate_config(cfg)

    def test_positive_levels_required(self):
        cfg = tiny_config()
        cfg.model.adapters.enabled = True
        cfg.model.adapters.pyramid.enabled = True
        cfg.model.adapters.pyramid.levels = (0, 1)
        from hagi.config import validate_config

        with pytest.raises(ValueError, match="positive integers"):
            validate_config(cfg)

    def test_rank_exceeds_hidden_rejected(self):
        cfg = tiny_config()
        cfg.model.adapters.enabled = True
        cfg.model.adapters.ttt_lora.enabled = True
        cfg.model.adapters.ttt_lora.rank = 2 * cfg.model.hidden_size
        from hagi.config import validate_config

        with pytest.raises(ValueError, match="rank"):
            validate_config(cfg)

    def test_simultaneous_pyramid_and_ttt_rejected(self):
        """Pyramid and TTT-LoRA are mutually exclusive by current policy."""
        cfg = tiny_config()
        cfg.model.adapters.enabled = True
        cfg.model.adapters.pyramid.enabled = True
        cfg.model.adapters.ttt_lora.enabled = True
        from hagi.config import validate_config

        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_config(cfg)


class TestOptimizerRouting:
    def test_pyramid_scale_in_adamw_not_muon(self):
        cfg = _adapter_cfg(cfg=tiny_config(), pyramid=True, pyramid_levels=(1,), ttt=False)
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        # Collect every trainable parameter id that the optimizer sees.
        muon_ids = set()
        adamw_ids = set()
        for group in opt.adamw.param_groups:
            for p in group["params"]:
                adamw_ids.add(id(p))
        if opt.muon is not None:
            for group in opt.muon.param_groups:
                for p in group["params"]:
                    muon_ids.add(id(p))
        # Pyramid scale must be in AdamW, never Muon.
        for block in model.blocks:
            if block.adapters is None:
                continue
            if block.adapters.pyramid is not None:
                scale_id = id(block.adapters.pyramid.scale)
                assert scale_id in adamw_ids, "pyramid scale must be AdamW"
                assert scale_id not in muon_ids, "pyramid scale must not be Muon"

    def test_lora_B_in_adamw_not_muon(self):
        cfg = _adapter_cfg(cfg=tiny_config(), pyramid=False, ttt=True, ttt_rank=4)
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        muon_ids = set()
        adamw_ids = set()
        for group in opt.adamw.param_groups:
            for p in group["params"]:
                adamw_ids.add(id(p))
        if opt.muon is not None:
            for group in opt.muon.param_groups:
                for p in group["params"]:
                    muon_ids.add(id(p))
        # TTT-LoRA lora_B must be in AdamW, never Muon.
        for block in model.blocks:
            if block.adapters is None:
                continue
            if block.adapters.ttt_lora is not None:
                b_id = id(block.adapters.ttt_lora.lora_B)
                assert b_id in adamw_ids, "lora_B must be AdamW"
                assert b_id not in muon_ids, "lora_B must not be Muon"

    def test_optimizer_is_bijection_pyramid(self):
        cfg = _adapter_cfg(cfg=tiny_config(), pyramid=True, pyramid_levels=(2,), ttt=False)
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        assigned = sum(len(g["params"]) for g in opt.adamw.param_groups)
        if opt.muon is not None:
            assigned += sum(len(g["params"]) for g in opt.muon.param_groups)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert assigned == trainable


class TestParamCount:
    def test_analytic_matches_real_pyramid(self):
        cfg = _adapter_cfg(cfg=tiny_config(), pyramid=True, pyramid_levels=(1, 2, 4), ttt=False)
        model = HAGI(cfg)
        counts = model.param_summary()
        real = sum(p.numel() for p in model.parameters())
        assert counts["total"] == real
        # Pyramid adds one scalar scale per block; no LoRA params expected.
        expected_adapter = cfg.model.num_layers * 1
        assert counts["adapter"] == expected_adapter

    def test_analytic_matches_real_ttt(self):
        cfg = _adapter_cfg(cfg=tiny_config(), pyramid=False, ttt=True, ttt_rank=4)
        model = HAGI(cfg)
        counts = model.param_summary()
        real = sum(p.numel() for p in model.parameters())
        assert counts["total"] == real
        # TTT-LoRA adds rank*hidden_size per block; no pyramid scale expected.
        expected_adapter = cfg.model.num_layers * (4 * cfg.model.hidden_size)
        assert counts["adapter"] == expected_adapter


class TestCheckpointRoundTrip:
    def test_pyramid_adapter_state_survives(self, tmp_path):
        from hagi.train.checkpoint import load_model, save_checkpoint

        cfg = _adapter_cfg(cfg=tiny_config(), pyramid=True, pyramid_levels=(1,), ttt=False)
        model = HAGI(cfg)
        path = save_checkpoint(model, cfg, 42, tmp_path)

        fresh = HAGI(_adapter_cfg(cfg=tiny_config(), pyramid=True, pyramid_levels=(1,), ttt=False))
        step, loaded = load_model(path, fresh)
        assert step == 42
        for (n1, a), (n2, b) in zip(
            model.named_parameters(), fresh.named_parameters(), strict=True
        ):
            assert torch.equal(a.detach(), b.detach()), f"{n1} differs after reload"

    def test_ttt_adapter_state_survives(self, tmp_path):
        from hagi.train.checkpoint import load_model, save_checkpoint

        cfg = _adapter_cfg(cfg=tiny_config(), pyramid=False, ttt=True, ttt_rank=4)
        model = HAGI(cfg)
        path = save_checkpoint(model, cfg, 42, tmp_path)

        fresh = HAGI(_adapter_cfg(cfg=tiny_config(), pyramid=False, ttt=True, ttt_rank=4))
        step, loaded = load_model(path, fresh)
        assert step == 42
        for (n1, a), (n2, b) in zip(
            model.named_parameters(), fresh.named_parameters(), strict=True
        ):
            assert torch.equal(a.detach(), b.detach()), f"{n1} differs after reload"

    def test_disabled_default_is_checkpoint_compatible(self, tmp_path):
        """A disabled-adapter checkpoint must load into a disabled model."""
        from hagi.train.checkpoint import load_model, save_checkpoint

        cfg = tiny_config()
        model = HAGI(cfg)
        path = save_checkpoint(model, cfg, 1, tmp_path)
        fresh = HAGI(tiny_config())
        step, _ = load_model(path, fresh)
        assert step == 1


class TestGenerationContract:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"model.sliding.window": 0},
        ],
        ids=["full"],
    )
    @pytest.mark.parametrize(
        "adapter_kind", ["pyramid", "ttt_lora"], ids=["pyramid", "ttt_lora"]
    )
    def test_incremental_decode_matches_full_with_adapter(self, overrides, adapter_kind):
        """The adapter delta depends only on `mixer_input`; KV-cache decode must match."""
        if adapter_kind == "pyramid":
            cfg = _adapter_cfg(
                cfg=tiny_config(**overrides), pyramid=True, pyramid_levels=(1,), ttt=False
            )
        else:
            cfg = _adapter_cfg(
                cfg=tiny_config(**overrides), pyramid=False, ttt=True, ttt_rank=4
            )
        model = HAGI(cfg).eval()
        ids = torch.randint(0, cfg.model.vocab_size, (1, 10))

        with torch.no_grad():
            full = model(ids, return_logits=True).logits
            model.reset_cache()
            model.allocate_cache(torch.float32, torch.device("cpu"))
            steps = []
            for t in range(10):
                steps.append(
                    model(
                        ids[:, t : t + 1],
                        positions=torch.arange(t, t + 1),
                        use_cache=True,
                        return_logits=True,
                    ).logits
                )
            model.reset_cache()
        incremental = torch.cat(steps, dim=1)
        assert float((full - incremental).abs().max()) < 1e-4
        assert float((full.argmax(-1) == incremental.argmax(-1)).float().mean()) == 1.0


class TestMergedHAGIAdapterAttachment:
    def test_merge_re_attaches_adapters(self):
        from hagi.model.merge import MergedHAGI

        cfg = tiny_config()
        cfg.model.hidden_size = 64
        cfg.model.num_layers = 2
        cfg.model.attention.num_query_heads = 2
        cfg.model.attention.num_kv_heads = 1
        cfg.model.attention.head_dim = 32
        cfg.model.ffn.intermediate_size = 64
        cfg.model.adapters.enabled = True
        cfg.model.adapters.pyramid.enabled = True
        cfg.model.adapters.pyramid.levels = (1,)
        from hagi.config import validate_config

        validate_config(cfg)
        cfg.merge.enabled = True
        cfg.merge.n_experts = 2
        merged = MergedHAGI(cfg)
        # Every final merged block must have its adapter attached.
        assert all(b.adapters is not None for b in merged.blocks)
        assert merged.blocks[0].adapters.pyramid.n_branches == 1
