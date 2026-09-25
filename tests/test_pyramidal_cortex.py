"""Characterization and research contracts for the pyramidal cortex MVP.

The cortex is a directed, same-token side channel between contiguous layer
levels.  It must be inert at initialization, acyclic by construction, and
compatible with both gradient checkpointing and incremental KV decode.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from hagi.config import Config, PyramidalCortexConfig, validate_config
from hagi.model.adaptive import AdaptiveComponent, adaptive_parameter_names
from hagi.model.cortex import PyramidalCortex, level_boundaries
from hagi.model.model import HAGI
from hagi.train.checkpoint import IncompatibleCheckpointError, load_model, save_checkpoint
from hagi.train.loop import Trainer, cast_model
from hagi.train.self_improve import _restore_adapters, _snapshot_adapters
from scripts.pyramidal_cortex_ab import load_exact_state, quality_supported
from scripts.ternary_precision_ab import (
    bitlinear_dtype_histogram,
    dtype_histogram,
)
from scripts.ternary_precision_ab import (
    make_config as make_precision_config,
)
from tests.conftest import TINY_VOCAB, assert_finite, tiny_config


def _cortex_config(**overrides) -> Config:
    values = {
        "model.num_layers": 4,
        "model.cortex.enabled": True,
        "model.cortex.num_levels": 2,
        "model.cortex.rank": 4,
        "model.cortex.link_strides": (1,),
        "model.cortex.residual_scale": 0.1,
    }
    values.update(overrides)
    return tiny_config(**values)


def _ids(cfg: Config) -> torch.Tensor:
    return torch.randint(0, cfg.model.vocab_size, (2, 12))


class TestAblationRunnerContracts:
    def test_precision_runner_lanes_have_explicit_dtype_contracts(self):
        for precision, fp32_master in (("fp32", False), ("bf16", False), ("bf16", True)):
            cfg = make_precision_config(
                seed=7,
                steps=1,
                precision=precision,
                ternary_fp32_master=fp32_master,
            )
            assert cfg.train.ternary_fp32_master is fp32_master
            model = HAGI(cfg)
            cast_model(model, precision, ternary_fp32_master=fp32_master)
            histograms = dtype_histogram(model), bitlinear_dtype_histogram(model)
            expected_master_dtype = "float32" if precision == "fp32" or fp32_master else "bfloat16"
            assert set(histograms[1]) == {expected_master_dtype}
            assert histograms[1][expected_master_dtype] > 0
            assert sum(histograms[0].values()) > 0

    def test_quality_verdict_requires_real_corpus_and_seed_majority(self):
        assert not quality_supported(
            execution_completed=True,
            num_seeds=1,
            cortex_only_deltas=[-1.0, -1.0, -1.0],
            cortex_lora_deltas=[-1.0, -1.0, -1.0],
        )
        assert quality_supported(
            execution_completed=True,
            num_seeds=3,
            cortex_only_deltas=[-1.0, -1.0, 1.0],
            cortex_lora_deltas=[-1.0, 1.0, -1.0],
            real_corpus=True,
        )
        assert not quality_supported(
            execution_completed=True,
            num_seeds=3,
            cortex_only_deltas=[-1.0, -1.0, 1.0],
            cortex_lora_deltas=[-1.0, 1.0, -1.0],
            real_corpus=False,
        )
        assert not quality_supported(
            execution_completed=True,
            num_seeds=3,
            cortex_only_deltas=[-1.0, 1.0, 1.0],
            cortex_lora_deltas=[-1.0, -1.0, -1.0],
            real_corpus=True,
        )
        assert not quality_supported(
            execution_completed=True,
            num_seeds=3,
            cortex_only_deltas=[float("nan"), -1.0, -1.0],
            cortex_lora_deltas=[-1.0, -1.0, -1.0],
            real_corpus=True,
        )
        assert not quality_supported(
            execution_completed=True,
            num_seeds=3,
            cortex_only_deltas=[-1.0, -1.0],
            cortex_lora_deltas=[-1.0, -1.0, -1.0],
            real_corpus=True,
        )

    def test_state_loader_rejects_mismatch_before_mutation(self):
        model = HAGI(tiny_config())
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        state = dict(before)
        key = next(iter(state))
        state[key] = state[key][:-1]
        with pytest.raises(ValueError, match="state mismatch|shape mismatch"):
            load_exact_state(model, state)
        assert all(
            torch.equal(model.state_dict()[name], tensor)
            for name, tensor in before.items()
        )


class TestGeometry:
    def test_contiguous_level_boundaries(self):
        assert level_boundaries(24, 4) == (5, 11, 17, 23)
        assert level_boundaries(6, 3) == (1, 3, 5)

    def test_layer_to_level_covers_every_layer_exactly_once(self):
        cfg = PyramidalCortexConfig(
            enabled=True,
            num_levels=3,
            rank=4,
            link_strides=(1, 2),
            residual_scale=1.0,
        )
        cortex = PyramidalCortex(4, 4, cfg)
        assert cortex.layer_to_level == (0, 0, 1, 2)
        assert [cortex.level_for_layer(i) for i in range(4)] == [0, 0, 1, 2]

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"model.cortex.num_levels": 1}, "at least 2"),
            ({"model.cortex.num_levels": 5}, "num_layers"),
            ({"model.cortex.rank": 0}, "rank"),
            ({"model.cortex.rank": 129}, "hidden_size"),
            ({"model.cortex.link_strides": ()}, "link_strides"),
            ({"model.cortex.link_strides": (0,)}, "positive"),
            ({"model.cortex.link_strides": (1, 1)}, "unique"),
            ({"model.cortex.link_strides": (2, 1)}, "sorted"),
            ({"model.cortex.residual_scale": 0.0}, "positive"),
            ({"model.cortex.residual_scale": float("nan")}, "finite"),
        ],
    )
    def test_invalid_config_is_rejected(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            _cortex_config(**overrides)

    def test_ttt_lora_and_cortex_can_coexist(self):
        cfg = _cortex_config(
            **{
                "model.adapters.enabled": True,
                "model.adapters.ttt_lora.enabled": True,
                "model.adapters.ttt_lora.rank": 4,
            }
        )
        model = HAGI(cfg)
        assert isinstance(model.cortex, PyramidalCortex)
        assert all(block.adapters is not None for block in model.blocks)
        assert all(block.adapters.ttt_lora is not None for block in model.blocks)


class TestInitialization:
    def test_disabled_path_has_no_cortex(self):
        model = HAGI(tiny_config())
        assert model.cortex is None
        assert model.param_summary()["cortex"] == 0

    def test_zero_links_reproduce_plain_model_bitwise(self):
        cfg = _cortex_config()
        cortex_model = HAGI(cfg).eval()
        reference = HAGI(tiny_config()).eval()
        incompatible = reference.load_state_dict(cortex_model.state_dict(), strict=False)
        assert incompatible.unexpected_keys
        assert not incompatible.missing_keys

        ids = _ids(cfg)
        with torch.no_grad():
            expected = reference(ids, return_logits=True)
            actual = cortex_model(ids, return_logits=True)
        assert torch.equal(expected.hidden, actual.hidden)
        assert torch.equal(expected.logits, actual.logits)

    def test_links_start_zero_and_need_two_steps_to_wake_projections(self):
        cfg = _cortex_config()
        model = HAGI(cfg).train()
        assert model.cortex is not None
        for link in model.cortex.links:
            assert float(link.weight.detach().abs().sum()) == 0.0

        ids, targets = _ids(cfg), torch.randint(0, TINY_VOCAB, (2, 12))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        model(ids, targets).loss.backward()
        assert all(link.weight.grad is not None for link in model.cortex.links)
        # With zero links the feature projections are intentionally silent on
        # step 1; otherwise zero-initialization would look like a dead module.
        assert all(
            projection.weight.grad is None
            or float(projection.weight.grad.detach().abs().sum()) == 0.0
            for projection in model.cortex.down
        )
        assert all(
            projection.weight.grad is None
            or float(projection.weight.grad.detach().abs().sum()) == 0.0
            for projection in model.cortex.up
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        model(ids, targets).loss.backward()
        for name, parameter in model.cortex.named_parameters():
            assert parameter.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"
            assert float(parameter.grad.detach().abs().sum()) > 0.0, f"zero gradient in {name}"

    def test_nonzero_links_change_output(self):
        cfg = _cortex_config()
        model = HAGI(cfg).eval()
        reference = HAGI(tiny_config()).eval()
        reference.load_state_dict(model.state_dict(), strict=False)
        assert model.cortex is not None
        for link in model.cortex.links:
            nn.init.normal_(link.weight, std=0.05)

        ids = _ids(cfg)
        with torch.no_grad():
            expected = reference(ids, return_logits=True).hidden
            actual = model(ids, return_logits=True).hidden
        assert not torch.equal(expected, actual)


class TestDirectionality:
    def test_edges_only_flow_from_earlier_levels(self):
        cfg = PyramidalCortexConfig(
            enabled=True,
            num_levels=3,
            rank=4,
            link_strides=(1, 2),
            residual_scale=1.0,
        )
        cfg.hidden_size = 4
        cortex = PyramidalCortex(4, 4, cfg)
        assert len(cortex.links) == 3  # 0->1, 0->2, 1->2
        identity = torch.eye(4)
        for projection in (*cortex.down, *cortex.up, *cortex.links):
            with torch.no_grad():
                projection.weight.copy_(identity)

        states = cortex.start_sequence()
        x0 = torch.full((1, 2, 4), 1.0)
        x1 = torch.full((1, 2, 4), 2.0)
        x2 = torch.full((1, 2, 4), 3.0)

        y0 = cortex.apply_boundary(x0, 0, states)
        y1 = cortex.apply_boundary(x1, 1, states)
        y2 = cortex.apply_boundary(x2, 2, states)

        assert torch.equal(y0, x0)  # level 0 only publishes a summary
        assert torch.equal(y1, x0 + x1)  # L0 -> L1
        assert torch.equal(y2, x0 + x1 + x2)  # L1 -> L2 plus L0 -> L2
        assert [state is not None for state in states] == [True, True, False]

        # Edge weights are independent: disabling 0->2 removes only that
        # source, not the adjacent 1->2 path.
        cortex._edge_index[(0, 2)]
        with torch.no_grad():
            cortex.links[cortex._edge_index[(0, 2)]].weight.zero_()
        states = cortex.start_sequence()
        cortex.apply_boundary(x0, 0, states)
        cortex.apply_boundary(x1, 1, states)
        y2_without_skip = cortex.apply_boundary(x2, 2, states)
        assert torch.equal(y2_without_skip, x1 + x2)


class TestTrainingAndRuntime:
    def test_adaptive_component_owns_cortex_parameters(self):
        model = HAGI(_cortex_config())
        assert isinstance(model.cortex, AdaptiveComponent)
        adaptive_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, AdaptiveComponent)
            for parameter in module.parameters()
        }
        assert adaptive_ids == {id(parameter) for parameter in model.cortex.parameters()}
        assert adaptive_parameter_names(model) == {
            f"cortex.{name}" for name, _ in model.cortex.named_parameters()
        }

    def test_bf16_cast_keeps_cortex_masters_fp32(self):
        from hagi.train.loop import cast_model

        model = HAGI(_cortex_config())
        cast_model(model, "bf16")
        assert model.blocks[0].attn.qkv_proj.weight.dtype == torch.bfloat16
        assert model.cortex is not None
        assert all(parameter.dtype == torch.float32 for parameter in model.cortex.parameters())
        out = model(_ids(model.cfg), return_logits=True)
        assert_finite(out.logits, "bf16 cortex logits")

    def test_cortex_only_freeze_base_trains_cortex(self):
        cfg = _cortex_config()
        cfg.train.adapt.freeze_base = True
        model = HAGI(cfg)
        Trainer(model, cfg)

        trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        cortex_ids = {id(parameter) for parameter in model.cortex.parameters()}
        assert trainable == cortex_ids

    def test_gradient_checkpointing_keeps_cortex_finite(self):
        cfg = _cortex_config(**{"train.grad_checkpointing": True})
        model = HAGI(cfg).train()
        ids, targets = _ids(cfg), torch.randint(0, TINY_VOCAB, (2, 12))
        loss = model(ids, targets).loss
        loss.backward()
        assert_finite(loss, "checkpointed cortex loss")
        assert model.cortex.links[0].weight.grad is not None

    def test_loop_depth_republishes_level_state_each_pass(self, monkeypatch):
        cfg = _cortex_config(**{"model.loop_depth": 2})
        model = HAGI(cfg).eval()
        assert model.cortex is not None
        calls = 0
        original = model.cortex.down[0].forward

        def counted(x):
            nonlocal calls
            calls += 1
            return original(x)

        monkeypatch.setattr(model.cortex.down[0], "forward", counted)
        model(_ids(cfg), return_logits=True)
        assert calls == 2

    def test_prefill_continuation_decode_matches_full_with_active_links(self):
        cfg = _cortex_config()
        model = HAGI(cfg).eval()
        assert model.cortex is not None
        for link in model.cortex.links:
            nn.init.normal_(link.weight, std=0.05)
        ids = torch.randint(0, TINY_VOCAB, (1, 10))

        with torch.no_grad():
            full = model(ids, return_logits=True).logits
            model.reset_cache()
            model.allocate_cache(torch.float32, torch.device("cpu"))
            prefill = model(ids[:, :4], use_cache=True, return_logits=True).logits
            continuation = [
                model(
                    ids[:, position : position + 1],
                    positions=torch.arange(position, position + 1),
                    use_cache=True,
                    return_logits=True,
                ).logits
                for position in range(4, ids.shape[1])
            ]
            model.reset_cache()
        combined = torch.cat([prefill, *continuation], dim=1)
        assert float((full - combined).abs().max()) < 1e-4
        assert torch.equal(full.argmax(-1), combined.argmax(-1))

    def test_incremental_decode_matches_full_with_active_links(self):
        cfg = _cortex_config()
        model = HAGI(cfg).eval()
        assert model.cortex is not None
        for link in model.cortex.links:
            nn.init.normal_(link.weight, std=0.05)
        ids = torch.randint(0, TINY_VOCAB, (1, 10))

        with torch.no_grad():
            full = model(ids, return_logits=True).logits
            model.reset_cache()
            model.allocate_cache(torch.float32, torch.device("cpu"))
            steps = [
                model(
                    ids[:, position : position + 1],
                    positions=torch.arange(position, position + 1),
                    use_cache=True,
                    return_logits=True,
                ).logits
                for position in range(ids.shape[1])
            ]
            model.reset_cache()
        incremental = torch.cat(steps, dim=1)
        assert float((full - incremental).abs().max()) < 1e-4
        assert torch.equal(full.argmax(-1), incremental.argmax(-1))


class TestParameterAndCheckpointContracts:
    def test_analytic_count_matches_real_module(self):
        cfg = _cortex_config(**{"model.cortex.num_levels": 3, "model.cortex.link_strides": (1, 2)})
        model = HAGI(cfg)
        counts = model.param_summary()
        assert counts["cortex"] == sum(parameter.numel() for parameter in model.cortex.parameters())
        assert counts["total"] == sum(parameter.numel() for parameter in model.parameters())

    def test_enabled_checkpoint_roundtrip(self, tmp_path):
        cfg = _cortex_config()
        model = HAGI(cfg)
        assert model.cortex is not None
        for link in model.cortex.links:
            nn.init.normal_(link.weight, std=0.05)
        path = save_checkpoint(model, cfg, 3, tmp_path)

        fresh = HAGI(_cortex_config())
        load_model(path, fresh)
        ids = _ids(cfg)
        model.eval()
        fresh.eval()
        with torch.no_grad():
            assert torch.equal(model(ids, return_logits=True).logits, fresh(ids, return_logits=True).logits)

    def test_enabled_cortex_checkpoint_rejects_disabled_model_strictly(self, tmp_path):
        cfg = _cortex_config()
        model = HAGI(cfg)
        path = save_checkpoint(model, cfg, 1, tmp_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        del payload["config"]["model"]["cortex"]
        disabled_payload = tmp_path / "disabled-cortex.pt"
        torch.save(payload, disabled_payload)

        with pytest.raises(IncompatibleCheckpointError, match="cortex"):
            load_model(disabled_payload, HAGI(tiny_config()))

    def test_old_config_without_cortex_still_loads(self, tmp_path):
        old_cfg = tiny_config()
        old_model = HAGI(old_cfg)
        path = save_checkpoint(old_model, old_cfg, 1, tmp_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        del payload["config"]["model"]["cortex"]
        migrated = tmp_path / "pre-cortex.pt"
        torch.save(payload, migrated)

        fresh = HAGI(tiny_config())
        step, loaded = load_model(migrated, fresh)
        assert step == 1
        assert not loaded.model.cortex.enabled

    def test_cortex_only_freeze_base_validation_allows_cortex_without_adapters(self):
        cfg = _cortex_config()
        cfg.train.adapt.freeze_base = True
        validate_config(cfg)

    def test_merge_ignores_expert_adaptive_state_and_builds_fresh_mixers(self):
        from hagi.model.merge import merge_experts

        cfg = _cortex_config(
            **{
                "model.num_layers": 2,
                "model.hidden_size": 64,
                "model.attention.num_query_heads": 2,
                "model.attention.num_kv_heads": 2,
                "model.attention.head_dim": 32,
                "model.ffn.intermediate_size": 128,
                "model.embedding.conv_kernel": 1,
                "model.cortex.num_levels": 2,
            }
        )
        cfg.merge.enabled = True
        cfg.merge.n_experts = 2
        cfg.merge.expert_hidden = 32
        cfg.merge.mixer_type = "swiglu"
        validate_config(cfg)
        expert_cfg = tiny_config(
            **{
                "model.num_layers": 2,
                "model.hidden_size": 32,
                "model.attention.num_query_heads": 1,
                "model.attention.num_kv_heads": 1,
                "model.attention.head_dim": 32,
                "model.ffn.intermediate_size": 64,
                "model.embedding.conv_kernel": 1,
            }
        )
        expert_cfg.model.cortex.enabled = True
        expert_cfg.model.cortex.num_levels = 2
        expert_cfg.model.cortex.rank = 4
        expert_cfg.model.cortex.link_strides = (1,)
        validate_config(expert_cfg)
        expert = HAGI(expert_cfg)
        assert expert.cortex is not None
        with torch.no_grad():
            for link in expert.cortex.links:
                link.weight.fill_(0.25)
        state = expert.state_dict()
        merged = merge_experts(cfg, [dict(state), dict(state)], n_mixers=1)
        assert merged.cortex is not None
        assert all(float(link.weight.detach().abs().sum()) == 0.0 for link in merged.cortex.links)
        assert len(merged.mixers) == 1

    def test_cortex_coexists_with_ttt_lora_and_merged_model(self):
        from hagi.model.merge import MergedHAGI

        cfg = _cortex_config(
            **{
                "model.num_layers": 2,
                "model.hidden_size": 64,
                "model.attention.num_query_heads": 2,
                "model.attention.num_kv_heads": 2,
                "model.attention.head_dim": 32,
                "model.ffn.intermediate_size": 128,
                "model.embedding.conv_kernel": 1,
                "model.adapters.enabled": True,
                "model.adapters.ttt_lora.enabled": True,
                "model.adapters.ttt_lora.rank": 4,
                "model.cortex.num_levels": 2,
            }
        )
        cfg.merge.enabled = True
        cfg.merge.n_experts = 2
        cfg.merge.expert_hidden = 32
        cfg.merge.mixer_type = "swiglu"
        validate_config(cfg)
        model = MergedHAGI(cfg)
        assert model.cortex is not None
        assert all(block.adapters is not None for block in model.blocks)
        assert all(block.adapters.ttt_lora is not None for block in model.blocks)
        with torch.no_grad():
            for link in model.cortex.links:
                link.weight.normal_(std=0.05)
        out = model(_ids(cfg), return_logits=True)
        assert_finite(out.logits, "merged cortex logits")

    def test_cortex_only_gradient_self_improvement_is_supported(self):
        from hagi.train.self_improve import self_improve

        cfg = _cortex_config(**{"model.loop_depth": 2})
        cfg.train.adapt.freeze_base = True
        cfg.train.max_steps = 1
        cfg.train.schedule.warmup_steps = 0
        cfg.train.learning_rate = 0.1
        model = HAGI(cfg)

        stats = self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            mode="gradient",
            n_new_tokens=2,
            max_iterations=1,
            kl_max=100.0,
        )
        assert len(stats.iterations) == 1
        assert stats.iterations[0].update_applied is True
        assert any(value != 0.0 for value in stats.iterations[0].adapter_values_after)

    def test_cortex_rollback_restores_all_adaptive_parameters(self):
        cfg = _cortex_config(**{"model.loop_depth": 2})
        cfg.train.adapt.freeze_base = True
        model = HAGI(cfg)
        Trainer(model, cfg)
        before = _snapshot_adapters(model)
        assert before
        with torch.no_grad():
            for parameter in model.cortex.parameters():
                parameter.add_(1.0)
        _restore_adapters(model, before)
        assert all(
            torch.equal(parameter.detach(), before[id(parameter)][0])
            and parameter.requires_grad is before[id(parameter)][1]
            for parameter in model.cortex.parameters()
        )
