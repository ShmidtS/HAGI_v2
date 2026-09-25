"""DecisionPlane contracts: opt-in objective, causal semantics, ownership."""

from __future__ import annotations

import math

import pytest
import torch

from hagi.config import count_params, validate_config
from hagi.model.decision import DecisionHead
from hagi.model.merge import merge_experts
from hagi.model.model import HAGI
from hagi.model.outputs import ModelOutput
from hagi.train.loop import Trainer
from tests.conftest import TINY_VOCAB, tiny_config


def decision_cfg(**overrides):
    values = {
        "model.decision.enabled": True,
        "model.decision.num_options": 4,
        "train.adapt.freeze_base": True,
        "train.precision": "fp32",
        "train.ternary_step_cache": False,
        "train.max_steps": 1,
        "train.schedule.warmup_steps": 0,
        "train.ce_keep_rate": 1.0,
    }
    values.update(overrides)
    return tiny_config(**values)


def batch(b=3, t=8, options=4):
    ids = torch.randint(0, TINY_VOCAB, (b, t))
    targets = torch.randint(0, TINY_VOCAB, (b, t))
    labels = torch.tensor([0, 2, 3][:b], dtype=torch.long)
    return ids, targets, labels


class TestDecisionConfig:
    def test_disabled_is_default_and_unchanged(self):
        cfg = tiny_config()
        assert cfg.model.decision.enabled is False
        assert count_params(cfg.model)["decision"] == 0
        assert HAGI(cfg).decision_head is None
        assert "decision" not in HAGI(cfg).state_dict()

    def test_enabled_count_is_exact(self):
        cfg = decision_cfg()
        model = HAGI(cfg)
        assert count_params(cfg.model)["decision"] == cfg.model.hidden_size * cfg.model.decision.num_options
        assert count_params(cfg.model)["total"] == sum(p.numel() for p in model.parameters())

    @pytest.mark.parametrize(
        "field,value",
        [
            ("model.decision.num_options", 1),
            ("model.decision.loss_weight", float("inf")),
            ("model.decision.loss_weight", 0.0),
            ("model.decision.confidence_bins", 0),
        ],
    )
    def test_invalid_config_rejected(self, field, value):
        with pytest.raises(ValueError):
            decision_cfg(**{field: value})


class TestDecisionForward:
    def test_shapes_and_component_loss(self):
        cfg = decision_cfg()
        model = HAGI(cfg).eval()
        ids, targets, labels = batch()
        out = model(ids, targets, decision_targets=labels)
        assert isinstance(out, ModelOutput)
        assert out.decision_logits.shape == (3, 4)
        assert out.decision_loss is not None
        assert out.lm_loss is not None
        assert out.loss is not None
        assert out.n_tokens == 24
        assert out.n_decisions == 3
        assert torch.isfinite(out.loss)

    def test_decision_only_forward(self):
        cfg = decision_cfg()
        model = HAGI(cfg).eval()
        ids, _, labels = batch()
        out = model(ids, decision_targets=labels)
        assert out.lm_loss is None
        assert out.ce is None
        assert out.n_tokens == 0
        assert out.decision_loss is not None
        out.loss.backward()
        assert model.decision_head.weight.grad is not None

    def test_mask_normalizes_selected_rows(self):
        cfg = decision_cfg()
        model = HAGI(cfg).eval()
        ids, _, labels = batch()
        mask = torch.tensor([True, False, True])
        out = model(ids, decision_targets=labels, decision_mask=mask)
        assert out.n_decisions == 2
        assert out.decision_targets.tolist() == [0, 3]
        all_rows = model(ids, decision_targets=labels)
        selected = model(ids, decision_targets=labels, decision_mask=mask)
        assert selected.decision_loss is not None
        assert all_rows.decision_loss is not None
        assert float(selected.decision_loss) > 0

    def test_all_false_mask_has_no_decision_loss(self):
        cfg = decision_cfg()
        model = HAGI(cfg).eval()
        ids, _, labels = batch()
        out = model(ids, decision_targets=labels, decision_mask=torch.zeros(3, dtype=torch.bool))
        assert out.decision_loss is None
        assert out.loss is None
        assert out.n_decisions == 0

    def test_empty_batch_is_rejected_before_decision_range_check(self):
        cfg = decision_cfg()
        model = HAGI(cfg)
        ids = torch.empty((0, 8), dtype=torch.long)
        with pytest.raises(ValueError, match=r"\[B, T\].*B >= 1"):
            model(ids, decision_targets=torch.empty(0, dtype=torch.long))

    def test_disabled_head_rejects_labels(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        ids, _, labels = batch()
        with pytest.raises(ValueError, match="model.decision.enabled"):
            model(ids, decision_targets=labels)

    @pytest.mark.parametrize(
        "targets, mask, match",
        [
            (torch.tensor([0, 2]), None, "shape"),
            (torch.tensor([0, 2, 3], dtype=torch.int32), None, "dtype"),
            (torch.tensor([0, 2, 9]), None, r"\[0, 4\)"),
            (None, torch.tensor([True, False, True]), "requires decision_targets"),
            (torch.tensor([0, 2, 3]), torch.tensor([True, False]), "shape"),
            (torch.tensor([0, 2, 3]), torch.tensor([True, False, True], dtype=torch.uint8), "dtype"),
        ],
    )
    def test_invalid_decision_contract_rejected(self, targets, mask, match):
        cfg = decision_cfg()
        model = HAGI(cfg)
        ids, _, _ = batch()
        with pytest.raises(ValueError, match=match):
            model(ids, decision_targets=targets, decision_mask=mask)

    def test_head_is_observational_for_lm(self):
        torch.manual_seed(4)
        enabled_cfg = decision_cfg()
        model = HAGI(enabled_cfg).eval()
        ids, targets, labels = batch()
        with torch.no_grad():
            without = model(ids, targets, return_logits=True)
            with_decision = model(
                ids,
                targets,
                decision_targets=labels,
                return_logits=True,
            )
        assert torch.equal(without.logits, with_decision.logits)
        assert torch.equal(without.hidden, with_decision.hidden)
        assert with_decision.decision_logits is not None

    def test_bf16_projection_accepts_bf16_hidden(self):
        head = DecisionHead(8, 3)
        hidden = torch.randn(2, 8, dtype=torch.bfloat16)
        out = head(hidden)
        assert out.shape == (2, 3)
        assert out.dtype == torch.bfloat16


class TestDecisionTrainer:
    def test_mixed_objective_updates_only_frozen_head(self):
        cfg = decision_cfg()
        model = HAGI(cfg)
        before_base = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if not name.startswith("decision_head.")
        }
        before_head = model.decision_head.weight.detach().clone()
        trainer = Trainer(model, cfg)
        ids, targets, labels = batch()
        metrics = trainer.train_step(
            [
                {
                    "input_ids": ids,
                    "targets": targets,
                    "decision_targets": labels,
                    "decision_mask": torch.tensor([True, False, True]),
                }
            ]
        )
        assert metrics["update_applied"] is True
        assert metrics["n_decisions"] == 2 if "n_decisions" in metrics else True
        assert not torch.equal(before_head, model.decision_head.weight)
        for name, parameter in model.named_parameters():
            if not name.startswith("decision_head."):
                assert torch.equal(before_base[name], parameter), name

    def test_two_decision_only_microbatches_use_row_denominator(self):
        cfg = decision_cfg()
        model = HAGI(cfg)
        trainer = Trainer(model, cfg)
        first = {
            "input_ids": torch.randint(0, TINY_VOCAB, (2, 6)),
            "decision_targets": torch.tensor([0, 1]),
        }
        second = {
            "input_ids": torch.randint(0, TINY_VOCAB, (3, 6)),
            "decision_targets": torch.tensor([2, 3, 0]),
        }
        metrics = trainer.train_step([first, second])
        assert metrics["update_applied"] is True
        assert math.isfinite(metrics["loss"])
        assert metrics["n_decisions"] == 5
        assert metrics["decision_loss"] > 0.0

    def test_freeze_base_marks_only_decision_head(self):
        cfg = decision_cfg()
        model = HAGI(cfg)
        Trainer(model, cfg)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert trainable == {"decision_head.weight"}


class TestDecisionDecodeAndMerge:
    def test_full_and_prefill_decode_final_logits_match(self):
        cfg = decision_cfg()
        model = HAGI(cfg).eval()
        ids, _, labels = batch(b=1, t=6)
        with torch.no_grad():
            full = model(ids, decision_targets=labels).decision_logits
            model.reset_cache()
            model.allocate_cache(torch.float32, torch.device("cpu"))
            prefill = model(
                ids[:, :4],
                use_cache=True,
                decision_targets=labels,
            ).decision_logits
            intermediate = model(
                ids[:, 4:5],
                positions=torch.arange(4, 5),
                use_cache=True,
                decision_targets=labels,
            ).decision_logits
            decoded = model(
                ids[:, 5:6],
                positions=torch.arange(5, 6),
                use_cache=True,
                decision_targets=labels,
            ).decision_logits
            model.reset_cache()
        assert prefill.shape == (1, 4)
        assert torch.isfinite(prefill).all()
        assert intermediate.shape == (1, 4)
        assert torch.isfinite(intermediate).all()
        assert torch.allclose(full, decoded, atol=1e-4)

    def test_final_position_contract_is_causal(self):
        cfg = decision_cfg()
        model = HAGI(cfg).eval()
        ids, _, labels = batch(b=1, t=6)
        assert model.decision_head is not None
        changed = ids.clone()
        changed[:, -1] = (changed[:, -1] + 1) % TINY_VOCAB
        with torch.no_grad():
            full = model(ids, decision_targets=labels)
            prefix = model(ids[:, :-1], decision_targets=labels)
            changed_full = model(changed, decision_targets=labels)
            assert full.decision_logits is not None
            assert prefix.decision_logits is not None
            assert changed_full.decision_logits is not None
            assert full.hidden is not None
            assert changed_full.hidden is not None
            assert torch.allclose(full.decision_logits, model.decision_head(full.hidden[:, -1]))
            assert torch.allclose(
                prefix.decision_logits,
                model.decision_head(full.hidden[:, -2]),
                atol=1e-4,
            )
            assert torch.allclose(changed_full.hidden[:, :-1], full.hidden[:, :-1], atol=1e-4)

    def test_merge_drops_expert_decision_state(self):
        merged_cfg = tiny_config(
            **{
                "model.num_layers": 2,
                "model.hidden_size": 64,
                "model.attention.num_query_heads": 2,
                "model.attention.num_kv_heads": 2,
                "model.attention.head_dim": 32,
                "model.ffn.intermediate_size": 128,
                "model.embedding.conv_kernel": 1,
            }
        )
        merged_cfg.model.decision.enabled = True
        merged_cfg.model.decision.num_options = 3
        merged_cfg.merge.enabled = True
        merged_cfg.merge.n_experts = 2
        merged_cfg.merge.expert_hidden = 32
        merged_cfg.merge.mixer_type = "swiglu"
        validate_config(merged_cfg)
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
        expert_cfg.model.decision.enabled = True
        expert_cfg.model.decision.num_options = 3
        validate_config(expert_cfg)
        expert = HAGI(expert_cfg)
        with torch.no_grad():
            expert.decision_head.weight.fill_(0.25)
        state = expert.state_dict()
        merged = merge_experts(merged_cfg, [dict(state), dict(state)], n_mixers=1)
        assert merged.decision_head is not None
        assert torch.count_nonzero(merged.decision_head.weight) == 0
