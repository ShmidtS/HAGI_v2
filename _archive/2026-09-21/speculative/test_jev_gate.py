"""Tests for the feature->adapter-delta JEV gate (no full generation needed)."""
from __future__ import annotations

import pytest
import torch

from hagi.inference.jev_gate import (
    FeatureVector,
    apply_scale_deltas,
    extract_features,
    feature_adapt_step,
    feature_weight,
    scale_deltas,
)
from hagi.model.adapters import BlockAdapter
from hagi.model.model import HAGI
from tests.conftest import tiny_config


def _adapter_cfg():
    cfg = tiny_config()
    cfg.model.adapters.enabled = True
    cfg.model.adapters.pyramid.enabled = True
    cfg.model.adapters.pyramid.levels = (1,)
    cfg.model.adapters.ttt_lora.enabled = False
    cfg.model.adapters.pyramid.residual_scale = 1.0
    from hagi.config import validate_config
    validate_config(cfg)
    return cfg


def _model_with_adapters():
    cfg = _adapter_cfg()
    return HAGI(cfg)


class TestFeatureVector:
    def test_valid_high_confidence(self):
        fv = FeatureVector(confidence=0.9, entropy=0.3, rep=0.05, n_tokens=10)
        assert fv.is_valid()

    def test_valid_boundaries(self):
        fv = FeatureVector(confidence=1.0, entropy=100.0, rep=1.0, n_tokens=0)
        assert fv.is_valid()

    def test_invalid_confidence_out_of_range(self):
        fv = FeatureVector(confidence=1.5, entropy=1.0, rep=0.1, n_tokens=5)
        assert not fv.is_valid()

    def test_invalid_negative_entropy(self):
        fv = FeatureVector(confidence=0.5, entropy=-0.1, rep=0.1, n_tokens=5)
        assert not fv.is_valid()

    def test_invalid_nonfinite(self):
        fv = FeatureVector(confidence=float("nan"), entropy=1.0, rep=0.1, n_tokens=5)
        assert not fv.is_valid()

    def test_invalid_negative_tokens(self):
        fv = FeatureVector(confidence=0.5, entropy=1.0, rep=0.1, n_tokens=-1)
        assert not fv.is_valid()


class TestExtractFeatures:
    def test_extracts_from_result_dict(self):
        result = {"confidence": 0.85, "entropy": 0.42, "rep": 0.12, "ntok": 32}
        fv = extract_features(result)
        assert fv.confidence == 0.85
        assert fv.entropy == 0.42
        assert fv.rep == 0.12
        assert fv.n_tokens == 32
        assert fv.is_valid()

    def test_defaults_for_missing_keys(self):
        fv = extract_features({})
        assert fv.confidence == 0.0
        assert fv.entropy == 0.0
        assert fv.rep == 1.0
        assert fv.n_tokens == 0
        # rep=1.0 is valid (in [0,1]), conf=0.0 is valid (in [0,1]),
        # entropy=0.0 is valid (>= 0), ntok=0 is valid (>= 0): so this is valid.
        assert fv.is_valid()

    def test_handles_none_logprobs(self):
        result = {"confidence": 0.5, "entropy": 1.0, "rep": 0.0, "ntok": 1,
                  "logprobs": None}
        fv = extract_features(result)
        assert fv.is_valid()


class TestFeatureWeight:
    def test_low_confidence_high_weight(self):
        fv = FeatureVector(confidence=0.05, entropy=2.0, rep=0.5, n_tokens=10)
        assert feature_weight(fv) > 0.5

    def test_high_confidence_low_weight(self):
        fv = FeatureVector(confidence=0.95, entropy=0.1, rep=0.0, n_tokens=10)
        assert feature_weight(fv) < 0.25

    def test_high_entropy_boosts_weight(self):
        fv_low = FeatureVector(confidence=0.5, entropy=0.1, rep=0.0, n_tokens=10)
        fv_high = FeatureVector(confidence=0.5, entropy=3.0, rep=0.0, n_tokens=10)
        assert feature_weight(fv_high) > feature_weight(fv_low)

    def test_high_rep_boost_weight(self):
        fv_clean = FeatureVector(confidence=0.5, entropy=1.0, rep=0.0, n_tokens=10)
        fv_rep = FeatureVector(confidence=0.5, entropy=1.0, rep=0.9, n_tokens=10)
        assert feature_weight(fv_rep) > feature_weight(fv_clean)

    def test_invalid_features_zero_weight(self):
        fv = FeatureVector(confidence=1.5, entropy=-1.0, rep=0.1, n_tokens=5)
        assert feature_weight(fv) == 0.0

    def test_weight_in_unit_interval(self):
        fv = FeatureVector(confidence=0.2, entropy=3.0, rep=0.9, n_tokens=10)
        w = feature_weight(fv)
        assert 0.0 <= w <= 1.0


class TestScaleDeltas:
    def test_high_confidence_low_rep_push_up(self):
        """High confidence + low rep → positive delta (amplify)."""
        model = _model_with_adapters()
        result = {"confidence": 0.95, "entropy": 0.2, "rep": 0.05, "ntok": 10}
        deltas = scale_deltas(model, result, lr=0.01)
        assert len(deltas) > 0
        # High confidence (0.95 - 0.5 = +0.45) * (1 - 0.05) = positive
        for delta in deltas.values():
            assert delta > 0.0

    def test_low_confidence_high_rep_negative(self):
        """Low confidence + high rep → negative delta (attenuate)."""
        model = _model_with_adapters()
        result = {"confidence": 0.05, "entropy": 3.0, "rep": 0.8, "ntok": 10}
        deltas = scale_deltas(model, result, lr=0.01)
        assert len(deltas) > 0
        for delta in deltas.values():
            assert delta < 0.0

    def test_invalid_features_empty_deltas(self):
        model = _model_with_adapters()
        result = {"confidence": 1.5, "entropy": -0.1, "rep": -0.5, "ntok": 10}
        deltas = scale_deltas(model, result)
        assert len(deltas) == 0

    def test_zero_feature_weight_empty_deltas(self):
        """When feature_weight is ~0, no deltas computed."""
        model = _model_with_adapters()
        result = {"confidence": 1.0, "entropy": 0.0, "rep": 0.0, "ntok": 1}
        deltas = scale_deltas(model, result, lr=0.01)
        # conf=1.0, ent=0.0, rep=0.0 → weight = 0 → empty
        assert len(deltas) == 0

    def test_delta_capped_at_max_abs(self):
        model = _model_with_adapters()
        result = {"confidence": 0.95, "entropy": 0.2, "rep": 0.05, "ntok": 10}
        deltas = scale_deltas(model, result, lr=100.0, max_abs_delta=0.1)
        for delta in deltas.values():
            assert abs(delta) <= 0.1

    def test_only_touches_pyramid_scale(self):
        model = _model_with_adapters()
        result = {"confidence": 0.5, "entropy": 1.5, "rep": 0.3, "ntok": 10}
        deltas = scale_deltas(model, result)
        for pid, delta in deltas.items():
            found = False
            for module in model.modules():
                if isinstance(module, BlockAdapter) and module.pyramid is not None:
                    if id(module.pyramid.scale) == pid:
                        found = True
            assert found, f"delta for pid {pid} did not map to a pyramid scale"


class TestApplyScaleDeltas:
    def test_applies_deltas_in_place(self):
        model = _model_with_adapters()
        original = {}
        for module in model.modules():
            if isinstance(module, BlockAdapter) and module.pyramid is not None:
                original[id(module.pyramid.scale)] = (
                    module.pyramid.scale.detach().clone()
                )
        result = {"confidence": 0.95, "entropy": 0.2, "rep": 0.05, "ntok": 10}
        deltas = scale_deltas(model, result, lr=0.01)
        n = apply_scale_deltas(model, deltas)
        assert n == len(deltas)
        for pid, delta in deltas.items():
            for module in model.modules():
                if isinstance(module, BlockAdapter) and module.pyramid is not None:
                    if id(module.pyramid.scale) == pid:
                        new_val = module.pyramid.scale.detach()
                        assert not torch.equal(new_val, original[pid]), (
                            "scale should have changed after apply"
                        )

    def test_returns_count_of_updates(self):
        model = _model_with_adapters()
        result = {"confidence": 0.95, "entropy": 0.2, "rep": 0.05, "ntok": 10}
        deltas = scale_deltas(model, result, lr=0.01)
        n = apply_scale_deltas(model, deltas)
        expected = sum(
            1
            for m in model.modules()
            if isinstance(m, BlockAdapter) and m.pyramid is not None
        )
        assert n == expected

    def test_empty_deltas_no_changes(self):
        model = _model_with_adapters()
        original = {}
        for module in model.modules():
            if isinstance(module, BlockAdapter) and module.pyramid is not None:
                original[id(module.pyramid.scale)] = (
                    module.pyramid.scale.detach().clone()
                )
        n = apply_scale_deltas(model, {})
        assert n == 0
        for module in model.modules():
            if isinstance(module, BlockAdapter) and module.pyramid is not None:
                assert torch.equal(
                    module.pyramid.scale.detach(),
                    original[id(module.pyramid.scale)],
                )

    def test_clamps_extreme_deltas(self):
        """Delta beyond max_abs_delta is clamped during scale_deltas, not apply."""
        model = _model_with_adapters()
        result = {"confidence": 0.99, "entropy": 0.1, "rep": 0.01, "ntok": 10}
        deltas = scale_deltas(model, result, lr=1000.0, max_abs_delta=0.05)
        for d in deltas.values():
            assert abs(d) <= 0.05

    def test_invalid_features_no_changes(self):
        model = _model_with_adapters()
        original = {}
        for module in model.modules():
            if isinstance(module, BlockAdapter) and module.pyramid is not None:
                original[id(module.pyramid.scale)] = (
                    module.pyramid.scale.detach().clone()
                )
        n = apply_scale_deltas(model, {})
        assert n == 0
        for module in model.modules():
            if isinstance(module, BlockAdapter) and module.pyramid is not None:
                assert torch.equal(
                    module.pyramid.scale.detach(),
                    original[id(module.pyramid.scale)],
                )


class TestFeatureAdaptStep:
    def test_full_step_returns_summary(self):
        model = _model_with_adapters()
        result = {"confidence": 0.8, "entropy": 0.5, "rep": 0.1, "ntok": 16}
        summary = feature_adapt_step(model, result, lr=0.01)
        assert summary["n_updated"] > 0
        assert summary["n_adapters"] == summary["n_updated"]
        assert "features" in summary
        assert summary["features"]["confidence"] == 0.8
        assert 0.0 < summary["weight"] <= 1.0

    def test_invalid_features_returns_no_updates(self):
        model = _model_with_adapters()
        result = {"confidence": float("nan"), "entropy": 1.0, "rep": 0.1, "ntok": 10}
        summary = feature_adapt_step(model, result)
        assert summary["n_updated"] == 0
        assert summary["weight"] == 0.0
        assert summary["features"] is None

    def test_step_is_bounded(self):
        model = _model_with_adapters()
        result = {"confidence": 0.95, "entropy": 0.1, "rep": 0.0, "ntok": 1}
        summary = feature_adapt_step(model, result, lr=1000.0, max_abs_delta=0.01)
        for pid, delta in summary["deltas"].items():
            assert abs(delta) <= 0.01

    def test_zero_init_adapter_gets_nonzero(self):
        """A zero-init adapter (scale=0) receives a non-zero delta when features demand it."""
        model = _model_with_adapters()
        for module in model.modules():
            if isinstance(module, BlockAdapter) and module.pyramid is not None:
                assert module.pyramid.scale.item() == 0.0
        result = {"confidence": 0.95, "entropy": 0.2, "rep": 0.05, "ntok": 10}
        feature_adapt_step(model, result, lr=0.01)
        got_nonzero = False
        for module in model.modules():
            if isinstance(module, BlockAdapter) and module.pyramid is not None:
                if abs(module.pyramid.scale.item()) > 0.0:
                    got_nonzero = True
        assert got_nonzero, "adapter scale must change from 0 when features trigger an update"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
