"""Tests for the generation-config geometry rules.

Two gen-4 launches failed on exactly these invariants, hours apart,
because a copied config silently keeps the old dimensions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_generation import geometry_ok, scale  # noqa: E402


def _base(width: int = 1152, q: int = 18, kv: int = 9, hd: int = 64) -> dict:
    return {
        "model": {
            "hidden_size": width,
            "attention": {
                "num_query_heads": q,
                "num_kv_heads": kv,
                "head_dim": hd,
            },
        },
        "merge": {"expert_hidden": 384, "n_experts": 3},
    }


def test_geometry_ok_accepts_a_consistent_config():
    ok, why = geometry_ok(_base())
    assert ok, why


def test_geometry_rejects_head_width_mismatch():
    """The exact gen-4 failure: q*head_dim must equal hidden_size."""
    cfg = _base(width=1152, q=6, kv=3)  # 6*64 = 384 != 1152
    ok, why = geometry_ok(cfg)
    assert not ok
    assert "1152" in why


def test_geometry_rejects_kv_above_q():
    """Keep the width consistent so the kv check is the one that fires."""
    cfg = _base(width=384, q=6, kv=12)
    ok, why = geometry_ok(cfg)
    assert not ok
    assert "num_kv_heads" in why


def test_geometry_rejects_missing_fields():
    ok, why = geometry_ok({"model": {"hidden_size": 8}})
    assert not ok
    assert "missing" in why


def test_scale_resizes_width_and_heads_together():
    """Scaling must move width, query heads and kv heads as one operation."""
    cfg = scale(_base(), width=3456, head_dim=64)
    ok, why = geometry_ok(cfg)
    assert ok, why
    assert cfg["model"]["hidden_size"] == 3456
    assert cfg["model"]["attention"]["num_query_heads"] == 54
    assert cfg["model"]["attention"]["num_kv_heads"] == 27


def test_scale_is_idempotent_on_a_matching_config():
    """Re-running the generator must not drift the geometry."""
    once = scale(_base(), width=3456, head_dim=64)
    twice = scale(once, width=3456, head_dim=64)
    assert once == twice


def test_scale_preserves_unrelated_fields():
    cfg = _base()
    cfg["train"] = {"learning_rate": 3e-4, "seed": 12345}
    out = scale(cfg, width=3456, head_dim=64)
    assert out["train"] == {"learning_rate": 3e-4, "seed": 12345}


def test_odd_kv_count_never_goes_to_zero():
    """A width of one head would divide kv by two to zero."""
    cfg = scale(_base(width=1152), width=64, head_dim=64)
    assert cfg["model"]["attention"]["num_kv_heads"] >= 1


def test_every_shipped_generation_config_is_consistent():
    """Regression guard on the real configs, not just synthetic ones."""
    root = Path(__file__).resolve().parents[1]
    checked = 0
    for path in sorted(root.glob("configs/*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict) or "model" not in cfg:
            continue
        model = cfg["model"]
        if "hidden_size" not in model:
            continue
        ok, why = geometry_ok(cfg)
        assert ok, f"{path.name}: {why}"
        checked += 1
    assert checked >= 3, f"expected the live generation configs, found {checked}"
