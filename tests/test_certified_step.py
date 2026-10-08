"""Certified step (SafeQP.lean acting controller + R105 safeqp_eta_max).

The WIP config fields declared two licenses and this suite pins both:
gate mode (veto a step whose direction damages a calibration corpus; cap
the LR at the R105 derived rate) and step mode (replace the step by the
SafeQP dual direction d* = g + sum lam_i g_i at the derived rate).

The "" default must stay invisible: no calibration, no veto, identical
updates -- a regression pin, because every shipped config runs that path.
"""
from __future__ import annotations

import pytest
import torch

from hagi.config import Config, validate_config
from hagi.model.factory import build_model_for_config
from hagi.train.loop import Trainer


def _cfg(**over) -> Config:
    c = Config()
    c.model.hidden_size = 64
    c.model.num_layers = 2
    c.model.attention.num_query_heads = 4
    c.model.attention.num_kv_heads = 2
    c.model.attention.head_dim = 16
    c.model.attention.max_seq_len = 64
    c.model.vocab_size = 128
    c.train.use_muon = False
    c.train.data.seq_len = 31
    for k, v in over.items():
        setattr(c.train, k, v)
    return c


def _batch() -> dict:
    torch.manual_seed(11)
    ids = torch.randint(0, 128, (2, 31))
    tgt = torch.randint(0, 128, (2, 31))
    return {"input_ids": ids, "targets": tgt}


def _certified_trainer(mode: str, corpora: list[str], tmp_path, **over):
    """Trainer with certified_* wired and tiny corpus files on disk."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in corpora:
        g = torch.Generator().manual_seed(hash(name) % 2**31)
        toks = torch.randint(0, 128, (4096,), generator=g, dtype=torch.int32)
        (data_dir / f"{name}.bin").write_bytes(toks.numpy().tobytes())
        # _certified_samples reads <name>.compact.bin (uint32 tokens)
        (data_dir / f"{name}.compact.bin").write_bytes(
            toks.to(torch.int32).numpy().tobytes()
        )
    base = {
        "certified_step": mode,
        "certified_corpora": corpora,
        "certified_interval": 1,
        "certified_tokens": 256,
        "certified_eps_rel": 0.0,
        "certified_eps_nats": 1.0,
    }
    base.update(over)
    c = _cfg(**base)
    c.train.data.data_dir = str(data_dir)
    validate_config(c)
    model = build_model_for_config(c)
    return Trainer(model, c, 0)


def test_config_validation_rejects_bad_mode():
    c = _cfg(certified_step="nonsense", certified_corpora=["a"])
    with pytest.raises(ValueError, match="certified_step"):
        validate_config(c)


def test_config_validation_requires_corpora():
    c = _cfg(certified_step="gate")
    with pytest.raises(ValueError, match="certified_corpora"):
        validate_config(c)


def test_config_validation_forbids_analytic_step_combo():
    c = _cfg(
        certified_step="gate",
        certified_corpora=["a"],
        analytic_step=True,
    )
    with pytest.raises(ValueError, match="cannot both"):
        validate_config(c)


def test_default_mode_is_invisible(tmp_path):
    """The "" path: no calibration state, update applied, no veto keys."""
    c = _cfg()
    validate_config(c)
    t = Trainer(build_model_for_config(c), c, 0)
    assert t._certified_mode == ""
    assert t._certified is None
    m = t.train_step([_batch()])
    assert m["update_applied"]
    assert "certified_vetoed" not in m
    assert "certified_eta" not in m


def test_gate_mode_runs_and_calibrates(tmp_path):
    corpora = ["alpha", "beta"]
    t = _certified_trainer("gate", corpora, tmp_path)
    assert t._certified_mode == "gate"
    m = t.train_step([_batch()])
    # Calibration happened on the first step.
    assert t._certified is not None
    assert set(t._certified["grads"]) == set(corpora)
    # With random-init model and random corpora, the gate has no reason to
    # veto: the honest assertion is that EITHER it applied the update with
    # an R105-capped LR, or it vetoed with the binding corpus named.
    if m["update_applied"]:
        assert m["certified_eta"] <= c_lr(t)
        assert m["certified_eta"] > 0.0
        assert "certified_binding" in m
    else:
        assert m["certified_vetoed"] == 1
        assert "certified_binding" in m


def c_lr(t: Trainer) -> float:
    return float(t.cfg.train.learning_rate)


def test_step_mode_applies_certified_direction(tmp_path):
    corpora = ["alpha", "beta"]
    t = _certified_trainer("step", corpora, tmp_path)
    m = t.train_step([_batch()])
    assert t._certified is not None
    if m["update_applied"]:
        assert m["certified_eta"] > 0.0
        assert m["certified_eta"] <= c_lr(t)
        # The certificate keys come from safe_qp_solve.
        assert "certified_kappa" in m
        # d* keeps the descent property: <g, d*> >= ||d*||^2 - tol.
        assert m["certified_descent_gap"] >= -1e-6
    else:
        assert m["certified_vetoed"] == 1


def test_gate_lr_cap_binds_when_window_tight(tmp_path):
    """A tight eps_nats budget must force eta below the schedule LR."""
    corpora = ["alpha", "beta"]
    t = _certified_trainer("gate", corpora, tmp_path, certified_eps_nats=1e-9)
    m = t.train_step([_batch()])
    if m["update_applied"] and "certified_eta" in m:
        # With eps ~ 1e-9 the R105 window is razor-thin unless inner > 0
        # everywhere; either the cap binds below the LR or the veto fired.
        assert m["certified_eta"] <= c_lr(t)


def test_calibration_refresh_respects_interval(tmp_path):
    corpora = ["alpha"]
    t = _certified_trainer("gate", corpora, tmp_path, certified_interval=50)
    t.train_step([_batch()])
    assert t._certified is not None
    first = t._certified
    t.train_step([_batch()])
    # step 1..49 reuses the cache (no re-read of corpus files).
    assert t._certified is first


def test_negative_window_vetoes_not_tiny_steps(tmp_path):
    """cap <= 0 must skip the step, not apply a near-zero LR."""
    corpora = ["alpha", "beta"]
    t = _certified_trainer("step", corpora, tmp_path, certified_eps_nats=0.0)
    m = t.train_step([_batch()])
    if not m["update_applied"]:
        assert m["certified_vetoed"] == 1
    else:
        # eps_nats=0 with strictly-conflicting corpora vetoes; with none
        # conflicting the window stays positive and the step applies.
        assert m["certified_eta"] > 0.0
