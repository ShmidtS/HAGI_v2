"""Reverse-recursion distill channel (RecursiveDistill.lean / FreeEnergy.lean P1b).

The runtime channel: a frozen external teacher (any geometry, loaded from a
checkpoint by its OWN payload config) supervises the compact student on the
same corpus states, blended into the LM objective. These tests pin the
mechanics -- teacher loading, KD sign/flow, row alignment under loss_mask,
and the alpha=0 degenerate case (pure CE, the tested baseline).
"""
from __future__ import annotations

import torch

from hagi.config import Config
from hagi.model.factory import build_model_for_config
from hagi.train.checkpoint import save_checkpoint
from hagi.train.loop import Trainer, _build_distill_teacher


def _cfg(merge: bool = False, distill: bool = False, teacher=None) -> Config:
    c = Config()
    c.model.hidden = 64
    c.model.layers = 2
    c.model.heads = 4
    c.model.vocab = 128
    c.model.sequence = 32
    c.merge.enabled = merge
    c.merge.distill = distill
    if teacher is not None:
        c.merge.distill_teacher = teacher
    return c


def _batch(masked: bool = False) -> dict:
    torch.manual_seed(7)
    ids = torch.randint(0, 128, (2, 31))
    tgt = torch.randint(0, 128, (2, 31))
    batch = {"input_ids": ids, "targets": tgt}
    if masked:
        mask = torch.ones_like(tgt, dtype=torch.bool)
        mask[:, :8] = False
        batch["loss_mask"] = mask
    return batch


def test_distill_requires_teacher_path(tmp_path):
    c = _cfg(distill=True)
    try:
        _build_distill_teacher(c, "cpu")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_distill_disabled_returns_none():
    t, path = _build_distill_teacher(_cfg(distill=False), "cpu")
    assert t is None and path == ""


def test_channel_end_to_end(tmp_path):
    torch.manual_seed(0)
    tcfg = _cfg(merge=True)
    tmodel = build_model_for_config(tcfg)
    outdir = tmp_path / "teacher"
    save_checkpoint(tmodel, tcfg, 10, outdir, keep_last=1)
    tpath = str(outdir / "step-0000010.pt")

    scfg = _cfg(distill=True, teacher=tpath)
    scfg.merge.distill_alpha = 0.5
    smodel = build_model_for_config(scfg)

    trainer = Trainer(smodel, scfg, 0)
    assert trainer._teacher is not None
    assert next(trainer._teacher.parameters()).requires_grad is False
    assert trainer._distill_alpha == 0.5

    m = trainer.train_step([_batch()])
    assert m["update_applied"]
    # A KL divergence is non-negative; a negative value means the student-side
    # log_softmax lost precision (the bf16 bug this channel shipped with).
    assert "kd" in m and m["kd"] >= 0.0

    # loss_mask: teacher target and student logits must use the SAME rows.
    m2 = trainer.train_step([_batch(masked=True)])
    assert m2["update_applied"]
    assert m2["kd"] >= 0.0


def test_alpha_zero_is_pure_ce(tmp_path):
    torch.manual_seed(1)
    tcfg = _cfg(merge=True)
    tmodel = build_model_for_config(tcfg)
    outdir = tmp_path / "teacher"
    save_checkpoint(tmodel, tcfg, 10, outdir, keep_last=1)
    tpath = str(outdir / "step-0000010.pt")

    batch = _batch()

    base_cfg = _cfg()
    torch.manual_seed(3)
    base = Trainer(build_model_for_config(base_cfg), base_cfg, 0)
    torch.manual_seed(2)
    m_base = base.train_step([batch])

    kd_cfg = _cfg(distill=True, teacher=tpath)
    kd_cfg.merge.distill_alpha = 0.0
    torch.manual_seed(3)
    with_kd = Trainer(build_model_for_config(kd_cfg), kd_cfg, 0)
    torch.manual_seed(2)
    m_kd = with_kd.train_step([batch])

    # alpha=0: the KD term must not move the objective off the CE baseline.
    assert abs(m_base["loss"] - m_kd["loss"]) < 1e-6
    assert with_kd._teacher is not None  # teacher loaded but weightless
