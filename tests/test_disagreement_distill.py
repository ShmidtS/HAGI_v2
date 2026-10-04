"""Шаг 4 плана формализации: Bregman-slice дистилляция — тесты.

Ядро: селекция токенов по mean pairwise JS учителей выделяет
расхождение (не консенсус); forward-KL цель на маске даёт градиент
ТОЛЬКО студенту; полный консенсус ⟹ пустая маска ⟹ нулевой канал
(R104: канал, которому нечему учиться, оптимизатор не выключает —
он просто не активируется). Знак JS: Σ p(log m − log p) = −KL.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.disagreement_distill import (  # noqa: E402
    disagreement_mask,
    disagreement_slice_loss,
    mean_pairwise_js,
)

V, R = 64, 64


def _teachers(strong_rows: int = 32) -> list[torch.Tensor]:
    torch.manual_seed(0)
    base = torch.randn(R, V)
    ts = [base + 0.01 * torch.randn(R, V) for _ in range(3)]
    ts[2][strong_rows:] += 3.0 * torch.randn(R - strong_rows, V)
    return ts


def test_js_nonneg_and_bounded():
    js = mean_pairwise_js([torch.softmax(t, -1) for t in _teachers()])
    assert js.min() >= -1e-9
    assert js.max() <= math.log(2) + 1e-9


def test_js_localizes_disagreement():
    ts = _teachers()
    js = mean_pairwise_js([torch.softmax(t, -1) for t in ts])
    assert js[32:].mean() > 5 * js[:32].mean()


def test_js_identical_teachers_zero():
    p = torch.softmax(torch.randn(R, V), -1)
    js = mean_pairwise_js([p, p.clone(), p.clone()])
    assert js.abs().max() < 1e-12


def test_mask_quantile_fraction():
    ts = _teachers()
    mask = disagreement_mask(ts, quantile=0.5)
    assert 0.3 < mask.float().mean() < 0.7


def test_mask_requires_two_teachers():
    t = torch.randn(R, V)
    with pytest.raises(ValueError):
        disagreement_mask([t])


def test_loss_grad_student_only():
    ts = _teachers()
    s = (ts[0] + 0.1 * torch.randn(R, V)).requires_grad_(True)
    loss = disagreement_slice_loss(s, ts, quantile=0.95)
    loss.backward()
    assert s.grad is not None
    assert s.grad.abs().sum() > 0
    assert not ts[0].requires_grad


def test_loss_consensus_empty_mask_zero():
    base = torch.randn(R, V)
    s = torch.randn(R, V, requires_grad=True)
    loss = disagreement_slice_loss(s, [base, base.clone(), base.clone()])
    loss.backward()
    assert loss.item() == 0.0
    assert s.grad.abs().sum() == 0.0


def test_loss_nonnegative():
    ts = _teachers()
    s = torch.randn(R, V, requires_grad=True)
    loss = disagreement_slice_loss(s, ts, quantile=0.5)
    assert loss.item() >= -1e-9
