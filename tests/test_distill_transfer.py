"""T3: тесты DistillTransfer (замена R130, 2609.38666/2607.15467).

Каждый тест пришпиливает runtime-порт к теореме T3 плана:

* ``transfer_bound`` — CE_q(θ) − CE_q(E) ≤ KL(p_E‖p_θ) + M·‖q−p_E‖₁;
* ``channel_efficiency`` — η = (CE_leafmean − CE_student)/twoGap в
  чистых натах, исторический γ-дефицит 0.002 vs 0.018 воспроизводится;
* ``forward_kl_teacher`` — арифметическая смесь (универсальность);
* ``reverse_kl_teacher`` — геометрическая смесь (модность), подавление
  minority;
* ``distill_floor_gap`` — потолок канала argmax vs full teacher.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.distill_transfer import (  # noqa: E402
    channel_efficiency,
    distill_floor_gap,
    forward_kl_teacher,
    reverse_kl_teacher,
    transfer_bound,
)

V = 50


class TestTransferBound:
    def test_zero_gap_zero_bound(self):
        # студент = учитель: KL = 0, M = 0 — граница 0
        lg = torch.randn(2, 8, V)
        b = transfer_bound(lg, lg.clone())
        assert b["kl"] == pytest.approx(0.0, abs=1e-12)
        assert b["bound"] == pytest.approx(0.0, abs=1e-12)

    def test_bound_positive_when_student_differs(self):
        torch.manual_seed(0)
        s = torch.randn(2, 8, V)
        t = s + 0.5
        b = transfer_bound(s, t)
        assert b["kl"] > 0.0
        assert b["bound"] >= b["kl"] - 1e-12

    def test_l1_term_enters_when_data_differs(self):
        torch.manual_seed(1)
        s = torch.randn(2, 4, V)
        t = s + 0.3 * torch.randn(2, 4, V)
        q = t + 1.0 * torch.randn(2, 4, V)
        b0 = transfer_bound(s, t)
        b1 = transfer_bound(s, t, logit_data=q)
        assert b1["l1_data_teacher"] > 0.0
        assert b1["bound"] >= b0["bound"]

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            transfer_bound(torch.randn(2, V), torch.randn(3, V))


class TestChannelEfficiency:
    def test_perfect_student_eta_one(self):
        # студент = ансамбль: η = gap/gap = 1
        assert channel_efficiency(3.5, 3.0, 3.0) == pytest.approx(1.0)

    def test_no_transfer_eta_zero(self):
        # студент = leafmean: η = 0
        assert channel_efficiency(3.5, 3.5, 3.0) == pytest.approx(0.0)

    def test_historical_gamma_deficit(self):
        # γ = 0.002 при twoGap = 0.1: η = 0.02 — исторический дефицит
        # (0.002 в натах прироста на цикл против требуемых 0.018)
        assert channel_efficiency(3.5, 3.498, 3.4) == pytest.approx(0.02)

    def test_negative_when_student_degrades(self):
        # CE студента выше leafmean: канал работает назад
        assert channel_efficiency(3.5, 3.6, 3.0) < 0.0

    def test_zero_gap_raises(self):
        with pytest.raises(ValueError):
            channel_efficiency(3.0, 3.0, 3.0)


class TestKLClosure:
    def test_forward_is_arithmetic_mix(self):
        # forward-KL аргмин = Σ w_i p_i: на двух учителях проверяем
        # экспоненту логит-выхода против прямой смеси вероятностей
        torch.manual_seed(2)
        a, b = torch.randn(4, V), torch.randn(4, V)
        f = forward_kl_teacher([a, b])
        mix = 0.5 * torch.softmax(a.double(), -1) + 0.5 * torch.softmax(b.double(), -1)
        assert torch.allclose(f.exp(), mix, atol=1e-10)

    def test_reverse_is_geometric_mix(self):
        torch.manual_seed(3)
        a, b = torch.randn(4, V), torch.randn(4, V)
        r = reverse_kl_teacher([a, b])
        geo = (torch.softmax(a.double(), -1) * torch.softmax(b.double(), -1)).sqrt()
        geo = geo / geo.sum(-1, keepdim=True)
        assert torch.allclose(r.exp(), geo, atol=1e-10)

    def test_reverse_suppresses_minority(self):
        # 2609.38666: reverse-KL подавляет minority (мода), forward — нет.
        # teacher A концентрирован на токене 0, teacher B слабо на 0.
        a = torch.zeros(1, V); a[0, 0] = 8.0          # концентрированный
        b = torch.zeros(1, V); b[0, 0] = 1.0; b[0, 1] = 1.0  # размазанный
        fwd = forward_kl_teacher([a, b]).exp()
        rev = reverse_kl_teacher([a, b]).exp()
        # forward держит хвост учителя B на токене 1 сильнее reverse
        assert fwd[0, 1] > rev[0, 1]

    def test_single_teacher_is_identity(self):
        a = torch.randn(2, V)
        f = forward_kl_teacher([a])
        assert torch.allclose(f.exp(), torch.softmax(a.double(), -1), atol=1e-10)
        r = reverse_kl_teacher([a])
        assert torch.allclose(r.exp(), torch.softmax(a.double(), -1), atol=1e-10)

    def test_weighted_forward_biases_majority(self):
        torch.manual_seed(4)
        a, b = torch.randn(2, V), torch.randn(2, V)
        f = forward_kl_teacher([a, b], weights=[0.9, 0.1])
        mix = 0.9 * torch.softmax(a.double(), -1) + 0.1 * torch.softmax(b.double(), -1)
        assert torch.allclose(f.exp(), mix, atol=1e-10)


class TestDistillFloor:
    def test_floor_gap_positive(self):
        assert distill_floor_gap(4.2, 3.6) == pytest.approx(0.6)

    def test_floor_gap_zero_when_no_dark_knowledge(self):
        assert distill_floor_gap(3.6, 3.6) == 0.0
