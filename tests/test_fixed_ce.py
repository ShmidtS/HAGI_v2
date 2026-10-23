"""Collapse/Cancellation-фиксы softmax и CE (arXiv:2610.31642).

Три живых бага, воспроизведённые тестами:

1. Softmax Collapse (fp32-клифф m ≈ 16.6, bf16 ≈ 5.5): стандартный CE
   округляет loss и целевой градиент до 0 за клиффом; фиксированный —
   точен до underflow экспоненты, градиент строки суммируется в 0.
2. Fused-путь (autograd через softplus-margin форму) — тот же инвариант.
3. Softmax Cancellation: KD-градиент q - p с раздельно округлёнными
   вероятностями имеет ненулевой пол у z_S → z_T; фиксированный
   (expm1/log1p) — точно 0 при равенстве.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from hagi.config import HeadConfig
from hagi.model.head import LMHead, _ChunkedCrossEntropy
from hagi.train.distill import fixed_log_softmax, soft_target_kl


def _margin_case(v: int = 8, margin: float = 20.0) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden/weight, дающие логиты [margin, 0, 0, ...] при target=0."""
    h = torch.zeros(1, v)
    h[0, 0] = 1.0
    w = torch.eye(v)
    # logits = h @ w.T = [1, 0, ...] -> нужно [margin, 0, ...]
    h[0, 0] = margin
    return h, w


class TestChunkedCollapse:
    def test_loss_beyond_fp32_cliff(self):
        h, w = _margin_case(margin=20.0)
        targets = torch.tensor([0])
        # Старая форма: lse = 20 + log(1 + 7e-20) -> 20.0 ровно, loss = 0.
        z = h @ w.T
        assert float(torch.logsumexp(z, -1) - z[0, 0]) == 0.0
        ce, _ = _ChunkedCrossEntropy.apply(h, w, targets, None, 4, 0.0, False)
        expected = math.log1p(7.0 * math.exp(-20.0))
        assert float(ce) == pytest.approx(expected, rel=1e-5)

    def test_gradient_zero_sum_and_target_alive(self):
        h, w = _margin_case(margin=20.0)
        targets = torch.tensor([0])
        h = h.clone().requires_grad_(True)
        ce, _ = _ChunkedCrossEntropy.apply(h, w, targets, None, 4, 0.0, False)
        ce.backward()
        # d ce / d logits = grad_h @ w.T (w = I): строка суммируется в 0 точно.
        row_sum = float(h.grad.sum())
        assert abs(row_sum) < 1e-12
        # Целевой член жив (старый путь давал ровно 0):
        assert h.grad[0, 0] < -1e-9

    def test_matches_reference_moderate_margin(self):
        torch.manual_seed(7)
        h = torch.randn(64, 16)
        w = torch.randn(32, 16)
        targets = torch.randint(0, 32, (64,))
        ce, z = _ChunkedCrossEntropy.apply(h, w, targets, None, 16, 0.001, False)
        ref = F.cross_entropy(h @ w.T, targets)
        assert torch.allclose(ce, ref, rtol=1e-5, atol=1e-6)
        lse_ref = torch.logsumexp(h @ w.T, dim=-1)
        assert torch.allclose(z, (lse_ref**2).mean(), rtol=1e-4)

    def test_gradcheck_fp64(self):
        torch.manual_seed(11)
        h = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        w = torch.randn(5, 6, dtype=torch.float64)
        targets = torch.tensor([0, 1, 2, 3])
        assert torch.autograd.gradcheck(
            lambda hh: _ChunkedCrossEntropy.apply(hh, w, targets, None, 3, 0.0, False)[0],
            (h,),
            atol=1e-8,
        )

    def test_bf16_margin_gradient_finite(self):
        h, w = _margin_case(margin=12.0)
        h_bf = h.to(torch.bfloat16).requires_grad_(True)
        w_bf = w.to(torch.bfloat16)
        targets = torch.tensor([0])
        ce, _ = _ChunkedCrossEntropy.apply(h_bf, w_bf, targets, None, 4, 0.0, False)
        ce.backward()
        assert torch.isfinite(h_bf.grad).all()
        # bf16-клифф ~5.5-6.2: margin 12 за клиффом, но целевой градиент жив.
        assert float(h_bf.grad.float().sum().abs()) < 1e-2  # ~0 в bf16-шуме
        assert float(h_bf.grad[0, 0]) < -1e-6


class TestFusedCollapse:
    def _head(self, v: int, h: int, fused: bool) -> LMHead:
        cfg = HeadConfig()
        cfg.fused_ce = fused
        cfg.z_loss_weight = 0.0
        head = LMHead(h, v, cfg, tied_weight=None)
        with torch.no_grad():
            head.projection.weight.zero_()
            head.logit_scale.fill_(1.0)
        return head

    def test_fused_beyond_cliff(self):
        v, h = 8, 8
        head = self._head(v, h, fused=True)
        with torch.no_grad():
            head.projection.weight.copy_(torch.eye(v, h) * 20.0)
        hidden = torch.zeros(1, h)
        hidden[0, 0] = 1.0
        hidden = hidden.requires_grad_(True)
        targets = torch.tensor([0])
        ce, _ = head.loss(hidden, targets)
        expected = math.log1p(7.0 * math.exp(-20.0))
        assert float(ce) == pytest.approx(expected, rel=1e-5)
        ce.backward()
        g_logits = hidden.grad @ head.weight.T  # d/d logits через identity-вес
        assert abs(float(g_logits.sum())) < 1e-12
        assert float(g_logits[0, 0]) < -1e-9

    def test_fused_and_chunked_agree(self):
        torch.manual_seed(3)
        v, h, n = 64, 32, 128
        fused = self._head(v, h, fused=True)
        chunked = self._head(v, h, fused=False)
        with torch.no_grad():
            fused.projection.weight.normal_(0, 0.5)
            chunked.projection.weight.copy_(fused.projection.weight)
        hidden = torch.randn(n, h)
        targets = torch.randint(0, v, (n,))
        ce_f, _ = fused.loss(hidden, targets)
        ce_c, _ = chunked.loss(hidden, targets)
        assert torch.allclose(ce_f, ce_c, rtol=1e-4)


class TestSampledCollapse:
    def test_sampled_beyond_cliff(self):
        v, h = 32, 16
        cfg = HeadConfig()
        cfg.sampled_softmax_k = 8
        head = LMHead(h, v, cfg, tied_weight=None)
        with torch.no_grad():
            head.projection.weight.zero_()
            head.projection.weight[0, 0] = 20.0
            for j in range(1, v):
                head.projection.weight[j, 1] = 1.0
            head.logit_scale.fill_(1.0)
        hidden = torch.zeros(1, h)
        hidden[0, 0] = 1.0
        hidden[0, 1] = 1.0
        targets = torch.tensor([0])
        torch.manual_seed(0)
        ce, _ = head.loss(hidden, targets)
        # lse по негативам >= log(8*e^1): m <= 20 - 1 = 19, но >> клиффа 16.6.
        assert float(ce) > 0.0
        assert float(ce) < 1e-3


class TestSoftTargetKL:
    def test_paper_example_exactness(self):
        # Пример из статьи: z_T = [2,1,0], z_S = z_T + [-1e-7, 1e-7, 0].
        z_t = torch.tensor([[2.0, 1.0, 0.0]])
        z_s = (z_t + torch.tensor([[-1e-7, 1e-7, 0.0]])).requires_grad_(True)
        loss = soft_target_kl(z_s, z_t, reduction="sum")
        loss.backward()
        exact = torch.tensor([[-4.595504e-8, 4.144190e-8, 4.513142e-9]])
        assert torch.allclose(z_s.grad, exact, rtol=1e-3, atol=1e-11)

    def test_zero_gradient_at_equality(self):
        z = torch.randn(5, 16)
        z_s = z.clone().requires_grad_(True)
        loss = soft_target_kl(z_s, z, reduction="sum")
        loss.backward()
        assert float(z_s.grad.abs().max()) == 0.0

    def test_matches_standard_kl_moderate(self):
        torch.manual_seed(5)
        z_s = torch.randn(32, 24)
        z_t = torch.randn(32, 24)
        fixed = soft_target_kl(z_s, z_t, reduction="batchmean")
        ref = F.kl_div(F.log_softmax(z_s, -1), F.softmax(z_t, -1), reduction="batchmean")
        assert torch.allclose(fixed, ref, rtol=1e-5, atol=1e-7)


class TestFixedLogSoftmax:
    def test_matches_reference(self):
        torch.manual_seed(9)
        z = torch.randn(64, 128)
        assert torch.allclose(fixed_log_softmax(z), F.log_softmax(z, -1), rtol=1e-5, atol=1e-6)

    def test_extreme_margin(self):
        z = torch.tensor([[20.0, 0.0, 0.0]])
        lq = fixed_log_softmax(z)
        expected = math.log1p(2.0 * math.exp(-20.0))
        assert float(-lq[0, 0]) == pytest.approx(expected, rel=1e-5)
        assert float(lq[0, 1]) == pytest.approx(-20.0, abs=1e-4)


class TestMarginTelemetry:
    def test_mean_logit_margin(self):
        v, h = 8, 8
        cfg = HeadConfig()
        head = LMHead(h, v, cfg, tied_weight=None)
        with torch.no_grad():
            head.projection.weight.copy_(torch.eye(v, h))
            head.logit_scale.fill_(1.0)
        hidden = torch.zeros(2, h)
        hidden[0, 0] = 1.0  # logits [1,0,...], target 0: m = 1 - log 7
        hidden[1, 1] = 1.0
        targets = torch.tensor([0, 1])
        m = head.mean_logit_margin(hidden, targets)
        assert m == pytest.approx(1.0 - math.log(7.0), abs=1e-4)
