"""Stage [4b]/[5b]/R245 instrumentation tests.

Pins in code:
* TokenWeightBias.lean ``token_weight_decomposition`` — the exact identity
  (finite sums, no asymptotics) on synthetic (T, g) with Cov != 0;
* OptimizerStage.lean [5b] consolidation reset — Adam first moments are
  cleared, second moments and Muon state untouched;
* DisagreementChain.lean ``alignment_factor_le_one`` — the sign-flip case:
  raw latent disagreement large, aligned small, alpha_align < 1.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from hagi.train.loop import TokenWeightBiasMeter, token_weight_decomposition
from hagi.train.optim import reset_adam_first_moments

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "growth" / "measure_disagreement_chain.py"
_spec = importlib.util.spec_from_file_location("measure_disagreement_chain", _SCRIPT)
mdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mdc)


def test_token_weight_decomposition_exact_identity():
    # Lean Hagi.token_weight_decomposition: (sum T_i g_i)/(sum T_i) =
    # mean + Cov(T,g)/Tbar — exact, 1e-12.
    samples = [(100.0, 0.30), (900.0, 0.42), (500.0, 0.35), (1500.0, 0.51)]
    res = token_weight_decomposition(samples)
    n = len(samples)
    total_t = sum(t for t, _ in samples)
    lhs = sum(t * g for t, g in samples) / total_t
    assert res["weighted_mean"] == pytest.approx(lhs, abs=1e-12)
    assert res["mean"] == pytest.approx(sum(g for _, g in samples) / n, abs=1e-12)
    tbar = total_t / n
    gbar = res["mean"]
    cov = sum((t - tbar) * (g - gbar) for t, g in samples) / n
    assert cov != 0.0  # correlated family: the bias is live
    assert res["cov_Tg_over_Tbar"] == pytest.approx(cov / tbar, abs=1e-12)
    # the identity itself
    assert res["weighted_mean"] == pytest.approx(
        res["mean"] + res["cov_Tg_over_Tbar"], abs=1e-12
    )


def test_token_weight_decomposition_uncorrelated_zero_bias():
    # zero_bias_iff_uncorrelated: construct g exactly orthogonal to T's
    # deviations, so the bias term is exactly zero.
    t = torch.tensor([100.0, 900.0, 500.0, 1500.0])
    g = torch.tensor([0.30, 0.50, 0.10, 0.62])
    dev_t = t - t.mean()
    dev_g = g - g.mean()
    # remove the component of dev_g along dev_t (Gram-Schmidt, fp64)
    dev_g = dev_g.double() - (dev_t.double() @ dev_g.double()) / (
        dev_t.double() @ dev_t.double()
    ) * dev_t.double()
    g = (g.mean() + dev_g).tolist()
    samples = list(zip(t.tolist(), g))
    res = token_weight_decomposition(samples)
    assert res["cov_Tg_over_Tbar"] == pytest.approx(0.0, abs=1e-12)
    assert res["weighted_mean"] == pytest.approx(res["mean"], abs=1e-12)


def test_token_weight_bias_meter_accumulates_and_resets():
    m = TokenWeightBiasMeter()
    m.add("math", 1000, 1.0)
    m.add("math", 2000, 3.0)
    m.add("code", 500, 2.0)
    res = m.take()
    assert res["n_domains"] == 2
    # math: T=3000, g=mean(1,3)=2; code: T=500, g=2 -> g constant => cov 0
    assert res["cov_Tg_over_Tbar"] == pytest.approx(0.0, abs=1e-12)
    assert m.take() == {}  # window consumed


class _FakeAdam(torch.optim.Optimizer):
    """Bare optimizer with Adam-style state for the reset test."""

    def __init__(self, params):
        super().__init__(params, defaults={})
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {
                    "exp_avg": torch.randn_like(p),
                    "exp_avg_sq": torch.rand_like(p) + 0.5,
                }

    @torch.no_grad()
    def step(self, closure=None):  # pragma: no cover - never stepped
        return None


def test_consolidation_reset_clears_first_moments_only():
    w = torch.nn.Parameter(torch.randn(4, 4))
    opt = _FakeAdam([w])
    sq_before = opt.state[w]["exp_avg_sq"].clone()
    assert opt.state[w]["exp_avg"].abs().sum() > 0
    cleared = reset_adam_first_moments(opt)
    assert cleared == 1
    assert float(opt.state[w]["exp_avg"].abs().sum()) == 0.0
    # exp_avg_sq untouched (beta1 reset is not a fresh optimizer)
    assert torch.equal(opt.state[w]["exp_avg_sq"], sq_before)
    # idempotent on empty state
    opt.state[w] = {}
    assert reset_adam_first_moments(opt) == 0


def test_consolidation_segments_config_default_off():
    from hagi.config import Config

    assert Config().train.consolidation_segments is False


def test_alpha_align_sign_flip_below_one():
    # alignment_factor_le_one: the same latent content expressed with
    # permuted/flipped factor columns reads as FULL raw disagreement
    # (raw counts every mismatched column at full energy), while the
    # Procrustes-aligned reading removes it exactly: alpha_align ~ 0.
    # (A bare sign flip is canonicalized away by SVD itself; a column
    # permutation with distinct spectra is the observable form.)
    torch.manual_seed(7)
    u, _ = torch.linalg.qr(torch.randn(64, 8))
    s = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.4], dtype=torch.float64)
    a = u.double() @ torch.diag(s)
    b = u.double()[:, torch.arange(7, -1, -1)] @ torch.diag(s)
    res = mdc.alpha_align(a, b, rank=8)
    assert res["E_raw"] > 0.0
    assert res["E_aligned"] < res["E_raw"]
    # alignment_factor_le_one: the aligned reading strictly compresses the
    # measured latent disagreement (0 <= alpha < 1) — raw overestimates.
    assert 0.0 <= res["alpha_align"] < 0.95


def test_alpha_align_identical_is_one():
    torch.manual_seed(11)
    a = torch.randn(32, 16)
    res = mdc.alpha_align(a, a.clone(), rank=4)
    assert res["alpha_align"] == pytest.approx(1.0, abs=1e-9)


def test_alpha_kept_rank_deficient_keeps_everything():
    torch.manual_seed(13)
    u, _ = torch.linalg.qr(torch.randn(20, 2))
    d = u * torch.tensor([3.0, 1.0])  # exactly rank 2
    res = mdc.alpha_kept(d, energy=0.95)
    assert res["alpha_kept"] == pytest.approx(1.0, abs=1e-9)
    assert res["rank_kept"] == 2


def test_alpha_kept_spread_disagreement_truncates():
    torch.manual_seed(17)
    d = torch.randn(32, 32) * 0.1  # near-isotropic spectrum
    res = mdc.alpha_kept(d, energy=0.95)
    assert 0.0 < res["alpha_kept"] <= 1.0 + 1e-9
    assert res["alpha_kept"] >= 0.95 - 1e-9  # k chosen to reach the target


def test_verdict_deficit_localizes():
    # deficit_localizes: one alpha far below the geometric share is named.
    v = mdc.verdict(
        {"alpha_align": 0.9, "alpha_kept": 0.95, "alpha_safe": 0.01, "alpha_cap": 0.9},
        gamma_req=0.018,
        gamma_meas=0.002,
    )
    assert v["below_share"] == ["alpha_safe"]
