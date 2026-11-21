"""b1.58 per-row ternary quantization (Energy/STEStep.lean contracts)."""
import torch
from hagi.lm import CertifiedStepLM


def test_ternary_level_scale_invariance():
    lm = CertifiedStepLM(k=4, dim=16, hidden=32, seed=1)
    W = lm.W1.detach().clone()
    lm.W1.data.copy_(W); lm.ternary_()
    q1 = lm.W1.detach().clone()
    s1 = W.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    lm.W1.data.copy_(W * 3.7); lm.ternary_()
    q2 = lm.W1.detach().clone()
    s2 = (W * 3.7).abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    # ternary_scale_invariance: Q(cW) == Q(W) as LEVELS
    assert torch.allclose(q1 / s1, q2 / s2, atol=1e-6)


def test_ternary_distortion_bound():
    # per-weight error <= s (clamp residues), <= s/2 in non-saturated mass
    lm = CertifiedStepLM(k=4, dim=16, hidden=32, seed=1)
    W0 = lm.W1.detach().clone()      # master, pre-quantization
    lm.ternary_()
    q = lm.W1.detach().clone()
    s = W0.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    err = (q - W0).abs()
    ratio = (W0 / s).abs()
    # exact residue profile (qTern_error_in / qTern_error_out):
    #   r < 0.5  -> err == r·s (rounds to 0)
    #   0.5<=r<=1.5 -> err <= s/2 (nearest grid)
    #   r > 1.5  -> err == (r−1)·s (clamp to ±s); qTern_pointwise_le:
    #   err <= |x| always
    small = ratio < 0.5
    mid = (ratio >= 0.5) & (ratio <= 1.5)
    sat = ratio > 1.5
    assert torch.allclose(err[small], ratio[small] * s.expand_as(W0)[small], atol=1e-5)
    assert float((err[mid] / s.expand_as(W0)[mid]).max()) <= 0.5 + 1e-6
    if sat.any():
        expect = (ratio[sat] - 1.0) * s.expand_as(W0)[sat]
        assert torch.allclose(err[sat], expect, atol=1e-5)
    assert float((err / W0.abs().clamp_min(1e-12)).max()) <= 1.0 + 1e-6


def test_ternary_levels_ternary():
    lm = CertifiedStepLM(k=4, dim=16, hidden=32, seed=1)
    W0s = {n: getattr(lm, n).detach().clone() for n in ("emb", "W1", "W2")}
    lm.ternary_()
    for name in ("emb", "W1", "W2"):
        q = getattr(lm, name)
        s = W0s[name].abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
        lvls = torch.round(q / s)
        assert float(lvls.abs().max()) <= 1.0 + 1e-6  # tern() grid only


def test_ternary_levels_stable_under_requantization():
    # repeated quantization keeps the LEVEL grid {-1,0,+1} (s may move,
    # but normalized levels stay ternary — gridQ_idem is for fixed-Δ)
    lm = CertifiedStepLM(k=4, dim=16, hidden=32, seed=1)
    W0 = lm.W1.detach().clone()
    lm.ternary_()
    q1 = lm.W1.detach().clone()
    s1 = W0.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    lm.ternary_()
    q2 = lm.W1.detach().clone()
    s2 = q1.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    assert torch.allclose(torch.round(q1 / s1), torch.round(q2 / s2))
    # energy does not grow: |q2| <= |q1| rowwise RMS (levels shrink-or-equal)
    assert float(q2.abs().mean()) <= float(q1.abs().mean()) + 1e-6
