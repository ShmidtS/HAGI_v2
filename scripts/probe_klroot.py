"""Probe: KL-Root-Kron multiplicative inverse-root tracking replaces the exact
ridge solve inside frozen-signs calibration.

Report math (KL_Root_Kron_Report_EN): maintain P ~ H^{-1/2} via
  C = P H P  (whitened covariance, H = h^T h / n)
  P <- P (I + eta/2 (I - C))     [no inverse / no eigendecomposition]
Then W2 = (1/n) P (P (h^T y))  is the ridge solution with H^{-1} = P P.

Gradient flows through h in BOTH h^T y and the output GEMM (P detached but
tracking) - a cheap approximation of the implicit gradient that made the
exact differentiable solve valuable (8.89% continuous vs 14.5% detached).

Compare on L5 k7 (heavy expert):
  exact diff-solve : continuous 8.89%, HONEST GPTQ 14.28%, 250 s
  target           : ~14% at 10x+ speedup
"""
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stub_import_tf  # noqa: F401
import dsv4_experts as de
from dsv4_refit_experts import _gptq_groups, soft_lim

L, K = 5, 7
dev = "cuda"
torch.set_grad_enabled(False)
P_rot = torch.load(f"dsv4_reduced/layer_{L}/P.pt", map_location=dev).float()
mu = torch.load(f"dsv4_reduced/layer_{L}/mu.pt", map_location=dev).float()
acts = torch.load("checkpoints_dsv4/pod_all_tokens/acts_layer5.pt", map_location="cpu", weights_only=False)
x, y = acts["7"]
z = ((x.float().to(dev) - mu) @ P_rot)
yf = y.float().to(dev)
experts = de.load_selected_experts(L, [K])
w1, w2, w3 = experts[K]
w1_rot = w1 @ P_rot
w3_rot = w3 @ P_rot
b1_0 = (mu.reshape(-1) @ w1.T).float()
b3_0 = (mu.reshape(-1) @ w3.T).float()

GS = 128


def run(eta=0.1, steps=400, lr=3e-3, label="klroot"):
    torch.set_grad_enabled(True)
    q1 = torch.where(w1_rot >= 0, 1.0, -1.0)
    q3 = torch.where(w3_rot >= 0, 1.0, -1.0)
    s1 = w1_rot.abs().mean(dim=1).clamp_min(1e-9).detach().clone().requires_grad_(True)
    s3 = w3_rot.abs().mean(dim=1).clamp_min(1e-9).detach().clone().requires_grad_(True)
    b1 = b1_0.clone().requires_grad_(True)
    b3 = b3_0.clone().requires_grad_(True)
    opt = torch.optim.Adam([s1, s3, b1, b3], lr=lr)
    n = z.shape[0]
    eye = torch.eye(z.shape[1] if False else 2048, device=dev)
    Pk = eye.clone()
    t0 = time.time()
    for st in range(steps):
        g = soft_lim(z @ (q1 * s1[:, None]).T + b1[None, :])
        u = soft_lim(z @ (q3 * s3[:, None]).T + b3[None, :])
        h = F.silu(g) * u
        with torch.no_grad():
            Hb = h.T @ h / n
            C = Pk @ Hb @ Pk
            if not torch.isfinite(C).all():
                print("    [C non-finite] abort", flush=True)
                break
            # adaptive step: keep the multiplicative update stable when the
            # whitened covariance is far from I (PSGD-style normalization)
            lam = (C.diagonal() ** 2).sum().sqrt().clamp_min(1e-9)  # ~spectral scale
            eta_i = min(eta, 1.0 / lam.item())
            Pk += (eta_i / 2.0) * (eye - C) @ Pk
        W2 = ((Pk @ (Pk @ (h.T @ yf))) / n).T
        yh = h @ W2.T
        loss = ((yh - yf) ** 2).sum() / (yf ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    dt = time.time() - t0
    torch.set_grad_enabled(False)
    with torch.no_grad():
        # final exact solve on the trained scales (report W2 honestly quantized)
        g = soft_lim(z @ (q1 * s1[:, None]).T + b1[None, :])
        u = soft_lim(z @ (q3 * s3[:, None]).T + b3[None, :])
        h = F.silu(g) * u
        Gm = h.T @ h
        print(f"    diag: mean {Gm.diagonal().mean().item():.3e} min {Gm.diagonal().min().item():.3e} | loss {loss.item():.4f}", flush=True)
        Gm.diagonal().add_(Gm.diagonal().mean().clamp_min(1e-6) * 1e-2)
        W2c = torch.linalg.solve(Gm, h.T @ yf).T.contiguous()
        ng = W2c.shape[1] // GS
        Wg = W2c.view(-1, ng, GS)
        sg_ = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / 7.0
        qg = (Wg / sg_).round().clamp(-7, 7)
        for _ in range(3):
            num = (qg * Wg).sum(-1, keepdim=True)
            den = (qg * qg).sum(-1, keepdim=True).clamp_min(1e-9)
            sg_ = (num / den).clamp_min(1e-9)
            qg = (Wg / sg_).round().clamp(-7, 7)
        s2g = sg_.squeeze(-1)
        Hh = (h.T @ h) / n
        q2 = _gptq_groups(W2c, Hh, s2g, gs=GS)
        yhq = h @ (q2 * s2g.repeat_interleave(GS, dim=1)).T
        rq = (((yhq - yf) ** 2).sum() / (yf ** 2).sum()).item()
    print(f"{label:24s} eta={eta}: continuous {math.sqrt(loss.item())*100:.2f}%  HONEST {math.sqrt(rq)*100:.2f}%  [{dt:.0f}s]", flush=True)


run(eta=0.05)
run(eta=0.2)
run(eta=0.5)
