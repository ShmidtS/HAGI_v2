"""Batched frozen-signs calibration (v4): G experts in one pass.

Math identical to calib_frozen_signs (differentiable ridge W2 solve each
step), but batched: bmm GEMMs, batched Cholesky, one Adam over [G, inter]
params, row-minibatch (mb) for gradient steps, per-expert patience.
This is the throughput lever for the full 43-layer refit.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dsv4_refit_experts import _gptq_groups, soft_lim  # noqa: E402


def calib_frozen_batch(pairs, steps=400, lr=3e-3, mb=512, patience=100, verbose=False):
    """pairs: list of dicts {w1_rot, w3_rot, b1, b3, z [n,4096], y [n,4096]}.
    Returns list of res tuples + resid (same format as calib_frozen_signs)."""
    dev = "cuda"
    G = len(pairs)
    Nmax = min(max(p["z"].shape[0] for p in pairs), 2048)
    Z = torch.zeros(G, Nmax, 4096, device=dev)
    Y = torch.zeros(G, Nmax, 4096, device=dev)
    M = torch.zeros(G, Nmax, device=dev)
    for i, p in enumerate(pairs):
        n = min(p["z"].shape[0], Nmax)
        Z[i, :n] = p["z"][:n]
        Y[i, :n] = p["y"][:n]
        M[i, :n] = 1.0
    q1 = torch.stack([torch.where(p["w1_rot"] >= 0, 1.0, -1.0) for p in pairs]).to(torch.bfloat16)  # [G,I,D]
    q3 = torch.stack([torch.where(p["w3_rot"] >= 0, 1.0, -1.0) for p in pairs]).to(torch.bfloat16)
    s1 = torch.stack([p["w1_rot"].abs().mean(1).clamp_min(1e-9) for p in pairs]).detach().clone().requires_grad_(True)
    s3 = torch.stack([p["w3_rot"].abs().mean(1).clamp_min(1e-9) for p in pairs]).detach().clone().requires_grad_(True)
    b1 = torch.stack([p["b1"] for p in pairs]).detach().clone().requires_grad_(True)
    b3 = torch.stack([p["b3"] for p in pairs]).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([s1, s3, b1, b3], lr=lr)
    best = torch.full((G,), float("inf"), device=dev)
    stall = torch.zeros(G, device=dev)
    active = torch.ones(G, dtype=torch.bool, device=dev)
    mbi = min(mb, Nmax)
    for st in range(steps):
        idx = torch.randint(0, Nmax, (mbi,), device=dev)
        m = M[:, idx].unsqueeze(-1)  # [G, mb, 1]
        zb = Z[:, idx]
        yb = Y[:, idx]
        _fp32 = os.environ.get("CB_FP32") == "1"
        if _fp32:
            W1q = (q1.float() * s1.unsqueeze(-1)).transpose(1, 2)  # [G,D,I]
            W3q = (q3.float() * s3.unsqueeze(-1)).transpose(1, 2)
            g = soft_lim(torch.bmm(zb, W1q) + b1.unsqueeze(1))
            u = soft_lim(torch.bmm(zb, W3q) + b3.unsqueeze(1))
        else:
            W1q = (q1 * s1.unsqueeze(-1).to(torch.bfloat16)).transpose(1, 2)  # [G,D,I]
            W3q = (q3 * s3.unsqueeze(-1).to(torch.bfloat16)).transpose(1, 2)
            g = soft_lim(torch.bmm(zb.to(torch.bfloat16), W1q).float() + b1.unsqueeze(1))
            u = soft_lim(torch.bmm(zb.to(torch.bfloat16), W3q).float() + b3.unsqueeze(1))
        h = F.silu(g) * u  # [G, mb, I]
        hm = h * m
        Gm = torch.bmm(hm.transpose(1, 2), h)  # [G, I, I]
        diag = Gm.diagonal(dim1=1, dim2=2)
        reg = diag.mean(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-4) * 1e-2  # [G,1,1]
        Gm = Gm + reg * torch.eye(h.shape[2], device=dev).unsqueeze(0)
        # thin experts: a minibatch may contain none of their rows -> Gm ~ 0
        n_active_rows = m.squeeze(-1).sum(1)  # [G]
        Gm = Gm + (n_active_rows < 2).float().view(G, 1, 1) * torch.eye(h.shape[2], device=dev).unsqueeze(0)
        Lc = torch.linalg.cholesky(Gm)
        rhs = torch.bmm(hm.transpose(1, 2), yb)  # [G, I, D]
        W2 = torch.cholesky_solve(rhs, Lc).transpose(1, 2)  # [G, D, I]
        yh = torch.bmm(h, W2.transpose(1, 2))
        diff2 = ((yh - yb) ** 2).sum(-1) * m.squeeze(-1)
        den = ((yb ** 2).sum(-1) * m.squeeze(-1)).sum(1).clamp_min(1e-12)
        resid = diff2.sum(1) / den
        loss = resid[active].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            better = (resid < best) & active
            best = torch.where(better, resid, best)
            stall = torch.where(better, torch.zeros_like(stall), stall + 1)
            active = active & (stall < patience)
            if not active.any():
                break
        if verbose and (st + 1) % 50 == 0:
            print(f"    step {st+1}: med resid {(best[active].median().item() if active.any() else float('nan'))**0.5*100:.2f}%", flush=True)
    # final: per-expert exact W2 on FULL rows + GPTQ int4 (cheap, single loop)
    results = []
    with torch.no_grad():
        s1d, s3d, b1d, b3d = s1.detach(), s3.detach(), b1.detach(), b3.detach()
        for i, p in enumerate(pairs):
            n = p["z"].shape[0]
            zz = p["z"][:Nmax]
            g = soft_lim(zz @ (q1[i].float() * s1d[i][:, None]).T + b1d[i][None, :])
            u = soft_lim(zz @ (q3[i].float() * s3d[i][:, None]).T + b3d[i][None, :])
            h = F.silu(g) * u
            hf = h.float()
            y = p["y"][:Nmax]
            Gm = hf.T @ hf
            Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
            W2c = torch.linalg.solve(Gm, hf.T @ y).T.contiguous()
            ng = W2c.shape[1] // 128
            Wg = W2c.view(-1, ng, 128)
            sg_ = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / 7.0
            qg = (Wg / sg_).round().clamp(-7, 7)
            for _ in range(3):
                num = (qg * Wg).sum(-1, keepdim=True)
                den2 = (qg * qg).sum(-1, keepdim=True).clamp_min(1e-9)
                sg_ = (num / den2).clamp_min(1e-9)
                qg = (Wg / sg_).round().clamp(-7, 7)
            s2g = sg_.squeeze(-1)
            Hh = (hf.T @ hf) / max(n, 1)
            try:
                q2 = _gptq_groups(W2c, Hh, s2g, gs=128)
            except RuntimeError:
                q2 = qg.view(-1, W2c.shape[1])
            W2q = q2 * s2g.repeat_interleave(128, dim=1)
            yhq = (F.silu(soft_lim(zz @ (q1[i].float() * s1d[i][:, None]).T + b1d[i][None, :]))
                   * soft_lim(zz @ (q3[i].float() * s3d[i][:, None]).T + b3d[i][None, :])).float() @ W2q.T
            resid = (((yhq - y) ** 2).sum() / (y ** 2).sum()).item()
            q1f = torch.where(p["w1_rot"] >= 0, 1.0, -1.0)
            q3f = torch.where(p["w3_rot"] >= 0, 1.0, -1.0)
            results.append(((q1f, s1d[i].clone(), q3f, s3d[i].clone(), q2, s2g, b1d[i].clone(), b3d[i].clone(), [0, 4096]), resid))
    return results
