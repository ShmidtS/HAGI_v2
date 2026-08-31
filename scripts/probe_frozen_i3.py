"""Frozen-grid calibration for i3i4: learn group scales s1,s3 [out,ng] and
biases b1,b3 by Adam (100 steps) with the exact differentiable ridge solve
for W2 inside the loss; W13 grid values frozen from GPTQ.

Baseline PTQ (no calibration): 6.34%. Frozen-signs analog gave -2pp on i1i4.
"""
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dsv4_refit_experts import soft_lim, _gptq_groups  # noqa: E402
import dsv4_experts as de  # noqa: E402

L, K = 5, 7
dev = "cuda"
torch.set_grad_enabled(False)
acts = torch.load(f"checkpoints_dsv4/pod_all_tokens/acts_layer{L}.pt", map_location="cpu", weights_only=False)
x_k, y_k = acts[str(K)]
del acts
x = x_k.float().to(dev)
y = y_k.float().to(dev)
E = de.load_expert_file(f"lossless_layers/layers_{L}_ffn.safetensors", f"layers.{L}.ffn", K)
w1, w2, w3 = E["w1"].to(dev), E["w2"].to(dev), E["w3"].to(dev)
P = torch.load(f"dsv4_reduced/layer_{L}/P.pt", map_location=dev).float()
mu = torch.load(f"dsv4_reduced/layer_{L}/mu.pt", map_location=dev).float()
z = (x - mu) @ P
yf = y
w1_rot = w1 @ P
w3_rot = w3 @ P
Hz = (z.T @ z) / z.shape[0]
b1_0 = (mu.reshape(-1) @ w1.T).float()
b3_0 = (mu.reshape(-1) @ w3.T).float()
GS = 128


def gptq_int(W, H, gs=GS, nlev=3, jit=1e-3):
    out_, in_ = W.shape
    ng = in_ // gs
    Wg = W.view(out_, ng, gs)
    sg = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / nlev
    for _ in range(4):
        q = (Wg / sg).round().clamp(-nlev, nlev)
        num = (q * Wg).sum(-1, keepdim=True)
        den = (q * q).sum(-1, keepdim=True).clamp_min(1e-9)
        sg = (num / den).clamp_min(1e-9)
    Wc = W.clone()
    Hf = H.float()
    eye = torch.eye(in_, device=W.device)
    Hi = torch.linalg.cholesky(Hf + jit * Hf.diag().mean() * eye)
    Hi = torch.cholesky_inverse(Hi)
    Hi = torch.linalg.cholesky(Hi + jit * Hi.diag().mean() * eye, upper=True)
    Q = torch.zeros_like(W)
    sg_full = sg.squeeze(-1).repeat_interleave(gs, dim=1)
    for c0 in range(0, in_, 128):
        c1 = min(c0 + 128, in_)
        Wb = Wc[:, c0:c1].clone()
        Sb = sg_full[:, c0:c1]
        Qb = torch.zeros_like(Wb)
        Err = torch.zeros_like(Wb)
        Hb = Hi[c0:c1, c0:c1]
        for j in range(c1 - c0):
            w = Wb[:, j]
            sc = Sb[:, j]
            q = (w / sc).round().clamp(-nlev, nlev)
            Qb[:, j] = q * sc
            Err[:, j] = (w - q * sc) / Hb[j, j]
            if j + 1 < c1 - c0:
                Wb[:, j + 1:] -= torch.outer(Err[:, j], Hb[j, j + 1:])
        Q[:, c0:c1] = Qb
        if c1 < in_:
            Wc[:, c1:] -= Err @ Hi[c0:c1, c1:]
    return Q, sg.squeeze(-1)


q1, s1g = gptq_int(w1_rot, Hz)  # q1: CONTINUOUS dequantized values
q3, s3g = gptq_int(w3_rot, Hz)
grid1 = (q1 / s1g.repeat_interleave(GS, dim=1)).round().clamp(-3, 3)  # int grid
grid3 = (q3 / s3g.repeat_interleave(GS, dim=1)).round().clamp(-3, 3)

torch.set_grad_enabled(True)
s1 = s1g.detach().clone().requires_grad_(True)
s3 = s3g.detach().clone().requires_grad_(True)
b1 = b1_0.clone().requires_grad_(True)
b3 = b3_0.clone().requires_grad_(True)
opt = torch.optim.Adam([s1, s3, b1, b3], lr=3e-3)
n = z.shape[0]
t0 = time.time()
for st in range(100):
    W1q = grid1 * s1.repeat_interleave(GS, dim=1)
    W3q = grid3 * s3.repeat_interleave(GS, dim=1)
    g = soft_lim(z @ W1q.T + b1[None, :])
    u = soft_lim(z @ W3q.T + b3[None, :])
    h = F.silu(g) * u
    Gm = h.T @ h
    Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
    W2c = torch.linalg.solve(Gm, h.T @ yf).T.contiguous()
    yh = h @ W2c.T
    loss = ((yh - yf) ** 2).sum() / (yf ** 2).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
dt = time.time() - t0
torch.set_grad_enabled(False)
with torch.no_grad():
    W1q = grid1 * s1.repeat_interleave(GS, dim=1)
    W3q = grid3 * s3.repeat_interleave(GS, dim=1)
    g = soft_lim(z @ W1q.T + b1[None, :])
    u = soft_lim(z @ W3q.T + b3[None, :])
    h = F.silu(g) * u
    Gm = h.T @ h
    Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
    W2c = torch.linalg.solve(Gm, h.T @ yf).T.contiguous()
    Hh = (h.T @ h) / n
    ng2 = W2c.shape[1] // GS
    Wg2 = W2c.view(-1, ng2, GS)
    sg2 = Wg2.abs().amax(-1, keepdim=True).clamp_min(1e-9) / 7.0
    qg2 = (Wg2 / sg2).round().clamp(-7, 7)
    for _ in range(3):
        num = (qg2 * Wg2).sum(-1, keepdim=True)
        den = (qg2 * qg2).sum(-1, keepdim=True).clamp_min(1e-9)
        sg2 = (num / den).clamp_min(1e-9)
        qg2 = (Wg2 / sg2).round().clamp(-7, 7)
    s2g = sg2.squeeze(-1)
    q2 = _gptq_groups(W2c, Hh, s2g, gs=GS)
    W2q = q2 * s2g.repeat_interleave(GS, dim=1)
    yhq = h @ W2q.T  # single-shot eval (chunk loop diverged from solve h - see debug)
    rq = ((yhq - yf) ** 2).sum() / (yf ** 2).sum()
    rq = rq.item()
print(f"frozen-grid i3i4: continuous {math.sqrt(loss.item())*100:.2f}%  HONEST {math.sqrt(rq)*100:.2f}%  [{dt:.0f}s]  (PTQ baseline 6.34%)", flush=True)
print(f"frozen-grid i3i4: continuous {math.sqrt(loss.item())*100:.2f}%  HONEST {math.sqrt(rq)*100:.2f}%  [{dt:.0f}s]  (PTQ baseline 6.34%)", flush=True)
