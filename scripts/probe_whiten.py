"""Partial whitening probe: z = (x-mu) @ P @ diag(lam^{-beta}) / g.

beta=0   -> plain eigen rotation (current recipe)
beta=0.5 -> full whitening: z-Hessian exactly I
lam from the calibration rows themselves (Var of rotated coords).
Weights absorb the scaling: W~ = W @ P @ diag(lam^{beta}) * g.
Group scales g128 adapt per group. int3 W13 GPTQ + int4 W2 GPTQ.

L5 k7 baseline (beta=0): 6.34%.
"""
import os
import sys

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

z0 = (x - mu) @ P  # rotated coords
lam = z0.var(dim=0).clamp_min(1e-12)  # eigenvalues of covariance
GS = 128


def run(beta, nlev13=3):
    s_diag = lam.pow(-beta)
    g = s_diag.log().mean().exp()  # geometric mean keeps overall scale
    z = z0 * s_diag[None, :] / g
    w1_rot = (w1 @ P) * (s_diag / g)[None, :]
    w3_rot = (w3 @ P) * (s_diag / g)[None, :]
    b1 = (mu.reshape(-1) @ w1.T).float()
    b3 = (mu.reshape(-1) @ w3.T).float()
    Hz = (z.T @ z) / z.shape[0]

    def gptq_int(W, H, gs=GS, nlev=nlev13, jit=1e-3):
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
        return Q

    W1q = gptq_int(w1_rot, Hz)
    W3q = gptq_int(w3_rot, Hz)
    g_ = soft_lim(z @ W1q.T + b1[None, :])
    u = soft_lim(z @ W3q.T + b3[None, :])
    h = F.silu(g_) * u
    Gm = h.T @ h
    Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
    W2c = torch.linalg.solve(Gm, h.T @ y).T.contiguous()
    Hh = (h.T @ h) / h.shape[0]
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
    outs = []
    for c0 in range(0, z.shape[0], 4096):
        zc = z[c0:c0 + 4096]
        gg = soft_lim(zc @ W1q.T + b1[None, :])
        uu = soft_lim(zc @ W3q.T + b3[None, :])
        outs.append((F.silu(gg) * uu) @ W2q.T)
    yh = torch.cat(outs)
    r = ((yh - y).norm() / y.norm()).item() * 100
    # condition of the z-Hessian (isotropy check)
    cond = (Hz.diag().max() / Hz.diag().min()).item()
    print(f"beta={beta:.2f}: resid {r:6.2f}%   z-H diag spread {cond:8.1f}x", flush=True)


for beta in (0.0, 0.2, 0.35, 0.45, 0.5):
    run(beta)
