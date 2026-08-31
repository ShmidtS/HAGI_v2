"""Compare W13 grids on L5 k7 with LS group scales + GPTQ feedback:
- int3 {−3..3} g128 (LEVELS[3] format-compatible) with/without GPTQ
- int3 g64
- 4-level {−2,−1,1,2} g64 GPTQ
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
x, y = acts[str(K)]
x = x.float().to(dev); y = y.float().to(dev)
E = de.load_expert_file(f"lossless_layers/layers_{L}_ffn.safetensors", f"layers.{L}.ffn", K)
w1, w2, w3 = E["w1"].to(dev), E["w2"].to(dev), E["w3"].to(dev)
P = torch.load(f"dsv4_reduced/layer_{L}/P.pt", map_location=dev).float()
mu = torch.load(f"dsv4_reduced/layer_{L}/mu.pt", map_location=dev).float()
z = (x - mu) @ P
w1_rot = w1 @ P.T; w3_rot = w3 @ P.T
Hz = (z.T @ z) / z.shape[0]
b1 = (mu @ w1_rot.T)[0].contiguous(); b3 = (mu @ w3_rot.T)[0].contiguous()
GS = 128


def snap(w, nlev, allow_mid, twolev):
    if twolev:
        q = (w.abs() > 1.5).float() * 2 + (w.abs() > 0).float()  # 0/1/2 magnitudes
        return torch.sign(w) * q
    q = w.round().clamp(-nlev, nlev)
    if not allow_mid:
        q = torch.where(q == 0, torch.sign(w), q)  # no zero allowed
    return q


def gptq_grid(W, H, gs, nlev=3, grid="full", jit=1e-3):
    """grid: full={-3..3}; odd={-3,-1,1,3}; twolev={-2,-1,1,2}"""
    out_, in_ = W.shape
    ng = in_ // gs
    Wg = W.view(out_, ng, gs)
    sg = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / nlev
    for _ in range(4):
        wq = (Wg / sg)
        if grid == "full":
            q = wq.round().clamp(-nlev, nlev)
        elif grid == "odd":
            q = wq.round().clamp(-nlev, nlev)
            q = torch.where(q.abs() == 2, torch.sign(q) * 3, q)
        else:  # twolev: snap to {-2,-1,1,2}
            a = wq.abs()
            q = torch.sign(wq) * torch.where(a > 1.5, torch.tensor(2.0, device=W.device), torch.where(a > 0.5, torch.tensor(1.0, device=W.device), torch.tensor(1.0, device=W.device)))
        num = (q * Wg).sum(-1, keepdim=True)
        den = (q * q).sum(-1, keepdim=True).clamp_min(1e-9)
        sg = (num / den).clamp_min(1e-9)
    Wc = W.clone()
    Hf = H.float(); eye = torch.eye(in_, device=W.device)
    Hi = torch.linalg.cholesky(Hf + jit * Hf.diag().mean() * eye)
    Hi = torch.cholesky_inverse(Hi)
    Hi = torch.linalg.cholesky(Hi + jit * Hi.diag().mean() * eye, upper=True)
    Q = torch.zeros_like(W)
    sg_full = sg.squeeze(-1).repeat_interleave(gs, dim=1)
    for c0 in range(0, in_, 128):
        c1 = min(c0 + 128, in_)
        Wb = Wc[:, c0:c1].clone(); Sb = sg_full[:, c0:c1]
        Qb = torch.zeros_like(Wb); Err = torch.zeros_like(Wb)
        Hb = Hi[c0:c1, c0:c1]
        for j in range(c1 - c0):
            w = Wb[:, j]; sc = Sb[:, j]
            wq = w / sc
            if grid == "full":
                qq = wq.round().clamp(-nlev, nlev)
            elif grid == "odd":
                qq = wq.round().clamp(-nlev, nlev)
                qq = torch.where(qq.abs() == 2, torch.sign(qq) * 3, qq)
            else:
                a = wq.abs()
                m = torch.where(a > 1.5, 2.0, 1.0)
                qq = torch.sign(wq) * m
            Qb[:, j] = qq * sc
            Err[:, j] = (w - qq * sc) / Hb[j, j]
            if j + 1 < c1 - c0:
                Wb[:, j + 1:] -= torch.outer(Err[:, j], Hb[j, j + 1:])
        Q[:, c0:c1] = Qb
        if c1 < in_:
            Wc[:, c1:] -= Err @ Hi[c0:c1, c1:]
    return Q


def cd_only(W, gs, nlev=3, grid="full"):
    out_, in_ = W.shape
    ng = in_ // gs
    Wg = W.view(out_, ng, gs)
    sg = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / nlev
    for _ in range(4):
        wq = (Wg / sg)
        if grid == "odd":
            q = wq.round().clamp(-nlev, nlev); q = torch.where(q.abs() == 2, torch.sign(q) * 3, q)
        else:
            q = wq.round().clamp(-nlev, nlev)
        num = (q * Wg).sum(-1, keepdim=True); den = (q * q).sum(-1, keepdim=True).clamp_min(1e-9)
        sg = (num / den).clamp_min(1e-9)
    return (q * sg).view(out_, in_)


def eval_w13(W1q, W3q):
    g = soft_lim(z @ W1q.T + b1[None, :]); u = soft_lim(z @ W3q.T + b3[None, :])
    h = F.silu(g) * u
    Gm = h.T @ h; Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
    W2c = torch.linalg.solve(Gm, h.T @ y).T.contiguous()
    Hh = (h.T @ h) / h.shape[0]
    ng2 = W2c.shape[1] // GS
    Wg2 = W2c.view(-1, ng2, GS)
    sg2 = Wg2.abs().amax(-1, keepdim=True).clamp_min(1e-9) / 7.0
    qg2 = (Wg2 / sg2).round().clamp(-7, 7)
    for _ in range(3):
        num = (qg2 * Wg2).sum(-1, keepdim=True); den = (qg2 * qg2).sum(-1, keepdim=True).clamp_min(1e-9)
        sg2 = (num / den).clamp_min(1e-9); qg2 = (Wg2 / sg2).round().clamp(-7, 7)
    s2g = sg2.squeeze(-1)
    q2 = _gptq_groups(W2c, Hh, s2g, gs=GS)
    W2q = q2 * s2g.repeat_interleave(GS, dim=1)
    outs = []
    for c0 in range(0, z.shape[0], 4096):
        zc = z[c0:c0 + 4096]
        gg = soft_lim(zc @ W1q.T + b1[None, :]); uu = soft_lim(zc @ W3q.T + b3[None, :])
        outs.append((F.silu(gg) * uu) @ W2q.T)
    yh = torch.cat(outs)
    return ((yh - y).norm() / y.norm()).item() * 100


W1q = gptq_grid(w1_rot, Hz, 128, grid="full"); W3q = gptq_grid(w3_rot, Hz, 128, grid="full")
print(f"int3 full g128 GPTQ   : {eval_w13(W1q, W3q):.2f}%")
W1q = gptq_grid(w1_rot, Hz, 64, grid="full"); W3q = gptq_grid(w3_rot, Hz, 64, grid="full")
print(f"int3 full g64  GPTQ   : {eval_w13(W1q, W3q):.2f}%")
W1q = cd_only(w1_rot, 128, grid="full"); W3q = cd_only(w3_rot, 128, grid="full")
print(f"int3 full g128 CD only: {eval_w13(W1q, W3q):.2f}%")
W1q = gptq_grid(w1_rot, Hz, 64, grid="twolev"); W3q = gptq_grid(w3_rot, Hz, 64, grid="twolev")
print(f"2lev {(-2,-1,1,2)} g64 GPTQ: {eval_w13(W1q, W3q):.2f}%")
