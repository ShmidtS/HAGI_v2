"""Untried identity transforms for the W2 leg:
(A) h-space rotation R=eigvec(Cov(h)): W2_rot = W2 @ R, GPTQ there, exact eval.
(B) y-space rotation Q (random orthogonal, QuIP#-style): W2_y = Q @ W2.
(C) both.
Baseline int3-W13 + int4-W2: 6.34%. W2-only ceiling (FP32 W2): ~3.5%?
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
g_ = torch.Generator(device=dev).manual_seed(0)
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
w1_rot, w3_rot = w1 @ P, w3 @ P
Hz = (z.T @ z) / z.shape[0]
b1 = (mu.reshape(-1) @ w1.T).float()
b3 = (mu.reshape(-1) @ w3.T).float()
GS = 128


def gptq_int(W, H, nlev=3, gs=GS, jit=1e-3):
    out_, in_ = W.shape
    ng = in_ // gs
    Wg = W.view(out_, ng, gs)
    sg = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / nlev
    for _ in range(4):
        q = (Wg / sg).round().clamp(-nlev, nlev)
        sg = ((q * Wg).sum(-1, keepdim=True) / (q * q).sum(-1, keepdim=True).clamp_min(1e-9)).clamp_min(1e-9)
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


def build_and_eval(gs13=128, gs2=128):
    globals()['GS'] = gs13
    W1q = gptq_int(w1_rot, Hz)
    W3q = gptq_int(w3_rot, Hz)
    g = soft_lim(z @ W1q.T + b1[None, :])
    u = soft_lim(z @ W3q.T + b3[None, :])
    h = F.silu(g) * u
    Gm = h.T @ h
    Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
    W2c2 = torch.linalg.solve(Gm, h.T @ y).T.contiguous()
    globals()['GS'] = gs2
    W2q2 = eval_w2(W2c2, h)
    globals()['GS'] = 128
    r = ((h @ W2q2.T - y).norm() / y.norm()).item() * 100
    mb = (2*2048*4096*3 + 4096*2048*4) / 8 / 1e6 + (2*2048*(4096//gs13)*2 + 4096*(2048//gs2)*2) / 1e6
    print(f"W13 g{gs13} + W2 g{gs2}: {r:6.2f}%  ~{mb:.1f} MB", flush=True)
    return r

W1q = gptq_int(w1_rot, Hz)
W3q = gptq_int(w3_rot, Hz)
g = soft_lim(z @ W1q.T + b1[None, :])
u = soft_lim(z @ W3q.T + b3[None, :])
h = F.silu(g) * u
Gm = h.T @ h
Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
W2c = torch.linalg.solve(Gm, h.T @ y).T.contiguous()

Covh = (h.T @ h) / h.shape[0]
dg = Covh.diag()
print(f"h-cov diag spread: {(dg.max()/dg.min()).item():.1f}x", flush=True)
ev, R = torch.linalg.eigh(Covh.cpu().to(torch.float64))
R = R.to(dev).float()

# random orthogonal Q (y-space)
A = torch.randn(4096, 4096, generator=g_, device=dev)
Q, _ = torch.linalg.qr(A)


def eval_w2(W2mat, h_use):
    """int4 g128 GPTQ on W2mat with h-Hessian, honest resid using h_use @ W2q.T (then un-rotate outside)."""
    Hh = (h_use.T @ h_use) / h_use.shape[0]
    ng2 = W2mat.shape[1] // GS
    Wg2 = W2mat.view(-1, ng2, GS)
    sg2 = Wg2.abs().amax(-1, keepdim=True).clamp_min(1e-9) / 7.0
    qg2 = (Wg2 / sg2).round().clamp(-7, 7)
    for _ in range(3):
        sg2 = ((qg2 * Wg2).sum(-1, keepdim=True) / (qg2 * qg2).sum(-1, keepdim=True).clamp_min(1e-9)).clamp_min(1e-9)
        qg2 = (Wg2 / sg2).round().clamp(-7, 7)
    s2g = sg2.squeeze(-1)
    q2 = _gptq_groups(W2mat, Hh, s2g, gs=GS)
    return q2 * s2g.repeat_interleave(GS, dim=1)


def resid(y_hat):
    return ((y_hat - y).norm() / y.norm()).item() * 100


# baseline
W2q = eval_w2(W2c, h)
print(f"baseline            : {resid(h @ W2q.T):6.2f}%  (=6.34 expected)", flush=True)

# (A) h-rotation
W2r = W2c @ R
hr = h @ R
W2rq = eval_w2(W2r, hr)
print(f"(A) h-rot eigen     : {resid(hr @ W2rq.T):6.2f}%", flush=True)
Hr = (hr.T @ hr) / hr.shape[0]
print(f"    rotated Hh diag spread: {(Hr.diag().max()/Hr.diag().min()).item():.2f}x", flush=True)

# (B) y-rotation
W2y = Q @ W2c
W2yq = eval_w2(W2y, h)
print(f"(B) y-rot random-orth: {resid((h @ W2yq.T) @ Q.T):6.2f}%", flush=True)

# (C) both
W2b = Q @ (W2c @ R)
W2bq = eval_w2(W2b, hr)
print(f"(C) both            : {resid((hr @ W2bq.T) @ Q.T):6.2f}%", flush=True)

# reference: W2 FP32 (W13-int3 cost only)
print(f"ref W2 FP32         : {resid(h @ W2c.T):6.2f}%", flush=True)

# finer groups exploit locality harder
for gsw in (64, 32):
    GSold = globals()["GS"]
    globals()["GS"] = gsw
    W2qw = eval_w2(W2c, h)
    globals()["GS"] = GSold
    print(f"W2 g{gsw:<4}           : {resid(h @ W2qw.T):6.2f}%", flush=True)


for a, b in ((128, 128), (64, 64), (64, 32), (128, 32), (64, 16)):
    build_and_eval(a, b)
