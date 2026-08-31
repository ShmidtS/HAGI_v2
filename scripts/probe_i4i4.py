"""Honest i4i4 probe: W13 int4 g128 (GPTQ over z-Hessian) + W2 int4 g128
(GPTQ over h-Hessian), bias compensation, soft_lim - the full decode math.

Baseline (current recipe A: sign-W13 + GPTQ-W2) = 16.32% on L5 k7.
Ceiling reference: FP32-W13 + int4-W2 = 5.67%.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dsv4_refit_experts import soft_lim, _gptq_groups  # noqa: E402

L = int(os.environ.get("L", "5"))
K = int(os.environ.get("K", "7"))
POD = "checkpoints_dsv4/pod_all_tokens"
RED = f"dsv4_reduced/layer_{L}"
GS = 128

torch.set_grad_enabled(False)
dev = "cuda"

acts = torch.load(os.path.join(POD, f"acts_layer{L}.pt"), map_location="cpu", weights_only=False)
x_k, y_k = acts[str(K)]
del acts
x = x_k.float().to(dev)
y = y_k.float().to(dev)

import dsv4_experts as de  # noqa: E402

E = de.load_expert_file(os.path.join("lossless_layers", f"layers_{L}_ffn.safetensors"), f"layers.{L}.ffn", K)
w1, w2, w3 = E["w1"].to(dev), E["w2"].to(dev), E["w3"].to(dev)

P = torch.load(os.path.join(RED, "P.pt"), map_location=dev).float()
mu = torch.load(os.path.join(RED, "mu.pt"), map_location=dev).float()
z = (x - mu) @ P
w1_rot = w1 @ P.T
w3_rot = w3 @ P.T
Hz = (z.T @ z) / z.shape[0]

b1 = (mu @ w1_rot.T)[0].contiguous()
b3 = (mu @ w3_rot.T)[0].contiguous()


def norm_resid(yh):
    return ((yh - y).norm() / y.norm()).item() * 100


def full_forward(W1q, W3q, W2q):
    outs = []
    for c0 in range(0, z.shape[0], 4096):
        zc = z[c0:c0 + 4096]
        g = soft_lim(zc @ W1q.T + b1[None, :])
        u = soft_lim(zc @ W3q.T + b3[None, :])
        outs.append((F.silu(g) * u) @ W2q.T)
    return torch.cat(outs)


def gptq_int(W, H, gs=GS, nlev=7, act_order=False, jit=1e-3):
    """Groupwise int GPTQ: per-row per-group scales, LDLQ feedback."""
    out_, in_ = W.shape
    ng = in_ // gs
    Wg = W.view(out_, ng, gs)
    sg = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / nlev
    for _ in range(3):
        q = (Wg / sg).round().clamp(-nlev, nlev)
        num = (q * Wg).sum(-1, keepdim=True)
        den = (q * q).sum(-1, keepdim=True).clamp_min(1e-9)
        sg = (num / den).clamp_min(1e-9)
    Wc = W.clone()
    perm = None
    if act_order:
        perm = torch.argsort(H.diag(), descending=True)
        Wc = Wc[:, perm]
        H = H[perm][:, perm]
    Hf = H.float()
    eye = torch.eye(in_, device=W.device)
    Hi = torch.linalg.cholesky(Hf + jit * Hf.diag().mean() * eye)
    Hi = torch.cholesky_inverse(Hi)
    Hi = torch.linalg.cholesky(Hi + jit * Hi.diag().mean() * eye, upper=True)
    Q = torch.zeros_like(W)
    block = 128
    # per-row scale lookup: group of column j (in permuted order)
    sg_full_rows = sg.squeeze(-1).repeat_interleave(gs, dim=1)  # [out_, in_]
    for c0 in range(0, in_, block):
        c1 = min(c0 + block, in_)
        Wb = Wc[:, c0:c1].clone()
        Sb = sg_full_rows[:, c0:c1]
        Qb = torch.zeros_like(Wb)
        Err = torch.zeros_like(Wb)
        Hb = Hi[c0:c1, c0:c1]
        for j in range(c1 - c0):
            w = Wb[:, j]
            s = Sb[:, j]
            q = (w / s).round().clamp(-nlev, nlev)
            Qb[:, j] = q * s
            Err[:, j] = (w - q * s) / Hb[j, j]
            if j + 1 < c1 - 0 - 1:
                Wb[:, j + 1:] -= torch.outer(Err[:, j], Hb[j, j + 1:])
        Q[:, c0:c1] = Qb
        if c1 < in_:
            Wc[:, c1:] -= Err @ Hi[c0:c1, c1:]
    if act_order:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(in_, device=W.device)
        Q = Q[:, inv]
    return Q


# --- variants ---
# A) current: sign-W13 + LS row scales + GPTQ-W2 (reference re-measure)
q1 = torch.where(w1_rot >= 0, 1.0, -1.0)
q3 = torch.where(w3_rot >= 0, 1.0, -1.0)
s1 = w1_rot.abs().mean(dim=1).clamp_min(1e-9)
s3 = w3_rot.abs().mean(dim=1).clamp_min(1e-9)
g = soft_lim(z @ (q1 * s1[:, None]).T + b1[None, :])
u = soft_lim(z @ (q3 * s3[:, None]).T + b3[None, :])
h = F.silu(g) * u
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
W2q_A = q2 * s2g.repeat_interleave(GS, dim=1)
print(f"A  sign-W13 + GPTQ-W2 (current)      {norm_resid(full_forward(q1 * s1[:, None], q3 * s3[:, None], W2q_A)):6.2f}%", flush=True)

# B) i4i4: int4-W13 GPTQ g128 + int4-W2 GPTQ g128
for ao in (False, True):
    W1q = gptq_int(w1_rot, Hz, act_order=ao)
    W3q = gptq_int(w3_rot, Hz, act_order=ao)
    g = soft_lim(z @ W1q.T + b1[None, :])
    u = soft_lim(z @ W3q.T + b3[None, :])
    h = F.silu(g) * u
    Gm = h.T @ h
    Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
    W2c = torch.linalg.solve(Gm, h.T @ y).T.contiguous()
    Hh = (h.T @ h) / h.shape[0]
    q2 = _gptq_groups(W2c, Hh, s2g, gs=GS)  # scales from this W2c variant
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
    tag = "actord" if ao else "plain"
    print(f"B  i4i4 GPTQ-W13 g128 ({tag}) + W2   {norm_resid(full_forward(W1q, W3q, W2q)):6.2f}%", flush=True)

# D) int2/int3 W13 (same gptq_int, lower levels)
for nlev_i, gs_i in [(1, 64), (1, 128), (2, 64), (3, 128)]:
    W1q = gptq_int(w1_rot, Hz, gs=gs_i, nlev=nlev_i)
    W3q = gptq_int(w3_rot, Hz, gs=gs_i, nlev=nlev_i)
    g = soft_lim(z @ W1q.T + b1[None, :])
    u = soft_lim(z @ W3q.T + b3[None, :])
    h = F.silu(g) * u
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
    bpw = (3 * (nlev_i.bit_length() + 1) * 8 - 5) // 8  # rough
    print(f"D  int{['','1','2','3'][nlev_i]}-W13 g{gs_i} + int4 W2        {norm_resid(full_forward(W1q, W3q, W2q)):6.2f}%", flush=True)

# C) size accounting
mb = (2 * 2048 * 4096 * 0.5 + 4096 * 2048 * 0.5) / 1e6
print(f"i4i4 size ~{mb:.1f} MB/expert (int4 all three matrices, g128 fp16 scales)", flush=True)
