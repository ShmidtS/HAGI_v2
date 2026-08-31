"""Ternary 3:4 structured W13 ({-a,0,+a}, 3-of-4 nonzero, Hy4 regime) +
int4-W2 GPTQ - the compression candidate (~8.5 MB/expert = 1.5x vs FP4).

Mask: per group of 4 input-columns keep 3 largest |w| (decided on original
W; LDLQ error feedback propagates but mask stays fixed).
Scale: per-row LS  a = mean|w_nonzero|  (also try group-g32 scale).
W2: same honest GPTQ int4 g128 pipeline as probe_i4i4.
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


def gptq_tern34(W, H, a_g=None, scale_group=0, jit=1e-3):
    """Ternary 3:4 GPTQ. a_g=None, scale_group=0 -> per-row scale; a_g=[out,ng] -> per-g."""
    out_, in_ = W.shape
    G = 4
    ng4 = in_ // G
    Wg = W.view(out_, ng4, G).abs()
    # fixed mask: keep 3 largest |w| per group of 4
    keep = Wg.argsort(dim=-1, descending=True)[..., :3]  # [out, ng4, 3]
    mask = torch.zeros(out_, ng4, G, device=W.device, dtype=torch.bool)
    mask.scatter_(-1, keep, True)
    mask = mask.view(out_, in_)

    Wc = W.clone()
    Hf = H.float()
    eye = torch.eye(in_, device=W.device)
    Hi = torch.linalg.cholesky(Hf + jit * Hf.diag().mean() * eye)
    Hi = torch.cholesky_inverse(Hi)
    Hi = torch.linalg.cholesky(Hi + jit * Hi.diag().mean() * eye, upper=True)
    Q = torch.zeros_like(W)
    block = 128
    for c0 in range(0, in_, block):
        c1 = min(c0 + block, in_)
        Wb = Wc[:, c0:c1].clone()
        mb = mask[:, c0:c1]
        Qb = torch.zeros_like(Wb)
        Err = torch.zeros_like(Wb)
        Hb = Hi[c0:c1, c0:c1]
        for j in range(c1 - c0):
            w = Wb[:, j]
            m = mb[:, j]
            if scale_group == 0:
                # per-row scale from the ORIGINAL row nonzeros (stable target)
                a = (W.abs() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            else:
                gg = (c0 + j) // scale_group
                a = a_g[:, gg]
            q = torch.zeros_like(w)
            q[m] = torch.where(w[m] >= 0, a[m], -a[m])
            Qb[:, j] = q
            Err[:, j] = (w - q) / Hb[j, j]
            if j + 1 < c1 - c0:
                Wb[:, j + 1:] -= torch.outer(Err[:, j], Hb[j, j + 1:])
        Q[:, c0:c1] = Qb
        if c1 < in_:
            Wc[:, c1:] -= Err @ Hi[c0:c1, c1:]
    return Q


def w2_pipeline(W1q, W3q):
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
    return q2 * s2g.repeat_interleave(GS, dim=1)


# global per-group scales variant (scale_group=32) needs a_g precomputed
def build_group_scales(W, sg_size=32):
    out_, in_ = W.shape
    ng = in_ // sg_size
    Wg = (W.view(out_, ng, sg_size).abs().sum(dim=2) / (sg_size * 3 / 4)).clamp_min(1e-9)
    return Wg  # [out, ng]


W1q = gptq_tern34(w1_rot, Hz, scale_group=0)
W3q = gptq_tern34(w3_rot, Hz, scale_group=0)
W2q = w2_pipeline(W1q, W3q)
print(f"T1 tern3:4 W13 (row scale) + int4 W2  {norm_resid(full_forward(W1q, W3q, W2q)):6.2f}%", flush=True)

# v2: g32 group scales
a_g1 = build_group_scales(w1_rot, 32)
a_g3 = build_group_scales(w3_rot, 32)
W1q = gptq_tern34(w1_rot, Hz, a_g=a_g1, scale_group=32)
W3q = gptq_tern34(w3_rot, Hz, a_g=a_g3, scale_group=32)
W2q = w2_pipeline(W1q, W3q)
print(f"T2 tern3:4 W13 (g32 scale)  + int4 W2  {norm_resid(full_forward(W1q, W3q, W2q)):6.2f}%", flush=True)

# v3: dense ternary (no 3:4 mask, all nonzero) per-row scale - reference



def gptq_tern_dense(W, H, jit=1e-3):
    out_, in_ = W.shape
    a = W.abs().sum(dim=1) / in_
    Wc = W.clone()
    Hf = H.float()
    eye = torch.eye(in_, device=W.device)
    Hi = torch.linalg.cholesky(Hf + jit * Hf.diag().mean() * eye)
    Hi = torch.cholesky_inverse(Hi)
    Hi = torch.linalg.cholesky(Hi + jit * Hi.diag().mean() * eye, upper=True)
    Q = torch.zeros_like(W)
    block = 128
    for c0 in range(0, in_, block):
        c1 = min(c0 + block, in_)
        Wb = Wc[:, c0:c1].clone()
        Qb = torch.zeros_like(Wb)
        Err = torch.zeros_like(Wb)
        Hb = Hi[c0:c1, c0:c1]
        for j in range(c1 - c0):
            w = Wb[:, j]
            q = torch.where(w >= 0, a, -a)
            Qb[:, j] = q
            Err[:, j] = (w - q) / Hb[j, j]
            if j + 1 < c1 - c0:
                Wb[:, j + 1:] -= torch.outer(Err[:, j], Hb[j, j + 1:])
        Q[:, c0:c1] = Qb
        if c1 < in_:
            Wc[:, c1:] -= Err @ Hi[c0:c1, c1:]
    return Q


W1q = gptq_tern_dense(w1_rot, Hz)
W3q = gptq_tern_dense(w3_rot, Hz)
W2q = w2_pipeline(W1q, W3q)
print(f"T3 tern dense W13 (row scale) + int4 W2 {norm_resid(full_forward(W1q, W3q, W2q)):6.2f}%", flush=True)
print("sizes: tern3:4 W13 ~4.2MB(2bit)+scales | dense tern 4.2MB | + int4 W2 4.2MB -> ~8.5MB = 1.48x vs FP4", flush=True)

