"""Honest A/B probe: GPTQ-W13 (binary + LDLQ feedback over the z-Hessian)
vs plain sign-W13, both followed by the same ridge-solve + GPTQ-W2 g128.

Runs on REAL activations (checkpoints_dsv4/pod_all_tokens/acts_layer{L}.pt).
Metric: ||y_pred - y_orig|| / ||y_orig|| per expert (norm resid, file-honest:
the SAME decode math as the generator: unpack -> z -> soft_lim -> silu -> W2).
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
print(f"L{L} k{K}: {x.shape[0]} real rows", flush=True)

# original expert weights (lossless FP4 -> fp32)
import dsv4_experts as de  # noqa: E402

E = de.load_expert_file(os.path.join("lossless_layers", f"layers_{L}_ffn.safetensors"), f"layers.{L}.ffn", K)
w1, w2, w3 = E["w1"].to(dev), E["w2"].to(dev), E["w3"].to(dev)

P = torch.load(os.path.join(RED, "P.pt"), map_location=dev).float()
mu = torch.load(os.path.join(RED, "mu.pt"), map_location=dev).float()
z = (x - mu) @ P  # [n, 4096]
w1_rot = w1 @ P.T  # rotated weights (what the refit quantizes)
w3_rot = w3 @ P.T

Hz = (z.T @ z) / z.shape[0]  # z-Hessian for W13 GPTQ


def norm_resid(yh):
    return ((yh - y).norm() / y.norm()).item() * 100


def ridge_w2(h):
    Gm = h.T @ h
    Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
    return torch.linalg.solve(Gm, h.T @ y).T.contiguous()


def gptq_w2(W2c, h, gs=128, nlev=7):
    ng = W2c.shape[1] // gs
    Wg = W2c.view(W2c.shape[0], ng, gs)
    sg_ = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / nlev
    qg = (Wg / sg_).round().clamp(-nlev, nlev)
    for _ in range(3):
        num = (qg * Wg).sum(-1, keepdim=True)
        den = (qg * qg).sum(-1, keepdim=True).clamp_min(1e-9)
        sg_ = (num / den).clamp_min(1e-9)
        qg = (Wg / sg_).round().clamp(-nlev, nlev)
    s2g = sg_.squeeze(-1)
    Hh = (h.T @ h) / h.shape[0]
    q2 = _gptq_groups(W2c, Hh, s2g, gs=gs, nlev=nlev)
    return q2 * s2g.repeat_interleave(gs, dim=1), s2g


def fp32_w13_h():
    g = soft_lim(z @ w1_rot.T)
    u = soft_lim(z @ w3_rot.T)
    return F.silu(g) * u


def yh_fp32_w2(W2c):
    return fp32_w13_h() @ W2c.T


# bias compensation: b = mu @ w.T makes z@w_rot.T + b == x@w.T EXACT
b1 = (mu @ w1_rot.T)[0].contiguous()  # [2048]
b3 = (mu @ w3_rot.T)[0].contiguous()

# bias compensation: b = mu @ w.T makes z@w_rot.T + b == x@w.T EXACT
b1 = (mu @ w1_rot.T)[0].contiguous()  # [2048]
b3 = (mu @ w3_rot.T)[0].contiguous()

# --- 0bb) bias + hard clamp ceiling: how close is our ARCHITECTURE itself? ---
def arch_ceiling(clamp_fn, label):
    g = clamp_fn(z @ w1_rot.T + b1[None, :])
    u = clamp_fn(z @ w3_rot.T + b3[None, :])
    h = F.silu(g) * u
    W2c = ridge_w2(h)
    yh = h @ W2c.T
    print(f"{label:44s} {norm_resid(yh):6.2f}%", flush=True)
    return W2c

hard = lambda t: t.clamp(min=-10, max=10)
arch_ceiling(hard, "0bb hard-clamp + bias + FP32 W2 (exact arch)")
arch_ceiling(soft_lim, "0bb soft_lim + bias + FP32 W2")

# --- C) W2 variants at the FP32-W13 ceiling (with bias, honest) ---
def fp32_h_bias():
    g = soft_lim(z @ w1_rot.T + b1[None, :])
    u = soft_lim(z @ w3_rot.T + b3[None, :])
    return F.silu(g) * u

h0 = fp32_h_bias()
W2c0 = ridge_w2(h0)
print(f"{'0b FP32-W13 + bias + FP32 ridge W2':44s} {norm_resid(h0 @ W2c0.T):6.2f}%", flush=True)

for gs_i, nlev_i, label in [(128, 7, "int4 g128"), (64, 7, "int4 g64"), (128, 15, "int5 g128"), (128, 31, "int6 g128")]:
    W2q, _ = gptq_w2(W2c0, h0, gs=gs_i, nlev=nlev_i)
    print(f"C FP32-W13 + bias + {label:22s} {norm_resid(h0 @ W2q.T):6.2f}%", flush=True)
    del W2q

# --- A) current recipe: sign + tie-break, LS row scales + GPTQ W2 ---
q1 = torch.where(w1_rot >= 0, 1.0, -1.0)
q3 = torch.where(w3_rot >= 0, 1.0, -1.0)
s1 = w1_rot.abs().mean(dim=1).clamp_min(1e-9)
s3 = w3_rot.abs().mean(dim=1).clamp_min(1e-9)


def run_variant(name, q1, s1, q3, s3):
    W1q = q1 * s1[:, None]
    W3q = q3 * s3[:, None]
    g = soft_lim(z @ W1q.T + b1[None, :])
    u = soft_lim(z @ W3q.T + b3[None, :])
    h = F.silu(g) * u
    W2c = ridge_w2(h)
    W2q, _ = gptq_w2(W2c, h)
    yh = full_forward(q1, s1, q3, s3, W2q)
    print(f"{name:44s} {norm_resid(yh):6.2f}%", flush=True)


def full_forward(q1, s1, q3, s3, W2q):
    outs = []
    for c0 in range(0, z.shape[0], 4096):
        zc = z[c0:c0 + 4096]
        g = soft_lim(zc @ (q1 * s1[:, None]).T + b1[None, :])
        u = soft_lim(zc @ (q3 * s3[:, None]).T + b3[None, :])
        outs.append((F.silu(g) * u) @ W2q.T)
    return torch.cat(outs)


run_variant("A sign-W13 + GPTQ-W2 g128 (current)", q1, s1, q3, s3)


# --- B) GPTQ-W13: binary + LDLQ feedback over Hz, frozen LS row scales ---
def gptq_bin(W, H, act_order=True, jit=1e-3):
    out_, in_ = W.shape
    s = W.abs().mean(dim=1).clamp_min(1e-9)  # frozen LS row scale
    Wc = W.clone()
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
    for c0 in range(0, in_, block):
        c1 = min(c0 + block, in_)
        Wb = Wc[:, c0:c1].clone()
        Qb = torch.zeros_like(Wb)
        Err = torch.zeros_like(Wb)
        Hb = Hi[c0:c1, c0:c1]
        for j in range(c1 - c0):
            w = Wb[:, j]
            q = torch.where(w >= 0, 1.0, -1.0)
            Qb[:, j] = q
            Err[:, j] = (w - q * s) / Hb[j, j]
            if j + 1 < c1 - c0:
                Wb[:, j + 1:] -= torch.outer(Err[:, j], Hb[j, j + 1:])
        Q[:, c0:c1] = Qb
        if c1 < in_:
            Wc[:, c1:] -= Err @ Hi[c0:c1, c1:]
    if act_order:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(in_, device=W.device)
        Q = Q[:, inv]
    return Q, s


q1b, s1b = gptq_bin(w1_rot, Hz)
q3b, s3b = gptq_bin(w3_rot, Hz)
run_variant("B GPTQ-W13 (LDLQ) + GPTQ-W2 g128", q1b, s1b, q3b, s3b)
