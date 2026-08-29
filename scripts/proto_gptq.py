# -*- coding: utf-8 -*-
"""GPTQ (LDLQ error feedback) for W2 int4: minimize ||h @ (W2-Ŵ2)^T||, not weight error.
Hessian = h^T h from real activations (already in our harness). Scalar int4 + block
error feedback, act-order permutation. Compare vs our per-row CD on the same expert."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
import dsv4_refit_experts as R

L, k = 5, 7
red = os.path.join("dsv4_reduced", f"layer_{L}")
P = torch.load(os.path.join(red, "P.pt"), map_location="cuda").float()
mu = torch.load(os.path.join(red, "mu.pt"), map_location="cuda").float()
acts = torch.load(os.path.join("checkpoints_dsv4", "pod_all_tokens", f"acts_layer{L}.pt"), map_location="cpu", weights_only=False)
x, y = acts[str(k)]
z = (x.float().cuda() - mu) @ P
yf = y.float().cuda()
ex = R.load_selected_experts(L, [k])
w1, w2, w3 = ex[k]
w1r = w1.float() @ P; w3r = w3.float() @ P
b1 = mu.reshape(-1) @ w1.float().T; b3 = mu.reshape(-1) @ w3.float().T

q1 = torch.sign(w1r); s1 = w1r.abs().mean(1, keepdim=True).clamp_min(1e-9)
q3 = torch.sign(w3r); s3 = w3r.abs().mean(1, keepdim=True).clamp_min(1e-9)
W1 = q1 * s1; W3 = q3 * s3
gg = R.soft_lim(z @ W1.T + b1[None, :]); uu = R.soft_lim(z @ W3.T + b3[None, :])
h = F.silu(gg) * uu
Gm = h.T @ h; Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
W2c = torch.linalg.solve(Gm, h.T @ yf).T  # [4096, 2048]


def quant_row(w, nlev, rounds=3):
    sg = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / nlev
    q = (w / sg).round().clamp(-nlev, nlev)
    for _ in range(rounds):
        num = (q * w).sum(-1, keepdim=True); den = (q * q).sum(-1, keepdim=True).clamp_min(1e-9)
        sg = (num / den).clamp_min(1e-9)
        q = (w / sg).round().clamp(-nlev, nlev)
    num = (q * w).sum(-1); den = (q * q).sum(-1).clamp_min(1e-9); s2 = num / den
    return q * s2[..., None]


def gptq_int4(W, H, nlev, block=128, act_order=True):
    """GPTQ with per-row scales + sequential error feedback over columns.
    W [out, in], H [in, in] = h^T h. Returns dequantized W."""
    out_, in_ = W.shape
    W = W.clone()
    if act_order:
        perm = torch.argsort(H.diag(), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
    # dead columns
    d = H.diag()
    dead = d <= 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0
    Hf = H.float()
    # Cholesky of inverse Hessian (standard GPTQ trick)
    Hinv = torch.linalg.cholesky(Hf + 1e-5 * torch.eye(in_, device=Hf.device) * Hf.diag().mean())
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    for col0 in range(0, in_, block):
        col1 = min(col0 + block, in_)
        Wb = W[:, col0:col1].clone()
        Qb = torch.zeros_like(Wb)
        Err = torch.zeros_like(Wb)
        Hinv_b = Hinv[col0:col1, col0:col1]
        for j in range(col1 - col0):
            w = Wb[:, j]
            d_chol = Hinv_b[j, j]
            q = quant_row(w.unsqueeze(1), nlev).squeeze(1)
            Qb[:, j] = q
            Err[:, j] = (w - q) / d_chol
            # feedback into remaining columns of the block
            if j + 1 < col1 - col0:
                Wb[:, j + 1:] -= torch.outer(Err[:, j], Hinv_b[j, j + 1:])
        Q[:, col0:col1] = Qb
        # feedback into remaining columns of the matrix
        if col1 < in_:
            W[:, col1:] -= Err @ Hinv[col0:col1, col1:]
    if act_order:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(in_, device=perm.device)
        Q = Q[:, inv]
    return Q


n = h.shape[0]
Hfull = (h.T @ h) / n
W2q = gptq_int4(W2c, Hfull, 7)
yh = h @ W2q.T
print(f"bin W13 + W2 int4 GPTQ(act-order): {((yh - yf).norm() / yf.norm()).item() * 100:.2f}% norm")

W2q2 = gptq_int4(W2c, Hfull, 7, act_order=False)
yh2 = h @ W2q2.T
print(f"bin W13 + W2 int4 GPTQ(no order) : {((yh2 - yf).norm() / yf.norm()).item() * 100:.2f}% norm")

W2q3 = gptq_int4(W2c, Hfull, 31)
yh3 = h @ W2q3.T
print(f"bin W13 + W2 int6 GPTQ          : {((yh3 - yf).norm() / yf.norm()).item() * 100:.2f}% norm")

# refs
for name, nlev in (("int4 CD (наш)", 7), ("int6 CD (наш)", 31)):
    W2qi = quant_row(W2c, nlev).squeeze(0) if W2c.dim() == 3 else quant_row(W2c, nlev)
    yh = h @ W2qi.T
    print(f"bin W13 + W2 {name}: {((yh - yf).norm() / yf.norm()).item() * 100:.2f}% norm")
