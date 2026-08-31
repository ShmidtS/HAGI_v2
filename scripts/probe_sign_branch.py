"""Sign-branch lattices: W13 quantized to {+/-a1 +/-a2 +/-a3} per (row,group),
plus full 8-level k-means codebook as upper bound of the family.

Baselines (L5 k7, honest): int3 uniform 6.34%, int4 5.84%, {−3,−1,1,3} 9.27%.
Storage: 3 branch scales = 6 B / 128 w = 0.37 bit; kmeans 8 lvls = 16 B/128 = 1 bit.
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
z = (x - mu) @ P
w1_rot, w3_rot = w1 @ P, w3 @ P
Hz = (z.T @ z) / z.shape[0]
b1 = (mu.reshape(-1) @ w1.T).float()
b3 = (mu.reshape(-1) @ w3.T).float()
GS = 128
SIGNS = torch.tensor(
    [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
    device=dev, dtype=torch.float32,
)  # [8,3]


def fit_codes(W, iters=12):
    """Per (row,group): init 3 branch scales from sorted thirds, alternate
    assignment/LS-refit of alpha. Returns codebook [out,ng,3] and idx [out,in]."""
    out_, in_ = W.shape
    ng = in_ // GS
    Wg = W.view(out_, ng, GS)  # [out,ng,128]
    srt, _ = Wg.sort(dim=-1)
    # init alphas: mean abs of terciles scaled 1:2:3
    t = GS // 3
    a = torch.stack([srt[..., :t].abs().mean(-1), srt[..., t:2 * t].abs().mean(-1), srt[..., 2 * t:].abs().mean(-1)], -1)
    a = a.clamp_min(1e-9)
    a = a / a.norm(dim=-1, keepdim=True) * 1.8  # match uniform scale roughly
    for _ in range(iters):
        codes = a @ SIGNS.T  # [out,ng,8]
        d = (Wg.unsqueeze(-1) - codes.unsqueeze(-2)).abs()  # [out,ng,128,8]
        idx = d.argmin(-1)  # [out,ng,128]
        sel = SIGNS[idx]  # [out,ng,128,3]
        num = (sel * Wg.unsqueeze(-1)).sum(-2)  # [out,ng,3]
        den = (sel * sel).sum(-2).clamp_min(1.0)
        a = (num / den).abs().clamp_min(1e-9)
    return a, idx.view(out_, in_)


def fit_kmeans(W, iters=12):
    out_, in_ = W.shape
    ng = in_ // GS
    Wg = W.view(out_, ng, GS)
    qs = torch.tensor([0.06, 0.19, 0.31, 0.44, 0.56, 0.69, 0.81, 0.94], device=dev)
    lv = torch.quantile(Wg.float(), qs, dim=-1).permute(1, 2, 0)  # [out,ng,8]
    idx = None
    for _ in range(iters):
        d = (Wg.unsqueeze(-1) - lv.unsqueeze(-2)).abs()
        idx = d.argmin(-1)
        onehot = F.one_hot(idx, 8).float()  # [out,ng,128,8]
        cnt = onehot.sum(-2).clamp_min(1e-9)
        lv = (onehot * Wg.unsqueeze(-1)).sum(-2) / cnt
    return lv, idx.view(out_, in_)


def gptq_cb(W, H, cb, idx):
    """GPTQ error feedback with a FROZEN per-(row,group) codebook.
    cb: decode fn value lookup: given row r and flat col c -> level."""
    out_, in_ = W.shape
    jit = 1e-3
    Hf = H.float()
    eye = torch.eye(in_, device=W.device)
    Hi = torch.linalg.cholesky(Hf + jit * Hf.diag().mean() * eye)
    Hi = torch.cholesky_inverse(Hi)
    Hi = torch.linalg.cholesky(Hi + jit * Hi.diag().mean() * eye, upper=True)
    Wc = W.clone()
    Q = torch.zeros_like(W)
    for c0 in range(0, in_, 128):
        c1 = min(c0 + 128, in_)
        Wb = Wc[:, c0:c1].clone()
        Qb = torch.zeros_like(Wb)
        Err = torch.zeros_like(Wb)
        Hb = Hi[c0:c1, c0:c1]
        for j in range(c1 - c0):
            w = Wb[:, j]
            idj = idx[:, c0 + j]  # [out] codebook index
            lev = cb(w, idj, c0 + j)
            Qb[:, j] = lev
            Err[:, j] = (w - lev) / Hb[j, j]
            if j + 1 < c1 - c0:
                Wb[:, j + 1:] -= torch.outer(Err[:, j], Hb[j, j + 1:])
        Q[:, c0:c1] = Qb
        if c1 < in_:
            Wc[:, c1:] -= Err @ Hi[c0:c1, c1:]
    return Q


def make_cb_branches(a):
    out_, ng, _ = a.shape
    codes = (a @ SIGNS.T)  # [out,ng,8]
    flat = codes.view(out_, -1)  # [out, in]
    def cb(w, idj, c):
        return flat.gather(1, (idj.unsqueeze(-1)).long().clamp(0, 7)).squeeze(-1) if False else torch.take_along_dim(flat, idj.clamp(0, 7).unsqueeze(1), 1).squeeze(1) * 0 + torch.take_along_dim(flat, idj.clamp(0, 7).unsqueeze(1), 1).squeeze(1)
    # simpler: value depends only on code index (assignment from fit; nearest to current w not required for probe)
    return cb


def run(name, qfun):
    W1q, W3q = qfun(w1_rot, Hz), qfun(w3_rot, Hz)
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
    yhq = h @ W2q.T
    r = ((yhq - y).norm() / y.norm()).item() * 100
    print(f"{name}: honest resid {r:6.2f}%", flush=True)


def q_branch(W, H):
    a, idx = fit_codes(W)
    out_, in_ = W.shape
    ng = in_ // GS
    codes = a @ SIGNS.T  # [out,ng,8]
    flat = codes.view(out_, ng, 1, 8).expand(-1, -1, GS, -1).reshape(out_, in_ * 8)
    def cb(w, idj, c):
        # nearest codeword on the CURRENT (feedback-updated) w
        col = flat[:, c * 8:(c + 1) * 8]
        j = (w.unsqueeze(1) - col).abs().argmin(1)
        return col.gather(1, j.unsqueeze(1)).squeeze(1)
    return gptq_cb(W, H, cb, idx)


def q_kmeans(W, H):
    lv, idx = fit_kmeans(W)
    out_, in_ = W.shape
    ng = in_ // GS
    flat = lv.view(out_, ng, 1, 8).expand(-1, -1, GS, -1).reshape(out_, in_ * 8)
    def cb(w, idj, c):
        # nearest codeword on the CURRENT (feedback-updated) w
        col = flat[:, c * 8:(c + 1) * 8]
        j = (w.unsqueeze(1) - col).abs().argmin(1)
        return col.gather(1, j.unsqueeze(1)).squeeze(1)
    return gptq_cb(W, H, cb, idx)


run("sign-3branch (unequal a, 3.4 bit)", q_branch)
run("kmeans-8lvl  (full cb, 4.0 bit)", q_kmeans)
