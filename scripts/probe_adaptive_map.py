"""Adaptive-precision map: per-expert error of the int2-W13 g64 + int4-W2
recipe vs routing frequency, over N experts of layer L.

Output: sorted error distribution, error-weighted-by-frequency impact,
and candidate upgrade thresholds (what % of experts would upgrade).
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stub_import_tf  # noqa: F401,E402
from dsv4_refit_experts import soft_lim, _gptq_groups  # noqa: E402

L = int(os.environ.get("L", "5"))
NEXP = int(os.environ.get("NEXP", "32"))
NLEV = int(os.environ.get("NLEV", "2"))
WGS = int(os.environ.get("WGS", "64"))
POD = "checkpoints_dsv4/pod_all_tokens"
RED = f"dsv4_reduced/layer_{L}"
GS = 128

torch.set_grad_enabled(False)
dev = "cuda"

acts = torch.load(os.path.join(POD, f"acts_layer{L}.pt"), map_location="cpu", weights_only=False)
keys = sorted([int(k) for k in acts.keys()])[:NEXP]

P = torch.load(os.path.join(RED, "P.pt"), map_location=dev).float()
mu = torch.load(os.path.join(RED, "mu.pt"), map_location=dev).float()

import dsv4_experts as de  # noqa: E402

def gptq_int(W, H, gs=128, nlev=7, jit=1e-3):
    """Groupwise int GPTQ (copied from probe_i4i4)."""
    out_, in_ = W.shape
    Wg = W.view(out_, in_ // gs, gs)
    sg = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / nlev
    for _ in range(3):
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



results = []
for idx, K in enumerate(keys):
    x_k, y_k = acts[str(K)]
    n_rows = x_k.shape[0]
    x = x_k.float().to(dev)
    y = y_k.float().to(dev)
    E = de.load_expert_file(os.path.join("lossless_layers", f"layers_{L}_ffn.safetensors"), f"layers.{L}.ffn", K)
    w1, w2, w3 = E["w1"].to(dev), E["w2"].to(dev), E["w3"].to(dev)
    z = (x - mu) @ P
    w1_rot = w1 @ P.T
    w3_rot = w3 @ P.T
    Hz = (z.T @ z) / n_rows
    b1 = (mu @ w1_rot.T)[0].contiguous()
    b3 = (mu @ w3_rot.T)[0].contiguous()

    W1q = gptq_int(w1_rot, Hz, gs=WGS, nlev=NLEV)
    W3q = gptq_int(w3_rot, Hz, gs=WGS, nlev=NLEV)
    g = soft_lim(z @ W1q.T + b1[None, :])
    u = soft_lim(z @ W3q.T + b3[None, :])
    h = F.silu(g) * u
    Gm = h.T @ h
    Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
    W2c = torch.linalg.solve(Gm, h.T @ y).T.contiguous()
    Hh = (h.T @ h) / n_rows
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
    for c0 in range(0, n_rows, 4096):
        zc = z[c0:c0 + 4096]
        gg = soft_lim(zc @ W1q.T + b1[None, :])
        uu = soft_lim(zc @ W3q.T + b3[None, :])
        outs.append((F.silu(gg) * uu) @ W2q.T)
    yh = torch.cat(outs)
    err = ((yh - y).norm() / y.norm()).item() * 100
    results.append((K, err, n_rows))
    if (idx + 1) % 8 == 0:
        print(f"  ... {idx+1}/{len(keys)}", flush=True)

errs = torch.tensor([r[1] for r in results])
freqs = torch.tensor([float(r[2]) for r in results])
print("\n=== per-expert error, int2-W13 g64 + int4-W2 ===")
for K, err, n in results:
    print(f"k{K:3d}: err {err:6.2f}%  rows {n}")
    torch.save({"err": err, "rows": n}, f"checkpoints_dsv4/adapt_L{L}_n{NLEV}g{WGS}_k{K}.pt") if False else None
errs_s, _ = errs.sort()
print(f"\nmin {errs_s[0]:.2f}%  p25 {errs_s[len(errs_s)//4]:.2f}%  median {errs_s[len(errs_s)//2]:.2f}%  p75 {errs_s[3*len(errs_s)//4]:.2f}%  max {errs_s[-1]:.2f}%")
for thr in (7, 8, 9, 10, 12):
    frac = (errs > thr).float().mean().item()
    wf = (freqs[errs > thr].sum() / freqs.sum()).item()
    print(f"upgrade@err>{thr}%: {frac*100:5.1f}% experts ({wf*100:5.1f}% of routed rows)")
