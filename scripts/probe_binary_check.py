"""Fresh single-expert error check (L5 k7): binary {-1,+1} W13 vs tern, int4 W2.
Same honest pipeline as probe_maxcomp (rotation, bias, GPTQ, ridge W2, int4 W2 GPTQ).
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dsv4_refit_experts import soft_lim, _gptq_groups  # noqa: E402
import dsv4_experts as de  # noqa: E402

L = int(os.environ.get("PROBE_L", "5"))
K = int(os.environ.get("PROBE_K", "7"))
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


def snap(w, levels):
    idx = torch.searchsorted(levels, w).clamp(1, len(levels) - 1)
    lo, hi = levels[idx - 1], levels[idx]
    return torch.where((w - lo).abs() < (hi - w).abs(), lo, hi)


def gptq_grid(W, H, gs, levels, jit=1e-3):
    out_, in_ = W.shape
    ng = in_ // gs
    Wg = W.view(out_, ng, gs)
    top = levels.abs().max().float()
    sg = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9) / top
    for _ in range(4):
        q = snap((Wg / sg).flatten(), levels).view(out_, ng, gs)
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
            q = snap(w / Sb[:, j], levels)
            lev = q * Sb[:, j]
            Qb[:, j] = lev
            Err[:, j] = (w - lev) / Hb[j, j]
            if j + 1 < c1 - c0:
                Wb[:, j + 1:] -= torch.outer(Err[:, j], Hb[j, j + 1:])
        Q[:, c0:c1] = Qb
        if c1 < in_:
            Wc[:, c1:] -= Err @ Hi[c0:c1, c1:]
    return Q


def run(name, lv13):
    W1q = gptq_grid(w1_rot, Hz, GS, lv13)
    W3q = gptq_grid(w3_rot, Hz, GS, lv13)
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
        sg2 = ((qg2 * Wg2).sum(-1, keepdim=True) / (qg2 * qg2).sum(-1, keepdim=True).clamp_min(1e-9)).clamp_min(1e-9)
        qg2 = (Wg2 / sg2).round().clamp(-7, 7)
    s2g = sg2.squeeze(-1)
    q2 = _gptq_groups(W2c, Hh, s2g, gs=GS)
    W2q = q2 * s2g.repeat_interleave(GS, dim=1)
    r = (((h @ W2q.T - y).norm()) / y.norm()).item() * 100
    print(f"{name}: {r:6.2f}%   (n={x.shape[0]})", flush=True)


run("binary {-1,+1} g128", torch.tensor([-1.0, 1.0], device=dev))
run("tern {-1,0,1} g128 ", torch.tensor([-1.0, 0.0, 1.0], device=dev))
run("twolev g128       ", torch.tensor([-2.0, -1.0, 1.0, 2.0], device=dev))
