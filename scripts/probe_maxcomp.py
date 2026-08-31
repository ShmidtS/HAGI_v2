"""MAX COMPRESSION matrix: W13 x W2 joint sweep with exact storage math.
W13 grids: twolev {-2,-1,1,2} (2b), tern {-1,0,1} (1.6b), i2_5lvl {-2..2} (2.32b)
W2 grids : int4 g32, int3 g32/g64, twolev g32, i2_5lvl g32
Baselines: i1i4 (W13 twolev g128 + W2 i4 g128) 16.3%...6.6MB FAIL;
int1 g128 9.1-9.6% @ ~7.5MB (W2 i4 g128).
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
M = 2048 * 4096  # params per W13 matrix; W2 = 2048*4096 same
MB = 1e6


def snap_custom(w, levels):
    """nearest level in `levels` (sorted tensor), returns q * per-group LS scale path is in caller."""
    idx = torch.searchsorted(levels, w)
    idx = idx.clamp(1, len(levels) - 1)
    lo = levels[idx - 1]
    hi = levels[idx]
    return torch.where((w - lo).abs() < (hi - w).abs(), lo, hi)


def gptq_grid(W, H, gs, levels, jit=1e-3):
    """GPTQ with arbitrary symmetric grid `levels` (e.g. {-2,-1,1,2}) and per-group LS scales."""
    out_, in_ = W.shape
    ng = in_ // gs
    Wg = W.view(out_, ng, gs)
    amax = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9)
    # init scale: match grid max
    top = levels.abs().max().float()
    sg = amax / top
    for _ in range(4):
        vals = Wg / sg
        q = snap_custom(vals.flatten(), levels).view(out_, ng, gs)
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
            vals = w / Sb[:, j]
            q = snap_custom(vals, levels)
            lev = q * Sb[:, j]
            Qb[:, j] = lev
            Err[:, j] = (w - lev) / Hb[j, j]
            if j + 1 < c1 - c0:
                Wb[:, j + 1:] -= torch.outer(Err[:, j], Hb[j, j + 1:])
        Q[:, c0:c1] = Qb
        if c1 < in_:
            Wc[:, c1:] -= Err @ Hi[c0:c1, c1:]
    return Q


def run(w13, w2cfg):
    name13, lv13, gs13, bits13 = w13
    name2, lv2, gs2, bits2 = w2cfg
    W1q = gptq_grid(w1_rot, Hz, gs13, lv13)
    W3q = gptq_grid(w3_rot, Hz, gs13, lv13)
    g = soft_lim(z @ W1q.T + b1[None, :])
    u = soft_lim(z @ W3q.T + b3[None, :])
    h = F.silu(g) * u
    Gm = h.T @ h
    Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
    W2c = torch.linalg.solve(Gm, h.T @ y).T.contiguous()
    W2q = gptq_grid(W2c, (h.T @ h) / h.shape[0], gs2, lv2)
    r = (((h @ W2q.T - y).norm()) / y.norm()).item() * 100
    sz = (2 * M * bits13 + M * bits2) / 8 / MB + (2 * (M / gs13) * 2 + (M / gs2) * 2) / MB
    comp = 12.6 / sz
    print(f"{name13:<14} + {name2:<16}: {r:6.2f}%  ~{sz:5.2f} MB  {comp:4.2f}x", flush=True)


TL13 = ("twolev g128", torch.tensor([-2., -1., 1., 2.], device=dev), 128, 2.0)
TERN13 = ("tern g128", torch.tensor([-1., 0., 1.], device=dev), 128, 8 / 5)
I2_13 = ("i2_5lvl g64", torch.tensor([-2., -1., 0., 1., 2.], device=dev), 64, 7 / 3)
I3_13 = ("int3 g128", torch.tensor([-3., -2., -1., 0., 1., 2., 3.], device=dev), 128, 3.0)

W2_I4G32 = ("W2 i4 g32", torch.arange(-7., 8., device=dev), 32, 4.0)
W2_I3G32 = ("W2 i3 g32", torch.arange(-3., 4., device=dev), 32, 3.0)
W2_I3G64 = ("W2 i3 g64", torch.arange(-3., 4., device=dev), 64, 3.0)
W2_TL = ("W2 twolev g32", torch.tensor([-2., -1., 1., 2.], device=dev), 32, 2.0)
W2_TERN = ("W2 tern g32", torch.tensor([-1., 0., 1.], device=dev), 32, 8 / 5)

run(TERN13, W2_I4G32)
run(TL13, W2_I4G32)
run(TL13, W2_I3G32)
run(I2_13, W2_I3G32)
run(TERN13, W2_I3G64)
run(TL13, W2_TL)
run(I3_13, W2_I3G32)
run(I2_13, W2_I4G32)
