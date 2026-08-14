"""Quick A/B test of the centering bug fix (8 experts, layer 0).

Bug: reduce computed the teacher target on CENTERED x (xc = x - mean),
so ternary experts learned FFN(x-mu) instead of FFN(x). Measured error
FFN(x-mu) vs FFN(x) = 25-27%.

Fix A (center PCA only, target on raw x):
    mu = x.mean(0); P = svd(xc); z = (x-mu)@P; target = FFN(x)@Q
    -> generation must center: z = (flat - mu) @ P

Fix B (no centering anywhere):
    P = svd(x); z = x@P; target = FFN(x)@Q
    -> generation unchanged: z = flat @ P

Run: python scripts/dsv4_test_fix.py [--steps 150] [--experts 8]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
import stub_import_tf  # noqa: F401
import torch
import torch.nn.functional as F
from safetensors import safe_open

import dsv4_experts as de

K = 512
Kp = 384
INTER = 4096


def load_expert(fp, layer, k, dev='cuda'):
    b = f'layers.{layer}.ffn.experts.{k}'
    with safe_open(fp, framework='pt', device='cpu') as f:
        w1 = de.dequant_fp4(f.get_tensor(f'{b}.w1.weight'), f.get_tensor(f'{b}.w1.scale')).to(dev)
        w2 = de.dequant_fp4(f.get_tensor(f'{b}.w2.weight'), f.get_tensor(f'{b}.w2.scale')).to(dev)
        w3 = de.dequant_fp4(f.get_tensor(f'{b}.w3.weight'), f.get_tensor(f'{b}.w3.scale')).to(dev)
    return w1.float(), w2.float(), w3.float()


def ffn(xin, w1, w2, w3):
    g = (xin @ w1.T).clamp(max=10.0)
    u = (xin @ w3.T).clamp(min=-10.0, max=10.0)
    return (F.silu(g) * u) @ w2.T


def qste(W):
    s = W.abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q = (W / s).clamp(-1, 1).round() * s
    return W + (q - W).detach()


def batched_forward(zb, W1, W3, W2):
    G = W1.shape[0]
    zb_stack = zb.unsqueeze(0).expand(G, -1, -1)
    w1q = qste(W1).transpose(1, 2)
    w3q = qste(W3).transpose(1, 2)
    w2q = qste(W2).transpose(1, 2)
    g = torch.bmm(zb_stack, w1q).clamp(max=10.0)
    u = torch.bmm(zb_stack, w3q).clamp(min=-10.0, max=10.0)
    h = F.silu(g) * u
    return torch.bmm(h, w2q).permute(1, 0, 2)


def train(z, targets, steps, bs):
    G = targets.shape[1]
    W1 = torch.nn.Parameter(torch.randn(G, INTER, K, device='cuda') * K**-0.5)
    W3 = torch.nn.Parameter(torch.randn(G, INTER, K, device='cuda') * K**-0.5)
    W2 = torch.nn.Parameter(torch.randn(G, Kp, INTER, device='cuda') * INTER**-0.5)
    o = torch.optim.Adam([W1, W3, W2], lr=2e-3)
    n = z.shape[0]
    for st in range(steps):
        lr = 2e-3 * 0.5 * (1 + math.cos(math.pi * st / steps))
        for g in o.param_groups:
            g['lr'] = lr
        idx = torch.randint(0, n, (bs,), device='cuda')
        zb, yb = z[idx], targets[idx]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            yp = batched_forward(zb, W1, W3, W2)
        o.zero_grad()
        F.mse_loss(yp.float(), yb).backward()
        o.step()
    with torch.no_grad():
        zb_stack = z.unsqueeze(0).expand(G, -1, -1)
        w1q = qste(W1).transpose(1, 2)
        w3q = qste(W3).transpose(1, 2)
        w2q = qste(W2).transpose(1, 2)
        g = torch.bmm(zb_stack, w1q).clamp(max=10.0)
        u = torch.bmm(zb_stack, w3q).clamp(min=-10.0, max=10.0)
        yf = torch.bmm(F.silu(g) * u, w2q).permute(1, 0, 2)
    res = (F.mse_loss(yf, targets) / F.mse_loss(targets, torch.zeros_like(targets))).item()
    return res, yf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--experts', type=int, default=8)
    args = ap.parse_args()

    x = torch.load('checkpoints_dsv4/pod/x_layer0.pt', map_location='cuda').float()
    x = x[:1000]
    mu = x.mean(0, keepdim=True)
    xc = x - mu

    fp = 'lossless_layers/layers_0_ffn.safetensors'
    ks = list(range(args.experts))

    # teacher outputs (raw x)
    targets_raw = []
    Qs = []
    for k in ks:
        w1, w2, w3 = load_expert(fp, 0, k)
        y = ffn(x, w1, w2, w3)  # FFN(x) — the CORRECT target
        _, _, Q = torch.svd_lowrank(y, q=Kp, niter=2)
        targets_raw.append(y @ Q)
        Qs.append(Q)
    targets_raw = torch.stack(targets_raw, dim=1)  # [N, G, Kp]

    # --- Fix A: center PCA, center z, target = FFN(x) ---
    _, _, Vt = torch.linalg.svd(xc, full_matrices=False)
    P_a = Vt.T[:, :K].contiguous()
    z_a = xc @ P_a
    t0 = time.time()
    res_a, _ = train(z_a, targets_raw, args.steps, 2048)
    print(f'Fix A (center PCA + center z, target=FFN(x)): residual={res_a*100:.3f}%  {time.time()-t0:.0f}s')

    # --- Fix B: no centering ---
    _, _, Vt = torch.linalg.svd(x, full_matrices=False)
    P_b = Vt.T[:, :K].contiguous()
    z_b = x @ P_b
    t0 = time.time()
    res_b, _ = train(z_b, targets_raw, args.steps, 2048)
    print(f'Fix B (no centering, target=FFN(x)):          residual={res_b*100:.3f}%  {time.time()-t0:.0f}s')

    # reference: the ORIGINAL (buggy) scheme residual for context
    # (targets on centered x, z centered) — from prior runs it was ~0.16%,
    # but it approximates the WRONG function FFN(x-mu).


if __name__ == '__main__':
    main()
