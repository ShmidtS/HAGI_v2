"""POD + ternary reduction of all DeepSeek-V4 experts in one layer.

Pipeline (per layer, resumable, batched):
  1. Load shared real activations x [N, 4096] (FFN input for this layer).
  2. PCA of x -> P [4096, K] (shared input projection); z = x @ P [N, K].
  3. Process experts in groups of G:
     a. dequant FP4 weights for G experts.
     b. y_g = ffn_g(x)  (teacher output on real input).
     c. svd_lowrank(y_g) -> Q_g [4096, Kp]; target_g = y_g @ Q_g.
     d. Batched ternary STE training of all G experts at once
        (amortizes kernel-launch overhead of the small per-expert matmuls).
     e. Save each expert's ternary weights + scales + Q_g (fp8).

Config (locked): K=512, Kp=384, inter=3072, steps=500, batch=2048,
cosine 2e-3, group G=8.

Usage:
  python scripts/dsv4_reduce_layer.py <layer> [--start 0] [--end 256] \
      [--steps 500] [--inter 3072] [--group 8] [--experts 5]

Resume: skips experts whose output file already exists.
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


K = 512          # reduced input dim
Kp = 384         # reduced output dim
X_PATH = 'checkpoints_dsv4/pod/x_layer{layer}.pt'
OUT_DIR = 'dsv4_reduced/layer_{layer}'


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
    """Batched ternary STE: forward = ternary, backward = identity."""
    s = W.abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q = (W / s).clamp(-1, 1).round() * s
    return W + (q - W).detach()


def ternarize(W):
    """Batched ternary quantize -> (q in {-1,0,1}, scale)."""
    s = W.abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
    q = (W / s).clamp(-1, 1).round()
    return q, s.squeeze(2)


def pack_ternary(q):
    """Pack ternary int8 {-1,0,1} [out, in] -> uint8 [out, ceil(in/5)] (5 trits/byte, base-3)."""
    t = (q + 1).to(torch.int32)  # {0,1,2}
    out, in_ = t.shape
    n = (in_ + 4) // 5
    padded = torch.zeros(out, n * 5, dtype=torch.int32)
    padded[:, :in_] = t
    packed = torch.zeros(out, n, dtype=torch.int32)
    for i in range(5):
        packed += padded[:, i::5] * (3 ** i)
    return packed.to(torch.uint8)


def batched_forward(zb, W1, W3, W2):
    """zb [B, K] -> [B, G, Kp] through G ternary experts."""
    G = W1.shape[0]
    zb_stack = zb.unsqueeze(0).expand(G, -1, -1)  # [G, B, K]
    w1q = qste(W1).transpose(1, 2)  # [G, K, inter]
    w3q = qste(W3).transpose(1, 2)
    w2q = qste(W2).transpose(1, 2)  # [G, inter, Kp]
    g = torch.bmm(zb_stack, w1q).clamp(max=10.0)  # [G, B, inter]
    u = torch.bmm(zb_stack, w3q).clamp(min=-10.0, max=10.0)
    h = F.silu(g) * u  # [G, B, inter]
    return torch.bmm(h, w2q).permute(1, 0, 2)  # [B, G, Kp]


def train_group(z, targets, inter, steps, bs, warm=None):
    """Train G experts batched. targets [N, G, Kp]; returns (ternary, scales, residual)."""
    G = targets.shape[1]
    W1 = torch.nn.Parameter(torch.randn(G, inter, K, device='cuda') * K**-0.5)
    W3 = torch.nn.Parameter(torch.randn(G, inter, K, device='cuda') * K**-0.5)
    W2 = torch.nn.Parameter(torch.randn(G, Kp, inter, device='cuda') * inter**-0.5)
    o = torch.optim.Adam([W1, W3, W2], lr=2e-3)
    n = z.shape[0]
    t0 = time.time()
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
        w1q, w1s = ternarize(W1.detach())
        w3q, w3s = ternarize(W3.detach())
        w2q, w2s = ternarize(W2.detach())
        G = W1.shape[0]
        w1eff = w1q * w1s.unsqueeze(2)  # [G, inter, K]
        w3eff = w3q * w3s.unsqueeze(2)
        w2eff = w2q * w2s.unsqueeze(2)
        zb_stack = z.unsqueeze(0).expand(G, -1, -1)  # [G, N, K]
        g = torch.bmm(zb_stack, w1eff.transpose(1, 2)).clamp(max=10.0)
        u = torch.bmm(zb_stack, w3eff.transpose(1, 2)).clamp(min=-10.0, max=10.0)
        yf = torch.bmm(F.silu(g) * u, w2eff.transpose(1, 2)).permute(1, 0, 2)
    res = (F.mse_loss(yf, targets) / F.mse_loss(targets, torch.zeros_like(targets))).item()
    return (w1q, w1s, w3q, w3s, w2q, w2s), res, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('layer', type=int)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=256)
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--inter', type=int, default=4096)
    ap.add_argument('--group', type=int, default=32)
    ap.add_argument('--experts', type=int, default=0, help='if >0, only process this many (debug)')
    args = ap.parse_args()

    layer = args.layer
    out_dir = OUT_DIR.format(layer=layer)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(layer * 1000)

    xpath = X_PATH.format(layer=layer)
    if not os.path.exists(xpath):
        x0 = X_PATH.format(layer=0)
        if os.path.exists(x0):
            print(f'layer {layer}: x_{layer} not found, using x_0 (fallback)', flush=True)
            x = torch.load(x0, map_location='cuda')
        else:
            print(f'FATAL: neither {xpath} nor {x0} exists. Run activation collection first.', file=sys.stderr)
            sys.exit(1)
    else:
        x = torch.load(xpath, map_location='cuda')
    # Fix A: PCA on CENTERED x (stable), but teacher target on RAW x.
    # The old code computed ffn(xc) — FFN(x-mu) differs from FFN(x) by 25-27%.
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    print(f'layer {layer}: x {tuple(xc.shape)}')

    p_path = os.path.join(out_dir, 'P.pt')
    mu_path = os.path.join(out_dir, 'mu.pt')
    if os.path.exists(p_path):
        P = torch.load(p_path, map_location='cuda')
        mu = torch.load(mu_path, map_location='cuda') if os.path.exists(mu_path) else mu
    else:
        _, _, Vt = torch.linalg.svd(xc, full_matrices=False)
        P = Vt.T[:, :K].contiguous()
        torch.save(P.cpu(), p_path)
        torch.save(mu.cpu(), mu_path)
    z = xc @ P

    fp = f'lossless_layers/layers_{layer}_ffn.safetensors'
    end = min(args.end, 256)
    total = end - args.start
    if args.experts:
        end = min(end, args.start + args.experts)
        total = args.experts
    done = 0
    t_start = time.time()

    k = args.start
    while k < end:
        # form a group of pending experts
        group = []
        while len(group) < args.group and k < end:
            if not os.path.exists(os.path.join(out_dir, f'expert_{k}.pt')):
                group.append(k)
            else:
                done += 1
            k += 1
        if not group:
            continue

        # teacher outputs + per-expert output basis
        targets = []
        Qs = []
        for gk in group:
            w1, w2, w3 = load_expert(fp, layer, gk)
            y = ffn(x, w1, w2, w3)  # FFN(x) on RAW input (was xc -> 26% bug)
            _, _, Q = torch.svd_lowrank(y, q=Kp, niter=2)  # [4096, Kp]
            targets.append(y @ Q)
            Qs.append(Q)
        targets = torch.stack(targets, dim=1)  # [N, G, Kp]
        Qs = torch.stack(Qs)  # [G, 4096, Kp]

        (w1q, w1s, w3q, w3s, w2q, w2s), res, dt = train_group(
            z, targets, args.inter, args.steps, 2048)

        for i, gk in enumerate(group):
            Q = Qs[i]  # [4096, Kp] fp32
            qscale = Q.abs().max(dim=0)[0] / 127.0  # [Kp]
            Q_i8 = (Q / qscale).round().clamp(-127, 127).to(torch.int8)
            torch.save({
                'w1': pack_ternary(w1q[i]).cpu(), 'w1_scale': w1s[i].cpu(),
                'w3': pack_ternary(w3q[i]).cpu(), 'w3_scale': w3s[i].cpu(),
                'w2': pack_ternary(w2q[i]).cpu(), 'w2_scale': w2s[i].cpu(),
                'Q': Q_i8.cpu(), 'Q_scale': qscale.cpu(),
                'residual': res,
            }, os.path.join(out_dir, f'expert_{gk}.pt'))
        done += len(group)

        elapsed = time.time() - t_start
        per_exp = elapsed / done
        eta = per_exp * (total - done)
        print(f'  experts {group[0]}..{group[-1]}: residual={res*100:.3f}%  '
              f'{dt:.1f}s/group  [{done}/{total}]  ETA {eta/60:.1f} min', flush=True)

    print(f'DONE layer {layer}: {done} experts in {time.time()-t_start:.1f}s '
          f'({done/(time.time()-t_start)*60:.1f} experts/min)')


if __name__ == '__main__':
    main()
