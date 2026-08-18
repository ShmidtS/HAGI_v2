"""Unifold round-trip: reduced experts vs lossless original, per layer, per expert.

No model, no text, no router: for each expert we draw a FRESH unifold signal
(bootstrap+jitter over the expert's z-manifold rows, eval seed — i.e. samples
the trained core has never seen), map it through the POD basis to x, and
compare
    exact  = ffn(x, lossless_fp4_weights)      (one expert dequantized at a time)
    reduced = ternary+int4 expert forward (z-space)
Relative error per expert, aggregated per layer and globally.

Memory: one layer's acts + ONE lossless expert at a time (~hundreds of MB).

Usage:
    python scripts/dsv4_roundtrip.py [--m 1024] [--every 1]
        [--layer-start 0] [--layer-end 42]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import stub_import_tf  # noqa: F401
import dsv4_generate_reduced as gr
from dsv4_generate_reduced import REDUCED, N_LAYERS
from dsv4_collect_x_accurate import load_selected_experts, ffn as ffn_exact
from dsv4_refit_experts import (
    POD, universal_signal, unpack_ternary, unpack_int4,
)

EVAL_SEED = 999_000


def reduced_forward(z: torch.Tensor, e: dict) -> torch.Tensor:
    """Ternary+int4 expert forward in z-space (same math as current_resid)."""
    Kk = z.shape[1]
    w1 = unpack_ternary(e['w1']).float().cuda()[:, :Kk] * e['w1_scale'].float().cuda()[:, None]
    w3 = unpack_ternary(e['w3']).float().cuda()[:, :Kk] * e['w3_scale'].float().cuda()[:, None]
    w2 = unpack_ternary(e['w2']).float().cuda() * e['w2_scale'].float().cuda()[:, None]
    Q = unpack_int4(e['Q']).float().cuda() * e['Q_scale'].float().cuda()[None, :]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        g = (z @ w1.T).clamp(max=10.0)
        u = (z @ w3.T).clamp(min=-10.0, max=10.0)
        w2 = w2[:, :g.shape[1]]  # trim base-3 packing pad on inter dim
        h = F.silu(g) * u
        y = (h @ w2.T) @ Q.T
    return y.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=1024, help='unifold samples per expert')
    ap.add_argument('--every', type=int, default=1, help='evaluate every Nth expert (quick pass)')
    ap.add_argument('--layer-start', type=int, default=0)
    ap.add_argument('--layer-end', type=int, default=N_LAYERS - 1)
    args = ap.parse_args()

    rows = []  # (layer, expert, resid_unifold, resid_saved)
    t00 = time.time()
    for L in range(args.layer_start, args.layer_end + 1):
        P = torch.load(os.path.join(REDUCED, f'layer_{L}', 'P.pt'), map_location='cuda').float()
        mu = torch.load(os.path.join(REDUCED, f'layer_{L}', 'mu.pt'), map_location='cuda').float()
        acts = torch.load(os.path.join(POD, f'acts_layer{L}.pt'),
                          map_location='cpu', weights_only=False)
        keys = sorted(acts.keys(), key=int)[:: args.every]
        t0 = time.time()
        for k in keys:
            ep = os.path.join(REDUCED, f'layer_{L}', f'expert_{k}.pt')
            if not os.path.exists(ep):
                continue
            e = torch.load(ep, map_location='cpu', weights_only=False)
            x_k, _ = acts[k]
            z_real = (x_k.float().cuda() - mu) @ P
            # fresh unifold draw (out-of-sample for the trained core)
            z_u = universal_signal(z_real, M=args.m, seed=EVAL_SEED + L * 1000 + int(k))
            x_u = mu + z_u @ P.T
            experts = load_selected_experts(L, [int(k)])
            w1, w2, w3 = experts[int(k)]
            y_ex = ffn_exact(x_u, w1, w2, w3)
            del experts, x_u
            y_rd = reduced_forward(z_u, e)
            resid = ((y_rd - y_ex) ** 2).mean().item() / y_ex.pow(2).mean().clamp_min(1e-12).item()
            rows.append((L, int(k), resid, float(e.get('residual', float('nan')))))
            del y_ex, y_rd, z_u, e
            torch.cuda.empty_cache()
        if rows:
            rl = [r[2] for r in rows if r[0] == L]
            rl_t = torch.tensor(rl)
            print(f'layer {L:2d}: n={len(rl):3d}  median={rl_t.median()*100:.4f}%  '
                  f'mean={rl_t.mean()*100:.4f}%  p90={rl_t.quantile(0.9)*100:.4f}%  '
                  f'max={rl_t.max()*100:.4f}%  [{time.time()-t0:.0f}s]', flush=True)

    if not rows:
        print('no experts evaluated')
        return
    allr = torch.tensor([r[2] for r in rows])
    print(f'\n=== unifold round-trip (layers {args.layer_start}-{args.layer_end}, '
          f'{len(rows)} experts, M={args.m}) ===')
    print(f'median  : {allr.median()*100:.4f}%')
    print(f'mean    : {allr.mean()*100:.4f}%')
    print(f'p90     : {allr.quantile(0.9)*100:.4f}%')
    print(f'p99     : {allr.quantile(0.99)*100:.4f}%')
    print(f'max     : {allr.max()*100:.4f}%')
    for t in (1e-4, 1e-3, 1e-2):
        n = (allr < t).sum().item()
        print(f'below {t*100:.2f}%: {n}/{len(rows)} ({100*n/len(rows):.1f}%)')
    worst = sorted(rows, key=lambda r: -r[2])[:10]
    print('worst 10:')
    for L, k, r, rs in worst:
        print(f'  layer {L:2d} expert {k:3d}: unifold {r*100:.4f}%  saved {rs*100:.4f}%')
    print(f'total {time.time()-t00:.0f}s')


if __name__ == '__main__':
    main()
