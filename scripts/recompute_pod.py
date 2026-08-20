"""Recompute the input POD basis (P) and mean (mu) from the full-vocab collection.

Loads `checkpoints_dsv4/pod_all_tokens/x_layer{L}.pt` (bf16) per layer, centers
by the per-layer mean, runs a full SVD, and writes `P.pt` / `mu.pt` into
`dsv4_reduced/layer_{L}/`.

Convention matches `dsv4_reduce_layer.py` exactly:
  mu = x.mean(0, keepdim=True)          # [1, 4096]
  P  = Vt.T[:, :K] from SVD of (x - mu) # [4096, K]

Usage:
  python scripts/recompute_pod.py [--k 512] [--src ...] [--dst ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import stub_import_tf  # noqa: F401
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=512)
    ap.add_argument('--src', default='checkpoints_dsv4/pod_all_tokens')
    ap.add_argument('--dst', default='dsv4_reduced')
    ap.add_argument('--layers', type=int, default=43)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    t0 = time.time()
    done = skipped = 0
    for L in range(args.layers):
        xp = os.path.join(args.src, f'x_layer{L}.pt')
        if not os.path.exists(xp):
            print(f'layer {L}: MISSING {xp}, skip', flush=True)
            skipped += 1
            continue
        x = torch.load(xp, map_location=args.device).float()  # [N, 4096]
        n = x.shape[0]
        if n < 2 * args.k:
            print(f'layer {L}: n={n} too small (<{2*args.k}), skip', flush=True)
            skipped += 1
            del x
            continue

        mu = x.mean(0, keepdim=True)                    # [1, 4096]
        xc = x - mu
        xc = torch.nan_to_num(xc, nan=0.0, posinf=0.0, neginf=0.0)
        _, _, Vt = torch.linalg.svd(xc, full_matrices=False)  # Vt [4096, 4096]
        P = Vt.T[:, : args.k].contiguous()              # [4096, K]

        out_dir = os.path.join(args.dst, f'layer_{L}')
        os.makedirs(out_dir, exist_ok=True)
        torch.save(P.cpu(), os.path.join(out_dir, 'P.pt'))
        torch.save(mu.cpu(), os.path.join(out_dir, 'mu.pt'))
        del x, xc, Vt, P
        done += 1
        print(f'layer {L}: n={n} -> P [{args.k}] saved ({time.time()-t0:.0f}s)', flush=True)

    print(f'done: {done}/{args.layers} layers recomputed, {skipped} skipped, '
          f'{time.time()-t0:.0f}s total', flush=True)


if __name__ == '__main__':
    main()
