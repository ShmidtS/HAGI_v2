"""Recompute the input POD basis (P) and mean (mu) from the full-vocab collection.

v2: randomized SVD (svd_lowrank, top-K only) on GPU + multi-process by layers.

Loads x_layer{L}.pt (bf16) per layer, centers by the per-layer mean, extracts
the top-K right singular vectors, writes P.pt / mu.pt into dsv4_reduced/layer_{L}/.

Convention matches dsv4_reduce_layer.py:
  mu = x.mean(0, keepdim=True)          # [1, 4096]
  P  = top-K right singular vectors     # [4096, K] (orthonormal)

Usage:
  python scripts/recompute_pod.py                       # all layers, lowrank, 4 procs
  python scripts/recompute_pod.py --n-procs 1           # single process
  python scripts/recompute_pod.py --svd full            # exact full SVD (slower)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import stub_import_tf  # noqa: F401
import torch


def svd_basis(x: torch.Tensor, k: int, lowrank: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (basis P [D, k], mean mu [1, D]) from row-major x [N, D]."""
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    xc = torch.nan_to_num(xc, nan=0.0, posinf=0.0, neginf=0.0)
    D = x.shape[1]
    if k >= D:
        # полный ранг: P = identity (поворот без потери), z = x - mu
        P = torch.eye(D, device=x.device, dtype=x.dtype)
    elif lowrank:
        _, _, V = torch.svd_lowrank(xc, q=k, niter=2)
        P = V
    else:
        _, _, Vt = torch.linalg.svd(xc, full_matrices=False)
        P = Vt.T[:, :k]
    return P.contiguous(), mu


def run_range(start: int, end: int, args) -> None:
    t0 = time.time()
    done = skipped = 0
    for L in range(start, end):
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
        P, mu = svd_basis(x, args.k, args.svd == 'lowrank')
        out_dir = os.path.join(args.dst, f'layer_{L}')
        os.makedirs(out_dir, exist_ok=True)
        torch.save(P.cpu(), os.path.join(out_dir, 'P.pt'))
        torch.save(mu.cpu(), os.path.join(out_dir, 'mu.pt'))
        del x, P, mu
        done += 1
        print(f'layer {L}: n={n} -> P [{args.k}] saved ({time.time()-t0:.0f}s)', flush=True)
    print(f'range {start}-{end}: {done} layers, {skipped} skipped, {time.time()-t0:.0f}s total', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=512)
    ap.add_argument('--src', default='checkpoints_dsv4/pod_all_tokens')
    ap.add_argument('--dst', default='dsv4_reduced')
    ap.add_argument('--layers', type=int, default=43)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--svd', choices=['lowrank', 'full'], default='lowrank')
    ap.add_argument('--n-procs', type=int, default=1,
                    help='split layers across N parallel processes (1 = single). WARNING: >1 on a shared iGPU can crash the GPU driver (BSOD) — keep 1.')
    ap.add_argument('--start-layer', type=int, default=0)
    ap.add_argument('--end-layer', type=int, default=None)
    args = ap.parse_args()
    args.end_layer = args.end_layer or args.layers

    if args.n_procs <= 1:
        run_range(args.start_layer, args.end_layer, args)
        return

    per = (args.end_layer - args.start_layer + args.n_procs - 1) // args.n_procs
    procs = []
    for i in range(args.n_procs):
        s = args.start_layer + i * per
        e = min(args.start_layer + (i + 1) * per, args.end_layer)
        if s >= e:
            break
        cmd = [sys.executable, '-u', os.path.abspath(__file__),
               '--start-layer', str(s), '--end-layer', str(e),
               '--k', str(args.k), '--src', args.src, '--dst', args.dst,
               '--device', args.device, '--svd', args.svd, '--n-procs', '1']
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()
    print(f'done: {len(procs)} processes finished', flush=True)


if __name__ == '__main__':
    main()
