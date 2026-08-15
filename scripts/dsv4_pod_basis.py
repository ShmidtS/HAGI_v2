"""Compute POD bases from collected attention/cross-layer activations.

POD = PCA/SVD on activations (not weights — weights are white noise). For a
data matrix X [N, D] the top-K right singular vectors V[:, :K] form the
orthonormal basis P [D, K]; the compressed code is z = X @ P, reconstruction
X_hat = z @ P^T, and residual = 1 - sum(top-K s^2)/sum(all s^2).

Outputs (checkpoints_dsv4/pod/):
  - P_kv_L{L}.pt   [512, r_kv]   KV-cache compression basis (pre-RoPE K==V)
  - P_q_L{L}.pt    [512, r_q]    Q compression basis (heads flattened)
  - P_x_in_L{L}.pt  [4096, r_x]  layer FFN-input subspace (per layer)
  - P_x_out_L{L}.pt [4096, r_x]  layer-output subspace (per layer)
  - P_x_global.pt  [4096, r_x]   shared cross-layer subspace (all x_L pooled)
  - rank_report.txt              variance captured vs rank (diagnostic)

Usage:
    python scripts/dsv4_pod_basis.py
    python scripts/dsv4_pod_basis.py --kv-rank 384 --x-rank 512 --q-rank 384
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def svd_basis(x: torch.Tensor, rank: int | None, use_cuda: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (basis [D, rank], mean [D], singular_values [min(N,D)]) for row-major X.

    SVD on CENTERED activations (Fix A): PCA on raw data is unstable and wastes
    the top component on the mean; centering keeps the basis on real variance.
    """
    if x.ndim == 3:  # Q: [N, heads, D] -> pool heads
        x = x.reshape(-1, x.shape[-1])
    dev = 'cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'
    xg = x.float().to(dev)
    mean = xg.mean(0)
    u, s, vh = torch.linalg.svd(xg - mean, full_matrices=False)
    rank = rank or x.shape[1]
    rank = min(rank, x.shape[1], len(s))
    return vh[:rank].T.cpu().contiguous(), mean.cpu(), s.cpu()


def variance_fractions(s: torch.Tensor) -> dict[int, float]:
    total = (s ** 2).sum().item()
    out = {}
    for k in (16, 32, 64, 128, 256, 384, 512, 1024):
        if k <= len(s):
            out[k] = (s[:k] ** 2).sum().item() / total * 100.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='checkpoints_dsv4/attention')
    ap.add_argument('--out', default='checkpoints_dsv4/pod')
    ap.add_argument('--kv-rank', type=int, default=384)
    ap.add_argument('--q-rank', type=int, default=384)
    ap.add_argument('--x-rank', type=int, default=512)
    ap.add_argument('--skip-xout', action='store_true', default=True,
                    help='skip multi-stream layer-output SVD (redundant with x_in for the base)')
    ap.add_argument('--no-cuda', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = []

    # layers present
    kv_files = sorted(glob.glob(os.path.join(args.src, 'kv_L*.pt')))
    layers = [int(os.path.basename(f).split('_L')[1].split('.')[0]) for f in kv_files]

    pooled_x_in = []

    for li in layers:
        kv = torch.load(os.path.join(args.src, f'kv_L{li}.pt'))
        pkv, mkv, skv = svd_basis(kv, args.kv_rank)
        torch.save(pkv, os.path.join(args.out, f'P_kv_L{li}.pt'))
        torch.save(mkv, os.path.join(args.out, f'mean_kv_L{li}.pt'))
        report.append(f'kv L{li}: ' + ' '.join(f'top-{k}={v:.3f}%' for k, v in variance_fractions(skv).items()))

        xin_path = os.path.join(args.src, f'x_L{li}.pt')
        if os.path.exists(xin_path):
            xin = torch.load(xin_path)
            pooled_x_in.append(xin)
            pin, minx, sin = svd_basis(xin, args.x_rank, use_cuda=not args.no_cuda)
            torch.save(pin, os.path.join(args.out, f'P_x_in_L{li}.pt'))
            torch.save(minx, os.path.join(args.out, f'mean_x_in_L{li}.pt'))
            report.append(f'x_in L{li}: ' + ' '.join(f'top-{k}={v:.3f}%' for k, v in variance_fractions(sin).items()))

        xout_path = os.path.join(args.src, f'xout_L{li}.pt')
        if os.path.exists(xout_path) and not args.skip_xout:
            xout = torch.load(xout_path)
            pout, mout, sout = svd_basis(xout, args.x_rank, use_cuda=not args.no_cuda)
            torch.save(pout, os.path.join(args.out, f'P_x_out_L{li}.pt'))
            torch.save(mout, os.path.join(args.out, f'mean_x_out_L{li}.pt'))
            report.append(f'x_out L{li}: ' + ' '.join(f'top-{k}={v:.3f}%' for k, v in variance_fractions(sout).items()))

        q_path = os.path.join(args.src, f'q_L{li}.pt')
        if os.path.exists(q_path):
            q = torch.load(q_path)
            pq, mq, sq = svd_basis(q, args.q_rank, use_cuda=not args.no_cuda)
            torch.save(pq, os.path.join(args.out, f'P_q_L{li}.pt'))
            torch.save(mq, os.path.join(args.out, f'mean_q_L{li}.pt'))
            report.append(f'q L{li} (heads pooled): ' + ' '.join(f'top-{k}={v:.3f}%' for k, v in variance_fractions(sq).items()))

    if pooled_x_in:
        xall = torch.cat(pooled_x_in, dim=0)
        pglobal, mglobal, sglobal = svd_basis(xall, args.x_rank)
        torch.save(pglobal, os.path.join(args.out, 'P_x_global.pt'))
        torch.save(mglobal, os.path.join(args.out, 'mean_x_global.pt'))
        report.append('x_in GLOBAL: ' + ' '.join(f'top-{k}={v:.3f}%' for k, v in variance_fractions(sglobal).items()))

    with open(os.path.join(args.out, 'rank_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report) + '\n')
    print('\n'.join(report))
    print(f'\nsaved POD bases to {args.out}', flush=True)


if __name__ == '__main__':
    main()
