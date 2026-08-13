"""Robust parallel per-expert reduce over layers 1..42.

Replaces the broken bash launcher (git-bash `$(jobs -r)` can't see background
jobs from a subshell). Uses subprocess PIDs for a reliable concurrency pool.
Resume-safe: skips layers whose 256 experts + P.pt already exist.

Usage: python scripts/dsv4_reduce_parallel.py [--conc 4] [--start 1] [--end 43]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REDUCED = os.path.join(_ROOT, 'dsv4_reduced')


def layer_done(L: int) -> bool:
    d = os.path.join(REDUCED, f'layer_{L}')
    if not os.path.exists(os.path.join(d, 'P.pt')):
        return False
    return all(os.path.exists(os.path.join(d, f'expert_{k}.pt')) for k in range(256))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conc', type=int, default=4)
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=43)
    args = ap.parse_args()

    pending = [L for L in range(args.start, args.end) if not layer_done(L)]
    print(f'{len(pending)} layers pending: {pending[:10]}{"..." if len(pending) > 10 else ""}', flush=True)

    procs: dict[int, subprocess.Popen] = {}
    t0 = time.time()

    def reap():
        for k in list(procs):
            if procs[k].poll() is not None:
                rc = procs[k].returncode
                print(f'layer {k} finished rc={rc} ({time.time()-t0:.0f}s)', flush=True)
                del procs[k]

    while pending or procs:
        reap()
        # launch while there is room and work
        while pending and len(procs) < args.conc:
            L = pending.pop(0)
            logf = open(os.path.join(_ROOT, f'dsv4_reduce_L{L}.log'), 'a', encoding='utf-8')
            p = subprocess.Popen(
                [sys.executable, os.path.join(_ROOT, 'scripts', 'dsv4_reduce_layer.py'), str(L)],
                stdout=logf, stderr=subprocess.STDOUT, cwd=_ROOT,
            )
            logf.close()
            procs[L] = p
            print(f'launched layer {L} (running={len(procs)})', flush=True)
        time.sleep(15)

    print(f'ALL REDUCE DONE in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
