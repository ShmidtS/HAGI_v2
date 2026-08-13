"""Loop dsv4_reduce_layer.py over all 43 layers (resumable).

Usage: python scripts/dsv4_reduce_all.py [--start 0] [--end 43]
"""
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
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=43)
    args = ap.parse_args()

    t0 = time.time()
    for L in range(args.start, args.end):
        if layer_done(L):
            print(f'[{L}/43] layer {L} done, skip', flush=True)
            continue
        print(f'[{L}/43] reducing layer {L}...', flush=True)
        r = subprocess.run(
            [sys.executable, os.path.join(_ROOT, 'scripts', 'dsv4_reduce_layer.py'), str(L)]
        )
        if r.returncode != 0:
            print(f'layer {L} FAILED (rc={r.returncode}), stopping', flush=True)
            sys.exit(1)
        print(f'[{L}/43] layer {L} done, elapsed {time.time()-t0:.0f}s', flush=True)
    print(f'ALL DONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
