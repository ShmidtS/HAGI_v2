#!/usr/bin/env python3
"""HAGI as a lossless archiver (the LM is the arithmetic coder's model).

  python scripts/hagi_archive.py compress   IN OUT [--domain NAME]
  python scripts/hagi_archive.py decompress IN OUT [--domain NAME]
  python scripts/hagi_archive.py bench      [N]     # ratio vs zlib on real chunks

The SAME checkpoint must decode an archive (the payload is tied to it).
"""
import argparse
import hashlib
import os
import pathlib
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compress", "decompress", "bench"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--domain", default=None)
    a = ap.parse_args()
    import numpy as np
    from hagi import archiver as A
    from hagi.unified import Unified
    u = Unified.load()
    if a.cmd == "compress":
        raw, out = A.compress_file(u, a.paths[0], a.paths[1], a.domain)
        print(f"compressed {a.paths[0]} -> {a.paths[1]}: {raw:,} -> {out:,} B ({out * 8 / max(raw, 1):.3f} b/B)")
    elif a.cmd == "decompress":
        data = A.decompress_file(u, a.paths[0], a.paths[1], a.domain)
        print(f"decompressed -> {a.paths[1]}: {len(data):,} B sha256={hashlib.sha256(data).hexdigest()}")
    else:
        n = int(a.paths[0]) if a.paths else 10
        files = sorted(pathlib.Path(u.__class__.__module__ and "checkpoints/domains").glob("ns_*.bin"))[-n:]
        print(f"{'file':14} {'raw':>8} {'ours':>8} {'b/B':>6} {'zlib b/B':>8} {'lossless':>8} {'c/s':>6} {'d/s':>6}")
        for f in files:
            d = f.read_bytes()[:65536]
            t0 = time.perf_counter(); c = A.compress_bytes(u, d); t1 = time.perf_counter()
            back = A.decompress_bytes(u, c); t2 = time.perf_counter()
            z = len(zlib.compress(d, 9))
            print(f"{f.name:14} {len(d):8} {len(c):8} {len(c) * 8 / len(d):6.3f} {z * 8 / len(d):8.3f} "
                  f"{str(back == d):>8} {len(d) / (t1 - t0):6.0f} {len(d) / (t2 - t1):6.0f}")


if __name__ == "__main__":
    main()
