#!/usr/bin/env python3
"""HAGI as a lossless archiver (the LM is the arithmetic coder's model).

  python scripts/hagi_archive.py compress   IN OUT [--domain NAME] [--prefix-head 0.8]
  python scripts/hagi_archive.py decompress IN OUT [--domain NAME] [--prefix FILE]
  python scripts/hagi_archive.py bench      [N]     # ratio vs zlib on real chunks

--prefix-head F : keep the first F of the file OUT of the archive and
  feed it to the decoder as known prefix -> stream codec HAGI4 with a
  multi-order context table (WAVE 264); the plain archive HAGI3 is
  kept when it is smaller. --prefix FILE supplies that head back.

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
    ap.add_argument("--prefix-head", type=float, default=None,
                    help="fraction of the file kept as a known prefix (stream codec)")
    ap.add_argument("--prefix", default=None, help="prefix file for decompress")
    a = ap.parse_args()
    import numpy as np
    from hagi import archiver as A
    from hagi.unified import Unified
    u = Unified.load()
    if a.cmd == "compress":
        import pathlib
        data = pathlib.Path(a.paths[0]).read_bytes()
        if a.prefix_head:
            cut = int(len(data) * a.prefix_head)
            pathlib.Path(a.paths[1] + ".prefix").write_bytes(data[:cut])
            payload = A.compress_stream(u.model, data[cut:], prefix=data[:cut], lam=0.70)
            pathlib.Path(a.paths[1]).write_bytes(payload)
            raw, out = len(data) - cut, len(payload)
            print(f"compressed {a.paths[0]} -> {a.paths[1]} (tail {raw:,} B, prefix {cut:,} B kept in {a.paths[1]}.prefix): "
                  f"{out:,} B ({out * 8 / max(raw, 1):.3f} b/B, format {payload[:5]!r})")
        else:
            raw, out = A.compress_file(u, a.paths[0], a.paths[1], a.domain)
            print(f"compressed {a.paths[0]} -> {a.paths[1]}: {raw:,} -> {out:,} B ({out * 8 / max(raw, 1):.3f} b/B)")
    elif a.cmd == "decompress":
        import pathlib
        if a.prefix:
            payload = pathlib.Path(a.paths[0]).read_bytes()
            data = A.decompress_stream(u.model, payload, prefix=pathlib.Path(a.prefix).read_bytes())
            pathlib.Path(a.paths[1]).write_bytes(data)
        else:
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
