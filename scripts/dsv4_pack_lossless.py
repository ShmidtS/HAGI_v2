"""Lossless-pack the DeepSeek-V4 experts with zstd.

Reads each per-layer safetensors from lossless_layers/ (packed FP4/FP8 weights
+ E8M0 scales), zstd-compresses the raw file bytes (streaming, memory-light),
writes dsv4_packed/<name>.zst, and verifies a byte-exact SHA256 roundtrip.

Measured composition of one layer file (3.46 GB):
  - I8 packed FP4 weights: 3.22 GB (93%) -> zstd ~96.4% (at the 3.86-bit entropy
    limit of the FP4 symbols)
  - F8_E8M0 scales: 0.20 GB (5.8%) -> zstd ~19.7% (4 values, ~1 bit/element real
    information, near the ~12.5% theoretical floor)
  - misc (gate, tid2eid, FP8): ~0.04 GB
Expected total: ~91% of the original (158.7 GB -> ~145 GB), all lossless.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import zstandard as zstd

LOSSLESS = "C:/HAGI_v2/lossless_layers"
PACKED = "C:/HAGI_v2/dsv4_packed"
LEVEL = 1
CHUNK = 8 * 1024 * 1024


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def compress_file(src: str, dst: str, level: int) -> None:
    cctx = zstd.ZstdCompressor(level=level)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        cctx.copy_stream(fin, fout)


def decompress_and_hash(path: str) -> str:
    dctx = zstd.ZstdDecompressor()
    h = hashlib.sha256()
    with open(path, "rb") as fin:
        reader = dctx.stream_reader(fin)
        while True:
            chunk = reader.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted(f for f in os.listdir(LOSSLESS) if f.endswith(".safetensors"))
    os.makedirs(PACKED, exist_ok=True)
    print(f"{len(files)} files, zstd level {LEVEL}", flush=True)

    manifest = {}
    total_in = total_out = 0
    t0 = time.time()
    for i, fname in enumerate(files):
        src = os.path.join(LOSSLESS, fname)
        dst = os.path.join(PACKED, fname + ".zst")
        n_in = os.path.getsize(src)
        compress_file(src, dst, LEVEL)
        n_out = os.path.getsize(dst)
        total_in += n_in
        total_out += n_out
        print(f"[{i + 1:2d}/{len(files)}] {fname}: {n_in / 1e9:.3f}GB -> "
              f"{n_out / 1e9:.3f}GB ({n_out / n_in * 100:.1f}%)", flush=True)
        manifest[fname] = {"in_bytes": n_in, "out_bytes": n_out, "ratio": n_out / n_in}

    print(f"compression done in {time.time() - t0:.1f}s: "
          f"{total_in / 1e9:.2f}GB -> {total_out / 1e9:.2f}GB "
          f"({total_out / total_in * 100:.1f}%)", flush=True)

    print("verifying byte-exact roundtrip (SHA256)...", flush=True)
    t0 = time.time()
    ok = True
    for i, fname in enumerate(files):
        src = os.path.join(LOSSLESS, fname)
        dst = os.path.join(PACKED, fname + ".zst")
        good = sha256_file(src) == decompress_and_hash(dst)
        ok = ok and good
        print(f"[{i + 1:2d}/{len(files)}] {fname}: "
              f"roundtrip {'OK' if good else 'MISMATCH'}", flush=True)

    print(f"verification done in {time.time() - t0:.1f}s: "
          f"{'ALL OK' if ok else 'FAILURES PRESENT'}", flush=True)

    with open(os.path.join(PACKED, "manifest.json"), "w") as f:
        json.dump({
            "level": LEVEL,
            "verified": ok,
            "total_in_bytes": total_in,
            "total_out_bytes": total_out,
            "overall_ratio": total_out / total_in,
            "files": manifest,
        }, f, indent=2)
    print(f"manifest written to {os.path.join(PACKED, 'manifest.json')}", flush=True)


if __name__ == "__main__":
    main()
