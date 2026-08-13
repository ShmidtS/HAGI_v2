"""Decompress the zstd-packed experts back to lossless_layers format.

Reads dsv4_packed/<name>.zst and writes the original safetensors bytes back to
a target directory (streaming, memory-light). Verification is via SHA256
(see manifest.json, which records per-file sizes and the global roundtrip flag).

Usage:
    python scripts/dsv4_unpack_lossless.py [target_dir]
"""

from __future__ import annotations

import os
import sys
import time

import zstandard as zstd

PACKED = "C:/HAGI_v2/dsv4_packed"
DEFAULT_TARGET = "C:/HAGI_v2/lossless_layers_restored"
CHUNK = 8 * 1024 * 1024


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    os.makedirs(target, exist_ok=True)
    dctx = zstd.ZstdDecompressor()
    files = sorted(f for f in os.listdir(PACKED) if f.endswith(".zst"))
    t0 = time.time()
    for i, fname in enumerate(files):
        src = os.path.join(PACKED, fname)
        dst = os.path.join(target, fname[: -len(".zst")])
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            dctx.copy_stream(fin, fout)
        print(f"[{i + 1:2d}/{len(files)}] {fname[: -len('.zst')]} restored", flush=True)
    print(f"restored {len(files)} files to {target} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
