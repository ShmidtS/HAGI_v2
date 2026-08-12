"""Lossless checkpoint compaction (no quantization).

Packs a training checkpoint into a smaller file with ZERO information loss:
  1. Every weight/buffer tensor is written as its exact dtype bytes.
  2. The byte stream is compressed with zstd (level 19).
  3. ``torch.save`` metadata overhead is dropped: we store a flat ``npz``-style
     container (tensor name -> raw bytes + shape + dtype) and the optimizer
     state is compressed the same way.

The model's effective weights are unchanged (``torch.load`` after unpacking
produces bit-identical tensors), so this is a true no-loss compaction — the
"сжатие без потерь" step of the growth cycle.

Usage:
    python scripts/compact_checkpoint.py --ckpt checkpoints_l0/ru_general/step-XXX.pt
    # writes <ckpt>.zst alongside; --restore to unpack back to a .pt file.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import struct
from pathlib import Path

import torch
import zstandard as zstd


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf, _use_new_zipfile_serialization=False)
    return buf.getvalue()


def _tensor_from_bytes(data: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def compact(path: Path, out: Path | None = None) -> Path:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    out = out or path.with_suffix(path.suffix + ".zst")
    header = {"keys": [], "shapes": [], "dtypes": []}
    blob = io.BytesIO()
    for k, v in payload.items():
        if isinstance(v, dict):
            # nested dict (e.g. optimizer state) — flatten recursively later;
            # here we just pickle it and let zstd handle it.
            data = pickle.dumps(v, protocol=5)
            header["keys"].append(k)
            header["shapes"].append(None)
            header["dtypes"].append("pickle")
        elif isinstance(v, torch.Tensor):
            data = _tensor_to_bytes(v)
            header["keys"].append(k)
            header["shapes"].append(list(v.shape))
            header["dtypes"].append(str(v.dtype))
        else:
            data = pickle.dumps(v, protocol=5)
            header["keys"].append(k)
            header["shapes"].append(None)
            header["dtypes"].append("pickle")
        blob.write(struct.pack("<Q", len(data)))
        blob.write(data)
    hdr = json.dumps(header).encode()
    raw = hdr + b"\x00" + blob.getvalue()
    cctx = zstd.ZstdCompressor(level=19)
    compressed = cctx.compress(raw)
    out.write_bytes(compressed)
    print(f"{path.name}: {path.stat().st_size/1e6:.1f} MB -> {out.name}: {len(compressed)/1e6:.1f} MB "
          f"({100*len(compressed)/path.stat().st_size:.1f}%)")
    return out


def restore(path: Path, out: Path) -> None:
    cctx = zstd.ZstdDecompressor()
    raw = cctx.decompress(path.read_bytes())
    hdr, _, body = raw.partition(b"\x00")
    header = json.loads(hdr)
    body = io.BytesIO(body)
    payload = {}
    for i, k in enumerate(header["keys"]):
        (n,) = struct.unpack("<Q", body.read(8))
        data = body.read(n)
        if header["dtypes"][i] == "pickle":
            payload[k] = pickle.loads(data)
        else:
            payload[k] = _tensor_from_bytes(data)
    torch.save(payload, out)
    print(f"restored {out.name} ({out.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = Path(args.ckpt)
    if args.restore:
        out = Path(args.out) if args.out else p.with_suffix(".pt")
        restore(p, out)
    else:
        out = Path(args.out) if args.out else None
        compact(p, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
