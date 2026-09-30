"""Concatenate packed shards into the flat `<name>.bin` the loader expects.

`src/hagi/data/dataset.py:44` resolves `data/<name>.compact.bin` (preferred)
or `data/<name>.bin`, and reads one flat stream per source. The packer writes
`data/<name>/shards/NNNNNN.bin` because a single multi-GB file is awkward to
write and resume. This bridges the two: shards in, one flat stream out.

Deleting the shards afterwards is opt-in via --remove-shards, since the flat
copy is redundant but the shards are the resumable form.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

CHUNK = 1 << 24  # 16M int32 = 64 MB per read/write, bounded regardless of size


def concat(name: str, data_dir: Path, remove: bool) -> tuple[int, int]:
    shard_dir = data_dir / name / "shards"
    shards = sorted(shard_dir.glob("*.bin"))
    if not shards:
        return (0, 0)
    out = data_dir / f"{name}.bin"
    tokens = 0
    with open(out, "wb") as handle:
        for shard in shards:
            data = np.fromfile(shard, dtype=np.int32)
            if data.size == 0:
                continue
            handle.write(data.tobytes())
            tokens += int(data.size)
            del data
    if remove:
        for shard in shards:
            os.remove(shard)
        try:
            shard_dir.rmdir()
            (data_dir / name).rmdir()
        except OSError:
            pass
    return (tokens, out.stat().st_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--remove-shards", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    names = sorted(p.name for p in data_dir.iterdir() if (p / "shards").is_dir())
    if args.only:
        names = [n for n in names if n in set(args.only)]

    total = 0
    for name in names:
        tokens, size = concat(name, data_dir, args.remove_shards)
        if tokens:
            total += tokens
            print(f"{name:18s} {tokens:>14,} tokens  {size/1e9:6.2f} GB")
    print(f"{'TOTAL':18s} {total:>14,} tokens")


if __name__ == "__main__":
    main()
