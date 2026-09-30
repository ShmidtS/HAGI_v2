"""Tokenize the raw JSONL corpora on the network share into packed .bin shards.

The packed corpus was deleted (24 GB, all of it under the old gemma id space).
The raw text is intact on the share, so this rebuilds it rather than
re-downloading 24 GB.

Tokenizer choice is settled by measurement, not preference: a self-trained
32k BPE on the same mix lost to gemma-262k-pruned-to-32k by roughly 2x on
every domain (637 vs 306 tokens per 1000 characters on Russian), at every
vocab size tried up to 262144. See docs/TOKENIZER_DECISION.md. So this packs
with gemma and reuses the project's existing compaction pass afterwards.

Files are processed independently and sequentially: one trainer, one job, and
the share is a network resource that should not be hammered in parallel.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import time
from pathlib import Path

BS = chr(92)
RAW = BS * 2 + "Home" + BS + "1" + BS + "raw"

# name, shard tokens. shard-tokens is per output shard, so a 5.8 GB source
# produces several 1M-token files rather than one enormous one.
SOURCES: list[tuple[str, str]] = [
    ("wikipedia_ru.jsonl",    "wikipedia_ru"),
    ("oscar_ru.jsonl",        "oscar_ru"),
    ("wikipedia_en.jsonl",    "wikipedia_en"),
    ("slimpajama.jsonl",      "slimpajama"),
    ("edu.jsonl",             "edu"),
    ("openwebmath.jsonl",     "openwebmath"),
    ("smoltalk.jsonl",        "smoltalk"),
    ("python_instruct.jsonl", "python_instruct"),
    ("tinystories.jsonl",     "tinystories"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="google/gemma-4-E2B-it")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--shard-tokens", type=int, default=1_000_000)
    ap.add_argument("--max-input-bytes", type=int, default=64 * 1024 * 1024)
    ap.add_argument("--only", nargs="*", help="restrict to these source names")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(args.only or [])
    todo = [(f, n) for f, n in SOURCES if not want or n in want]

    for filename, name in todo:
        src = os.path.join(RAW, filename)
        if not os.path.exists(src):
            print(f"SKIP {name}: {filename} not on the share", flush=True)
            continue
        dest = out_dir / (name if not want else name)
        if dest.exists() and any(dest.glob("*.bin")):
            print(f"SKIP {name}: already packed", flush=True)
            continue
        if dest.exists():
            import shutil
            shutil.rmtree(dest)
        size_mb = os.path.getsize(src) / 1e6
        print(f"packing {name} ({filename}, {size_mb:.0f} MB)", flush=True)
        t0 = time.time()
        cmd = [
            sys.executable, "scripts/research/prepare_training_data.py", "prepare",
            src, str(dest),
            "--tokenizer", args.tokenizer,
            "--source-name", name,
            "--shard-tokens", str(args.shard_tokens),
            "--max-input-bytes", str(args.max_input_bytes),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  FAILED rc={proc.returncode}")
            print("  " + (proc.stderr or "").strip()[-600:])
            continue
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        for line in tail:
            print(f"  {line}")
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print("\nmanifest:")
    for filename, name in SOURCES:
        dest = out_dir / (name if not want else name)
        if dest.exists() and any(dest.glob("*.bin")):
            shards = sorted(dest.glob("*.bin"))
            total = sum(p.stat().st_size for p in shards)
            print(f"  {name:18s} {len(shards):3d} shards  {total/1e9:6.2f} GB")


if __name__ == "__main__":
    main()
