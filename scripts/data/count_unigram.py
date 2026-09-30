"""Count token unigram frequencies over the packed .bin corpus.

The unigram distribution is the zero-order source model. Feeding
``log p_unigram`` into the LM head as a FIXED logit bias means the network only
has to learn the *conditional* correction on top of the zero-order code — the
information-theoretic definition of residual coding. Measured on this corpus it
removes ~4.4 nats/token of trivial work at step 0 (unigram entropy 8.06 nats vs
ln V = 12.48).

Usage:
    python scripts/count_unigram.py --data-dir data --vocab-size 262144
    python scripts/count_unigram.py --limit-per-file 200000000   # subsample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def count(data_dir: Path, vocab_size: int, limit_per_file: int | None) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=np.int64)
    for path in sorted(data_dir.glob("*.bin")):
        arr = np.memmap(path, dtype=np.uint32, mode="r")
        take = len(arr) if limit_per_file is None else min(len(arr), limit_per_file)
        block = 1 << 24
        for start in range(0, take, block):
            chunk = np.asarray(arr[start : min(start + block, take)])
            counts += np.bincount(chunk, minlength=vocab_size)
        print(f"{path.name}: {take / 1e6:.1f}M tokens counted", flush=True)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--vocab-size", type=int, default=262144)
    ap.add_argument("--limit-per-file", type=int, default=None)
    ap.add_argument("--out", default=None, help="default: <data-dir>/unigram.npy")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    counts = count(data_dir, args.vocab_size, args.limit_per_file)
    out = Path(args.out) if args.out else data_dir / "unigram.npy"
    np.save(out, counts)

    total = counts.sum()
    probs = counts[counts > 0] / total
    entropy = float(-(probs * np.log(probs)).sum())
    print(f"total {total / 1e6:.1f}M tokens | distinct {int((counts > 0).sum())}/{args.vocab_size}")
    print(f"unigram entropy {entropy:.4f} nats ({entropy / np.log(2):.4f} bits) | ln V {np.log(args.vocab_size):.4f}")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
