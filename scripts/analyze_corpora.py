"""Corpus correlation analysis for expert-domain sorting.

For each ``<name>.compact.bin`` corpus, sample a fixed number of tokens, build
a unigram profile (log-probability over the compact vocab), and compute the
pairwise correlation matrix. Weakly-correlated corpora are the best expert
domains (each small expert specializes on a decorrelated source, so the
ternary merge later recombines genuinely different subspaces).

Outputs:
    - pairwise correlation matrix (Pearson on log-profiles)
    - a suggested greedy partition into ``n_groups`` groups, each containing
      maximally decorrelated corpora (an expert domain per group).

Usage:
    python scripts/analyze_corpora.py --data-dir data --samples 20000000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def unigram_profile(path: Path, vocab_size: int, samples: int) -> np.ndarray:
    arr = np.memmap(path, dtype=np.uint32, mode="r")
    take = min(len(arr), samples)
    counts = np.zeros(vocab_size, dtype=np.int64)
    block = 1 << 24
    for start in range(0, take, block):
        chunk = np.asarray(arr[start : min(start + block, take)])
        counts += np.bincount(chunk, minlength=vocab_size)
    total = counts.sum()
    p = (counts + 1.0) / (total + vocab_size)  # smoothed
    return np.log(p)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def greedy_partition(names: list[str], corr: np.ndarray, n_groups: int) -> list[list[str]]:
    """Greedily assign corpora to groups minimizing intra-group correlation.

    Each corpus goes to the group whose current members it correlates with the
    least on average. Balances group sizes.
    """
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    # Seed: n_groups most-mutually-decorrelated corpora.
    n = len(names)
    score = [sum(abs(corr[i, j]) for j in range(n)) for i in range(n)]
    seeds = sorted(range(n), key=lambda i: score[i])[:n_groups]
    for g, i in enumerate(seeds):
        groups[g].append(names[i])
    rest = [i for i in range(n) if i not in seeds]
    # Assign by minimal average |corr| to the target group.
    for i in sorted(rest, key=lambda i: score[i]):
        best_g = min(
            range(n_groups),
            key=lambda g: (
                sum(abs(corr[i, names.index(m)]) for m in groups[g]) / max(1, len(groups[g])),
                len(groups[g]),
            ),
        )
        groups[best_g].append(names[i])
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--samples", type=int, default=20_000_000)
    ap.add_argument("--groups", type=int, default=3)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.compact.bin"))
    if not paths:
        print("no .compact.bin corpora found")
        return 1
    names = [p.name.replace(".compact.bin", "") for p in paths]
    print(f"corpora ({len(names)}): {', '.join(names)}")
    print(f"sampling {args.samples/1e6:.0f}M tokens per corpus\n")

    profs = []
    for p in paths:
        prof = unigram_profile(p, args.vocab_size, args.samples)
        profs.append(prof)
        print(f"  {p.name}: done", flush=True)

    n = len(names)
    corr = np.zeros((n, n))
    print(f"\n{'':18s}" + "".join(f"{m[:7]:>8s}" for m in names))
    for i in range(n):
        row = []
        for j in range(n):
            c = pearson(profs[i], profs[j]) if i != j else 1.0
            corr[i, j] = c
            row.append(f"{c:8.3f}")
        print(f"{names[i][:18]:18s}" + "".join(row))

    groups = greedy_partition(names, corr, args.groups)
    print(f"\ngreedy partition into {args.groups} decorrelated groups:")
    for g, members in enumerate(groups):
        # intra-group mean |corr|
        idx = [names.index(m) for m in members]
        intra = np.mean([abs(corr[i, j]) for i in idx for j in idx if i != j]) if len(idx) > 1 else 0.0
        print(f"  group {g}: {members}  (mean |corr| {intra:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
