"""Rebuild .compact2.bin directly from raw .bin (262144-space) using a full map.

The recompaction path in compact_vocab.py builds 65536->32768 from the 65536-space
.compact.bin, losing the original 262144-space identity. For inference we need the
FULL map 262144->32768 (tokenizer emits old-space ids). Rebuilding from .bin makes
map and binaries consistent by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from compact_vocab import build_map, rewrite_stream  # noqa: E402

data_dir = Path("data")
counts = np.load(data_dir / "unigram.npy")
keep = list(range(min(256, len(counts)))) + [1, 0, 3]
old_to_new, new_to_old = build_map(counts, 32768, keep)

total_mass = counts.sum()
print(f"vocabulary {len(counts)} -> {len(new_to_old)}")
print(f"token mass retained: {counts[new_to_old].sum() / total_mass:.6f}")

fallback = int(old_to_new[3])
assert fallback >= 0, "unk token not retained"

np.savez(data_dir / "vocab_map.npz", old_to_new=old_to_new, new_to_old=new_to_old)
np.save(data_dir / "unigram.compact.npy", counts[new_to_old])
print(f"saved: {data_dir / 'vocab_map.npz'}")

for source in sorted((data_dir).glob("*.bin")):
    if source.name.endswith((".compact.bin", ".compact2.bin")):
        continue
    dest = source.with_suffix(".compact2.bin")
    written, replaced = rewrite_stream(source, dest, old_to_new, fallback)
    print(f"{source.name} -> {dest.name}: {written / 1e6:.1f}M tokens, {replaced / max(written, 1):.6f} unk")

print("done")
