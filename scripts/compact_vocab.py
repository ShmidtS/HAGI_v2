"""Compact the vocabulary to the ids that actually occur in the corpus.

A tokenizer's alphabet is chosen for coverage across many languages and domains;
a specific corpus uses a subset of it. Measured on this corpus: 176,512 of
262,144 ids ever appear, and the top 131,072 by frequency carry 99.93% of all
token mass.

Emitting logits for unreachable symbols costs twice. The codebook holds ``V*H``
parameters whether or not a row is ever selected, and the softmax normalizes over
all V columns, so the model must actively push unreachable logits down — capacity
spent suppressing symbols that cannot occur.

This script builds a bidirectional id map and rewrites the ``.bin`` streams
against it. The tokenizer is unchanged; encode and decode go through the map, so
the compaction is invisible above the data layer.

Usage:
    python scripts/count_unigram.py --data-dir data          # counts first
    python scripts/compact_vocab.py --data-dir data --target-vocab 131072
    # then set model.vocab_size to the reported new size

Outputs (all in ``--data-dir``):
    vocab_map.npz        old_to_new [V_old] int32 (-1 = dropped), new_to_old [V_new] int32
    <name>.compact.bin   rewritten token streams
    unigram.compact.npy  counts in the new id space
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BLOCK = 1 << 24


def build_map(
    counts: np.ndarray, target_vocab: int, keep_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Map the most frequent ids into a dense range, dropping the rest.

    Args:
        counts: ``[V_old]`` occurrence counts.
        target_vocab: size of the new alphabet.
        keep_ids: ids that must survive regardless of frequency (EOS, PAD, and
            any special token the tokenizer or serialization format relies on —
            dropping one of those silently corrupts every document boundary).

    Returns:
        ``(old_to_new, new_to_old)``; ``old_to_new`` is -1 for dropped ids.

    Raises:
        ValueError: if ``target_vocab`` cannot hold the reserved ids.
    """
    v_old = len(counts)
    if target_vocab > v_old:
        raise ValueError(f"target_vocab {target_vocab} exceeds current vocabulary {v_old}")
    reserved = sorted(set(keep_ids))
    if len(reserved) > target_vocab:
        raise ValueError(f"{len(reserved)} reserved ids do not fit in target_vocab {target_vocab}")

    ranked = np.argsort(counts)[::-1]
    chosen: list[int] = list(reserved)
    seen = set(reserved)
    for old_id in ranked:
        if len(chosen) >= target_vocab:
            break
        old_id = int(old_id)
        if old_id not in seen:
            chosen.append(old_id)
            seen.add(old_id)

    new_to_old = np.array(chosen, dtype=np.int32)
    old_to_new = np.full(v_old, -1, dtype=np.int32)
    old_to_new[new_to_old] = np.arange(len(new_to_old), dtype=np.int32)
    return old_to_new, new_to_old


def rewrite_stream(
    source: Path, dest: Path, old_to_new: np.ndarray, fallback: int, drop: bool = False
) -> tuple[int, int]:
    """Rewrite a token stream through the id map.

    Dropped ids are either replaced by ``fallback`` (the V33 default: keeps
    positions and document lengths, but makes UNK a common token the model
    learns to emit) or **deleted** (``drop=True``: the stream is filtered, so
    UNK appears only when the tokenizer itself produced it). Deletion shifts
    positions, but document boundaries are marked by EOS (a reserved id, never
    dropped), so ``doc_ids = cumsum(is_eos)`` in the loader stays correct.

    Returns:
        ``(tokens_written, tokens_remapped_or_dropped)``.
    """
    tokens = np.memmap(source, dtype=np.uint32, mode="r")
    total = len(tokens)
    replaced = 0
    written = 0
    with open(dest, "wb") as out:
        for start in range(0, total, BLOCK):
            chunk = np.asarray(tokens[start : min(start + BLOCK, total)], dtype=np.int64)
            mapped = old_to_new[chunk]
            missing = mapped < 0
            replaced += int(missing.sum())
            if drop:
                if missing.any():
                    mapped = mapped[~missing]
            else:
                mapped[missing] = fallback
            if mapped.size:
                out.write(mapped.astype(np.uint32).tobytes())
                written += mapped.size
    return written, replaced


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--counts", default=None, help="default: <data-dir>/unigram.npy")
    ap.add_argument("--target-vocab", type=int, default=131072)
    ap.add_argument("--eos-token-id", type=int, default=1)
    ap.add_argument("--pad-token-id", type=int, default=0)
    ap.add_argument(
        "--reserve-below",
        type=int,
        default=256,
        help="keep every id below this value (tokenizers place special tokens and byte fallbacks low)",
    )
    ap.add_argument("--unk-token-id", type=int, default=3, help="old id used for dropped tokens")
    ap.add_argument(
        "--drop",
        action="store_true",
        help="delete dropped ids from the stream instead of remapping to UNK. "
        "Keeps UNK out of the training data (the model does not learn to emit "
        "<unk>), at a ~6% corpus-mass loss. Document boundaries (EOS) are "
        "preserved. Use this for a clean retrain.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    counts_path = Path(args.counts) if args.counts else data_dir / "unigram.npy"
    if not counts_path.exists():
        raise SystemExit(f"counts not found: {counts_path}\nrun scripts/count_unigram.py first")
    counts = np.load(counts_path)

    keep = list(range(min(args.reserve_below, len(counts))))
    keep += [args.eos_token_id, args.pad_token_id, args.unk_token_id]
    old_to_new, new_to_old = build_map(counts, args.target_vocab, keep)

    total_mass = counts.sum()
    kept_mass = counts[new_to_old].sum()
    print(f"vocabulary {len(counts)} -> {len(new_to_old)}")
    print(f"token mass retained: {kept_mass / total_mass:.6f}")
    print(f"eos {args.eos_token_id} -> {int(old_to_new[args.eos_token_id])}")
    print(f"pad {args.pad_token_id} -> {int(old_to_new[args.pad_token_id])}")

    fallback = int(old_to_new[args.unk_token_id])
    if fallback < 0:
        raise SystemExit(f"unk token {args.unk_token_id} was not retained; pass a reserved id")

    map_path = data_dir / "vocab_map.npz"
    np.savez(map_path, old_to_new=old_to_new, new_to_old=new_to_old)
    print(f"saved: {map_path}")

    new_counts = np.zeros(len(new_to_old), dtype=np.int64)
    new_counts[:] = counts[new_to_old]
    np.save(data_dir / "unigram.compact.npy", new_counts)
    print(f"saved: {data_dir / 'unigram.compact.npy'}")

    # Recompact from the *current* compact streams when they exist (a second
    # compaction run: 131072 -> 65536), else from the raw tokenizer streams.
    # The map's input space must match the file's id space; for a recompaction
    # that is the existing compact space, which is len(old_to_new) = len(counts).
    #
    # ``unigram.compact.npy`` must be rebuilt for the *final* map, not the
    # previous one: the per-id counts shift when ids are dropped again (dropped
    # ids become UNK in the stream, and the retained ids' ranks change). The
    # old code left the counts from the first pass in place, so the head's
    # log-prior disagreed with the training stream by orders of magnitude —
    # the model then learned UNK as a high-frequency token and emitted <unk>
    # at inference. Recompute counts from the rewritten stream so the prior is
    # the actual empirical distribution.
    candidates = sorted(data_dir.glob("*.compact.bin"))
    if candidates:
        new_counts = np.zeros(len(new_to_old), dtype=np.int64)
        fallback_id = int(old_to_new[args.unk_token_id])
        for source in candidates:
            dest = source.with_name(source.name.replace(".compact.bin", ".compact2.bin"))
            written, replaced = rewrite_stream(source, dest, old_to_new, fallback, drop=args.drop)
            # Count ids directly from the rewritten stream (blocked to bound
            # memory); this is the empirical distribution the model will see.
            tokens = np.memmap(dest, dtype=np.uint32, mode="r")
            for start in range(0, len(tokens), BLOCK):
                chunk = np.asarray(tokens[start : min(start + BLOCK, len(tokens))], dtype=np.int64)
                new_counts += np.bincount(chunk, minlength=len(new_to_old))
            mode = "dropped" if args.drop else "unk"
            print(f"{source.name} -> {dest.name}: {written / 1e6:.1f}M tokens, "
                  f"{replaced / max(written, 1):.6f} {mode}")
        np.save(data_dir / "unigram.compact.npy", new_counts)
        print(f"rebuilt: {data_dir / 'unigram.compact.npy'} (from final stream)")
        return 0

    for source in sorted(data_dir.glob("*.bin")):
        if source.name.endswith(".compact.bin"):
            continue
        dest = source.with_suffix(".compact.bin")
        written, replaced = rewrite_stream(source, dest, old_to_new, fallback, drop=args.drop)
        mode = "dropped" if args.drop else "unk"
        print(f"{source.name} -> {dest.name}: {written / 1e6:.1f}M tokens, "
              f"{replaced / max(written, 1):.6f} {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
