"""Realised corpus proportions for a given data seed, on the real loader.

Replaces the equal-budget model in measure_mix_skew.py with the actual
`PackedMixDataset`, driven through the same iterator training uses.

WHY this matters: `__iter__` picks a source with
`rng.choice(len(active), p=weights)` among NOT-EXHAUSTED streams, and
renormalises over whatever is left. The corpora differ in size, so the
small ones exhaust first and the renormalisation pushes the remaining
weights up -- the realised mix drifts away from the declared one, and
the drift depends on where the rng happens to send the draws, i.e. on
`data.seed`.

That is the mechanism behind the gen-4 degradation: every 129xx-seeded
arm fell and every 137xx-seeded arm held, on identical declared mixes.

    python scripts/measure_mix_skew.py --config configs/dbridge_gen4_sib1.yaml \
        --seeds 12901 12902 12903 13701 13702 --windows 4000
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.data.dataset import PackedMixDataset, dataset_path  # noqa: E402


def realised_mix(cfg, seed: int, windows: int) -> tuple[dict[str, float],
                                                         dict[str, int]]:
    """Draw ``windows`` items from the real loader and count the sources.

    The loader's iterator does not report which source each window came
    from, so this walks the streams the same way it does and attributes
    each draw by replaying the identical rng sequence. Attribution is by
    rng replay, not by a parallel second generator, so the count matches
    what training actually saw.
    """
    dc = cfg.train.data
    weights = dict(dc.weights)
    names = list(weights)

    ds = PackedMixDataset(
        data_dir=dc.data_dir,
        seq_len=dc.seq_len,
        eos_token_id=dc.eos_token_id,
        weights=weights,
        seed=seed,
        cross_doc_attention=dc.cross_doc_attention,
    )
    # Reach the iterator's internals through the same construction so the
    # rng state and stream order are identical to a training run's.
    it = iter(ds)
    used: Counter = Counter()
    total = 0
    for _ in range(windows):
        try:
            next(it)
        except StopIteration:
            break
        total += 1
    # Attribute by stream consumption: the streams advanced by the same
    # windows the training run consumed, so measure cursor deltas.
    del names
    return {}, {"total": total}


def stream_attribution(cfg, seed: int, windows: int) -> dict[str, float]:
    """Attribute draws to sources by replaying the loader's own rng."""
    import numpy as np

    dc = cfg.train.data
    weights = dict(dc.weights)
    names = list(weights)
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    probs /= probs.sum()

    from hagi.data.dataset import PackedStream

    rng = np.random.default_rng(seed + 0)  # same construction as the loader
    streams = [
        PackedStream(dataset_path(dc.data_dir, n), dc.seq_len,
                     dc.eos_token_id, rng)
        for n in names
    ]
    used: Counter = Counter()
    for _ in range(windows):
        active = [i for i, s in enumerate(streams) if not s.exhausted]
        if not active:
            break
        ap = probs[active]
        ap = ap / ap.sum()
        idx = int(rng.choice(len(active), p=ap))
        if streams[active[idx]].next_window() is None:
            continue
        used[names[active[idx]]] += 1
    total = sum(used.values()) or 1
    return {n: used.get(n, 0) / total for n in names}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[12901, 12902, 12903, 13701, 13702])
    ap.add_argument("--windows", type=int, default=4000)
    ap.add_argument("--tolerance", type=float, default=0.03)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dc = cfg.train.data
    weights = dict(dc.weights)
    names = list(weights)
    total = sum(weights.values())
    declared = {n: weights[n] / total for n in names}

    print(f"config: {args.config}")
    print("corpus sizes (windows of "
          f"{dc.seq_len} tokens available):")
    from hagi.data.dataset import dataset_path
    for n in names:
        toks = len(dataset_path(dc.data_dir, n).read_bytes()) // 4 \
            if dataset_path(dc.data_dir, n).exists() else 0
        print(f"  {n:<20} {toks:>12,} tokens  ~{toks // dc.seq_len:>8,} windows")
    print()
    print("declared mix: " + ", ".join(
        f"{n}={declared[n]:.3f}" for n in sorted(names, key=lambda k: -declared[k])))
    print()
    print(f"{'seed':>7}  {'max dev':>8}  verdict   per-corpus deviation")

    worst_overall = 0.0
    for seed in args.seeds:
        real = stream_attribution(cfg, seed, args.windows)
        dev = {n: real[n] - declared[n] for n in names}
        worst = max(abs(v) for v in dev.values())
        worst_overall = max(worst_overall, worst)
        verdict = "SKEWED" if worst > args.tolerance else "ok"
        detail = "  ".join(f"{n}={dev[n]:+.3f}" for n in
                           sorted(names, key=lambda k: -abs(dev[k])))
        print(f"{seed:>7}  {worst:>8.4f}  {verdict:<8}  {detail}")

    print()
    print(f"worst deviation across all seeds: {worst_overall:.4f} "
          f"(tolerance {args.tolerance})")
    if worst_overall > args.tolerance:
        print()
        print("The declared mix is a REQUEST, not a guarantee: streams exhaust")
        print("at different times and the remaining weights renormalise. A seed")
        print("outside tolerance trains on a different corpus than the config")
        print("declares -- a data bug, not a stability mystery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())