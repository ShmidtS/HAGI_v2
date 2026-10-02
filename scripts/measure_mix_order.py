"""Order sensitivity of the packed mix: is a seed just a permutation?

The gen-4 degradation traced to `data.seed`, but NOT to the realised
mix -- `measure_mix_skew.py` shows the aggregate proportions within 1%
of the declared ones for every seed tried, and the stream starts at the
same offset. What differs is the ORDER in which sources are drawn.

That matters for optimisation independently of data volume: if a seed
produces long runs of a single corpus, the optimiser sees a stretch of
one distribution followed by a stretch of another, instead of a mixture
every step. The gradient statistics of those two regimes differ even when
the totals are identical, and a run of them is a plausible mechanism for
a late-training instability that a seed change removes.

    python scripts/measure_mix_order.py --config configs/dbridge_gen4_sib1.yaml \
        --seeds 12901 12902 12903 13701 13702
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.data.dataset import PackedStream, dataset_path  # noqa: E402


def draw_order(weights: dict[str, float], seed: int, draws: int,
               data_dir: str, seq_len: int, eos: int) -> list[str]:
    """Replay the loader's source selection, returning the draw order."""
    names = list(weights)
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    streams = [
        PackedStream(dataset_path(data_dir, n), seq_len, eos, rng)
        for n in names
    ]
    order: list[str] = []
    for _ in range(draws):
        active = [i for i, s in enumerate(streams) if not s.exhausted]
        if not active:
            break
        ap = probs[active]
        ap = ap / ap.sum()
        idx = int(rng.choice(len(active), p=ap))
        if streams[active[idx]].next_window() is None:
            continue
        order.append(names[active[idx]])
    return order


def run_lengths(order: list[str]) -> list[int]:
    """Lengths of maximal same-source runs."""
    out: list[int] = []
    for name in order:
        if out and order[len(out) - 1] == name:
            out[-1] += 1
        else:
            out.append(1)
    return out


def block_purity(order: list[str], block: int = 32) -> float:
    """Mean share of the modal source inside each block of ``block``.

    Under a perfectly mixed stream this is the declared share of the
    largest corpus; higher means the optimiser sees longer single-domain
    stretches.
    """
    if not order:
        return 0.0
    counts: list[float] = []
    for i in range(0, len(order) - block + 1, block):
        chunk = order[i:i + block]
        counts.append(max(Counter(chunk).values()) / len(chunk))
    return sum(counts) / len(counts) if counts else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[12901, 12902, 12903, 13701, 13702])
    ap.add_argument("--draws", type=int, default=6000)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dc = cfg.train.data
    weights = dict(dc.weights)

    print(f"config: {args.config}")
    print(f"draws={args.draws}  block={args.block} windows "
          f"(a batch is {cfg.train.batch_size} windows)")
    print()
    print(f"{'seed':>7} {'max run':>9} {'mean run':>9} {'p99 run':>8} "
          f"{'block purity':>13}")

    rows = []
    for seed in args.seeds:
        order = draw_order(weights, seed, args.draws, dc.data_dir,
                           dc.seq_len, dc.eos_token_id)
        runs = sorted(run_lengths(order), reverse=True)
        mean = sum(runs) / len(runs)
        p99 = runs[max(0, int(len(runs) * 0.01) - 1)]
        purity = block_purity(order, args.block)
        rows.append((seed, runs[0], mean, p99, purity))
        print(f"{seed:>7} {runs[0]:>9} {mean:>9.2f} {p99:>8} {purity:>13.4f}")

    print()
    falls = [r for r in rows if r[0] in (12901, 12902, 12903)]
    holds = [r for r in rows if r[0] in (13701, 13702)]
    if falls and holds:
        f_max = max(r[1] for r in falls)
        h_max = max(r[1] for r in holds)
        print(f"longest run, falling seeds (129xx): {f_max}")
        print(f"longest run, holding seeds (137xx): {h_max}")
        if f_max > h_max:
            print()
            print("The falling seeds produce longer same-corpus stretches. Same")
            print("aggregate mix, same stream start -- different ORDER, and with")
            print("it longer stretches where the optimiser sees a single")
            print("distribution. That is the mechanism worth acting on.")
        else:
            print()
            print("Run length does NOT separate the two families, so order is")
            print("not the mechanism either. The seed changes something else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())