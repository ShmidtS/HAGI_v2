"""Which corpus regions do the dataloader workers actually read?

`PackedMixDataset.__iter__` derives each worker's rng as
`default_rng(seed + worker_id * 7919)`, so with `num_workers = 2` two
workers draw from the same corpora at INDEPENDENT offsets. Each worker
then consumes its own stream cursors, and a worker that runs ahead
silently takes a different part of the corpus than its sibling.

That is the remaining candidate for how `data.seed` acts: not the mix
(measured within 1%), not the start (all cursor 0), not the order
(short runs), but WHICH PART of each corpus the two workers between them
cover -- and therefore whether a run spends its steps on the head of a
corpus, on its tail, or on both.

    python scripts/measure_worker_coverage.py --config configs/dbridge_gen4_sib1.yaml \
        --steps 1600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.data.dataset import PackedStream, dataset_path  # noqa: E402


def worker_coverage(weights, seed, worker_id, steps, batch_size, data_dir,
                    seq_len, eos) -> dict[str, dict]:
    """Where one worker got to after ``steps`` steps.

    A worker is handed ``batch_size`` windows per step, so over ``steps``
    steps it draws ``steps * batch_size`` windows.
    """
    names = list(weights)
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    probs /= probs.sum()
    rng = np.random.default_rng(seed + worker_id * 7919)
    streams = [
        PackedStream(dataset_path(data_dir, n), seq_len, eos, rng)
        for n in names
    ]
    draws = steps * batch_size
    for _ in range(draws):
        active = [i for i, st in enumerate(streams) if not st.exhausted]
        if not active:
            break
        ap = probs[active]
        ap = ap / ap.sum()
        idx = int(rng.choice(len(active), p=ap))
        if streams[active[idx]].next_window() is None:
            continue
    # measured AFTER the draws: the cursor is what the worker advanced.
    out = {}
    for n, st in zip(names, streams):
        out[n] = {
            "windows": st.cursor // seq_len,
            "frac": st.cursor / max(st.n_tokens, 1),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[12901, 12902, 12903, 13701, 13702])
    ap.add_argument("--steps", type=int, default=1600)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dc = cfg.train.data
    bs = cfg.train.batch_size
    nw = dc.num_workers
    weights = dict(dc.weights)

    print(f"config: {args.config}")
    print(f"num_workers={nw}  batch_size={bs}  steps={args.steps}  "
          f"-> {args.steps * bs} windows per worker")
    print()
    print("Fraction of each corpus consumed per worker (1.0 = exhausted):")
    print()

    for seed in args.seeds:
        print(f"seed {seed}")
        for w in range(nw):
            cov = worker_coverage(weights, seed, w, args.steps, bs,
                                  dc.data_dir, dc.seq_len, dc.eos_token_id)
            cells = "  ".join(
                f"{n}={cov[n]['frac']:.3f}" for n in
                sorted(cov, key=lambda k: -cov[k]["frac"]))
            print(f"   worker {w}: {cells}")
        # the spread between workers is what decides whether the pair
        # covers a corpus evenly or both crowd the same end
        fr = [worker_coverage(weights, seed, w, args.steps, bs, dc.data_dir,
                              dc.seq_len, dc.eos_token_id)
              for w in range(nw)]
        worst = max(abs(fr[0][n]["frac"] - fr[1][n]["frac"]) for n in fr[0]) \
            if nw > 1 else 0.0
        print(f"   -> largest worker disagreement on any corpus: {worst:.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())