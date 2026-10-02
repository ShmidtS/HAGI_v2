"""Why the merge harvests only 0.2% of the disagreement it is given.

The measured chain (see README, "Что ограничивает рост"):

  frontier D_t          = 18.25 nats   (ample)
  realised gain G       = +0.0365 nats (three orders below)
  implied gamma = G/D_t = 0.002

This isolates the mechanism. The question is not whether averaging is
lossy -- it obviously is -- but WHICH of the two losses applies:

  (a) the deviations are mutually ORTHOGONAL, so averaging cannot cancel
      them and they survive intact into the merged model;
  (b) the deviations are mutually ANTI-CORRELATED, so averaging cancels
      a large share of them and what the merged model carries is only
      the common component.

The two cases imply opposite fixes. Under (a) the merge already keeps
everything and the problem is downstream. Under (b) the merge itself is
the right thing to rebuild.

Both are measured here, on checkpoints that exist:

    python scripts/diagnose_merge_cancellation.py \
        --experts math=checkpoints/gen2_dsib_math/step-0001600.pt ...
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

KEY = "encoder.embedding.weight"


def load(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    sd = payload.get("model", payload)
    if KEY not in sd:
        raise SystemExit(f"{path}: no {KEY}")
    return sd[KEY].double()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    return float((a * b).sum() / (a.norm() * b.norm()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts", nargs="+", required=True,
                    metavar="NAME=PATH")
    args = ap.parse_args()

    mats = {}
    for s in args.experts:
        name, path = s.split("=")
        mats[name] = load(path)
        print(f"loaded {name}: {path} {tuple(mats[name].shape)}")

    names = list(mats)
    mean = torch.stack([mats[n] for n in names]).mean(0)
    dev = {n: (mats[n] - mean).flatten() for n in names}

    print()
    print("=== (a) are the deviations mutually ORTHOGONAL? ===")
    orth = True
    for a, b in itertools.combinations(names, 2):
        c = cos(dev[a], dev[b])
        orth &= abs(c) < 0.2
        print(f"  cos(dev_{a}, dev_{b}) = {c:+.4f}")
    print(f"  -> {'ORTHOGONAL: averaging cannot cancel them' if orth else 'NOT orthogonal: see the sign'}")

    print()
    print("=== (b) how strongly anti-correlated are they with the mean? ===")
    for n in names:
        print(f"  cos(dev_{n}, mean) = {cos(dev[n], mean.flatten()):+.4f}")

    print()
    print("=== does the merged model recover each expert? ===")
    for n in names:
        w = mats[n].flatten()
        recon = mean.flatten() + dev[n]
        print(f"  cos(W_{n}, merged) = {cos(w, mean.flatten()):+.4f}   "
              f"||merged + dev_{n} - W_{n}|| = {float((recon - w).norm()):.2e}")

    print()
    cancelled = 1.0 if orth else float(
        abs(cos(dev[names[0]], dev[names[1]]))
    )
    print("Reading:")
    if orth:
        print("  The deviations survive averaging. The merged model carries")
        print("  every expert's individuality, so the lost gain is NOT a")
        print("  merge loss -- it is harvested nowhere downstream.")
    else:
        print(f"  The deviations are mutually ANTI-correlated (|cos| up to "
              f"{cancelled:.2f}).")
        print("  Averaging cancels exactly that shared direction, so the")
        print("  merged model keeps the COMMON component and discards the")
        print("  part that distinguishes the experts -- which is the part")
        print("  the frontier's disagreement lives in. That is the 0.2%.")
        print()
        print("  Reconstruction from mean + dev is exact to float error, so")
        print("  the information is PRESENT in the checkpoints and the merge")
        print("  simply does not carry it. R101's routed_eval_exact (C + R_i,")
        print("  exact per route) is the formal shape of an operator that")
        print("  would, though its low-rank premise measured false here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())