"""Measure whether R101's shared core is worth building on real experts.

The go/no-go behind `factorized_merge`. Two claims are proved:
`factorized_budget` (the parameter saving is real when r << d) and
`factorized_error_bound` (the quality cost is (1/N) sum delta_i). What is
NOT proved -- and is measured here -- is whether the residual spectrum of
{W_i - C} is low rank. If it is not, the scheme costs parameters and
saves nothing.

Run:
    python scripts/measure_shared_core.py [--step 1600]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.factorized_merge import decomposition_gaps, factorized_error_bound
from hagi.train.spectral import rank_from_spectrum

EMBED_KEY = "encoder.embedding.weight"

DEFAULT_EXPERTS = [
    "checkpoints/gen2_dsib_math/step-0001600.pt",
    "checkpoints/gen2_dsib_lang/step-0001600.pt",
    "checkpoints/gen2_dsib_code/step-0001600.pt",
]


def load_experts(paths: list[str]) -> list[torch.Tensor]:
    mats = []
    for p in paths:
        payload = torch.load(p, map_location="cpu", weights_only=True)
        sd = payload.get("model", payload)
        if EMBED_KEY not in sd:
            raise SystemExit(f"{p}: no {EMBED_KEY} in the checkpoint")
        mats.append(sd[EMBED_KEY].double())
    return mats


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum() / (a.norm() * b.norm()))


def residual_ranks(residual: torch.Tensor, fractions: list[float]) -> dict:
    sv = torch.linalg.svdvals(residual)
    energy = sv.pow(2)
    total = float(energy.sum())
    out: dict[float, int] = {}
    for f in fractions:
        tol = (1.0 - f) * total
        out[f] = rank_from_spectrum(sv, total, tol)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts", nargs="+", default=DEFAULT_EXPERTS)
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.90, 0.95, 0.99],
                    help="cumulative residual energy to retain")
    args = ap.parse_args()

    mats = load_experts(args.experts)
    print("experts:")
    for p, m in zip(args.experts, mats):
        print(f"  {p:52s} {tuple(m.shape)}  ||W||_F={float(m.norm()):.1f}")

    print("\npairwise cosine (are the experts even distinct?):")
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            print(f"  cos(W{i}, W{j}) = {cosine(mats[i], mats[j]):+.4f}")

    core = torch.stack(mats).mean(0)
    gaps = decomposition_gaps(mats, core)
    print("\ndelta_i for the mean core:", [round(g, 3) for g in gaps])
    print("factorized_error_bound    : %.4f" % factorized_error_bound(gaps))
    print("relative residual ||W_i - C|| / ||W_i||:")
    for i, m in enumerate(mats):
        print("  expert%d  %.4f" % (i, float((m - core).norm()) / float(m.norm())))

    print("\nis the residual LOW RANK? (the open empirical item)")
    print("  a low-rank residual is the premise; a full-rank one means the")
    print("  factorization saves parameters and buys no compression.")
    d = mats[0].shape[-1]
    for i, m in enumerate(mats):
        r = residual_ranks(m - core, args.fractions)
        for f in args.fractions:
            k = r[f]
            print(f"  expert{i}: {k:>5}/{d} comps for {f:.0%} of residual "
                  f"energy ({k / d:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())