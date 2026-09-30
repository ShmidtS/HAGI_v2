"""Per-matrix Newton-Schulz audit (reviewer round-3 -> Lean round-43).

For every Muon-managed 2D weight in a checkpoint:
  - sigma_i / ||X||_F spectrum (svdvals)
  - share of sigma < 1e-3 * ||X||_F  (the mass NS-5 provably cannot
    push to ~0.5 per the corrected ns_iter_bound sigma_k <= a^k sigma_0)
  - needed steps s_i = ceil(ln(sigma*/sigma_min)/ln a) for sigma* = 0.5

Prescription: if the sub-1e-3 share is small, fixed ns_steps=5 is fine;
if large, per-matrix s_i (kappa audit) should drive the schedule.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

A_SHRINK = 3.4445  # ns_iter_bound coefficient (Lean, corrected round-43)
SIGMA_STAR = 0.5


def ns_steps_for(sigma_min: float, sigma_star: float = SIGMA_STAR) -> int:
    if sigma_min <= 0 or sigma_min >= sigma_star:
        return 0
    return math.ceil(math.log(sigma_star / sigma_min) / math.log(A_SHRINK))


def audit(path: str) -> int:
    from hagi.train.checkpoint import load_payload

    sd = load_payload(path, "cpu")["model"]
    print(f"checkpoint: {path}")
    rows = []
    for k, v in sd.items():
        if v.ndim != 2 or "embedding" in k or "head" in k or "mixer" in k:
            continue  # Muon domain: body 2D weights only
        X = v.float()
        nrm = float(X.norm())
        if nrm == 0:
            continue
        s = torch.linalg.svdvals(X) / nrm
        smin = float(s.min())
        share_small = float((s < 1e-3).float().mean())
        need = ns_steps_for(smin)
        rows.append((k, tuple(s.shape), smin, share_small, need))
    print(f"{'weight':52s} {'shape':14s} {'smin':>9s} {'%<1e-3':>7s} {'s*':>4s}")
    for k, shp, smin, share, need in rows:
        print(f"{k:52s} {str(shp):14s} {smin:9.2e} {100*share:6.1f}% {need:4d}")
    if rows:
        avg_share = sum(r[3] for r in rows) / len(rows)
        max_need = max(r[4] for r in rows)
        print(f"\nmean share sigma<1e-3: {100*avg_share:.1f}% | max needed NS steps: {max_need}")
        print("VERDICT:", "fixed ns_steps=5 adequate" if avg_share < 0.05
              else "per-matrix s_i schedule warranted (kappa audit)")
    return 0


if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/dbridge_gen2_merged_had/step-0001600.pt"
    raise SystemExit(audit(ck))
