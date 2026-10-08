#!/usr/bin/env python
"""§20 decisive experiment: per-layer covariance spectrum of a checkpoint.

ALGORITHMS.md §20 (R181) -- the DECISIVE EXPERIMENT, not a benchmark:

    intrinsic dimension + covariance spectrum PER LAYER:
    before merge / after merge / after Grow->Merge cycles -- a check
    that the fiber theory (cortex U + orthogonal low-rank fibers V_i)
    describes the REAL geometry rather than interpreting it post hoc.

For each 2D weight matrix W in the checkpoint this script computes the
singular values (SVD), the energy share of the top-r directions
(energy = s^2, per the Eckart-Young reading of §20's covariance
spectrum), and the intrinsic-dimension proxy d90 = the smallest number
of components reaching 90% of the total energy. Run it on a leaf, a
merged and a joint checkpoint (``--checkpoint`` is repeatable) and
compare the tables: the theory predicts the merged/joint bodies stay
low-rank per layer (r << d) with the rank concentrated in the fiber
blocks, not spread like a dense re-mix.

Pure analysis: CPU or GPU (whatever the checkpoint device needs), no
training, no writes beyond the report. Honest boundary (§20): this is
a MEASUREMENT, not a theorem check -- the 90% threshold is the report
convention, the theory makes no claim at exactly 90%.

Usage::

    python scripts/growth/measure_fiber_geometry.py \
        --checkpoint checkpoints/sib/step-0001000.pt \
        --checkpoint checkpoints/merged/step-0001000.pt \
        --out reports/fiber_geometry.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

# The §20 report convention: d90 = components to reach this energy share.
ENERGY_TARGET = 0.90


def energy_profile(w: torch.Tensor) -> dict[str, float]:
    """Covariance-spectrum profile of one weight matrix.

    Args:
        w: a 2D weight matrix ``[out, in]`` (any dtype; computed in
            float64 -- the spectrum tail near 90% energy is exactly
            where fp32 cancellation bites).

    Returns:
        ``{"rows", "cols", "rank", "energy_top1", "energy_top10",
        "d90"}`` where energy_topX is the share of total squared
        singular values carried by the top X directions and ``d90`` is
        the number of components reaching 90% energy.
    """
    if w.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got {tuple(w.shape)}")
    s = torch.linalg.svdvals(w.detach().to(torch.float64))
    energy = s * s
    total = float(energy.sum())
    if total <= 0.0:
        # A zero matrix carries no spectrum; every share is 0 and d90
        # is undefined -- reported as 0 rather than crashed on.
        return {
            "rows": int(w.shape[0]), "cols": int(w.shape[1]),
            "rank": 0, "energy_top1": 0.0, "energy_top10": 0.0, "d90": 0,
        }
    shares = energy / total
    cum = torch.cumsum(shares, dim=0)
    d90 = int(torch.searchsorted(cum, torch.tensor(ENERGY_TARGET)).item()) + 1
    return {
        "rows": int(w.shape[0]), "cols": int(w.shape[1]),
        "rank": int((s > s[0] * 1e-10).sum().item()) if s.numel() else 0,
        "energy_top1": float(shares[0]),
        "energy_top10": float(shares[:10].sum()),
        "d90": d90,
    }


def checkpoint_spectrum(path: Path, device: str = "cpu") -> dict[str, dict]:
    """Per-layer spectra of every 2D weight in a checkpoint payload."""
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    out: dict[str, dict] = {}
    for name, tensor in state.items():
        t = tensor
        # Head-adjacent tables (embedding codebook, lm_head) are excluded:
        # §20 is about the BODY geometry (cortex + fibers), and a V x d
        # codebook's spectrum is dominated by token frequency, not fibers.
        if t.ndim == 2 and t.numel() >= 2 and "embedding" not in name \
                and "lm_head" not in name and "head" not in name:
            out[name] = energy_profile(t)
    return out


def format_table(label: str, spectra: dict[str, dict]) -> str:
    lines = [f"# {label}", f"{'layer':<48}{'shape':>14}{'top1':>8}"
             f"{'top10':>8}{'d90':>6}"]
    for name, p in spectra.items():
        shape = f"{p['rows']}x{p['cols']}"
        lines.append(
            f"{name:<48}{shape:>14}"
            f"{p['energy_top1']:>8.3f}{p['energy_top10']:>8.3f}{p['d90']:>6}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="checkpoint path; repeat for leaf/merged/joint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/fiber_geometry.json")
    args = ap.parse_args()

    report: dict[str, dict] = {}
    for ckpt in args.checkpoint:
        path = Path(ckpt)
        if not path.is_file():
            print(f"missing checkpoint: {path}", file=sys.stderr)
            return 1
        label = path.parent.name + "/" + path.stem
        spectra = checkpoint_spectrum(path, args.device)
        report[label] = spectra
        print(format_table(label, spectra))

    out = _REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
