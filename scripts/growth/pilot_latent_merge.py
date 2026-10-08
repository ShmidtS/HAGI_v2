"""Pilot: fixed-width latent merge of the three fresh latent-line leaves.

Implements the R242-R244 pipeline on real checkpoints:
  shared root = mean of the leaf state dicts (weight space)
  per-tensor: factorize_delta -> latent_align -> root_contrast_merge
              -> spectral_compress
Output: a merged ROOT state dict at the SAME width (no 3x), the
contrast fibers kept aside (R_expert, never compressed), and a BPW
report (FactorizedBPW arithmetic). The root model is then evaluated
on the domain mix for a first quality signal.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hagi.model.latent_merge import (  # noqa: E402
    factorize_delta,
    latent_align,
    root_contrast_merge,
    spectral_compress,
)

LEAVES = [
    "checkpoints/latent_leaf_math/best.pt",
    "checkpoints/latent_leaf_lang/best.pt",
    "checkpoints/latent_leaf_code/best.pt",
]
ENERGY = 0.9
ROOT_ENERGY = 0.95
OUT = ROOT / "checkpoints" / "latent_gen1_root"


def is_factorizable(name: str, t: torch.Tensor) -> bool:
    return t.ndim == 2 and min(t.shape) >= 64 and "embedding" not in name


def main() -> None:
    sds = [
        torch.load(p, map_location="cpu", weights_only=False)["model"]
        for p in LEAVES
    ]
    keys = [k for k in sds[0] if is_factorizable(k, sds[0][k])]
    root_sd = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]}
    # non-factorizable tensors (norms, biases, scales, embedding): plain mean
    merged = dict(root_sd)
    total_bits = 0.0
    total_d2 = 0.0
    for k in keys:
        W_shared = root_sd[k].double()
        facts = [factorize_delta(sd[k].double(), W_shared, energy=ENERGY) for sd in sds]
        al = latent_align(facts)
        rc = root_contrast_merge(al, W_shared)
        Ur, Vr = spectral_compress(rc["root_factors"], energy=ROOT_ENERGY)
        W_root = W_shared + (Ur @ Vr.T).double()
        merged[k] = W_root.float()
        d_out, d_in = W_shared.shape
        r = Ur.shape[1]
        import math

        total_bits += math.log2(3) * (d_out + d_in) * r + 16 * (d_out + d_in + r)
        total_d2 += d_out * d_in
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": merged,
        "config": torch.load(LEAVES[0], map_location="cpu", weights_only=False)["config"],
        "completed_steps": 0,
        "meta": {
            "pipeline": "factorize->latent-align->root/contrast->spectral-compress",
            "energy": ENERGY,
            "root_energy": ROOT_ENERGY,
            "bpw_factorized_branch": total_bits / max(total_d2, 1),
        },
    }
    torch.save(payload, OUT / "root.pt")
    print(f"merged tensors: {len(keys)} of {len(sds[0])}")
    print(f"BPW (factorized branch, log2(3)+fp16 scales): {total_bits / max(total_d2, 1):.4f}")
    print(f"saved: {OUT / 'root.pt'}")


if __name__ == "__main__":
    main()
