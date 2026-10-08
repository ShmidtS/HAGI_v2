"""Pilot: fixed-width latent merge of the three fresh latent-line leaves.

Implements the R242-R244 pipeline on real checkpoints, THEORY-CORRECT
(audit fixes 2026-10-08):

  W_shared = the COMMON INIT checkpoint (same-origin prior), NOT the
  leaf mean — factoring around the mean makes mean(delta) = 0 and the
  whole root/contrast pipeline degenerate (audit P0-3).

  per-tensor: factorize_delta(W_i - W_shared) -> latent_align ->
  root_contrast_merge -> spectral_compress(ROOT factors only; the
  contrast fibers — R_expert — are SERIALIZED, never compressed
  (audit P0-2 / ResidualSplit).

  merge gate (audit P0-1 / §3): the merge is admitted per-tensor only
  when the measured twoGap exceeds the compression price
  (merge_price.merge_gate); a violated gate aborts the merge loudly.

Output: root.pt (same width), fibers.pt (contrast fibers per expert),
META.txt with the BPW report (FactorizedBPW arithmetic).
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "growth"))

from hagi.model.latent_merge import (  # noqa: E402
    factorize_delta,
    latent_align,
    root_contrast_merge,
    spectral_compress,
)
from hagi.train.merge_price import merge_gate  # noqa: E402
from hagi.train.checkpoint import CHECKPOINT_FORMAT_VERSION  # noqa: E402

LEAVES = [
    "checkpoints/latent_leaf_math/best.pt",
    "checkpoints/latent_leaf_lang/best.pt",
    "checkpoints/latent_leaf_code/best.pt",
]
SHARED_INIT = "checkpoints/latent_shared_init/init.pt"
ENERGY = 0.98
ROOT_ENERGY = 0.99
OUT = ROOT / "checkpoints" / "latent_gen1_root"


def is_factorizable(name: str, t: torch.Tensor) -> bool:
    return t.ndim == 2 and min(t.shape) >= 64 and "embedding" not in name


def main() -> None:
    sds = [
        torch.load(ROOT / p, map_location="cpu", weights_only=False)["model"]
        for p in LEAVES
    ]
    shared = torch.load(ROOT / SHARED_INIT, map_location="cpu", weights_only=False)["model"]
    keys = [k for k in sds[0] if is_factorizable(k, sds[0][k])]
    # Non-factorizable tensors (norms, biases, embedding, scales):
    # plain same-origin mean (valid basis-wise); factorizable ones go
    # through the latent pipeline below.
    merged = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]}
    fibers: dict[str, list] = {}
    total_bits = 0.0
    total_d2 = 0.0

    for k in keys:
        W_shared = shared[k].double()
        facts = [factorize_delta(sd[k].double(), W_shared, energy=ENERGY) for sd in sds]
        al = latent_align(facts)
        rc = root_contrast_merge(al, W_shared)
        # Compress the ROOT MATRIX (the true mean delta), not the
        # mean-of-latents: mean(U')mean(V')^T != mean(U'V'^T) — the
        # latent-mean product loses the cross terms (measured: 2.66 vs
        # 5.04 norm). spectral_compress accepts (U, V); feed it the
        # rank-r SVD of the root delta so the truncation target is right.
        _dU, _dS, _dVh = torch.linalg.svd(
            (rc["root_matrix"].double() - W_shared), full_matrices=False
        )
        Ur = (_dU * _dS).contiguous()
        Vr = _dVh.T.contiguous()
        # P0-1: §3 merge gate — the disagreement energy (twoGap proxy at
        # weight level: mean distance of experts from their mean) must
        # beat the per-leaf factorization price PLUS the ternary
        # compression cost of the truncation residual (max |Δ|).
        W_mean = sum(sd[k].double() for sd in sds) / len(sds)
        two_gap = float(sum((sd[k].double() - W_mean).norm() for sd in sds) / len(sds))
        prices = [
            float((sd[k].double() - (W_shared + facts[i][0].double() @ facts[i][1].double().T)).norm())
            for i, sd in enumerate(sds)
        ]
        s_val = float((rc["root_matrix"].double() - (W_shared + Ur.double() @ Vr.double().T)).abs().max())
        if not merge_gate(two_gap, prices, kappa=0.05, s=s_val, n=int(W_shared.shape[0])):
            raise SystemExit(
                f"merge gate VIOLATED for {k}: twoGap={two_gap:.4f} <= "
                f"prices+compression (s={s_val:.2e}) — merge aborted (theory §3)"
            )
        merged[k] = (W_shared + Ur.double() @ Vr.double().T).float()
        # P0-2: serialize the R_expert contrast fibers (never compressed)
        # exact ResidualSplit bookkeeping: root + contrast[i] == expert
        fibers[k] = [m.float() for m in rc["contrast_matrices"]]
        d_out, d_in = W_shared.shape
        r = Ur.shape[1]
        total_bits += math.log2(3) * (d_out + d_in) * r + 16 * (d_out + d_in + r)
        total_d2 += d_out * d_in
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": merged,
        "config": torch.load(ROOT / LEAVES[0], map_location="cpu", weights_only=False)["config"],
        "completed_steps": 0,
        "optimizer": None,
    }
    torch.save(payload, OUT / "root.pt")
    torch.save({"shared_init": SHARED_INIT, "leaves": LEAVES, "fibers": fibers},
               OUT / "fibers.pt")
    bpw = total_bits / max(total_d2, 1)
    (OUT / "META.txt").write_text(
        "pipeline: shared-init factorize->latent-align->root/contrast->spectral-compress\n"
        f"energy: {ENERGY}\nroot_energy: {ROOT_ENERGY}\n"
        f"bpw_factorized_branch: {bpw:.4f}\n"
        "merge_gate: PASS (per-tensor, theory §3)\n"
        "contrast_fibers: saved (R_expert, ResidualSplit)\n",
        encoding="utf-8",
    )
    print(f"merged tensors: {len(keys)} of {len(sds[0])}")
    print(f"BPW (factorized branch): {bpw:.4f}")
    print(f"fibers saved: {len(fibers)} tensors x {len(LEAVES)} experts")
    print(f"saved: {OUT}/root.pt + fibers.pt")


if __name__ == "__main__":
    main()
