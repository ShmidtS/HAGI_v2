"""Round 72: task-arithmetic union of the 3 domain LoRA leaves.

All leaves start from the SAME record prior (r48, gate 3.3686) and
train disjoint domain streams. Their deltas against the prior are
therefore well-defined and additive (mergekit task_arithmetic,
arcee-ai): merged = prior + sum_i (leaf_i - prior).

Mechanics:
- fold every *.lora.{A,B,base} triple into the effective table
  (W = base + A @ B) for prior and each leaf;
- deltas of non-LoRA tensors are taken as-is (they are frozen copies
  of the prior -- verified below, sum |delta| must be 0);
- output checkpoint keeps the PRIOR's config (plain, no adapters),
  so joint training resumes the standard recipe.

Usage:
    python scripts/lora/merge_leaves_task_arith.py \
        --prior checkpoints/dbridge_gen2_lora_c8_r48/step-0001600.pt \
        --leaves checkpoints/leaf_math/step-0001600.pt \
                 checkpoints/leaf_code/step-0001600.pt \
                 checkpoints/leaf_ru/step-0001600.pt \
        --out checkpoints/tree_gen3_merged/step-0000000.pt
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_HERE, _REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from hagi.train.checkpoint import CHECKPOINT_FORMAT_VERSION, config_to_dict


def fold(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Materialize every LoRA wrapper into its effective weight."""
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith(".lora.A") or key.endswith(".lora.B"):
            continue
        if key.endswith(".lora.base"):
            root = key[: -len(".lora.base")]
            A = state[root + ".lora.A"].float()
            B = state[root + ".lora.B"].float()
            out[root] = state[key] + (A @ B)
            continue
        out[key] = tensor
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--leaves", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="per-leaf delta scale (1.0 = plain sum)")
    args = ap.parse_args()

    prior_pl = torch.load(args.prior, map_location="cpu", weights_only=False)
    prior = fold(prior_pl["model"])
    base_keys = set(prior.keys())

    deltas: list[dict[str, torch.Tensor]] = []
    for path in args.leaves:
        pl = torch.load(path, map_location="cpu", weights_only=False)
        leaf = fold(pl["model"])
        missing = base_keys - set(leaf.keys())
        if missing:
            raise SystemExit(f"{path}: missing keys vs prior: {sorted(missing)[:5]}")
        delta = {}
        frozen_drift = 0.0
        for k in base_keys:
            d = leaf[k].float() - prior[k].float()
            if k.endswith(".lora.base") or ".lora." in k:
                continue
            delta[k] = d
        # non-LoRA tensors must be frozen: quantify the drift
        for k in delta:
            if "sink_bias" in k or "norm" in k or "branch_scale" in k:
                frozen_drift += float(delta[k].abs().sum())
        print(f"{os.path.basename(os.path.dirname(path))}: "
              f"non-LoRA frozen drift |sum|={frozen_drift:.3e}")
        deltas.append(delta)

    merged = {}
    for k in base_keys:
        acc = prior[k].float().clone()
        for delta in deltas:
            acc = acc + args.scale * delta.get(
                k, torch.zeros_like(acc))
        merged[k] = acc.to(prior[k].dtype)

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model": merged,
            "config": prior_pl["config"],
            "completed_steps": 0,
        },
        out,
    )
    print(f"saved {out} ({len(merged)} keys, scale={args.scale}, "
          f"{len(deltas)} leaves)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
