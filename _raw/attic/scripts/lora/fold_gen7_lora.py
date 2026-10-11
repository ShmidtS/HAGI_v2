"""Fold the gen7 LoRA checkpoint into a plain MergedHAGI checkpoint.

The rank channel (scripts/lora/lora_gen7_rank.py) trains TableLoRA
wrappers on the frozen gen6 joint's tables; eval_domains needs a
plain MergedHAGI checkpoint to strict-load. fold() materializes
W = base + A@B per wrapper (the same algebra as
merge_leaves_task_arith.fold, which lives in a sibling script and
handles the .lora.{A,B,base} triples this checkpoint carries).

Usage:
    python scripts/lora/fold_gen7_lora.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
_REPO = _HERE.parent.parent
for _p in (_HERE, _REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from hagi.config import load_config  # noqa: E402
from hagi.train.checkpoint import CHECKPOINT_FORMAT_VERSION, config_to_dict  # noqa: E402


def fold(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Materialize every LoRA wrapper into its effective weight."""
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith(".lora.A") or key.endswith(".lora.B"):
            continue
        if key.endswith(".lora.base"):
            # wrapper modules carry lora.{A,B,base} as direct children:
            # encoder.embedding.lora.base -> encoder.embedding.weight
            root = key[: -len(".lora.base")] + ".weight"
            A = state[key[:-len(".base")] + ".A"].float()
            B = state[key[:-len(".base")] + ".B"].float()
            out[root] = state[key] + (A @ B)
        else:
            out[key] = tensor
    return out


def main() -> int:
    src = _REPO / "checkpoints/dbridge_gen7_lora/step-0000800.pt"
    out = _REPO / "checkpoints/dbridge_gen7_lora_folded/step-0000800.pt"
    pl = torch.load(src, map_location="cpu", weights_only=False)
    folded = fold(pl["model"])
    cfg = load_config(_REPO / "configs/dbridge_gen6_joint.yaml")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": folded,
        "config": config_to_dict(cfg),
        "completed_steps": int(pl.get("completed_steps", 800)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    n_lora = sum(1 for k in folded if ".lora." in k)
    print(f"folded {src.name} -> {out}")
    print(f"  lora keys remaining: {n_lora} (must be 0)")
    print(f"  table keys: {sum(1 for k in folded if 'embedding.weight' in k or 'projection.weight' in k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
