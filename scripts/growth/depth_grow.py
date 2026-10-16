"""Grow a trained root to a deeper body, function-preserving (identity blocks).

Usage:
    python scripts/growth/depth_grow.py \
        --resume checkpoints/gen51_cooldown/best.pt --target-layers 7 \
        --out checkpoints/gen51_depth7/best.pt

The grown checkpoint reproduces the source model bit-for-bit (zeroed
out_proj/down on the appended blocks); the cycle then trains the new
capacity through the usual leaves/joint path.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from hagi.config import CHECKPOINT_FORMAT_VERSION  # noqa: E402
from hagi.model.depth_grow import (  # noqa: E402
    fix_old_residual_scale_,
    grow_checkpoint,
    old_residual_scale,
    zero_new_blocks_,
)
from hagi.model.factory import build_model_for_config  # noqa: E402
from hagi.train.checkpoint import config_from_dict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--target-layers", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity-batch", type=int, default=0,
                    help="when >0, verify logits parity on a random batch")
    args = ap.parse_args()

    payload = grow_checkpoint(args.resume, args.target_layers)
    cfg = config_from_dict(payload["config"])
    assert cfg.model.num_layers == args.target_layers

    model = build_model_for_config(cfg)
    old_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    old_layers = old_payload["config"]["model"]["num_layers"]
    model.load_state_dict(payload["model"], strict=False)
    zero_new_blocks_(model, old_layers)
    loop_depth = int(payload["config"]["model"].get("loop_depth", 1))
    fix_old_residual_scale_(
        model, old_layers, old_residual_scale(old_layers, loop_depth)
    )
    model.eval()

    if args.parity_batch > 0:
        torch.manual_seed(0)
        idx = torch.randint(0, cfg.model.vocab_size, (1, args.parity_batch))
        with torch.no_grad():
            # source model for parity
            old_cfg_dict = dict(old_payload["config"])
            old_model = build_model_for_config(config_from_dict(old_cfg_dict))
            old_model.load_state_dict(old_payload["model"])
            old_model.eval()
            a = old_model(idx, return_logits=True).logits
            b = model(idx, return_logits=True).logits
        max_diff = (a - b).abs().max().item()
        print(f"parity max |logit diff| = {max_diff:.3e}")
        if max_diff != 0.0:
            print("FAIL: grown model is not bit-for-bit identical")
            return 1

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": payload["config"],
        "model": model.state_dict(),
        "completed_steps": payload["completed_steps"],
    }
    torch.save(state, out.with_suffix(".tmp"))
    out.with_suffix(".tmp").replace(out)
    print(f"grown checkpoint written: {out} ({old_layers} -> {args.target_layers} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
