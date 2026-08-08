"""Hierarchical expert merge: level-0 experts -> level-1 groups -> final.

The growing hypothesis asks whether *hierarchical* merging (merge a few
experts, train the mixer, merge those, train again) beats *flat* merging
(merge all N experts in one step). This script builds both from a set of
trained level-0 expert checkpoints and reports the merged model sizes.

Level structure (default 16 experts, groups of 4):
    level 0: 16 experts H=128
    level 1: 4 groups of 4 -> 4 models H=512 (each with 1 mixer)
    level 2: 4 models H=512 -> 1 final H=2048 (with 1 mixer)

The level-1 models are written as checkpoints so they can be joint-trained
(``scripts/train.py --config <l1cfg> --resume <l1ckpt>``) before being merged
into the final level-2 model. ``--flat`` builds the direct 16->1 merge for
comparison.

Usage:
    python scripts/merge_hierarchical.py \
        --config configs/v48_merged.yaml \
        --experts checkpoints_experts/ru/step-0000050.pt ... (16 paths) \
        --group-size 4 --out-dir checkpoints_hier

    python scripts/merge_hierarchical.py --config ... --experts ... --flat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import CHECKPOINT_FORMAT_VERSION, load_config  # noqa: E402
from hagi.model.merge import merge_experts  # noqa: E402
from hagi.train.checkpoint import config_to_dict, load_payload  # noqa: E402


def _write_ckpt(model, cfg, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "config": config_to_dict(cfg),
        "completed_steps": 0,
    }
    torch.save(payload, path)
    print(f"wrote {path}")


def _load_states(paths: list[str], device: str) -> list[dict]:
    return [load_payload(p, device)["model"] for p in paths]


def _scale_heads(cfg, hidden_size: int) -> None:
    """Scale attention head counts to match a hidden size.

    The merged config is written for the final H (e.g. 512 with 8 heads). For
    intermediate hierarchical levels the hidden size is smaller, so the head
    count must scale down proportionally (head_dim stays fixed).
    """
    hd = cfg.model.attention.head_dim
    nq = hidden_size // hd
    if nq < 1:
        raise ValueError(f"hidden_size {hidden_size} too small for head_dim {hd}")
    cfg.model.attention.num_query_heads = nq
    cfg.model.attention.num_kv_heads = max(1, nq // 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hierarchical expert merge")
    parser.add_argument("--config", default="configs/v48_merged.yaml")
    parser.add_argument("--experts", nargs="+", required=True, help="level-0 expert checkpoints")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--out-dir", default="checkpoints_hier")
    parser.add_argument("--flat", action="store_true", help="direct N->1 merge (no hierarchy)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n = len(args.experts)
    if n % args.group_size != 0:
        raise SystemExit(f"n_experts {n} must be divisible by group_size {args.group_size}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states = _load_states(args.experts, args.device)
    eh = cfg.merge.expert_hidden  # level-0 expert hidden (128)

    if args.flat:
        # Direct N->1: one merged model H = n * eh.
        flat_cfg = load_config(args.config)
        flat_cfg.model.hidden_size = n * eh
        flat_cfg.merge.n_experts = n
        flat_cfg.merge.expert_hidden = eh
        _scale_heads(flat_cfg, n * eh)
        model = merge_experts(flat_cfg, states, n_mixers=1, mixer_init_scale=0.0)
        _write_ckpt(model, flat_cfg, out_dir / "flat" / "step-0000000.pt")
        counts = model.param_summary()
        print(f"flat {n}->1: total {counts['total']/1e6:.1f}M | body {counts['body']/1e6:.1f}M")
        return 0

    # Hierarchical: group level-0 experts into level-1 models, then merge those.
    n_groups = n // args.group_size
    level1_paths: list[Path] = []
    for gi in range(n_groups):
        group = states[gi * args.group_size : (gi + 1) * args.group_size]
        l1_hidden = args.group_size * eh
        l1_cfg = load_config(args.config)
        l1_cfg.model.hidden_size = l1_hidden
        l1_cfg.merge.n_experts = args.group_size
        l1_cfg.merge.expert_hidden = eh
        _scale_heads(l1_cfg, l1_hidden)
        l1 = merge_experts(l1_cfg, group, n_mixers=1, mixer_init_scale=0.0)
        p = out_dir / f"level1_{gi}" / "step-0000000.pt"
        _write_ckpt(l1, l1_cfg, p)
        level1_paths.append(p)
        counts = l1.param_summary()
        print(f"level1_{gi}: {args.group_size}->1 H={l1_hidden} total {counts['total']/1e6:.1f}M")

    # Level 2: merge the level-1 models (drop their mixers, add a fresh one).
    l1_states = _load_states([str(p) for p in level1_paths], args.device)
    final_hidden = n * eh
    final_cfg = load_config(args.config)
    final_cfg.model.hidden_size = final_hidden
    final_cfg.merge.n_experts = n_groups
    final_cfg.merge.expert_hidden = l1_hidden
    _scale_heads(final_cfg, final_hidden)
    final = merge_experts(
        final_cfg, l1_states, n_mixers=1, mixer_init_scale=0.0, drop_expert_mixers=True
    )
    _write_ckpt(final, final_cfg, out_dir / "final" / "step-0000000.pt")
    counts = final.param_summary()
    print(f"final: {n_groups}->1 H={final_hidden} total {counts['total']/1e6:.1f}M | body {counts['body']/1e6:.1f}M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
