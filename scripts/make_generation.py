"""Generate a whole generation's configs with consistent geometry.

Deriving the next generation by copying the previous generation's YAML
and editing paths silently keeps the OLD dimensions. Two launches of
gen-4 failed that way in a row:

1. ``hidden_size`` stayed at the template's value while ``init_from``
   pointed at a wider parent, so loading raised an embedding shape
   mismatch. A sibling inherits the PARENT's width; only
   ``expert_hidden`` scales as ``parent_width / n_experts``.
2. ``num_query_heads`` stayed at the template's value, so
   ``q * head_dim != hidden_size``.

This script takes the parent's geometry and the n_experts of the merge
and writes the sibling, merged and joint configs with every derived
field consistent. It refuses to emit a config that violates
``q * head_dim == hidden_size`` rather than failing hours later on the
GPU.

Usage::

    python scripts/make_generation.py --parent-width 1152 \
        --generation 4 --n-experts 3 --head-dim 64 \
        --parent-joint checkpoints/dbridge_gen3_joint/step-0001600.pt \
        --template-sibling configs/dbridge_gen4_sib1.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Fields that must scale with the width. Leaving any of them at the
# template's value produces a config that loads on paper and explodes on
# the first forward pass.
SCALING = {
    "model": ("hidden_size",),
    "model.attention": ("num_query_heads", "num_kv_heads"),
    "merge": ("expert_hidden",),
}


def _get(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def geometry_ok(cfg: dict) -> tuple[bool, str]:
    """Check the invariant that actually bites: q*head_dim == hidden_size."""
    hidden = _get(cfg, "model.hidden_size")
    q = _get(cfg, "model.attention.num_query_heads")
    kv = _get(cfg, "model.attention.num_kv_heads")
    hd = _get(cfg, "model.attention.head_dim")
    if None in (hidden, q, kv, hd):
        return False, "missing model/attention fields"
    if q * hd != hidden:
        return False, f"num_query_heads*head_dim = {q * hd} != hidden_size {hidden}"
    if kv > q:
        return False, f"num_kv_heads {kv} > num_query_heads {q}"
    return True, "ok"


def scale(cfg: dict, width: int, head_dim: int) -> dict:
    """Return a copy of ``cfg`` resized for ``width``."""
    out = copy.deepcopy(cfg)
    _set(out, "model.hidden_size", width)
    _set(out, "model.attention.num_query_heads", width // head_dim)
    _set(out, "model.attention.num_kv_heads", max(1, (width // head_dim) // 2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-width", type=int, required=True,
                    help="hidden size of the parent joint model")
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--n-experts", type=int, default=3)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--parent-joint", required=True,
                    help="checkpoint the siblings init from")
    ap.add_argument("--template-sibling", required=True)
    ap.add_argument("--template-merged", required=True)
    ap.add_argument("--template-joint", required=True)
    ap.add_argument("--mixes", nargs="+", required=True,
                    metavar="NAME=SEED=CORPUS:WEIGHT,...",
                    help="one per sibling, e.g. math=12901=openwebmath:0.45,edu:0.25")
    ap.add_argument("--merge-checkpoint-step", type=int, default=0,
                    help="if > 0, merged init_from this step of the merge dir")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if args.parent_width % args.head_dim:
        raise SystemExit(
            f"parent width {args.parent_width} is not divisible by head_dim {args.head_dim}"
        )
    if args.parent_width % args.n_experts:
        raise SystemExit(
            f"parent width {args.parent_width} is not divisible by n_experts {args.n_experts}"
        )

    gen = args.generation
    sib_width = args.parent_width
    merged_width = args.parent_width * args.n_experts
    expert_hidden = sib_width // args.n_experts

    tmpl_sib = yaml.safe_load((ROOT / args.template_sibling).read_text(encoding="utf-8"))
    tmpl_merged = yaml.safe_load((ROOT / args.template_merged).read_text(encoding="utf-8"))
    tmpl_joint = yaml.safe_load((ROOT / args.template_joint).read_text(encoding="utf-8"))

    made: list[tuple[Path, dict]] = []

    # --- siblings -------------------------------------------------------
    for spec in args.mixes:
        name, seed, mix = spec.split("=")
        cfg = scale(tmpl_sib, sib_width, args.head_dim)
        _set(cfg, "merge.expert_hidden", expert_hidden)
        cfg["train"]["checkpoint_dir"] = f"checkpoints/dbridge_gen{gen}_sib_{name}"
        cfg["train"]["init_from"] = args.parent_joint
        cfg["train"]["data"]["seed"] = int(seed)
        weights: dict[str, float] = {}
        for item in mix.split(","):
            corpus, weight = item.split(":")
            weights[corpus] = float(weight)
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise SystemExit(f"{name}: weights sum to {total}, expected 1.0")
        cfg["train"]["data"]["weights"] = weights
        ok, why = geometry_ok(cfg)
        if not ok:
            raise SystemExit(f"sibling {name}: {why}")
        made.append((ROOT / f"configs/dbridge_gen{gen}_sib_{name}.yaml", cfg))

    # --- merged ---------------------------------------------------------
    merged = scale(tmpl_merged, merged_width, args.head_dim)
    _set(merged, "merge.expert_hidden", expert_hidden)
    _set(merged, "merge.n_experts", args.n_experts)
    merged["train"]["checkpoint_dir"] = f"checkpoints/dbridge_gen{gen}_merged"
    merged["train"]["init_from"] = ""
    step = args.merge_checkpoint_step or 0
    merged["merge"]["expert_checkpoints"] = [
        f"checkpoints/dbridge_gen{gen}_sib_{spec.split('=')[0]}/step-{step:07d}.pt"
        for spec in args.mixes
    ]
    ok, why = geometry_ok(merged)
    if not ok:
        raise SystemExit(f"merged: {why}")
    made.append((ROOT / f"configs/dbridge_gen{gen}_merged.yaml", merged))

    # --- joint ----------------------------------------------------------
    joint = scale(tmpl_joint, merged_width, args.head_dim)
    _set(joint, "merge.expert_hidden", expert_hidden)
    _set(joint, "merge.n_experts", args.n_experts)
    joint["train"]["checkpoint_dir"] = f"checkpoints/dbridge_gen{gen}_joint"
    joint["train"]["init_from"] = (
        f"checkpoints/dbridge_gen{gen}_merged/step-{args.merge_checkpoint_step:07d}.pt"
        if args.merge_checkpoint_step
        else f"checkpoints/dbridge_gen{gen}_merged/step-0000000.pt"
    )
    ok, why = geometry_ok(joint)
    if not ok:
        raise SystemExit(f"joint: {why}")
    made.append((ROOT / f"configs/dbridge_gen{gen}_joint.yaml", joint))

    print(f"gen{gen}: siblings H={sib_width} (q={sib_width // args.head_dim}, "
          f"kv={max(1, (sib_width // args.head_dim) // 2)}), "
          f"merged/joint H={merged_width} (q={merged_width // args.head_dim}, "
          f"kv={max(1, (merged_width // args.head_dim) // 2)}), "
          f"expert_hidden={expert_hidden}")
    for path, cfg in made:
        print(f"  {'WRITE' if args.write else 'would write'} {path.relative_to(ROOT)}")
        if args.write:
            path.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
    if not args.write:
        print("\n(dry run; pass --write to emit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
