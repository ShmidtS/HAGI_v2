"""Generate configs for one ladder level at a chosen leaf width.

Why this exists: the project has a measured cost/quality curve for a leaf
(800 steps, honest data): H=64 -> 0.93 nats for 108 s, H=128 -> 1.84 nats
for 157 s, H=384 -> 3.56 nats for 384 s. Per second of GPU time H=128 is the
optimum (11.70 nats/1000 s against 8.64 and 9.27), so the leaf width is a
choice and it is recorded here rather than assumed.

The three leaves differ ONLY in `train.data.seed` and `model.init_seed`.
`init_seed` had a dead guard -- `if cfg.model.init_seed:` skipped the value
0, which is the default, so no leaf in the project had ever been seeded.
Measured consequence: different seeds move leaf quality by +0.005 nats, i.e.
nothing at this budget, but they do make the leaves distinct bodies, which is
what the F3 lift needs in order to do anything at all.

Usage:
    python scripts/make_ladder_configs.py --leaf-hidden 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from hagi.config import ffn_width, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEAF_SEEDS = (1001, 1002, 1003)


def leaf_config(hidden: int, seed: int, steps: int, weights: dict) -> dict:
    heads = hidden // 64
    while heads > 1 and hidden % heads:
        heads -= 1
    return {
        "model": {
            "vocab_size": 32768,
            "hidden_size": hidden,
            "num_layers": 3,
            "loop_depth": 2,
            # Distinct per leaf. 0 is a valid seed; train.py now applies it.
            "init_seed": 0 if seed == LEAF_SEEDS[0] else seed,
            "attention": {
                "num_query_heads": heads,
                "num_kv_heads": max(1, heads // 3),
                "head_dim": 64,
                "max_seq_len": 4096,
            },
            "embedding": {"tie_lm_head": False, "conv_kernel": 1},
            "ternary": {"enabled": False},
            # These leaves were trained with expansion 1.0; the 8/3 default
            # builds a wider mixer and the lift then fails on a shape
            # mismatch (.omc/attempts/f3_tree_vs_flat_2026-09-26.md).
            "ffn": {"expansion": 1.0},
            "head": {"unigram_prior": False, "sampled_proposal": "uniform"},
            "cortex": {"enabled": False},
            "adapters": {"enabled": False},
            "decision": {"enabled": False},
        },
        "train": {
            "max_steps": steps,
            "checkpoint_interval": max(1, steps // 2),
            "precision": "bf16",
            "ternary_fp32_master": False,
            "batch_size": 8,
            "grad_accum_steps": 4,
            "learning_rate": 3e-4,
            "checkpoint_dir": f"checkpoints/leaf_h{hidden}_s{seed}",
            "data": {
                "data_dir": "data",
                "seq_len": 1024,
                "seed": seed,
                "weights": dict(weights),
            },
        },
    }


def parent_config(hidden: int, steps: int, weights: dict, tag: str) -> dict:
    child = leaf_config(hidden, LEAF_SEEDS[0], steps, weights)
    model = child["model"]
    return {
        "model": {
            **model,
            "hidden_size": hidden * 3,
            "init_seed": 0,
            "attention": {
                **model["attention"],
                "num_query_heads": model["attention"]["num_query_heads"] * 3,
                "num_kv_heads": model["attention"]["num_kv_heads"] * 3,
            },
            "cortex": {
                "enabled": True,
                "mode": "root",
                "num_levels": 3,
                "rank": 64,
                "link_strides": [1, 2],
                "residual_scale": 0.1,
            },
        },
        "merge": {
            "enabled": True,
            "mixer_type": "ternary_f3",
            "n_experts": 3,
            "expert_hidden": hidden,
            "ternary_depth": 1,
            "ternary_lift_mode": "parent_preserving",
            "ternary_tree_schema_version": 1,
            "freeze_experts": False,
            "expert_checkpoints": [
                f"checkpoints/leaf_h{hidden}_s{s}/step-{steps:07d}.pt" for s in LEAF_SEEDS
            ],
        },
        "train": {
            **child["train"],
            "max_steps": steps,
            "checkpoint_dir": f"checkpoints/parent_h{hidden}_{tag}",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaf-hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()

    weights = dict(load_config(ROOT / "configs/f3_leaf_s1001.yaml").train.data.weights)
    out = ROOT / "configs"
    for seed in LEAF_SEEDS:
        path = out / f"leaf_h{args.leaf_hidden}_s{seed}.yaml"
        cfg = leaf_config(args.leaf_hidden, seed, args.steps, weights)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"wrote {path.name}  ffn_width={ffn_width(load_config(path).model)}")

    for tag in ("cortex", "flat"):
        path = out / f"parent_h{args.leaf_hidden}_{tag}.yaml"
        cfg = (
            parent_config(args.leaf_hidden, args.steps, weights, tag)
            if tag == "cortex"
            else flat_config(args.leaf_hidden, args.steps, weights)
        )
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        loaded = load_config(path)
        print(f"wrote {path.name}  hidden={loaded.model.hidden_size} "
              f"ffn_width={ffn_width(loaded.model)} mixer={loaded.merge.mixer_type}")
    return 0


def flat_config(hidden: int, steps: int, weights: dict) -> dict:
    cfg = parent_config(hidden, steps, weights, "flat")
    cfg["model"]["cortex"] = {"enabled": False}
    cfg["merge"] = {
        "enabled": True,
        "mixer_type": "hadamard",
        "n_experts": 3,
        "expert_hidden": hidden,
        "mixer_init_scale": 0.0,
        "mixer_rank": 64,
        "freeze_experts": False,
        "expert_checkpoints": cfg["merge"]["expert_checkpoints"],
    }
    return cfg


if __name__ == "__main__":
    raise SystemExit(main())
