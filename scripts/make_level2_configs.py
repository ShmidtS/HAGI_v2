"""Generate level-2 expert configs for recursive growth.

Level-1 produced a merged H=512 model (4 experts H=128, block-diagonal merge,
mixer joint-training). Level-2 grows recursively: each level-2 expert is a
*copy* of that merged H=512 model (the shared prior), specialized on a single
domain, then block-diagonally merged into H=2048.

The key difference from level-1: experts do NOT start from random weights
(init_seed). They start from the merged level-1 checkpoint via ``--init-from``,
so the block-diagonal merge at level-2 is a faithful concatenation of N
subspaces grown from one shared prior — exactly the method's recursive step.

Each level-2 expert is a MergedHAGI (same architecture as the merged ckpt:
BlockRMSNorm, mixers), so ``--init-from`` loads the weights strictly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
EXPERTS = CONFIGS / "experts"
LEVEL2 = CONFIGS / "level2"

# domain -> {corpus: weight} (same split as level-1)
DOMAINS: dict[str, dict[str, float]] = {
    "RU": {"wikipedia_ru": 1.0, "oscar_ru": 1.0},
    "EN": {"edu": 1.0, "slimpajama": 1.0, "wikipedia_en": 1.0},
    "MATH": {"openwebmath": 1.0},
    "CODE": {"python_instruct": 1.0},
}

# The merged level-1 model is the shared prior for every level-2 expert.
MERGED_L1 = ROOT / "checkpoints_v48_merged" / "step-0004000.pt"

# Level-2 experts are H=512 copies of the merged model.
HIDDEN = 512
NUM_LAYERS = 3
QUERY_HEADS = 8
KV_HEADS = 4
HEAD_DIM = 64

SAFE_BATCH = 256


def _load(name: str) -> dict:
    with open(CONFIGS / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dump(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, width=100)


def make_level2_expert_configs(expert_steps: int) -> dict[str, Path]:
    """Write one level-2 expert config per domain. Returns {domain: path}."""
    paths: dict[str, Path] = {}
    for domain, weights in DOMAINS.items():
        # Base: the merged level-1 config (H=512, MergedHAGI geometry).
        cfg = _load("v48_merged.yaml")
        # max_steps is a hard safety ceiling, NOT the target. Train to
        # saturation (validation-CE plateau).
        cfg["train"]["max_steps"] = max(expert_steps, 100000)
        cfg["train"]["batch_size"] = SAFE_BATCH
        # No init_seed: level-2 experts start from the merged level-1
        # checkpoint (--init-from), not from random weights.
        cfg["model"].pop("init_seed", None)
        cfg["train"]["data"]["weights"] = weights
        # Train to saturation, not a fixed step count.
        cfg["train"]["logging"]["exact_ce_interval"] = 100
        cfg["train"]["logging"]["exact_ce_rows"] = 512
        cfg["train"]["saturation_patience"] = 20
        cfg["train"]["saturation_tol"] = 0.01
        cfg["train"]["saturation_min_steps"] = 2000
        cfg["train"]["checkpoint_dir"] = f"checkpoints_level2/{domain.lower()}"
        cfg["train"]["checkpoint_interval"] = 1000
        cfg["train"]["checkpoint_keep_last"] = 2
        cfg["train"]["schedule"]["warmup_steps"] = min(500, expert_steps // 4)
        # The merged level-1 checkpoint is the shared prior.
        cfg["train"]["init_from"] = str(MERGED_L1)
        path = LEVEL2 / f"level2_{domain.lower()}.yaml"
        _dump(cfg, path)
        paths[domain] = path
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate level-2 expert configs")
    parser.add_argument("--expert-steps", type=int, default=20000)
    args = parser.parse_args()

    paths = make_level2_expert_configs(args.expert_steps)
    print("level-2 expert configs:")
    for d, p in paths.items():
        print(f"  {d:5s} -> {p}")
    print(f"shared prior: {MERGED_L1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
