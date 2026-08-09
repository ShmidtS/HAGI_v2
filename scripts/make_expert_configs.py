"""Generate per-corpus expert configs + merged + from-scratch baseline configs.

The V47 merge machinery is already built (MergedHAGI, merge_experts). What is
missing is the *pipeline*: N small experts (H=128) each trained to saturation on
a single domain corpus, then block-diagonally merged into one H=N*128 model, then
a short joint-training run to teach the blocks to interact.

This script writes the configs. The training/merge/joint steps are driven by
scripts/run_experts_pipeline.py.

Corpus -> domain mapping (from the available .bin streams):
    RU    : wikipedia_ru + oscar_ru
    EN    : edu + slimpajama + wikipedia_en
    MATH  : openwebmath
    CODE  : python_instruct   (small corpus; saturates fast)

Each expert config is v46_1b.yaml (the fast compiled H=128 geometry) with the
data mix pinned to its domain and its own checkpoint dir. The merged config is
v47_merged.yaml with the expert checkpoint paths filled in. The baseline config
is a from-scratch H=512 model for the honest wall-time comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
EXPERTS = CONFIGS / "experts"

# domain -> {corpus: weight}
DOMAINS: dict[str, dict[str, float]] = {
    "RU": {"wikipedia_ru": 1.0, "oscar_ru": 1.0},
    "EN": {"edu": 1.0, "slimpajama": 1.0, "wikipedia_en": 1.0},
    "MATH": {"openwebmath": 1.0},
    "CODE": {"python_instruct": 1.0},
}

EXPERT_BASE = "v46_1b.yaml"
MERGED_BASE = "v47_merged.yaml"

# Safe max batch under torch.compile on ROCm. Measured: batch>=320 with
# compile_model=True produces non-finite rest gradients (rest=nan/inf) and
# silently skips every optimizer update (a torch.compile buffer-reuse bug on
# this ROCm build). batch=256 is the largest batch that stays finite, and it
# gives +13% throughput over 192 (1.24M vs 1.10M tok/s) with identical loss
# trajectory. compile OFF has no such limit but is 1.5x slower overall.
SAFE_BATCH = 256


def _load(name: str) -> dict:
    with open(CONFIGS / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _ensure_merged_base() -> None:
    """Make sure the merged base config exists.

    ``v47_merged.yaml`` is the template the merged/baseline configs are derived
    from. It is not tracked by git (untracked artifact), so it can disappear
    between runs. If it is missing, rebuild it from ``v48_merged.yaml`` (which
    is always regenerated) by stripping the expert paths and resetting the
    training schedule to the base form.
    """
    if (CONFIGS / MERGED_BASE).exists():
        return
    fallback = CONFIGS / "v48_merged.yaml"
    if not fallback.exists():
        raise FileNotFoundError(
            f"neither {MERGED_BASE} nor v48_merged.yaml exists; run the pipeline once"
        )
    cfg = _load("v48_merged.yaml")
    cfg["merge"]["expert_checkpoints"] = []
    cfg["merge"]["freeze_experts"] = False
    cfg["train"]["max_steps"] = 20000
    cfg["train"]["batch_size"] = 192
    cfg["train"]["checkpoint_dir"] = "checkpoints_v47_merged"
    cfg["train"]["checkpoint_interval"] = 1000
    cfg["train"]["checkpoint_keep_last"] = 3
    cfg["train"]["schedule"]["warmup_steps"] = 500
    _dump(cfg, CONFIGS / MERGED_BASE)
    print(f"[rebuilt] {MERGED_BASE} from v48_merged.yaml")


def _dump(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, width=100)


def make_expert_configs(expert_steps: int) -> dict[str, Path]:
    """Write one config per domain. Returns {domain: path}."""
    paths: dict[str, Path] = {}
    for domain, weights in DOMAINS.items():
        cfg = _load(EXPERT_BASE)
        # max_steps is a hard safety ceiling, NOT the training target. The
        # model trains until it reaches a validation-CE plateau (saturation),
        # which may take far more than ``expert_steps``. The corpus is an
        # infinite stream (PackedStream wraps at EOF), so data never runs out;
        # saturation is the only stop condition.
        cfg["train"]["max_steps"] = max(expert_steps, 100000)
        cfg["train"]["batch_size"] = SAFE_BATCH
        # Same-initialization: every expert must start from the same random
        # weights (the method's core requirement). Pin the model init seed so
        # all experts draw identical weights and the block-diagonal merge is a
        # faithful concatenation of N subspaces grown from one shared prior.
        cfg["model"]["init_seed"] = 1234
        cfg["train"]["data"]["weights"] = weights
        # Train to saturation (validation-CE plateau), not a fixed step count.
        # exact_ce is measured every 100 steps; stop once it has not improved
        # by more than 0.01 over 20 samples (2000 steps). max_steps is a hard
        # ceiling, not the target.
        cfg["train"]["logging"]["exact_ce_interval"] = 100
        cfg["train"]["logging"]["exact_ce_rows"] = 512
        cfg["train"]["saturation_patience"] = 20
        cfg["train"]["saturation_tol"] = 0.01
        cfg["train"]["saturation_min_steps"] = 2000
        cfg["train"]["checkpoint_dir"] = f"checkpoints_experts/{domain.lower()}"
        cfg["train"]["checkpoint_interval"] = 1000
        cfg["train"]["checkpoint_keep_last"] = 2
        # Experts are trained to saturation on one domain; a shorter warmup is
        # fine and saves wall-time.
        cfg["train"]["schedule"]["warmup_steps"] = min(500, expert_steps // 4)
        path = EXPERTS / f"expert_{domain.lower()}.yaml"
        _dump(cfg, path)
        paths[domain] = path
    return paths


def make_merged_config(expert_paths: dict[str, Path], joint_steps: int, expert_steps: int) -> Path:
    """Write the merged config with expert checkpoints filled in.

    Experts train to a validation-CE plateau (not a fixed step count), so each
    expert's checkpoint is its highest-numbered ``step-*.pt`` (the saturation
    point), not ``step-{expert_steps}``. ``expert_steps`` is only the hard
    ceiling.
    """
    from hagi.train.checkpoint import latest_checkpoint

    cfg = _load(MERGED_BASE)
    ckpts = []
    for d in DOMAINS:
        latest = latest_checkpoint(ROOT / f"checkpoints_experts/{d.lower()}")
        if latest is None:
            latest = ROOT / f"checkpoints_experts/{d.lower()}/step-{expert_steps:07d}.pt"
        ckpts.append(str(latest))
    cfg["merge"]["expert_checkpoints"] = ckpts
    # The merge script writes step-0000000.pt; joint training resumes from it.
    cfg["train"]["max_steps"] = joint_steps
    cfg["train"]["batch_size"] = SAFE_BATCH
    cfg["train"]["checkpoint_dir"] = "checkpoints_v48_merged"
    cfg["train"]["checkpoint_interval"] = 500
    cfg["train"]["checkpoint_keep_last"] = 2
    cfg["train"]["schedule"]["warmup_steps"] = min(200, joint_steps // 4)
    path = CONFIGS / "v48_merged.yaml"
    _dump(cfg, path)
    return path


def make_baseline_config(baseline_steps: int) -> Path:
    """From-scratch H=512 model for the honest wall-time comparison."""
    cfg = _load(MERGED_BASE)
    cfg["merge"]["enabled"] = False
    cfg["merge"]["expert_checkpoints"] = []
    cfg["train"]["max_steps"] = baseline_steps
    cfg["train"]["batch_size"] = SAFE_BATCH
    cfg["train"]["checkpoint_dir"] = "checkpoints_v48_baseline"
    cfg["train"]["checkpoint_interval"] = 500
    cfg["train"]["checkpoint_keep_last"] = 2
    cfg["train"]["schedule"]["warmup_steps"] = min(500, baseline_steps // 4)
    path = CONFIGS / "v48_baseline.yaml"
    _dump(cfg, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V48 expert/merged/baseline configs")
    parser.add_argument("--expert-steps", type=int, default=20000)
    parser.add_argument("--joint-steps", type=int, default=2000)
    parser.add_argument("--baseline-steps", type=int, default=20000)
    args = parser.parse_args()

    _ensure_merged_base()
    expert_paths = make_expert_configs(args.expert_steps)
    merged = make_merged_config(expert_paths, args.joint_steps, args.expert_steps)
    baseline = make_baseline_config(args.baseline_steps)
    print("expert configs:")
    for d, p in expert_paths.items():
        print(f"  {d:5s} -> {p}")
    print(f"merged  -> {merged}")
    print(f"baseline-> {baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
