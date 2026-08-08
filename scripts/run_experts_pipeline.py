"""Run the full V48 pipeline: train N experts -> merge -> joint train.

Steps:
  1. Train each domain expert (H=128) to saturation on its corpus.
  2. Block-diagonally merge the N experts into one H=N*128 model.
  3. Short joint-training run to teach the blocks to interact.

Usage:
    python scripts/run_experts_pipeline.py --expert-steps 20000 --joint-steps 2000
    python scripts/run_experts_pipeline.py --expert-steps 20000 --joint-steps 2000 --dry-run
    python scripts/run_experts_pipeline.py --expert-steps 20000 --joint-steps 2000 --only-merge

The pipeline is resumable: each stage skips work whose checkpoint already
exists. ``--only-merge`` runs just the merge (after experts are trained).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CONFIGS = ROOT / "configs"
EXPERTS = CONFIGS / "experts"

DOMAINS = ["RU", "EN", "MATH", "CODE"]


def _run(cmd: list[str], dry: bool) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    if dry:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def _expert_ckpt(domain: str, steps: int) -> Path:
    return ROOT / f"checkpoints_experts/{domain.lower()}/step-{steps:07d}.pt"


def _merged_ckpt() -> Path:
    return ROOT / "checkpoints_v48_merged/step-0000000.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V48 expert-merge pipeline")
    parser.add_argument("--expert-steps", type=int, default=20000)
    parser.add_argument("--joint-steps", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-merge", action="store_true")
    args = parser.parse_args()

    # 0. Generate configs.
    _run(
        [
            sys.executable,
            str(SCRIPTS / "make_expert_configs.py"),
            "--expert-steps", str(args.expert_steps),
            "--joint-steps", str(args.joint_steps),
        ],
        args.dry_run,
    )

    # 1. Train each expert.
    if not args.only_merge:
        for domain in DOMAINS:
            ckpt = _expert_ckpt(domain, args.expert_steps)
            if ckpt.exists():
                print(f"\n[skip] {domain} expert already trained: {ckpt}")
                continue
            _run(
                [
                    sys.executable,
                    str(SCRIPTS / "train.py"),
                    "--config", str(EXPERTS / f"expert_{domain.lower()}.yaml"),
                ],
                args.dry_run,
            )

    # 2. Merge.
    merged = _merged_ckpt()
    if merged.exists():
        print(f"\n[skip] merged checkpoint already exists: {merged}")
    elif args.dry_run:
        expert_paths = [_expert_ckpt(d, args.expert_steps) for d in DOMAINS]
        _run(
            [
                sys.executable,
                str(SCRIPTS / "merge_experts.py"),
                "--config", str(CONFIGS / "v48_merged.yaml"),
                "--experts", *[str(p) for p in expert_paths],
                "--out", str(merged),
            ],
            args.dry_run,
        )
    else:
        expert_paths = [_expert_ckpt(d, args.expert_steps) for d in DOMAINS]
        missing = [p for p in expert_paths if not p.exists()]
        if missing:
            print(f"ERROR: missing expert checkpoints: {missing}", file=sys.stderr)
            return 1
        _run(
            [
                sys.executable,
                str(SCRIPTS / "merge_experts.py"),
                "--config", str(CONFIGS / "v48_merged.yaml"),
                "--experts", *[str(p) for p in expert_paths],
                "--out", str(merged),
            ],
            args.dry_run,
        )

    # 3. Joint training.
    if not args.only_merge:
        _run(
            [
                sys.executable,
                str(SCRIPTS / "train.py"),
                "--config", str(CONFIGS / "v48_merged.yaml"),
                "--resume", str(merged),
            ],
            args.dry_run,
        )

    print("\nV48 pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
