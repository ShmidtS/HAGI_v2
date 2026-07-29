"""Automated ablation experiment runner for HAGI-2.

Runs training with a base YAML config and per-experiment CLI overrides,
logging step-level metrics to a CSV file.

Usage:
    python scripts/run_ablation.py \\
        --base-config configs/smollm2.yaml \\
        --overrides "baseline:" "no_ib:train.w_rate=0,train.w_distortion=0" \\
        --steps 1000 --data-dir data --device cpu
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from hagi.train._rocm_fsdp_stub import install as _install_rocm_fsdp_stub
_install_rocm_fsdp_stub()

from hagi.config import load_config
from hagi.model.model import HAGI
from hagi.data.sequential import build_sequential_dataloader
from hagi.train.loop import train

logger = logging.getLogger(__name__)


def parse_overrides(raw: list[str]) -> dict[str, dict[str, str]]:
    experiments: dict[str, dict[str, str]] = {}
    for item in raw:
        if ":" not in item:
            raise ValueError(f"override '{item}' missing ':' separator")
        name, kv_str = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty experiment name in '{item}'")
        overrides: dict[str, str] = {}
        if kv_str.strip():
            for kv in kv_str.split(","):
                kv = kv.strip()
                if "=" not in kv:
                    raise ValueError(f"invalid key=value pair: '{kv}'")
                k, v = kv.split("=", 1)
                overrides[k.strip()] = v.strip()
        experiments[name] = overrides
    return experiments


def apply_override(cfg, key: str, value: str):
    parts = key.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    current = getattr(obj, parts[-1])
    if isinstance(current, bool):
        setattr(obj, parts[-1], value.lower() in ("true", "1", "yes"))
    elif isinstance(current, int):
        setattr(obj, parts[-1], int(value))
    elif isinstance(current, float):
        setattr(obj, parts[-1], float(value))
    else:
        setattr(obj, parts[-1], value)


METRICS = [
    "experiment", "step", "loss", "bpt", "masked_ce", "rate", "rate_bits",
    "distortion", "posterior_entropy", "top2_mass", "avg_confidence",
    "grad_norm", "grad_rms", "lr", "exit_halted",
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="HAGI ablation experiment runner")
    parser.add_argument("--base-config", default="configs/smollm2.yaml")
    parser.add_argument("--overrides", nargs="*", default=[], help="name:k=v,k2=v2 ...")
    parser.add_argument("--steps", type=int, default=None, help="Max steps per experiment")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="logs/ablations")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    experiments = parse_overrides(args.overrides)
    if not experiments:
        experiments = {"baseline": {}}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    logger.info(f"Device: {device} | Output: {csv_path} | {len(experiments)} experiments")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS)
        writer.writeheader()

        for exp_name, overrides in experiments.items():
            logger.info(f"=== {exp_name} ===")
            cfg = load_config(path=args.base_config)
            if args.steps is not None:
                cfg.train.max_steps = args.steps
            for k, v in overrides.items():
                apply_override(cfg, k, v)

            model = HAGI(cfg).to(device)
            n = sum(p.numel() for p in model.parameters())
            logger.info(f"Params: {n/1e6:.1f}M")
            dataloader = build_sequential_dataloader(cfg, data_dir=args.data_dir)

            for metrics in train(model, dataloader, cfg, start_step=0):
                row = {"experiment": exp_name}
                for field in METRICS:
                    if field == "experiment":
                        continue
                    val = metrics.get(field, float("nan"))
                    row[field] = val if isinstance(val, (int, float, bool)) else float("nan")
                writer.writerow(row)
            f.flush()
            logger.info(f"[{exp_name}] complete")
    logger.info(f"Done: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
