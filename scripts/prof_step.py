"""Profile one train_step: forward/backward/optimizer split + top kernels.

Bandwidth-bound iGPU (Radeon 8060S). Prints the wall-time breakdown so we can
see whether the step is host-bound, bandwidth-bound, or launch-bound, and which
kernels dominate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hagi.config import load_config
from hagi.data.dataset import build_dataloader
from hagi.model.model import HAGI
from hagi.train.loop import Trainer, configure_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v43_1b.yaml")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--grad-checkpointing", action="store_true")
    args = parser.parse_args()

    configure_runtime()
    torch.manual_seed(0)
    overrides: dict[str, object] = {
        "train.data.num_workers": 0,
        "train.checkpoint_interval": 0,
        "train.grad_checkpointing": args.grad_checkpointing,
    }
    cfg = load_config(args.config, **overrides)
    device = torch.device("cuda")
    model = HAGI(cfg).to(device)
    trainer = Trainer(model, cfg)
    batches = iter(build_dataloader(cfg))

    # Warmup
    for _ in range(2):
        micro = [next(batches) for _ in range(cfg.train.grad_accum_steps)]
        trainer.train_step(micro)

    micro = [next(batches) for _ in range(cfg.train.grad_accum_steps)]
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        trainer.train_step(micro)
        torch.cuda.synchronize()

    print(f"=== grad_checkpointing={args.grad_checkpointing} ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    print("\n=== CPU time (host-bound check) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
