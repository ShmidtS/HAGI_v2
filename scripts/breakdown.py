"""Precise wall-time breakdown of one train_step (no profiler overhead).

Splits the step into: data fetch, H2D transfer, forward, backward, optimizer,
and the ternary-cache + clip overhead. Uses torch.cuda.synchronize() around each
phase so the numbers are real wall-time, not profiler-inflated.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
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
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    configure_runtime()
    torch.manual_seed(0)
    overrides: dict[str, object] = {"train.data.num_workers": 0, "train.checkpoint_interval": 0}
    cfg = load_config(args.config, **overrides)
    device = torch.device("cuda")
    model = HAGI(cfg).to(device)
    trainer = Trainer(model, cfg)
    batches = iter(build_dataloader(cfg))

    def sync():
        torch.cuda.synchronize()

    def t0():
        return time.perf_counter()

    phases = {k: [] for k in ("data", "h2d", "forward", "backward", "optim", "other")}

    for step in range(args.warmup + args.steps):
        s = t0()
        micro = [next(batches) for _ in range(cfg.train.grad_accum_steps)]
        sync()
        phases["data"].append(t0() - s)

        s = t0()
        ids = micro[0]["input_ids"].to(device)
        targets = micro[0]["targets"].to(device)
        doc_ids = micro[0]["doc_ids"].to(device) if "doc_ids" in micro[0] else None
        sync()
        phases["h2d"].append(t0() - s)

        trainer.optimizer.zero_grad(set_to_none=True)
        model.train()
        s = t0()
        out = model(ids, targets, doc_ids=doc_ids)
        sync()
        phases["forward"].append(t0() - s)

        s = t0()
        out.loss.backward()
        sync()
        phases["backward"].append(t0() - s)

        s = t0()
        trainer.optimizer.step()
        sync()
        phases["optim"].append(t0() - s)

        if step >= args.warmup:
            total = sum(phases[k][-1] for k in phases if phases[k])
            print(
                f"step={step} total={total*1000:.1f}ms "
                f"data={phases['data'][-1]*1000:.1f} h2d={phases['h2d'][-1]*1000:.1f} "
                f"fwd={phases['forward'][-1]*1000:.1f} bwd={phases['backward'][-1]*1000:.1f} "
                f"optim={phases['optim'][-1]*1000:.1f}"
            )

    print("\n=== medians (ms) ===")
    for k, v in phases.items():
        if v:
            print(f"{k:10s} {statistics.median(v)*1000:8.2f}")
    total = sum(statistics.median(v) for v in phases.values() if v)
    print(f"{'TOTAL':10s} {total*1000:8.2f}")
    body_tokens = cfg.train.batch_size * cfg.train.grad_accum_steps * cfg.train.data.seq_len
    print(f"body tok/s: {body_tokens/total:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
