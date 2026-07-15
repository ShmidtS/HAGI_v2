"""Finite-training diagnostics for channel-correct training runs."""

from __future__ import annotations

import argparse
import json
import math

import torch


def collect_diagnostics(metrics: list[dict]) -> dict:
    keys = ("loss", "grad_norm", "grad_rms", "correction_alignment")
    return {
        "all_finite": all(math.isfinite(float(metric[key])) for metric in metrics for key in keys),
        "max_grad_norm": max((float(metric["grad_norm"]) for metric in metrics), default=0.0),
        "max_grad_rms": max((float(metric["grad_rms"]) for metric in metrics), default=0.0),
        "loss_start": float(metrics[0]["loss"]) if metrics else None,
        "loss_end": float(metrics[-1]["loss"]) if metrics else None,
        "correction_alignment_start": float(metrics[0]["correction_alignment"]) if metrics else None,
        "correction_alignment_end": float(metrics[-1]["correction_alignment"]) if metrics else None,
        "steps": len(metrics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from hagi_v4.config import load_config
    from hagi_v4.model.hagi_v4 import HAGIv4
    from hagi_v4.train.loop import train_step
    from hagi_v4.train.optim import build_optimizer

    cfg = load_config(args.config, **{"train.max_steps": args.steps, "train.distill_enabled": False})
    cfg.model.attention.max_seq_len = min(cfg.model.attention.max_seq_len, 32)
    cfg.train.seq_len = min(cfg.train.seq_len, 32)
    cfg.train.batch_size = 1
    model = HAGIv4(cfg).to(args.device)
    optimizer = build_optimizer(model, cfg)
    metrics = []
    for step in range(args.steps):
        microbatches = []
        for micro_idx in range(cfg.train.grad_accum_steps):
            ids = torch.randint(0, cfg.model.vocab_size, (1, cfg.train.seq_len), device=args.device)
            microbatches.append({"input_ids": ids, "targets": ids.clone(), "fingerprint": (step, micro_idx)})
        metric = train_step(model, microbatches, optimizer, cfg, step)
        metric["parameters_finite"] = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
        metric["gradients_finite"] = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()
        )
        metrics.append(metric)
    result = collect_diagnostics(metrics)
    result["all_finite"] = result["all_finite"] and all(
        metric["parameters_finite"] and metric["gradients_finite"] for metric in metrics
    )
    print(json.dumps(result, indent=2))
    return 0 if result["all_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
