"""Shared utilities for training test and profiling scripts."""

from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger(__name__)


def setup_training(config_path: str = "configs/8gb_canonical.yaml", device: str | None = None):
    """Load config, build model and optimizer. Returns (cfg, model, optimizer, device, n_params)."""
    from hagi_v4.config import load_config
    from hagi_v4.model.hagi_v4 import HAGIv4
    from hagi_v4.train.optim import build_optimizer

    cfg = load_config(path=config_path)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HAGIv4(cfg).to(dev)
    model.train()
    optimizer = build_optimizer(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Params: {n_params / 1e6:.1f}M | B={cfg.train.batch_size} T={cfg.train.seq_len}")
    return cfg, model, optimizer, dev, n_params


def run_training_steps(
    model,
    optimizer,
    cfg,
    device,
    n_steps: int,
    teacher=None,
    log_interval: int = 20,
    per_step_timing: bool = False,
) -> tuple[list[float], float]:
    """Run n_steps of training with random data. Returns (losses, total_time)."""
    from hagi_v4.train.loop import train_step

    losses = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        input_ids = torch.randint(0, cfg.model.vocab_size, (cfg.train.batch_size, cfg.train.seq_len), device=device)
        targets = input_ids.clone()
        batch = {"input_ids": input_ids, "targets": targets}

        t_step = None
        if per_step_timing:
            t_step0 = time.perf_counter()
            metrics = train_step(model, batch, optimizer, cfg, step, teacher=teacher)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_step = time.perf_counter() - t_step0
        else:
            metrics = train_step(model, batch, optimizer, cfg, step, teacher=teacher)

        losses.append(metrics["loss"])

        early_log = 5 if per_step_timing else 10
        if step < early_log or step % log_interval == 0:
            vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            if t_step is not None:
                logger.info(
                    f"  step {step:3d}/{n_steps} | loss={metrics['loss']:.4f} | {t_step:.2f}s | VRAM={vram:.3f} GB"
                )
            else:
                logger.info(
                    f"  step {step:3d}/{n_steps} | loss={metrics['loss']:.4f} | "
                    f"lr={metrics['lr']:.6f} | grad={metrics['grad_norm']:.3f} | "
                    f"mask={metrics['mask_ratio']:.2f} | VRAM={vram:.2f} GB"
                )

    total_time = time.perf_counter() - t0
    return losses, total_time


def log_summary(n_params: int, losses: list[float], total_time: float, n_steps: int, peak_vram: float):
    """Log training summary statistics."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Params:     {n_params / 1e6:.1f}M")
    logger.info(f"  Peak VRAM:  {peak_vram:.3f} GB")
    logger.info(f"  Total time: {total_time:.1f}s | Avg step: {total_time / n_steps:.2f}s")
    logger.info(f"  Loss: start={losses[0]:.4f} -> end={losses[-1]:.4f} (delta={losses[-1] - losses[0]:+.4f})")
    logger.info(f"  Min loss: {min(losses):.4f} at step {losses.index(min(losses))}")
    logger.info("=" * 60)
