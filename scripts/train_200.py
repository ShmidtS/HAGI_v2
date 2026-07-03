"""HAGI V4 — 200-step training test with distillation. All params from YAML."""

from __future__ import annotations

import logging
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    from hagi_v4.config import load_config
    from hagi_v4.model.hagi_v4 import HAGIv4
    from hagi_v4.train.distillation import DistillationTeacher, transfer_embeddings
    from hagi_v4.train.loop import train_step
    from hagi_v4.train.optim import build_optimizer

    cfg = load_config(path="configs/8gb_canonical.yaml")
    device = torch.device("cuda")

    model = HAGIv4(cfg).to(device)
    model.train()
    optimizer = build_optimizer(model, cfg)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Params: {n_params / 1e6:.1f}M | B={cfg.train.batch_size} T={cfg.train.seq_len}")

    # Embedding transfer
    transfer_embeddings(model, cfg.train.distill_embed_teacher)

    # Load teacher
    teacher = DistillationTeacher(cfg.train.distill_teacher)
    teacher._load()

    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.perf_counter()

    n_steps = 200
    for step in range(n_steps):
        input_ids = torch.randint(0, cfg.model.vocab_size, (cfg.train.batch_size, cfg.train.seq_len), device=device)
        targets = input_ids.clone()
        batch = {"input_ids": input_ids, "targets": targets}

        metrics = train_step(model, batch, optimizer, cfg, step, teacher=teacher)
        losses.append(metrics["loss"])

        if step < 10 or step % 20 == 0:
            vram = torch.cuda.max_memory_allocated() / 1e9
            logger.info(
                f"  step {step:3d}/{n_steps} | loss={metrics['loss']:.4f} | "
                f"lr={metrics['lr']:.6f} | grad={metrics['grad_norm']:.3f} | "
                f"mask={metrics['mask_ratio']:.2f} | VRAM={vram:.2f} GB"
            )

    total = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    teacher.free()

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Params:     {n_params / 1e6:.1f}M")
    logger.info(f"  Peak VRAM:  {peak_vram:.3f} GB")
    logger.info(f"  Total time: {total:.1f}s | Avg step: {total / n_steps:.2f}s")
    logger.info(f"  Loss: start={losses[0]:.4f} -> end={losses[-1]:.4f} (delta={losses[-1] - losses[0]:+.4f})")
    logger.info(f"  Min loss: {min(losses):.4f} at step {losses.index(min(losses))}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
