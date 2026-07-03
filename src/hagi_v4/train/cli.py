"""HAGI V4 training CLI. All params from YAML config.

Features:
  - Embedding transfer from SmolLM2-135M at init
  - Online KL distillation from SmolLM2-360M (phase 1)
  - Pure CE after teacher freed (phase 2)

Usage:
    hagi4-train --config configs/8gb_canonical.yaml [--dry-run] [--steps N]
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="HAGI V4 training")
    parser.add_argument("--config", default="configs/8gb_canonical.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-distill", action="store_true")
    parser.add_argument("--no-embed-transfer", action="store_true")
    args = parser.parse_args()

    from hagi_v4.config import load_config
    from hagi_v4.model.hagi_v4 import HAGIv4
    from hagi_v4.model.masking import create_random_mask

    overrides = {}
    if args.steps is not None:
        overrides["train.max_steps"] = args.steps
    if args.no_distill:
        overrides["train.distill_enabled"] = False
    cfg = load_config(path=args.config, **overrides)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    logger.info(f"Device: {device} | Config: {args.config}")

    model = HAGIv4(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params / 1e6:.1f}M")

    if cfg.train.distill_enabled and not args.no_embed_transfer:
        from hagi_v4.train.distillation import transfer_embeddings

        transfer_embeddings(model, cfg.train.distill_embed_teacher)

    if args.dry_run:
        B, T = 2, min(cfg.train.seq_len, 128)
        input_ids = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
        targets = input_ids.clone()
        masked_ids, mask = create_random_mask(
            input_ids, mask_ratio=cfg.model.masking.mask_ratio, mask_token_id=cfg.model.masking.mask_token_id
        )
        model.train()
        output = model(masked_ids, targets=targets, mask=mask)
        logger.info(f"Loss: {output.loss.item():.4f}")
        if device.type == "cuda":
            logger.info(f"VRAM: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
        return 0

    teacher = None
    if cfg.train.distill_enabled:
        from hagi_v4.train.distillation import DistillationTeacher

        teacher = DistillationTeacher(cfg.train.distill_teacher)
        teacher._load()

    from hagi_v4.train.loop import train

    def dataloader():
        for _ in range(cfg.train.max_steps):
            ids = torch.randint(0, cfg.model.vocab_size, (cfg.train.batch_size, cfg.train.seq_len), device=device)
            yield {"input_ids": ids, "targets": ids.clone()}

    logger.info(f"Training: {cfg.train.max_steps} steps")
    for metrics in train(model, dataloader(), cfg, log_interval=100, teacher=teacher):
        logger.info(f"step {metrics['step']} | loss={metrics['loss']:.4f} | lr={metrics['lr']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
