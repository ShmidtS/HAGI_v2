"""HAGI V4 training entry point. All params from YAML config.

Reads .bin token files from data/ directory using mix.json ratios.
Performs embedding transfer + online KL distillation from SmolLM2.

Usage:
    python scripts/train_v4.py --config configs/8gb_canonical.yaml
    python scripts/train_v4.py --dry-run
    python scripts/train_v4.py --no-distill
    python scripts/train_v4.py --data-dir data --steps 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch


def setup_file_logging(log_dir: str = "logs") -> str:
    """Add file handler to root logger. Returns log file path."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{log_dir}/train_{timestamp}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return log_path


logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="HAGI V4 training on real data")
    parser.add_argument("--config", default="configs/8gb_canonical.yaml")
    parser.add_argument("--data-dir", default="data", help="Directory with .bin files + mix.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-distill", action="store_true")
    parser.add_argument("--no-embed-transfer", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--log-dir", default="logs", help="Directory for log files")
    args = parser.parse_args()

    if not args.dry_run:
        log_path = setup_file_logging(args.log_dir)
        logger.info(f"Log file: {log_path}")

    from hagi_v4.config import load_config
    from hagi_v4.model.hagi_v4 import HAGIv4
    from hagi_v4.model.masking import create_random_mask

    overrides = {}
    if args.steps is not None:
        overrides["train.max_steps"] = args.steps
    if args.no_distill:
        overrides["train.distill_enabled"] = False
    if args.checkpoint_dir is not None:
        overrides["train.checkpoint_dir"] = args.checkpoint_dir
    cfg = load_config(path=args.config, **overrides)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    logger.info(f"Device: {device} | Config: {args.config} | Data: {args.data_dir}")

    model = HAGIv4(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params / 1e6:.1f}M")

    # Resume from checkpoint
    start_step = 0
    ckpt_extra: dict = {}
    if args.resume:
        from hagi_v4.train.checkpoint import resume_from_checkpoint

        start_step, _, ckpt_extra = resume_from_checkpoint(cfg.train.checkpoint_dir, model, device=str(device))
        if start_step > 0:
            logger.info(f"Resumed from step {start_step}")
        else:
            logger.info("No checkpoint found — starting from scratch")

    # Embedding transfer (skip if resuming)
    if cfg.train.distill_enabled and not args.no_embed_transfer and start_step == 0:
        from hagi_v4.train.distillation import transfer_embeddings

        transfer_embeddings(model, cfg.train.distill_embed_teacher)

    if args.dry_run:
        B, T = 2, min(cfg.train.seq_len, 128)
        input_ids = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
        targets = input_ids.clone()
        masked_ids, mask = create_random_mask(
            input_ids,
            mask_ratio=cfg.model.masking.mask_ratio,
            mask_token_id=cfg.model.masking.mask_token_id,
        )
        from hagi_v4.train.losses import LossAggregator

        aggregator = LossAggregator(cfg)
        model.train()
        output = model(masked_ids, targets=targets, mask=mask)
        total_loss = aggregator(output, targets, mask)
        logger.info(f"Loss: {total_loss.item():.4f}")
        if device.type == "cuda":
            logger.info(f"VRAM: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
        return 0

    # Load distillation teacher (KL distillation only, not embedding transfer)
    teacher = None
    if cfg.train.distill_enabled and getattr(cfg.train, "distill_kl_enabled", False):
        from hagi_v4.train.distillation import DistillationTeacher

        teacher = DistillationTeacher(cfg.train.distill_teacher)
        teacher.load()
        if teacher.is_loaded and device.type == "cuda":
            logger.info(f"Teacher VRAM: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

    # Build sequential cycling dataloader (v1-style curriculum)
    from hagi_v4.data.sequential import build_sequential_dataloader

    dataloader = build_sequential_dataloader(cfg, data_dir=args.data_dir, start_step=start_step)
    if "dataloader" in ckpt_extra:
        dataloader.load_state_dict(ckpt_extra["dataloader"])
        logger.info(f"Dataloader state restored: {ckpt_extra['dataloader']}")
    logger.info(f"Sequential cycling dataloader from {args.data_dir}")

    from hagi_v4.train.loop import train

    logger.info(f"Training: {cfg.train.max_steps} steps, B={cfg.train.batch_size} T={cfg.train.seq_len}")
    if teacher is not None and teacher.is_loaded:
        distill_end = int(cfg.train.max_steps * cfg.train.distill_end_frac)
        logger.info(
            f"Distillation: steps 0->{distill_end} "
            f"(alpha {cfg.train.distill_alpha_start}->{cfg.train.distill_alpha_end}, T={cfg.train.distill_temperature})"
        )

    for metrics in train(
        model, dataloader, cfg, log_interval=1, teacher=teacher, start_step=start_step, resume_extra=ckpt_extra
    ):
        loss = metrics["loss"]
        bits_per_token = loss / 0.6931
        conf = metrics.get("avg_confidence", 0.0)
        ext = metrics.get("extrinsic_info", 0.0)
        par = metrics.get("parity", 0.0)
        logger.info(
            f"step {metrics['step']} | loss={loss:.4f} | bpt={bits_per_token:.2f} | "
            f"lr={metrics['lr']:.6f} | grad={metrics['grad_norm']:.3f} | "
            f"conf={conf:.3f} | ext={ext:.2f} | par={par:.4f}"
        )
    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
