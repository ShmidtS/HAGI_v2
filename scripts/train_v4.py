"""HAGI training entry point. All params come from the YAML config.

Usage:
    python scripts/train_v4.py --config configs/smollm2.yaml
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
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return log_path


logger = logging.getLogger(__name__)


def format_training_metrics(metrics: dict) -> str:
    return (
        f"step {metrics['step']} | loss={metrics['loss']:.4f} | bpt={metrics.get('bpt', float('nan')):.2f} | "
        f"lr={metrics['lr']:.6f} | grad={metrics['grad_norm']:.3f} | "
        f"conf={metrics.get('avg_confidence', 0.0):.3f} | "
        f"masked_ce={metrics.get('masked_ce', float('nan')):.4f} | "
        f"rate={metrics.get('rate', float('nan')):.4f} | "
        f"distortion={metrics.get('distortion', float('nan')):.4f} | "
        f"entropy={metrics.get('posterior_entropy', float('nan')):.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="HAGI training on real data")
    parser.add_argument("--config", default="configs/smollm2.yaml")
    parser.add_argument("--data-dir", default="data", help="Directory with .bin files + mix.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-distill", action="store_true")
    parser.add_argument("--no-embed-transfer", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--log-dir", default="logs", help="Directory for log files")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from checkpoint. Bare '--resume' picks the latest step-*.pt.",
    )
    args = parser.parse_args()

    log_path = setup_file_logging(args.log_dir)
    logger.info(f"Log file: {log_path}")

    from hagi_v4.config import load_config
    from hagi_v4.model.model import HAGI

    overrides = {}
    if args.steps is not None:
        overrides["train.max_steps"] = args.steps
    if args.no_distill:
        overrides["train.distill.enabled"] = False
    if args.checkpoint_dir is not None:
        overrides["train.checkpoint_dir"] = args.checkpoint_dir
    cfg = load_config(path=args.config, **overrides)

    from hagi_v4.train.checkpoint import assert_fresh_checkpoint_root, latest_checkpoint, load_checkpoint_payload

    assert_fresh_checkpoint_root(cfg.train.checkpoint_dir)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    logger.info(f"Device: {device} | Config: {args.config} | Data: {args.data_dir}")

    teacher = None
    if cfg.train.distill.enabled is True:
        from hagi_v4.train.distillation import create_distillation_teacher

        teacher = create_distillation_teacher(cfg, device)

    model = HAGI(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params / 1e6:.1f}M")

    start_step = 0
    optimizer_state = None
    if args.resume is not None:
        from hagi_v4.train.checkpoint import load_model_checkpoint

        ckpt_path = args.resume if args.resume != "latest" else latest_checkpoint(cfg.train.checkpoint_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"--resume: no step-*.pt found in {cfg.train.checkpoint_dir}")
        completed, _ = load_model_checkpoint(ckpt_path, model, str(device))
        start_step = completed
        optimizer_state = load_checkpoint_payload(ckpt_path, str(device)).get("optimizer")
        logger.info(f"Resumed from {ckpt_path} at step {completed} (optimizer state: {'yes' if optimizer_state else 'no'})")

    if cfg.train.distill.enabled is True and not args.no_embed_transfer and start_step == 0:
        from hagi_v4.train.distillation import transfer_embeddings

        transfer_embeddings(model, cfg.train.distill.embed_teacher)

    if args.dry_run:
        B, T = 2, min(cfg.train.seq_len, 128)
        input_ids = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
        valid_target_mask = torch.ones_like(input_ids, dtype=torch.bool)
        from hagi_v4.train.losses import LossAggregator

        aggregator = LossAggregator(cfg)
        model.train()
        output = model(
            input_ids,
            targets=input_ids.clone(),
            semantic_unknown_mask=torch.zeros_like(valid_target_mask),
            prediction_mask=valid_target_mask,
            valid_target_mask=valid_target_mask,
            attention_mode="causal",
        )
        total_loss = aggregator(output, step=0)
        logger.info(f"Loss: {total_loss.item():.4f}")
        if device.type == "cuda":
            logger.info(f"VRAM: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
        return 0

    from hagi_v4.data.sequential import build_sequential_dataloader

    dataloader = build_sequential_dataloader(cfg, data_dir=args.data_dir)
    logger.info(f"Sequential cycling dataloader from {args.data_dir}")

    from hagi_v4.train.loop import train

    logger.info(f"Training: {cfg.train.max_steps} steps, B={cfg.train.batch_size} T={cfg.train.seq_len}")
    if teacher is not None and teacher.is_loaded:
        distill_end = int(cfg.train.max_steps * cfg.train.distill.end_frac)
        logger.info(
            f"Distillation: steps 0->{distill_end} "
            f"(alpha {cfg.train.distill.alpha_start}->{cfg.train.distill.alpha_end}, T={cfg.train.distill.temperature})"
        )

    for metrics in train(
        model, dataloader, cfg, log_interval=1, teacher=teacher,
        start_step=start_step, optimizer_state=optimizer_state,
    ):
        logger.info(format_training_metrics(metrics))
    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
