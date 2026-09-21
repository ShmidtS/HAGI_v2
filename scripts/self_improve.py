#!/usr/bin/env python3
"""Bounded pyramid-only online self-improvement CLI for HAGI.

Opt-in entry point: compose existing production primitives (generate -> exact
CE score -> adapter-only Trainer.train_step -> KL guard -> checkpoint) without
changing the default generation or training paths.

Usage:
    python scripts/self_improve.py \
        --config configs/level0_ab/ru_baseline.yaml \
        --prompt-ids 1 2 3 4 5 \
        --n-new-tokens 32 --max-iterations 3 --kl-max 0.05 \
        --device cpu --checkpoint-dir checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from hagi.config import Config, load_config
from hagi.model.model import HAGI
from hagi.train.checkpoint import (
    latest_checkpoint,
    load_model,
    load_payload,
    save_checkpoint,
)
from hagi.train.loop import Trainer
from hagi.train.self_improve import self_improve


def _apply_self_improvement_contract(
    cfg: Config,
    args: argparse.Namespace,
    *,
    completed_steps: int = 0,
) -> Config:
    """Turn a training YAML into the bounded self-improvement contract."""
    cfg.model.adapters.enabled = True
    cfg.model.adapters.pyramid.enabled = True
    cfg.model.adapters.pyramid.levels = (1,)
    cfg.model.adapters.ttt_lora.enabled = False
    cfg.model.head.sampled_softmax_k = 0
    cfg.model.loop_depth = 2
    cfg.train.adapt.freeze_base = True
    cfg.train.max_steps = completed_steps + args.max_iterations
    cfg.train.learning_rate = args.lr
    cfg.train.batch_size = 1
    cfg.train.grad_accum_steps = 1
    cfg.train.ce_keep_rate = 1.0
    cfg.train.ce_keep_mode = "bernoulli"
    cfg.train.schedule.warmup_steps = 0
    cfg.train.schedule.inverse_sqrt_stable = False
    cfg.train.logging.exact_ce_interval = 0
    cfg.train.compile_model = False
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded pyramid-only online self-improvement (opt-in).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/level0_ab/ru_baseline.yaml",
        help="YAML config (load_config applies overrides below).",
    )
    parser.add_argument(
        "--prompt-ids",
        type=int,
        nargs="*",
        required=True,
        help="Prompt as integer token ids (required: no external tokenizer).",
    )
    parser.add_argument("--n-new-tokens", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--kl-max", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the highest-numbered checkpoint in --checkpoint-dir.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompt_ids = list(args.prompt_ids)
    if any(t < 0 or t >= cfg.model.vocab_size for t in prompt_ids):
        raise SystemExit(f"prompt token id out of range for vocab_size={cfg.model.vocab_size}")

    cfg = _apply_self_improvement_contract(cfg, args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = HAGI(cfg).to(device).eval()

    start_step = 0
    trainer: Trainer | None = None
    if args.resume:
        ckpt_dir = Path(args.checkpoint_dir)
        latest = latest_checkpoint(ckpt_dir)
        if latest is not None:
            start_step, restored_cfg = load_model(latest, model, device=str(device))
            cfg = _apply_self_improvement_contract(
                restored_cfg,
                args,
                completed_steps=start_step,
            )
            payload = load_payload(latest, device=str(device))
            opt_state = payload.get("optimizer")
            trainer = Trainer(model, cfg, start_step)
            if opt_state is not None:
                trainer.load_optimizer_state(opt_state)
            print(f"[self_improve] resumed from step {start_step}", file=sys.stderr)
        else:
            print(
                f"[self_improve] --resume requested but no checkpoint in {ckpt_dir}",
                file=sys.stderr,
            )
    if trainer is None:
        trainer = Trainer(model, cfg, start_step)

    start = time.time()
    stats = self_improve(
        model=model,
        cfg=cfg,
        prompt_ids=prompt_ids,
        n_new_tokens=args.n_new_tokens,
        max_iterations=args.max_iterations,
        kl_max=args.kl_max,
        ce_min_improve=0.0,
        eos_token_id=cfg.train.data.eos_token_id,
        pad_token_id=cfg.train.data.pad_token_id,
        trainer=trainer,
    )
    elapsed = time.time() - start

    start_step += sum(item.update_applied for item in stats.iterations)
    path = save_checkpoint(
        model, cfg, start_step, Path(args.checkpoint_dir), optimizer=trainer.optimizer
    )

    report = {
        "checkpoint": str(path),
        "step": start_step,
        "duration_s": round(elapsed, 3),
        "iterations": len(stats.iterations),
        "stopped": stats.stopped,
        "best_ce": stats.best_ce,
        "final_pre_ce": stats.iterations[-1].pre_ce if stats.iterations else None,
        "final_post_ce": stats.iterations[-1].post_ce if stats.iterations else None,
        "final_kl": stats.iterations[-1].kl_div if stats.iterations else None,
        "adapter_values_after": stats.iterations[-1].adapter_values_after if stats.iterations else [],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
