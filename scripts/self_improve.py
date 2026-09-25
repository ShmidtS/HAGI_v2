#!/usr/bin/env python3
"""Bounded online self-improvement CLI for HAGI.

Opt-in entry point: compose existing production primitives (generate -> exact
CE score -> adapter-only update -> KL guard -> checkpoint) without changing the
default generation or training paths. ``gradient`` uses the Pyramidal Contour;
``rls`` uses TTT-LoRA and allocates no optimizer. A run that applied no accepted
update persists nothing and exits 2; an applied update is operational progress
only, never quality evidence.

RLS checkpoints contain only the model/config, not the persistent RLS
accumulators, so RLS ``--resume`` is deliberately unsupported.

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
    if args.mode == "rls":
        cfg.model.adapters.pyramid.enabled = False
        cfg.model.adapters.ttt_lora.enabled = True
    else:
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
        description="Bounded online self-improvement (opt-in).",
    )
    parser.add_argument(
        "--mode",
        choices=("gradient", "rls"),
        default="gradient",
        help="Update contour: gradient uses Pyramidal adapters; rls uses TTT-LoRA.",
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

    if args.mode == "rls" and args.resume:
        raise SystemExit(
            "--mode rls does not support --resume: model checkpoints omit "
            "persistent RLS accumulators"
        )

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
            payload = load_payload(latest, device=str(device))
            opt_state = payload.get("optimizer")
            if (
                not isinstance(opt_state, dict)
                or set(opt_state) != {"muon", "adamw"}
                or not all(isinstance(part, dict) for part in opt_state.values())
            ):
                raise SystemExit(
                    "gradient --resume requires optimizer state with muon/adamw parts"
                )
            start_step, restored_cfg = load_model(latest, model, device=str(device))
            cfg = _apply_self_improvement_contract(
                restored_cfg,
                args,
                completed_steps=start_step,
            )
            trainer = Trainer(model, cfg, start_step)
            try:
                trainer.load_optimizer_state(opt_state)
            except Exception as exc:
                raise SystemExit(
                    f"gradient --resume optimizer state is incompatible: {exc}"
                ) from exc
            print(f"[self_improve] resumed from step {start_step}", file=sys.stderr)
        else:
            # ``--resume`` means resume, not "start a fresh run". Falling through
            # here makes a missing parent indistinguishable from a successful
            # recovery: automation that always passes --resume would silently
            # train from scratch and write a new checkpoint under the lineage it
            # expected to continue.
            raise SystemExit(
                f"--resume requested but no checkpoint in {ckpt_dir}"
            )
    if args.mode == "gradient" and trainer is None:
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
        mode=args.mode,
        trainer=trainer,
    )
    elapsed = time.time() - start

    # A rejected run must leave no checkpoint behind. The loop scores its own
    # generated trajectory, so a run that applied no accepted update has nothing
    # worth persisting; writing one anyway would hand the next `--resume` a
    # step-*.pt that looks like real progress while being an untouched parent.
    # Exit 2 mirrors the existing guard convention in this repo
    # (scripts/qwen_pyramid_smoke.py) so automation can branch on the code.
    accepted_updates = stats.accepted_updates
    # "Accepted" is an in-memory guard decision; "applied" is a written delta.
    # RLS warmup legitimately accumulates rows while lora_B is unchanged, and a
    # model-only checkpoint cannot carry those accumulators, so persisting on
    # acceptance alone would publish an unchanged parent under a new step
    # number and drop the very state the next run would need.
    applied_updates = sum(1 for item in stats.iterations if item.update_applied)
    start_step += applied_updates
    rejected = applied_updates == 0
    path = None
    if not rejected:
        optimizer = trainer.optimizer if trainer is not None else None
        path = save_checkpoint(
            model,
            cfg,
            start_step,
            Path(args.checkpoint_dir),
            optimizer=optimizer,
        )

    report = {
        "checkpoint": str(path) if path is not None else None,
        "mode": args.mode,
        "step": start_step,
        "duration_s": round(elapsed, 3),
        "iterations": len(stats.iterations),
        "accepted_updates": accepted_updates,
        "applied_updates": applied_updates,
        "stopped": stats.stopped,

        "best_ce": stats.best_ce,
        "final_pre_ce": stats.iterations[-1].pre_ce if stats.iterations else None,
        "final_post_ce": stats.iterations[-1].post_ce if stats.iterations else None,
        "final_kl": stats.iterations[-1].kl_div if stats.iterations else None,
        "delta_rms_frac": (
            stats.iterations[-1].delta_rms_frac if stats.iterations else None
        ),
        "adapter_values_after": stats.iterations[-1].adapter_values_after if stats.iterations else [],
        # Operational success only. This loop is scored on the same window it
        # trained on and has no immutable external holdout, so it can never by
        # itself be quality evidence (see docs/RESEARCH_FRONTIER.md).
        "quality_supported": False,
        "production_promotion": False,
    }
    print(json.dumps(report, indent=2))
    if rejected:
        print(
            "[self_improve] no applied update: nothing persisted; refusing to "
            "checkpoint a run that left the model unchanged",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
