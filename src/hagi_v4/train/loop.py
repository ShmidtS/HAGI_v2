"""Training loop for HAGI V8.

V8 changes vs V7:
  1. Simplified loss hierarchy (3 levels, 4 aux instead of 7)
  2. FOXP2 made optional (disabled by default for clean from-scratch)
  3. No frozen embeddings (trainable from scratch)
  4. Distillation optional (not required for basic training)
  5. AWGN annealing preserved (channel robustness)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.masking import (
    adaptive_mask_ratio,
    create_random_mask,
    progressive_mask_ratio,
)
from hagi_v4.model.outputs import ModelOutput
from hagi_v4.train.losses import LossAggregator
from hagi_v4.train.optim import CombinedOptimizer, build_optimizer

logger = logging.getLogger(__name__)


def configure_runtime() -> None:
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def lr_at(step: int, cfg: HAGIv4Config) -> float:
    """Linear warmup from 0, then stable, then cosine decay."""
    tc = cfg.train
    warmup = tc.warmup_steps
    max_steps = tc.max_steps
    base_lr = tc.learning_rate
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    stable_end = int(max_steps * 0.8)
    if step < stable_end:
        return base_lr
    decay_steps = max(max_steps - stable_end, 1)
    progress = (step - stable_end) / decay_steps
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def cast_to_bf16(model: nn.Module) -> None:
    model.to(torch.bfloat16)


def grad_norm(model: nn.Module) -> float:
    """Compute gradient norm (no clipping/scaling)."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    total_sq = torch.stack([g.pow(2).sum() for g in grads]).sum()
    return total_sq.sqrt().item()


def _two_phase_mask_ratio(step: int, cfg: HAGIv4Config) -> float:
    """Three-phase mask ratio for LLaDA-compatible masked LM."""
    if not cfg.train.use_two_phase_schedule:
        return cfg.model.masking.mask_ratio
    max_steps = cfg.train.max_steps
    split1 = int(max_steps * cfg.train.two_phase_split)
    split2 = int(max_steps * 0.8)
    if step < split1:
        return cfg.train.phase1_mask_ratio
    if step < split2:
        return cfg.train.phase2_mask_ratio
    return getattr(cfg.train, "phase3_mask_ratio", 0.50)


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: CombinedOptimizer,
    cfg: HAGIv4Config,
    step: int,
    teacher=None,
    loss_aggregator: LossAggregator | None = None,
    mask_warmup_steps: int = 20000,
    adaptive_mask_state: dict | None = None,
    distill_end_step: int = 0,
) -> dict:
    """Single training step with gradient accumulation."""
    model.train()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    targets = batch["targets"].to(device)

    mc = cfg.model.masking
    if mc.use_adaptive_erasure and adaptive_mask_state is not None:
        mask_ratio = adaptive_mask_state.get("ratio", mc.mask_ratio)
    elif cfg.train.use_two_phase_schedule:
        mask_ratio = _two_phase_mask_ratio(step, cfg)
    elif mc.use_progressive:
        mask_ratio = progressive_mask_ratio(step, mask_warmup_steps, 0.15, mc.mask_ratio)
    else:
        mask_ratio = mc.mask_ratio
    masked_ids, mask = create_random_mask(input_ids, mask_ratio, mc.mask_token_id)

    if loss_aggregator is None:
        loss_aggregator = LossAggregator(cfg)

    accum = cfg.train.grad_accum_steps
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    output: ModelOutput | None = None

    for micro_idx in range(accum):
        awgn_sigma = 0.0
        if cfg.train.awgn_enabled:
            awgn_end = int(cfg.train.max_steps * cfg.train.awgn_end_frac)
            if step < awgn_end:
                progress = step / max(awgn_end, 1)
                awgn_sigma = cfg.train.awgn_sigma_start * (1.0 - progress) + cfg.train.awgn_sigma_end * progress
        output = model(masked_ids, targets=targets, mask=mask, step=step, awgn_sigma=awgn_sigma)
        loss = loss_aggregator(output, targets, mask, step=step)

        use_distill = (
            teacher is not None
            and getattr(teacher, "_loaded", False)
            and cfg.train.distill_enabled
            and step % cfg.train.distill_every == 0
            and step <= distill_end_step
        )
        if use_distill:
            from hagi_v4.train.distillation import alpha_at

            alpha = alpha_at(
                step,
                cfg.train.distill_alpha_start,
                cfg.train.distill_alpha_end,
                cfg.train.max_steps,
                cfg.train.distill_end_frac,
            )
            with torch.inference_mode():
                teacher_hidden = teacher.get_hidden(input_ids)
            if teacher_hidden is not None:
                distill_loss = teacher.distillation_loss_chunked(
                    student_hidden=output.hidden,
                    teacher_hidden=teacher_hidden,
                    student_lm_head_weight=model.lm_head.weight,
                    targets=targets,
                    ce_loss=loss,
                    mask=mask,
                    alpha=alpha,
                )
                loss = distill_loss
                del teacher_hidden

        if not torch.isfinite(loss).all():
            logger.warning(f"Step {step} micro {micro_idx}: non-finite loss — skipping")
            continue

        scaled_loss = loss / accum
        scaled_loss.backward()
        total_loss += loss.item()

    gn = grad_norm(model)

    optimizer.step()

    if output is None:
        return {"loss": 0.0, "step": step, "grad_norm": gn}

    with torch.no_grad():
        if output.logits is not None:
            probs = F.softmax(output.logits.float(), dim=-1)
            avg_confidence = probs.max(dim=-1).values.mean().item()
        elif output.ce_loss is not None:
            avg_confidence = max(0.0, 1.0 - output.ce_loss.item() / 10.0)
        else:
            avg_confidence = 0.5
    if mc.use_adaptive_erasure and adaptive_mask_state is not None:
        adaptive_mask_state["ratio"] = adaptive_mask_ratio(
            avg_confidence,
            adaptive_mask_state.get("ratio", mc.mask_ratio),
            adaptation_rate=mc.adaptation_rate,
        )

    return {
        "loss": total_loss / max(accum, 1),
        "parity": float(output.aux.parity.detach()) if output.aux.parity is not None else 0.0,
        "extrinsic_info": float(output.aux.extrinsic_info.detach()) if output.aux.extrinsic_info is not None else 0.0,
        "rate_distortion": float(output.aux.rate_distortion.detach())
        if output.aux.rate_distortion is not None
        else 0.0,
        "whiteness": float(output.aux.whiteness.detach()) if output.aux.whiteness is not None else 0.0,
        "contrastive": float(output.aux.contrastive.detach()) if output.aux.contrastive is not None else 0.0,
        "avg_confidence": avg_confidence,
        "grad_norm": gn,
        "mask_ratio": mask_ratio,
        "lr": lr_at(step, cfg),
        "step": step,
    }


def train(
    model: nn.Module,
    dataloader,
    cfg: HAGIv4Config,
    log_interval: int = 100,
    teacher=None,
    start_step: int = 0,
    resume_extra: dict | None = None,
) -> Iterator[dict]:
    from hagi_v4.train.checkpoint import save_checkpoint

    configure_runtime()

    if torch.cuda.is_available() and cfg.train.precision == "bf16":
        cast_to_bf16(model)

    optimizer = build_optimizer(model, cfg)
    loss_aggregator = LossAggregator(cfg)
    adaptive_mask_state = {"ratio": cfg.model.masking.mask_ratio} if cfg.model.masking.use_adaptive_erasure else None

    if resume_extra:
        opt_state = resume_extra.get("optimizer")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
                logger.info("Optimizer state restored from checkpoint")
            except (ValueError, KeyError, RuntimeError) as exc:
                logger.warning(f"Optimizer state mismatch — starting fresh: {exc}")
        if "adaptive_mask" in resume_extra and adaptive_mask_state is not None:
            adaptive_mask_state["ratio"] = resume_extra["adaptive_mask"]["ratio"]
            logger.info(f"Adaptive mask ratio restored: {adaptive_mask_state['ratio']:.4f}")
        if "rng" in resume_extra:
            rng = resume_extra["rng"]
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state(rng["cuda"].cpu().to(torch.uint8))
            logger.info("RNG state restored from checkpoint")

    step = start_step
    distill_end_step = int(cfg.train.max_steps * cfg.train.distill_end_frac)
    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_interval = cfg.train.checkpoint_interval
    ckpt_keep = cfg.train.checkpoint_keep_last

    for batch in dataloader:
        if step >= cfg.train.max_steps:
            break

        if teacher is not None and getattr(teacher, "_loaded", False) and step == distill_end_step:
            logger.info(f"Step {step}: distillation ended — freeing teacher")
            teacher.free()

        metrics = train_step(
            model,
            batch,
            optimizer,
            cfg,
            step,
            teacher,
            loss_aggregator=loss_aggregator,
            adaptive_mask_state=adaptive_mask_state,
            distill_end_step=distill_end_step,
        )
        if step % log_interval == 0:
            yield metrics

        if ckpt_interval > 0 and step > 0 and step % ckpt_interval == 0:
            extra = {
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                },
            }
            if adaptive_mask_state is not None:
                extra["adaptive_mask"] = {"ratio": adaptive_mask_state["ratio"]}
            if hasattr(dataloader, "state_dict"):
                extra["dataloader"] = dataloader.state_dict()
            save_checkpoint(model, optimizer, cfg, step, ckpt_dir, ckpt_keep, extra=extra)

        step += 1

    if ckpt_interval > 0:
        extra = {
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }
        if adaptive_mask_state is not None:
            extra["adaptive_mask"] = {"ratio": adaptive_mask_state["ratio"]}
        if hasattr(dataloader, "state_dict"):
            extra["dataloader"] = dataloader.state_dict()
        save_checkpoint(model, optimizer, cfg, step, ckpt_dir, ckpt_keep, extra=extra)
