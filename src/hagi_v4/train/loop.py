"""Training loop for HAGI V4.

Key fixes from log analysis:
1. grad_accum_steps IMPLEMENTED — micro-batch loop with loss scaling
2. Suffix masking removed — only random + span (suffix causes loss 8-16)
3. Soft grad norm scaling (not clipping) — scale grads by 1/max(1, norm/target)
4. Warmup starts from 0 — lr=0 at step 0, linear ramp to max
5. manual_bf16 — model cast to bf16, no autocast
6. Teacher under inference_mode + del + empty_cache before backward
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import torch
from torch import nn

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.masking import (
    create_random_mask,
    progressive_mask_ratio,
)
from hagi_v4.model.outputs import ModelOutput
from hagi_v4.train.distillation import alpha_at, temperature_at
from hagi_v4.train.losses import LossAggregator
from hagi_v4.train.optim import CombinedOptimizer, build_optimizer

logger = logging.getLogger(__name__)


def configure_runtime() -> None:
    """Set CUDA runtime flags. Call once at startup, not at import time."""
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


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


def sample_mask_pattern() -> str:
    """Only random masking — span causes loss 6-12 and grad spikes."""
    return "random"


def create_mask(input_ids, pattern, mask_ratio, mask_token_id, span_length=3):
    return create_random_mask(input_ids, mask_ratio, mask_token_id)


def cast_to_bf16(model: nn.Module) -> None:
    model.to(torch.bfloat16)


def soft_grad_scale(model: nn.Module, target: float = 1.0) -> float:
    """Scale gradients by 1/max(1, norm/target). Not clipping — uniform scaling."""
    norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    if norm > target:
        scale = target / norm
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
    return norm


def _two_phase_mask_ratio(step: int, cfg: HAGIv4Config) -> float:
    """Two-phase mask ratio: low in fitting phase, high in compression phase."""
    if not cfg.train.use_two_phase_schedule:
        return cfg.model.masking.mask_ratio
    split_step = int(cfg.train.max_steps * cfg.train.two_phase_split)
    if step < split_step:
        return cfg.train.phase1_mask_ratio
    return cfg.train.phase2_mask_ratio


def _two_phase_coherence_weight(step: int, cfg: HAGIv4Config) -> float:
    """Two-phase coherence weight: minimal early, stronger late."""
    if not cfg.train.use_two_phase_schedule:
        return cfg.train.w_coherence
    split_step = int(cfg.train.max_steps * cfg.train.two_phase_split)
    if step < split_step:
        return cfg.train.phase1_w_coherence
    return cfg.train.phase2_w_coherence


def _log_grade_variance(model: nn.Module, hidden: torch.Tensor, step: int, cfg: HAGIv4Config) -> dict:
    """Log per-grade activation variance for capacity monitoring."""
    if not cfg.train.log_grade_variance:
        return {}
    if step % cfg.train.grade_log_interval != 0:
        return {}
    gdr = model.gdr
    b = gdr._bounds
    var_scalar = hidden[..., b[0] : b[1]].float().var(dim=(0, 1)).sum().item()
    var_vector = hidden[..., b[1] : b[2]].float().var(dim=(0, 1)).sum().item()
    var_bivector = hidden[..., b[2] : b[3]].float().var(dim=(0, 1)).sum().item()
    var_trivector = hidden[..., b[3] : b[4]].float().var(dim=(0, 1)).sum().item()
    return {
        "var_scalar": var_scalar,
        "var_vector": var_vector,
        "var_bivector": var_bivector,
        "var_trivector": var_trivector,
    }


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: CombinedOptimizer,
    cfg: HAGIv4Config,
    step: int,
    teacher=None,
    loss_aggregator: LossAggregator | None = None,
    mask_warmup_steps: int = 20000,
    grad_norm_target: float = 1.0,
) -> dict:
    """Single training step with gradient accumulation."""
    model.train()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    targets = batch["targets"].to(device)

    mc = cfg.model.masking
    if cfg.train.use_two_phase_schedule:
        mask_ratio = _two_phase_mask_ratio(step, cfg)
    elif mc.use_progressive:
        mask_ratio = progressive_mask_ratio(step, mask_warmup_steps, 0.15, mc.mask_ratio)
    else:
        mask_ratio = mc.mask_ratio
    pattern = sample_mask_pattern()
    masked_ids, mask = create_mask(input_ids, pattern, mask_ratio, mc.mask_token_id, mc.span_length)

    w_coherence = cfg.train.w_coherence
    if cfg.train.use_two_phase_schedule:
        w_coherence = _two_phase_coherence_weight(step, cfg)

    if loss_aggregator is None:
        loss_aggregator = LossAggregator(cfg)

    accum = cfg.train.grad_accum_steps
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    output: ModelOutput | None = None

    for micro_idx in range(accum):
        output = model(masked_ids, targets=targets, mask=mask, step=step)
        loss = loss_aggregator(output, targets, mask, w_coherence_override=w_coherence)

        use_distill = (
            teacher is not None
            and getattr(teacher, "_loaded", False)
            and cfg.train.distill_enabled
            and step % cfg.train.distill_every == 0
        )
        if use_distill:
            alpha = alpha_at(
                step,
                cfg.train.distill_alpha_start,
                cfg.train.distill_alpha_end,
                cfg.train.max_steps,
                cfg.train.distill_end_frac,
            )
            if cfg.train.distill_use_temp_anneal:
                temperature = temperature_at(
                    step,
                    cfg.train.max_steps,
                    cfg.train.distill_temp_start,
                    cfg.train.distill_temp_end,
                    cfg.train.distill_end_frac,
                )
            else:
                temperature = cfg.train.distill_temperature
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
                    temperature=temperature,
                    alpha=alpha,
                )
                loss = distill_loss
                del teacher_hidden
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if not torch.isfinite(loss).all():
            logger.warning(f"Step {step} micro {micro_idx}: non-finite loss — skipping")
            continue

        scaled_loss = loss / accum
        scaled_loss.backward()
        total_loss += loss.item()

    grad_norm = soft_grad_scale(model, grad_norm_target)
    optimizer.step()

    if output is None:
        return {"loss": 0.0, "step": step, "grad_norm": grad_norm}

    grade_vars = _log_grade_variance(model, output.hidden, step, cfg)

    result = {
        "loss": total_loss / max(accum, 1),
        "moe_aux": float(output.aux.moe_lb.detach()) if output.aux.moe_lb is not None else 0.0,
        "gdr_router": float(output.aux.gdr_router.detach()) if output.aux.gdr_router is not None else 0.0,
        "grad_norm": grad_norm,
        "mask_ratio": mask_ratio,
        "mask_pattern": pattern,
        "lr": lr_at(step, cfg),
        "step": step,
    }
    result.update(grade_vars)
    return result


def train(
    model: nn.Module,
    dataloader,
    cfg: HAGIv4Config,
    log_interval: int = 100,
    teacher=None,
    start_step: int = 0,
) -> Iterator[dict]:
    from hagi_v4.train.checkpoint import save_checkpoint

    configure_runtime()

    if torch.cuda.is_available() and cfg.train.precision == "bf16":
        cast_to_bf16(model)

    optimizer = build_optimizer(model, cfg)
    loss_aggregator = LossAggregator(cfg)
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

        metrics = train_step(model, batch, optimizer, cfg, step, teacher, loss_aggregator=loss_aggregator)
        if step % log_interval == 0:
            yield metrics

        if ckpt_interval > 0 and step > 0 and step % ckpt_interval == 0:
            save_checkpoint(model, optimizer, cfg, step, ckpt_dir, ckpt_keep)

        step += 1

    if ckpt_interval > 0:
        save_checkpoint(model, optimizer, cfg, step, ckpt_dir, ckpt_keep)
