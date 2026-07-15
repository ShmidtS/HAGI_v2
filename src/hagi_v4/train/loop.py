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
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.masking import (
    adaptive_mask_ratio,
    create_erasure_mask,
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


def muon_lr_at(step: int, cfg: HAGIv4Config) -> float:
    """Linear warmup from 0, then stable, then cosine decay for Muon."""
    tc = cfg.train
    warmup = tc.warmup_steps
    max_steps = tc.max_steps
    base_lr = tc.muon_lr
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    stable_end = int(max_steps * 0.8)
    if step < stable_end:
        return base_lr
    decay_steps = max(max_steps - stable_end, 1)
    progress = (step - stable_end) / decay_steps
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: CombinedOptimizer, step: int, cfg: HAGIv4Config) -> None:
    """Apply LR schedule to optimizer param_groups."""
    lr_adam = lr_at(step, cfg)
    lr_muon = muon_lr_at(step, cfg)
    for group in optimizer.param_groups:
        if group.get("_muon", False):
            group["lr"] = lr_muon
        else:
            group["lr"] = lr_adam


def cast_to_bf16(model: nn.Module) -> None:
    model.to(torch.bfloat16)


def gradient_stats(model: nn.Module, train_cfg) -> tuple[float, float]:
    """Report raw global norm and element RMS, then optionally clip."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0, 0.0
    total_sq = torch.stack([g.pow(2).sum() for g in grads]).sum()
    total_elements = sum(g.numel() for g in grads)
    raw_norm = total_sq.sqrt().item()
    grad_rms = (total_sq / max(total_elements, 1)).sqrt().item()
    max_grad_norm = getattr(train_cfg, "max_grad_norm", None)
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    return raw_norm, grad_rms


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
    microbatches: list[dict],
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
    mc = cfg.model.masking
    if mc.use_adaptive_erasure and adaptive_mask_state is not None:
        mask_ratio = adaptive_mask_state.get("ratio", mc.mask_ratio)
    elif cfg.train.use_two_phase_schedule:
        mask_ratio = _two_phase_mask_ratio(step, cfg)
    elif mc.use_progressive:
        mask_ratio = progressive_mask_ratio(step, mask_warmup_steps, 0.15, mc.mask_ratio)
    else:
        mask_ratio = mc.mask_ratio
    if loss_aggregator is None:
        loss_aggregator = LossAggregator(cfg)

    accum = cfg.train.grad_accum_steps
    if len(microbatches) != accum:
        raise ValueError(f"expected {accum} microbatches, got {len(microbatches)}")
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    output: ModelOutput | None = None
    outputs: list[ModelOutput] = []
    all_finite = True

    for micro_idx, batch in enumerate(microbatches):
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        mask = create_erasure_mask(input_ids, mask_ratio)
        awgn_sigma = 0.0
        if cfg.train.awgn_enabled:
            awgn_end = int(cfg.train.max_steps * cfg.train.awgn_end_frac)
            if step < awgn_end:
                progress = step / max(awgn_end, 1)
                awgn_sigma = cfg.train.awgn_sigma_start * (1.0 - progress) + cfg.train.awgn_sigma_end * progress
        output = model(input_ids, targets=targets, mask=mask, step=step, awgn_sigma=awgn_sigma)
        outputs.append(output)
        loss = loss_aggregator(output, targets, mask, step=step)

        use_distill = (
            teacher is not None
            and getattr(teacher, "_loaded", False)
            and cfg.train.distill_enabled
            and step < distill_end_step
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
                    reconstruction_loss=loss,
                    mask=mask,
                    align_projection=model.distill_align,
                    alpha=alpha,
                )
                loss = distill_loss
                del teacher_hidden

        if not torch.isfinite(loss).all():
            logger.warning(f"Step {step} micro {micro_idx}: non-finite loss — cancelling update")
            all_finite = False
            break

        scaled_loss = loss / accum
        scaled_loss.backward()
        total_loss += loss.item()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    all_finite = all_finite and all(torch.isfinite(grad).all().item() for grad in grads)
    if not all_finite:
        optimizer.zero_grad(set_to_none=True)
        for param in model.parameters():
            param.grad = None
        return {
            "loss": total_loss / max(accum, 1),
            "parity": 0.0,
            "correction_alignment": 0.0,
            "rate_distortion": 0.0,
            "whiteness": 0.0,
            "contrastive": 0.0,
            "avg_confidence": 0.0,
            "grad_norm": 0.0,
            "grad_rms": 0.0,
            "mask_ratio": mask_ratio,
            "lr": lr_at(step, cfg) if hasattr(cfg.train, "warmup_steps") else 0.0,
            "step": step,
            "update_applied": False,
            "all_finite": False,
        }

    gn, grad_rms = gradient_stats(model, cfg.train)
    optimizer.step()

    if output is None:
        return {"loss": 0.0, "step": step, "grad_norm": gn, "grad_rms": grad_rms}

    with torch.no_grad():
        confidences = []
        for micro_output in outputs:
            if micro_output.logits is not None:
                probs = F.softmax(micro_output.logits.float(), dim=-1)
                confidences.append(probs.max(dim=-1).values.mean().item())
            elif micro_output.ce_loss is not None:
                confidences.append(max(0.0, 1.0 - micro_output.ce_loss.item() / 10.0))
            else:
                confidences.append(0.5)
        avg_confidence = sum(confidences) / max(len(confidences), 1)
    if mc.use_adaptive_erasure and adaptive_mask_state is not None:
        adaptive_mask_state["ratio"] = adaptive_mask_ratio(
            avg_confidence,
            adaptive_mask_state.get("ratio", mc.mask_ratio),
            adaptation_rate=mc.adaptation_rate,
        )

    def mean_aux(name: str) -> float:
        values = [getattr(item.aux, name) for item in outputs if getattr(item.aux, name) is not None]
        return sum(float(value.detach()) for value in values) / max(len(values), 1)

    return {
        "loss": total_loss / max(accum, 1),
        "parity": mean_aux("parity"),
        "correction_alignment": mean_aux("correction_alignment"),
        "rate_distortion": mean_aux("rate_distortion"),
        "whiteness": mean_aux("whiteness"),
        "contrastive": mean_aux("contrastive"),
        "avg_confidence": avg_confidence,
        "grad_norm": gn,
        "grad_rms": grad_rms,
        "mask_ratio": mask_ratio,
        "lr": lr_at(step, cfg) if hasattr(cfg.train, "warmup_steps") else 0.0,
        "step": step,
        "update_applied": True,
        "all_finite": True,
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
                raise RuntimeError(f"optimizer state is incompatible with strict resume: {exc}") from exc
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

    data_iter = iter(dataloader)
    while step < cfg.train.max_steps:
        if hasattr(dataloader, "set_optimizer_step"):
            dataloader.set_optimizer_step(step)
        microbatches = list(islice(data_iter, cfg.train.grad_accum_steps))
        if len(microbatches) != cfg.train.grad_accum_steps:
            break

        set_lr(optimizer, step, cfg)

        if teacher is not None and getattr(teacher, "_loaded", False) and step == distill_end_step:
            logger.info(f"Step {step}: distillation ended — freeing teacher")
            teacher.free()

        metrics = train_step(
            model,
            microbatches,
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

        if not metrics["update_applied"]:
            continue

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
            save_checkpoint(model, optimizer, cfg, step + 1, ckpt_dir, ckpt_keep, extra=extra)

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
