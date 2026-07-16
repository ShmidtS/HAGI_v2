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
from torch import nn

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.masking import (
    adaptive_mask_ratio,
    create_physical_corruption_mask,
    create_semantic_corruption,
    progressive_mask_ratio,
)
from hagi_v4.train.losses import LossAggregator, selected_cross_entropy, suffix_cross_entropy
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
    masked_ce_sum = 0.0
    masked_rows = 0
    suffix_ce_sum = 0.0
    suffix_rows = 0
    confidence_sum = 0.0
    top2_mass_sum = 0.0
    entropy_sum = 0.0
    posterior_rows = 0
    suffix_tasks = 0
    total_tasks = 0
    semantic_unknowns = 0
    valid_targets = 0
    physical_corruptions = 0
    physical_positions = 0
    fallback_confidence_sum = 0.0
    fallback_confidence_count = 0
    aux_sums = {name: 0.0 for name in ("parity", "correction_alignment", "rate_distortion", "whiteness", "contrastive")}
    aux_counts = {name: 0 for name in aux_sums}
    all_finite = True

    for micro_idx, batch in enumerate(microbatches):
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        valid_target_mask = batch["valid_target_mask"].to(device)
        semantic_generator = torch.Generator(device=device).manual_seed(step * accum * 2 + micro_idx * 2)
        physical_generator = torch.Generator(device=device).manual_seed(step * accum * 2 + micro_idx * 2 + 1)
        semantic_unknown_mask, prediction_mask, is_suffix = create_semantic_corruption(
            valid_target_mask,
            generator=semantic_generator,
            random_ratio=mask_ratio,
        )
        physical_corruption_mask = create_physical_corruption_mask(
            input_ids,
            mask_ratio,
            generator=physical_generator,
        )
        awgn_sigma = 0.0
        if cfg.train.awgn_enabled:
            awgn_end = int(cfg.train.max_steps * cfg.train.awgn_end_frac)
            if step < awgn_end:
                progress = step / max(awgn_end, 1)
                awgn_sigma = cfg.train.awgn_sigma_start * (1.0 - progress) + cfg.train.awgn_sigma_end * progress
        output = model(
            input_ids,
            targets=targets,
            semantic_unknown_mask=semantic_unknown_mask,
            prediction_mask=prediction_mask,
            valid_target_mask=valid_target_mask,
            physical_corruption_mask=physical_corruption_mask,
            step=step,
            awgn_sigma=awgn_sigma,
        )
        loss = loss_aggregator(output, targets, prediction_mask, step=step)

        use_distill = (
            teacher is not None
            and getattr(teacher, "_loaded", False)
            and cfg.train.distill_enabled is True
            and step < distill_end_step
        )
        if use_distill:
            from hagi_v4.train.distillation import alpha_at

            visibility_mask = valid_target_mask & ~semantic_unknown_mask
            alpha = alpha_at(
                step,
                cfg.train.distill_alpha_start,
                cfg.train.distill_alpha_end,
                cfg.train.max_steps,
                cfg.train.distill_end_frac,
            )
            with torch.inference_mode():
                teacher_result = teacher.get_hidden(input_ids, visibility_mask=visibility_mask)
            if teacher_result is not None:
                loss = teacher.distillation_loss_chunked(
                    student_hidden=output.hidden,
                    teacher_hidden=teacher_result.hidden,
                    reconstruction_loss=loss,
                    visibility_mask=teacher_result.visibility_mask,
                    align_projection=model.distill_align,
                    alpha=alpha,
                )
                del teacher_result

        if not torch.isfinite(loss).all():
            logger.warning(f"Step {step} micro {micro_idx}: non-finite loss — cancelling update")
            all_finite = False
            break

        scaled_loss = loss / accum
        scaled_loss.backward()
        total_loss += loss.item()

        with torch.no_grad():
            suffix_tasks += int(is_suffix.sum().item())
            total_tasks += is_suffix.numel()
            semantic_unknowns += int((semantic_unknown_mask & valid_target_mask).sum().item())
            valid_targets += int(valid_target_mask.sum().item())
            physical_corruptions += int(physical_corruption_mask.sum().item())
            physical_positions += physical_corruption_mask.numel()
            for name in aux_sums:
                value = getattr(output.aux, name)
                if value is not None:
                    aux_sums[name] += float(value.detach())
                    aux_counts[name] += 1

            if output.logits is None or output.prediction_indices is None:
                if output.ce_loss is not None:
                    fallback_confidence_sum += max(0.0, 1.0 - output.ce_loss.detach().item() / 10.0)
                    fallback_confidence_count += 1
            else:
                logits = output.logits.detach()
                prediction_indices = output.prediction_indices.detach().to(targets.device)
                selected_targets = targets.flatten().index_select(0, prediction_indices).to(logits.device)
                suffix_by_position = is_suffix.unsqueeze(1).expand_as(valid_target_mask).flatten()
                suffix_predictions = suffix_by_position.index_select(0, prediction_indices).to(logits.device)
                row_count = logits.shape[0]
                masked_ce_sum += selected_cross_entropy(logits, selected_targets).item() * row_count
                masked_rows += row_count
                suffix_count = int(suffix_predictions.sum().item())
                if suffix_count:
                    suffix_ce_sum += (
                        suffix_cross_entropy(logits, selected_targets, suffix_predictions).item() * suffix_count
                    )
                    suffix_rows += suffix_count

                log_probabilities = logits.float().log_softmax(dim=-1)
                probabilities = log_probabilities.exp()
                confidence_sum += probabilities.max(dim=-1).values.sum().item()
                top2_mass_sum += probabilities.topk(min(2, logits.shape[-1]), dim=-1).values.sum(dim=-1).sum().item()
                entropy_sum += (-(probabilities * log_probabilities).sum(dim=-1)).sum().item()
                posterior_rows += row_count
                del logits, prediction_indices, selected_targets, suffix_predictions, log_probabilities, probabilities
        del output, loss, scaled_loss

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    all_finite = all_finite and all(torch.isfinite(grad).all().item() for grad in grads)
    if not all_finite:
        optimizer.zero_grad(set_to_none=True)
        for param in model.parameters():
            param.grad = None
        raise FloatingPointError(f"training update {step} has non-finite loss or gradients; gradients cleared")

    gn, grad_rms = gradient_stats(model, cfg.train)
    optimizer.step()

    if total_tasks == 0:
        return {"loss": 0.0, "step": step, "grad_norm": gn, "grad_rms": grad_rms}

    masked_ce = masked_ce_sum / masked_rows if masked_rows else float("nan")
    suffix_ce = suffix_ce_sum / suffix_rows if suffix_rows else float("nan")
    if posterior_rows:
        avg_confidence = confidence_sum / posterior_rows
        top2_mass = top2_mass_sum / posterior_rows
        posterior_entropy = entropy_sum / posterior_rows
    else:
        avg_confidence = fallback_confidence_sum / fallback_confidence_count if fallback_confidence_count else 0.5
        top2_mass = float("nan")
        posterior_entropy = float("nan")
    if mc.use_adaptive_erasure and adaptive_mask_state is not None:
        adaptive_mask_state["ratio"] = adaptive_mask_ratio(
            avg_confidence,
            adaptive_mask_state.get("ratio", mc.mask_ratio),
            adaptation_rate=mc.adaptation_rate,
        )

    def mean_aux(name: str) -> float:
        return aux_sums[name] / max(aux_counts[name], 1)

    return {
        "loss": total_loss / max(accum, 1),
        "objective_loss": total_loss / max(accum, 1),
        "masked_ce": masked_ce,
        "suffix_ce": suffix_ce,
        "bpt": masked_ce / math.log(2.0),
        "top2_mass": top2_mass,
        "posterior_entropy": posterior_entropy,
        "suffix_task_ratio": suffix_tasks / max(total_tasks, 1),
        "random_task_ratio": (total_tasks - suffix_tasks) / max(total_tasks, 1),
        "semantic_mask_ratio": semantic_unknowns / max(valid_targets, 1),
        "physical_corruption_ratio": physical_corruptions / max(physical_positions, 1),
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
) -> Iterator[dict]:
    from hagi_v4.train.checkpoint import assert_fresh_checkpoint_root, save_checkpoint

    configure_runtime()
    assert_fresh_checkpoint_root(cfg.train.checkpoint_dir)

    if torch.cuda.is_available() and cfg.train.precision == "bf16":
        cast_to_bf16(model)

    optimizer = build_optimizer(model, cfg)
    loss_aggregator = LossAggregator(cfg)
    adaptive_mask_state = {"ratio": cfg.model.masking.mask_ratio} if cfg.model.masking.use_adaptive_erasure else None

    step = 0
    distill_end_step = int(cfg.train.max_steps * cfg.train.distill_end_frac)
    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_interval = cfg.train.checkpoint_interval
    ckpt_keep = cfg.train.checkpoint_keep_last
    last_checkpoint_step = None

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

        completed_updates = step + 1
        if ckpt_interval > 0 and completed_updates % ckpt_interval == 0:
            save_checkpoint(model, cfg, completed_updates, ckpt_dir, ckpt_keep)
            last_checkpoint_step = completed_updates

        step += 1

    if ckpt_interval > 0 and last_checkpoint_step != step:
        save_checkpoint(model, cfg, step, ckpt_dir, ckpt_keep)
