"""Training loop for the ternary RD-channel LM.

The model is a CAUSAL generative LM: every batch trains next-token prediction
with a causal (or causal-dominant) attention mask, exactly matching the
inference path. There is no bidir-MLM/suffix/parity/AWGN machinery — those
belonged to the abandoned self-inflicted-channel design and broke the
train/infer alignment.

Attention-mode curriculum: causal is dominant from step 0 (the inference
regime); a small soft_causal/bidir slice adds a denser representation
gradient early and is ramped out by mid-training. This avoids the
out-of-distribution-at-generation failure of pure-bidir warmup.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from hagi_v4.config import Config
from hagi_v4.train.losses import LossAggregator
from hagi_v4.train.optim import CombinedOptimizer, build_optimizer

logger = logging.getLogger(__name__)


def configure_runtime() -> None:
    import os

    # Bounded caching allocator: varying prediction sizes fragment Windows
    # VRAM without this; empty_cache() every step keeps reserved bounded.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.6")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _maybe_release_caches(step: int, interval: int = 1) -> None:
    if interval <= 0:
        return
    if step > 0 and step % interval == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup from 0, then stable, then cosine decay to 10% floor."""
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
    min_lr_ratio = 0.1
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def muon_lr_at(step: int, cfg: Config) -> float:
    """Muon LR schedule (same shape as lr_at, scaled to muon_lr)."""
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
    min_lr_ratio = 0.1
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def set_lr(optimizer: CombinedOptimizer, step: int, cfg: Config) -> None:
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


def _sample_attention_mode(step: int, cfg: Config) -> tuple[str, float | None]:
    """Causal-dominant attention-mode curriculum.

    Returns (attention_mode, soft_beta). Causal is the inference regime and is
    dominant from step 0; soft_causal/bidir add a denser representation
    gradient early and are ramped out by mid-training.
    """
    warmup = max(int(cfg.train.warmup_steps), 1)
    stable_end = int(cfg.train.max_steps * 0.8)
    progress = min(1.0, step / warmup)
    late = min(1.0, step / max(stable_end, 1))
    r = float(torch.rand(1).item())
    causal_prob = 0.50 + 0.10 * late
    soft_prob = 0.30 + 0.10 * late
    if r < causal_prob:
        return "causal", None
    if r < causal_prob + soft_prob:
        return "soft_causal", 0.5 + progress * 2.5
    return "bidir", None


def _causal_next_token_loss(output, input_ids, targets, valid_target_mask):
    """Next-token shifted CE from flat logits/indices for causal-style modes.

    For causal/prefix/soft_causal the model selects ALL valid positions and
    returns logits as [n_valid, V]; we shift targets by one to form the
    next-token objective that matches inference.
    """
    T = input_ids.size(1)
    idx = output.prediction_indices.to(output.logits.device)
    t = idx % T
    keep = t < (T - 1)
    if not keep.any():
        return output.logits.new_zeros(())
    idx_keep = idx[keep]
    target_idx = idx_keep + 1
    shift_logits = output.logits[keep]
    shift_targets = targets.view(-1).index_select(0, target_idx.to(targets.device)).to(shift_logits.device)
    target_valid = (
        valid_target_mask.view(-1)
        .index_select(0, target_idx.to(valid_target_mask.device))
        .to(shift_logits.device)
    )
    if target_valid.any():
        return F.cross_entropy(shift_logits[target_valid], shift_targets[target_valid])
    return output.logits.new_zeros(())


def train_step(
    model: nn.Module,
    microbatches: list[dict],
    optimizer: CombinedOptimizer,
    cfg: Config,
    step: int,
    teacher=None,
    loss_aggregator: LossAggregator | None = None,
    distill_end_step: int = 0,
) -> dict:
    """Single training step with gradient accumulation (causal next-token)."""
    model.train()
    device = next(model.parameters()).device
    if loss_aggregator is None:
        loss_aggregator = LossAggregator(cfg)

    accum = cfg.train.grad_accum_steps
    if len(microbatches) != accum:
        raise ValueError(f"expected {accum} microbatches, got {len(microbatches)}")
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    masked_ce_sum = 0.0
    masked_rows = 0
    confidence_sum = 0.0
    conf_rows = 0
    top2_mass_sum = 0.0
    entropy_sum = 0.0
    posterior_rows = 0
    aux_sums = {name: 0.0 for name in ("rate", "distortion", "perception", "ternary_bias", "moe_lb", "attn_entropy")}
    aux_counts = {name: 0 for name in aux_sums}
    all_finite = True

    for micro_idx, batch in enumerate(microbatches):
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        valid_target_mask = batch["valid_target_mask"].to(device)
        images = batch.get("images")
        if images is not None:
            images = images.to(device)
        spectrograms = batch.get("spectrograms")
        if spectrograms is not None:
            spectrograms = spectrograms.to(device)

        attention_mode, soft_beta = _sample_attention_mode(step, cfg)
        # Causal/prefix/soft_causal: predict at ALL valid positions, nothing
        # erased (matches inference). bidir: would use masked prediction, but
        # we keep the causal next-token objective for all modes so the model
        # always trains the generation signal.
        out_unknown_mask = torch.zeros_like(valid_target_mask)
        out_prediction_mask = valid_target_mask

        output = model(
            input_ids,
            targets=None,
            semantic_unknown_mask=out_unknown_mask,
            prediction_mask=out_prediction_mask,
            valid_target_mask=valid_target_mask,
            images=images,
            spectrograms=spectrograms,
            attention_mode=attention_mode,
            soft_beta=soft_beta,
        )
        output.ce_loss = _causal_next_token_loss(output, input_ids, targets, valid_target_mask)
        loss = loss_aggregator(output, step=step)

        use_distill = (
            teacher is not None
            and getattr(teacher, "_loaded", False)
            and cfg.train.distill.enabled is True
            and step < distill_end_step
            and getattr(model, "distill_align", None) is not None
        )
        if use_distill:
            from hagi_v4.train.distillation import alpha_at

            visibility_mask = valid_target_mask
            alpha = alpha_at(
                step,
                cfg.train.distill.alpha_start,
                cfg.train.distill.alpha_end,
                cfg.train.max_steps,
                cfg.train.distill.end_frac,
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
            for name in aux_sums:
                value = getattr(output.aux, name)
                if value is not None:
                    aux_sums[name] += value.detach().item()
                    aux_counts[name] += 1

            if output.logits is not None and output.prediction_indices is not None:
                logits = output.logits.detach()
                prediction_indices = output.prediction_indices.detach().to(targets.device)
                selected_targets = targets.flatten().index_select(0, prediction_indices).to(logits.device)
                row_count = logits.shape[0]
                if output.ce_loss is not None:
                    masked_ce_sum += output.ce_loss.detach().item() * row_count
                masked_rows += row_count
                with torch.no_grad():
                    want_post = (step % 20 == 0)
                    chunk_rows = 256
                    for ci in range(0, logits.shape[0], chunk_rows):
                        chunk = logits[ci : ci + chunk_rows].float()
                        log_p = chunk.log_softmax(dim=-1)
                        p = log_p.exp()
                        confidence_sum += p.max(dim=-1).values.sum().item()
                        if want_post:
                            top2_mass_sum += p.topk(min(2, chunk.shape[-1]), dim=-1).values.sum(dim=-1).sum().item()
                            entropy_sum += -(p * log_p).sum(dim=-1).sum().item()
                conf_rows += row_count
                if want_post:
                    posterior_rows += row_count
                del logits, prediction_indices, selected_targets

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

    if masked_rows == 0:
        return {"loss": 0.0, "step": step, "grad_norm": gn, "grad_rms": grad_rms}

    masked_ce = masked_ce_sum / masked_rows
    avg_confidence = confidence_sum / conf_rows if conf_rows else float("nan")
    if posterior_rows:
        top2_mass = top2_mass_sum / posterior_rows
        posterior_entropy = entropy_sum / posterior_rows
    else:
        top2_mass = min(1.0, max(0.0, avg_confidence * 1.5)) if conf_rows else float("nan")
        posterior_entropy = float("nan") if not conf_rows else max(0.0, -math.log(max(avg_confidence, 1e-8)))

    def mean_aux(name: str) -> float:
        return aux_sums[name] / max(aux_counts[name], 1)

    return {
        "loss": total_loss / max(accum, 1),
        "objective_loss": total_loss / max(accum, 1),
        "masked_ce": masked_ce,
        "bpt": masked_ce / math.log(2.0),
        "top2_mass": top2_mass,
        "posterior_entropy": posterior_entropy,
        "rate": mean_aux("rate"),
        "distortion": mean_aux("distortion"),
        "perception": mean_aux("perception"),
        "ternary_bias": mean_aux("ternary_bias"),
        "moe_lb": mean_aux("moe_lb"),
        "attn_entropy": mean_aux("attn_entropy"),
        "avg_confidence": avg_confidence,
        "grad_norm": gn,
        "grad_rms": grad_rms,
        "lr": lr_at(step, cfg),
        "step": step,
        "update_applied": True,
        "all_finite": True,
    }


def train(
    model: nn.Module,
    dataloader,
    cfg: Config,
    log_interval: int = 100,
    teacher=None,
    start_step: int = 0,
    optimizer_state: dict | None = None,
) -> Iterator[dict]:
    from hagi_v4.train.checkpoint import assert_fresh_checkpoint_root, save_checkpoint

    configure_runtime()
    assert_fresh_checkpoint_root(cfg.train.checkpoint_dir)

    if torch.cuda.is_available() and cfg.train.precision == "bf16":
        cast_to_bf16(model)

    optimizer = build_optimizer(model, cfg)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    loss_aggregator = LossAggregator(cfg)

    step = start_step
    distill_end_step = int(cfg.train.max_steps * cfg.train.distill.end_frac)
    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_interval = cfg.train.checkpoint_interval
    ckpt_keep = cfg.train.checkpoint_keep_last
    last_checkpoint_step = step if step > 0 else None

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
            model, microbatches, optimizer, cfg, step,
            teacher, loss_aggregator=loss_aggregator, distill_end_step=distill_end_step,
        )

        _maybe_release_caches(step, interval=1)

        if step % log_interval == 0:
            yield metrics

        if not metrics["update_applied"]:
            continue

        completed_updates = step + 1
        if ckpt_interval > 0 and completed_updates % ckpt_interval == 0:
            save_checkpoint(model, cfg, completed_updates, ckpt_dir, ckpt_keep, optimizer=optimizer)
            last_checkpoint_step = completed_updates

        step += 1

    if ckpt_interval > 0 and last_checkpoint_step != step:
        save_checkpoint(model, cfg, step, ckpt_dir, ckpt_keep, optimizer=optimizer)
