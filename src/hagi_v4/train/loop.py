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
    create_physical_corruption_mask,
    create_semantic_corruption,
    progressive_mask_ratio,
)
from hagi_v4.train.losses import LossAggregator, selected_cross_entropy, suffix_cross_entropy
from hagi_v4.train.optim import CombinedOptimizer, build_optimizer

logger = logging.getLogger(__name__)


def _suffix_probability(step: int, cfg: HAGIv4Config) -> float:
    """V18 suffix curriculum — conservative (language-first, generate-late).

    Early (step < 2*warmup): base 0.30 (V15 language-win regime).
    Mid (2*warmup .. 80%*max_steps): stay at base — focus on language.
    Late (>= 80%*max_steps): ramp to 0.85 to align with LLaDA generation.

    V17 error: ramped to 0.7 by step 10k which killed suffix_ce early.
    V18 keeps suffix low while language is being learned, ramps only at the
    end to match the inference distribution.
    """
    base = float(getattr(cfg.train, "suffix_probability", 0.30))
    early_end = 2 * max(int(getattr(cfg.train, "warmup_steps", 1000)), 1)
    late_start = int(cfg.train.max_steps * 0.80)
    if step < early_end:
        return base
    if step < late_start:
        return base
    # Phase 3: linear ramp base -> 0.85 over the final 20% of training.
    span = max(cfg.train.max_steps - late_start, 1)
    t = (step - late_start) / span
    return base + t * (0.85 - base)


def configure_runtime() -> None:
    import os

    # V12: ``garbage_collection_threshold`` triggers CUDA GC when the
    # caching allocator's reserved memory exceeds this fraction of total
    # VRAM. Without it, fragmentation from varying-size prediction masks
    # (n_predicted positions change every step) causes reserved memory to
    # grow unboundedly. Combined with ``expandable_segments:True`` this
    # keeps reserved memory bounded.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.6")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _maybe_release_caches(step: int, interval: int = 1) -> None:
    """V12: release PyTorch's caching allocator reserved-but-unused memory.

    On Windows ``expandable_segments`` is unsupported, so the caching
    allocator cannot grow segments dynamically — it must reserve whole
    blocks. When prediction mask sizes vary step-to-step (different
    n_predicted positions), the allocator fragments: freed blocks from
    a large step cannot satisfy a small step's request, so it reserves
    MORE memory. Observed: reserved grew 0.16 → 9.3 GB over 100 steps
    without this call; with it, reserved stays at ~1 GB.

    ``empty_cache()`` is ~1 ms on RTX 3070, negligible vs the ~1.3 s
    per-step training cost.
    """
    if interval <= 0:
        return
    if step > 0 and step % interval == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()


def lr_at(step: int, cfg: HAGIv4Config) -> float:
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
    # V20: cosine decay to 10% floor (not 0) so late-stage training
    # still makes progress. V19 decayed to 0, causing the late collapse
    # (mce 3.0 -> 5.7 after step 1000).
    min_lr_ratio = 0.1
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def muon_lr_at(step: int, cfg: HAGIv4Config) -> float:
    """Linear warmup from 0, then stable, then cosine decay to 10% floor for Muon."""
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
    conf_rows = 0
    top2_mass_sum = 0.0
    entropy_sum = 0.0
    posterior_rows = 0
    suffix_tasks = 0
    total_tasks = 0
    semantic_unknowns = 0
    valid_targets = 0
    physical_corruptions = 0
    physical_positions = 0
    aux_sums = {
        name: 0.0
        for name in (
            "parity",
            "correction_alignment",
            "rate_distortion",
            "whiteness",
            "contrastive",
            "parity_diversity",
        )
    }
    aux_counts = {name: 0 for name in aux_sums}
    all_finite = True

    for micro_idx, batch in enumerate(microbatches):
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        valid_target_mask = batch["valid_target_mask"].to(device)
        # V12: optional multimodal inputs — passed through when the batch
        # provides them, ``None`` otherwise. The model's ``forward`` already
        # gates the multimodal path on ``images is not None or spectrograms
        # is not None`` AND ``self.multimodal_enabled``, so a text-only batch
        # is handled correctly even when multimodal is enabled in config.
        images = batch.get("images")
        if images is not None:
            images = images.to(device)
        spectrograms = batch.get("spectrograms")
        if spectrograms is not None:
            spectrograms = spectrograms.to(device)
        semantic_generator = torch.Generator(device=device).manual_seed(step * accum * 2 + micro_idx * 2)
        physical_generator = torch.Generator(device=device).manual_seed(step * accum * 2 + micro_idx * 2 + 1)
        # V17: suffix curriculum — low early (language), ramp late (generate align).
        suffix_prob = _suffix_probability(step, cfg)
        semantic_unknown_mask, prediction_mask, is_suffix = create_semantic_corruption(
            valid_target_mask,
            generator=semantic_generator,
            random_ratio=mask_ratio,
            random_probability=1.0 - suffix_prob,
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
        # V20: UniLM-style mixed attention per batch. Fixed probabilities
        # (40% bidir, 30% prefix-LM, 30% causal) ensure all three modes
        # are exercised every step regardless of max_steps. This avoids
        # the V19 failure where suffix_prob curriculum (tied to max_steps)
        # never ramped up in short smoke tests, leaving the model unable
        # to generate causally.
        r = float(torch.rand(1).item())
        if r < 0.40:
            attention_mode = "bidir"
            prefix_len = None
        elif r < 0.70:
            attention_mode = "prefix"
            # Use first 25% of the sequence as the bidirectional prefix
            # (model context), the rest as the causal generation target.
            seq_len = input_ids.size(1)
            prefix_len = max(seq_len // 4, 1)
        else:
            attention_mode = "causal"
            prefix_len = None
        if attention_mode == "bidir":
            # MLM: use the standard masked prediction path.
            out_prediction_mask = prediction_mask
            out_unknown_mask = semantic_unknown_mask
        else:
            # Causal/prefix AR: predict at ALL valid positions (no input
            # masking). The model returns logits as [n_valid, V]; we
            # reshape to [B, T, V] for next-token shifted CE.
            out_prediction_mask = valid_target_mask
            out_unknown_mask = torch.zeros_like(valid_target_mask)
        output = model(
            input_ids,
            targets=targets if attention_mode == "bidir" else None,
            semantic_unknown_mask=out_unknown_mask,
            prediction_mask=out_prediction_mask,
            valid_target_mask=valid_target_mask,
            physical_corruption_mask=physical_corruption_mask,
            step=step,
            awgn_sigma=awgn_sigma,
            images=images,
            spectrograms=spectrograms,
            attention_mode=attention_mode,
            prefix_len=prefix_len,
        )
        # V20: For causal/prefix modes, compute next-token CE (shifted
        # targets) directly from flat logits and indices — no [B,T,V]
        # expansion needed. For bidir, keep the masked-CE path.
        if attention_mode in ("causal", "prefix"):
            T = input_ids.size(1)
            idx = output.prediction_indices.to(output.logits.device)  # [n_sel], flat b*T+t
            t = idx % T
            # Only positions with t < T-1 have a next token to predict.
            mask = t < (T - 1)
            if attention_mode == "prefix" and prefix_len is not None:
                mask = mask & (t >= prefix_len)
            if mask.any():
                idx_keep = idx[mask]
                target_idx = idx_keep + 1  # next position
                shift_logits = output.logits[mask]
                shift_targets = targets.view(-1).index_select(0, target_idx.to(targets.device)).to(shift_logits.device)
                target_valid = (
                    valid_target_mask.view(-1)
                    .index_select(0, target_idx.to(valid_target_mask.device))
                    .to(shift_logits.device)
                )
                if target_valid.any():
                    output.ce_loss = F.cross_entropy(
                        shift_logits[target_valid],
                        shift_targets[target_valid],
                    )
                else:
                    output.ce_loss = output.logits.new_zeros(())
            else:
                output.ce_loss = output.logits.new_zeros(())
            loss = loss_aggregator(output, targets, None, step=step)
        else:
            loss = loss_aggregator(output, targets, prediction_mask, step=step)

        use_distill = (
            teacher is not None
            and getattr(teacher, "_loaded", False)
            and cfg.train.distill_enabled is True
            and step < distill_end_step
            and getattr(model, "distill_align", None) is not None
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
                    # ``.item()`` fully extracts the scalar and releases the
                    # tensor reference immediately, preventing the aux tensors
                    # (which carry autograd state) from accumulating across
                    # microbatches.
                    aux_sums[name] += value.detach().item()
                    aux_counts[name] += 1

            if output.logits is not None and output.prediction_indices is not None:
                # Detach first to avoid holding the autograd graph across
                # microbatches. The logits tensor can be large ([N, V]) and
                # keeping it attached until the end of the loop was a major
                # contributor to the observed ~10 GB VRAM growth.
                logits = output.logits.detach()
                prediction_indices = output.prediction_indices.detach().to(targets.device)
                selected_targets = targets.flatten().index_select(0, prediction_indices).to(logits.device)
                suffix_by_position = is_suffix.unsqueeze(1).expand_as(valid_target_mask).flatten()
                suffix_predictions = suffix_by_position.index_select(0, prediction_indices).to(logits.device)
                row_count = logits.shape[0]
                # V14: reuse forward CE when available (avoid recompute every microbatch).
                if output.ce_loss is not None:
                    masked_ce_sum += output.ce_loss.detach().item() * row_count
                else:
                    masked_ce_sum += selected_cross_entropy(logits, selected_targets).item() * row_count
                masked_rows += row_count
                # V16: conf from actual max-prob every step (never fake 0.5).
                # V17: top2/entropy every 20 steps for cost; skip log NaN when empty.
                with torch.no_grad():
                    log_probabilities = logits.float().log_softmax(dim=-1)
                    probabilities = log_probabilities.exp()
                    confidence_sum += probabilities.max(dim=-1).values.sum().item()
                    conf_rows += row_count
                    if step % 20 == 0:
                        top2_mass_sum += (
                            probabilities.topk(min(2, logits.shape[-1]), dim=-1).values.sum(dim=-1).sum().item()
                        )
                        entropy_sum += (-(probabilities * log_probabilities).sum(dim=-1)).sum().item()
                        posterior_rows += row_count
                    del log_probabilities, probabilities
                suffix_count = int(suffix_predictions.sum().item())
                if suffix_count:
                    suffix_ce_sum += (
                        suffix_cross_entropy(logits, selected_targets, suffix_predictions).item() * suffix_count
                    )
                    suffix_rows += suffix_count
                del logits, prediction_indices, selected_targets, suffix_predictions

            # Free decoder side_info tensors that hold autograd graph refs.
            # ``side_info`` accumulates ``extrinsic_norms`` (a list of tensors
            # with graph) and ``parity_residual`` every forward pass; without
            # explicit cleanup they persist until the next forward, which
            # compounds across microbatches.
            side_info = getattr(output, "side_info", None)
            if side_info is not None and isinstance(side_info, dict):
                side_info.pop("extrinsic_norms", None)
                side_info.pop("parity_residual", None)
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
    # suffix_ce is nan only when truly no suffix rows this step (valid signal).
    suffix_ce = suffix_ce_sum / suffix_rows if suffix_rows else float("nan")
    avg_confidence = confidence_sum / conf_rows if conf_rows else float("nan")
    if posterior_rows:
        top2_mass = top2_mass_sum / posterior_rows
        posterior_entropy = entropy_sum / posterior_rows
    else:
        # Off-gate steps: finite placeholders from conf when available.
        top2_mass = min(1.0, max(0.0, avg_confidence * 1.5)) if conf_rows else float("nan")
        posterior_entropy = float("nan") if not conf_rows else max(0.0, -math.log(max(avg_confidence, 1e-8)))
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
        "parity_diversity": mean_aux("parity_diversity"),
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

        # V12: release caching allocator's reserved memory every step.
        # On Windows the allocator fragments when prediction mask sizes
        # vary, causing reserved memory to grow to ~10 GB. ``empty_cache``
        # is ~1 ms, negligible vs ~1.3 s per-step cost.
        _maybe_release_caches(step, interval=20)

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
