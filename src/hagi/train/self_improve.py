"""Opt-in online self-improvement for HAGI.

The wrapper composes existing production primitives without changing their
contracts:

1. Generate a deterministic trajectory with :func:`hagi.inference.generate`.
2. Re-score that trajectory with exact CE through ``HAGI.forward``.
3. Apply one adapter-only optimizer step through ``Trainer.train_step``.
4. Re-score the same trajectory and measure ``KL(p_pre || p_post)``.
5. Repeat until a hard iteration, KL, non-finite-loss, or plateau bound.

``generate.py`` remains inference-only. This module is the opt-in entry point;
ordinary generation and ordinary external-data training do not call it. With
``train.adapt.freeze_base=True``, the existing trainer freezes the base and
routes only adapter parameters to AdamW. The base tensors are therefore not
updated by this loop.

The KL guard is deliberately distribution-based rather than a reward model:
the pre-update model is the reference for its own generated trajectory, and a
large ``KL(p_pre || p_post)`` stops feedback before the adapter can collapse
onto a narrow self-generated distribution. Exact full-vocabulary KL is used
for the bounded generated window; no sampled-softmax receiver is allowed.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from hagi.config import Config
from hagi.inference.generate import generate
from hagi.model.model import HAGI
from hagi.train.loop import Trainer

logger = logging.getLogger(__name__)


def _deepcopy_optimizer_state(state_dict: dict) -> dict:
    """Return an independent optimizer-state snapshot.

    ``Optimizer.state_dict()`` contains live tensor objects. A shallow mapping
    therefore keeps aliasing the optimizer as it continues to step; deepcopy is
    required for a rollback snapshot that can actually be restored later.
    """
    return copy.deepcopy(state_dict)


@dataclass
class SelfImproveResult:
    """Metrics for one generate -> score -> update iteration."""

    iteration: int
    generated_ids: list[int]
    pre_ce: float
    post_ce: float
    kl_div: float
    update_applied: bool
    adapter_values_after: list[float] = field(default_factory=list)


@dataclass
class SelfImproveStats:
    """Aggregated result returned by :func:`self_improve`."""

    iterations: list[SelfImproveResult] = field(default_factory=list)
    stopped: str = "max_iterations"
    best_ce: float | None = None
    accepted_updates: int = 0


def _snapshot_adapters(model: HAGI) -> dict[int, torch.Tensor]:
    """Capture trainable adapter param values for transactional rollback.

    Only adapter parameters are trainable when freeze_base=True (the base is
    frozen), so this is a small, cheap snapshot. Non-trainable params are
    skipped. Values are cloned and detached so they are independent of any
    subsequent optimizer step.
    """
    from hagi.model.adapters import BlockAdapter

    snapshot: dict[int, torch.Tensor] = {}
    for module in model.modules():
        if isinstance(module, BlockAdapter):
            for p in module.parameters(recurse=True):
                if p.requires_grad:
                    snapshot[id(p)] = p.detach().clone()
    return snapshot


def _restore_adapters(model: HAGI, snapshot: dict[int, torch.Tensor]) -> None:
    """Roll trainable adapter params back to a snapshot.

    Matches by parameter identity (id) so partial snapshots are safe.
    """
    restored = 0
    for module in model.modules():
        for p in module.parameters(recurse=True):
            stored = snapshot.get(id(p))
            if stored is not None and p.requires_grad:
                p.data.copy_(stored)
                restored += 1
    if restored == 0 and snapshot:
        raise RuntimeError("_restore_adapters: no matching trainable params found")


def _validate_loop(
    cfg: Config,
    *,
    n_new_tokens: int,
    max_iterations: int,
    patience: int,
    ce_min_improve: float,
    kl_max: float,
) -> None:
    """Validate the opt-in contract before allocating an optimizer."""
    if not cfg.model.adapters.enabled:
        raise ValueError("model.adapters.enabled=True is required for self-improvement")
    if not cfg.train.adapt.freeze_base:
        raise ValueError("train.adapt.freeze_base=True is required for adapter-only updates")
    if cfg.model.loop_depth <= 1:
        raise ValueError("model.loop_depth > 1 is required for repeated adapter input")
    if cfg.model.head.sampled_softmax_k != 0:
        raise ValueError("head.sampled_softmax_k must be 0 for exact self-improvement CE")
    if n_new_tokens < 1:
        raise ValueError("n_new_tokens must be >= 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if ce_min_improve < 0 or not math.isfinite(ce_min_improve):
        raise ValueError("ce_min_improve must be finite and >= 0")
    if kl_max < 0 or not math.isfinite(kl_max):
        raise ValueError("kl_max must be finite and >= 0")
    if cfg.train.ce_keep_rate != 1.0:
        raise ValueError("train.ce_keep_rate must be 1.0 for exact generated-trajectory CE")
    if not cfg.model.adapters.pyramid.enabled and not cfg.model.adapters.ttt_lora.enabled:
        raise ValueError("at least one adapter contour must be enabled")
    if cfg.model.adapters.pyramid.enabled and cfg.model.adapters.ttt_lora.enabled:
        raise ValueError(
            "pyramid and ttt_lora adapters are currently mutually exclusive: "
            "enable only one contour for self-improvement"
        )


def _adapter_values(model: HAGI) -> list[float]:
    """Return learnable adapter gates for diagnostics."""
    values: list[float] = []
    for module in model.modules():
        pyramid = getattr(module, "pyramid", None)
        if pyramid is not None and hasattr(pyramid, "scale"):
            values.append(float(pyramid.scale.detach().item()))
        lora = getattr(module, "lora_B", None)
        if lora is not None:
            values.append(float(lora.detach().float().mean().item()))
    return values


def _generate_trajectory(
    model: HAGI,
    prompt_ids: list[int],
    *,
    n_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Generate greedily and remove finished-row padding."""
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    output = generate(
        model,
        prompt,
        max_new_tokens=n_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        repetition_window=128,
        min_new_tokens=n_new_tokens,
        use_cache=True,
    )
    sequence = output.token_ids[0]
    pads = (sequence == pad_token_id).nonzero(as_tuple=True)[0]
    if pads.numel():
        sequence = sequence[: int(pads[0])]
    return sequence


def _score(
    model: HAGI,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Return exact CE and full logits for a generated trajectory."""
    model.eval()
    with torch.no_grad():
        output = model(input_ids, targets, return_logits=True)
    if output.ce is None or output.logits is None:
        raise RuntimeError("HAGI.forward did not return CE and logits")
    return float(output.ce.detach()), output.logits.detach()


def _kl_pre_post(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    *,
    n_generated_positions: int,
) -> float:
    """Compute exact KL(pre-update || post-update) on generated positions.

    Uses the analytically exact softmax-difference identity
    ``KL(p||q) = sum p * (log p - log q)`` on the generated window only, so the
    guard never inspects pre/post on prompt or padding positions. Returns a
    strictly non-negative scalar (tiny negative float32 roundoff is clamped to
    0.0).
    """
    if n_generated_positions < 1:
        raise ValueError("KL requires at least one generated position")
    pre = pre_logits[:, -n_generated_positions:].float()
    post = post_logits[:, -n_generated_positions:].float()
    pre_log_probs = F.log_softmax(pre, dim=-1)
    post_log_probs = F.log_softmax(post, dim=-1)
    pre_probs = pre_log_probs.exp()
    # KL(p_pre || p_post) = sum p_pre * (log p_pre - log p_post) >= 0.
    kl = (pre_probs * (pre_log_probs - post_log_probs)).sum(dim=-1)
    # Clamp only tiny negative float32 noise from rounding; true KL >= 0.
    return max(float(kl.mean()), 0.0)


def self_improve(
    model: HAGI,
    cfg: Config,
    prompt_ids: list[int],
    *,
    n_new_tokens: int = 16,
    max_iterations: int = 8,
    ce_min_improve: float = 0.0,
    patience: int = 3,
    kl_max: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    trainer: Trainer | None = None,
) -> SelfImproveStats:
    """Run bounded generate -> exact-CE -> adapter-update iterations.

    The function is intentionally explicit: callers opt in by invoking it. It
    does not mutate ``cfg`` or the default generation path. The same generated
    trajectory is scored before and after each update, so the CE change and KL
    are directly comparable.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    _validate_loop(
        cfg,
        n_new_tokens=n_new_tokens,
        max_iterations=max_iterations,
        patience=patience,
        ce_min_improve=ce_min_improve,
        kl_max=kl_max,
    )
    if len(prompt_ids) + n_new_tokens > cfg.model.attention.max_seq_len:
        raise ValueError("prompt plus generation budget exceeds attention.max_seq_len")

    eos = cfg.train.data.eos_token_id if eos_token_id is None else eos_token_id
    pad = cfg.train.data.pad_token_id if pad_token_id is None else pad_token_id
    if eos == pad:
        raise ValueError("eos_token_id and pad_token_id must differ")
    if any(t == pad for t in prompt_ids):
        raise ValueError("prompt_ids must not contain the pad token_id")

    if trainer is None:
        trainer = Trainer(model, cfg)
    stats = SelfImproveStats()
    best_ce: float | None = None
    no_improve = 0

    for iteration in range(max_iterations):
        sequence = _generate_trajectory(
            model,
            prompt_ids,
            n_new_tokens=n_new_tokens,
            eos_token_id=eos,
            pad_token_id=pad,
        )
        n_generated = sequence.numel() - len(prompt_ids)
        if n_generated < 1:
            stats.stopped = "empty_trajectory"
            logger.warning("iteration %d: generation produced no new tokens", iteration)
            break

        input_ids = sequence[:-1].unsqueeze(0)
        targets = sequence[1:].unsqueeze(0)
        pre_ce, pre_logits = _score(model, input_ids, targets)

        # Snapshot trainable adapter params AND optimizer state BEFORE the step,
        # so a KL/non-finite guard breach can roll the update back as a precise
        # transaction. The base is frozen and excluded from the optimizer, so
        # only adapter params + optimizer momentum move on a rejected step.
        # Injected lightweight trainers may not expose an optimizer; in that
        # case the adapter snapshot still protects the model state.
        param_snapshot = _snapshot_adapters(model)
        optimizer = getattr(trainer, "optimizer", None)
        opt_snapshot = (
            # Deep-copy: optimizer.state_dict() returns live tensor objects
            # that keep mutating as the optimizer steps. A shallow snapshot
            # therefore restores nothing on rollback.
            _deepcopy_optimizer_state(optimizer.state_dict())
            if optimizer is not None
            else None
        )
        pre_step = getattr(trainer, "step", None)

        model.train()
        metrics = trainer.train_step(
            [
                {
                    "input_ids": input_ids,
                    "targets": targets,
                    "loss_mask": torch.ones_like(targets, dtype=torch.bool),
                }
            ]
        )
        update_applied = bool(metrics.get("update_applied", False))

        # Diagnostics stay FAITHFUL: post_ce/kl_div are the real post-update
        # values regardless of whether the guard accepts or rejects. Keep both
        # post-step states as recovery points if rollback itself fails.
        post_param_snapshot = (
            _snapshot_adapters(model) if optimizer is not None and update_applied else None
        )
        post_opt_snapshot = (
            _deepcopy_optimizer_state(optimizer.state_dict())
            if optimizer is not None and update_applied
            else None
        )
        post_ce, post_logits = _score(model, input_ids, targets)
        kl_div = _kl_pre_post(
            pre_logits,
            post_logits,
            n_generated_positions=n_generated,
        )

        reject = False
        if not (math.isfinite(pre_ce) and math.isfinite(post_ce) and math.isfinite(kl_div)):
            stats.stopped = "nonfinite_ce_or_kl"
            logger.warning("iteration %d: non-finite CE or KL; stopping", iteration)
            reject = True
        elif kl_div > kl_max:
            stats.stopped = "kl_bound"
            logger.warning("iteration %d: KL %.4f exceeded bound %.4f", iteration, kl_div, kl_max)
            reject = True

        if reject:
            # Transactional rollback of the rejected step. A rejected guard
            # step must not count toward completed_steps for the resume
            # horizon. Restore optimizer state before touching model parameters
            # so that a parameter-only load cannot proceed from a corrupted
            # momentum state. Trainer.train_step() increments its counter
            # only after optimizer.step() succeeds.
            if update_applied:
                try:
                    if optimizer is not None and opt_snapshot is not None:
                        optimizer.load_state_dict(opt_snapshot)
                    _restore_adapters(model, param_snapshot)
                except Exception:
                    # Best-effort recovery to the coherent post-step state. The
                    # raised error still aborts before CLI checkpoint saving.
                    try:
                        if optimizer is not None and post_opt_snapshot is not None:
                            optimizer.load_state_dict(post_opt_snapshot)
                        if post_param_snapshot is not None:
                            _restore_adapters(model, post_param_snapshot)
                    except Exception:
                        pass  # leave live post-step state; outer raise still aborts
                    raise
                finally:
                    # The rejected step must never advance the resume horizon,
                    # even when rollback or recovery fails.
                    if pre_step is not None and hasattr(trainer, "step"):
                        trainer.step = pre_step
            update_applied = False
        else:
            if best_ce is None or post_ce < best_ce - ce_min_improve:
                best_ce = post_ce
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    stats.stopped = "ce_plateau"
                    # Count this accepted update before recording the plateau
                    # iteration, so accepted_updates and stats.iterations stay
                    # consistent: the plateau iteration is the last entry.
                    stats.accepted_updates += 1
                    stats.iterations.append(
                        SelfImproveResult(
                            iteration=iteration,
                            generated_ids=sequence[len(prompt_ids) :].tolist(),
                            pre_ce=pre_ce,
                            post_ce=post_ce,
                            kl_div=kl_div,
                            update_applied=update_applied,
                            adapter_values_after=_adapter_values(model),
                        )
                    )
                    logger.info("iteration %d: CE plateau; stopping", iteration)
                    break
            stats.accepted_updates += 1

        stats.iterations.append(
            SelfImproveResult(
                iteration=iteration,
                generated_ids=sequence[len(prompt_ids) :].tolist(),
                pre_ce=pre_ce,
                post_ce=post_ce,
                kl_div=kl_div,
                update_applied=update_applied,
                adapter_values_after=_adapter_values(model),
            )
        )

        # A rejected KL/non-finite guard ends the search: the model is back at
        # the pre-state, and retrying from it would re-enter the same region.
        if reject:
            break

    stats.best_ce = best_ce
    return stats


__all__ = ["SelfImproveResult", "SelfImproveStats", "_restore_adapters", "_snapshot_adapters", "self_improve"]
