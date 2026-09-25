"""Opt-in online self-improvement for HAGI.

The wrapper composes existing production primitives without changing their
contracts:

1. Generate a deterministic trajectory with :func:`hagi.inference.generate`.
2. Re-score that trajectory with exact CE through ``HAGI.forward``.
3. Apply one adapter-only update — an optimizer step through
   ``Trainer.train_step`` (``mode="gradient"``) or a closed-form anchored-RLS
   fit of ``lora_B`` from hidden-state features through :meth:`TttRls.step`
   (``mode="rls"``).
4. Re-score the same trajectory and measure ``KL(p_pre || p_post)``.
5. Repeat until a hard iteration, KL, non-finite-loss, or plateau bound.

``mode="rls"`` generates **once** and reuses that trajectory for every
iteration, so the autoregressive cost is amortized over the whole loop instead
of paid per update; it also allocates no optimizer at all. Both modes share the
guards, the transactional rollback, and the frozen-base contract.

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
import weakref
from dataclasses import dataclass, field
from typing import Literal, Protocol

import torch
import torch.nn.functional as F

from hagi.config import Config
from hagi.inference.generate import generate
from hagi.model.model import HAGI
from hagi.train.loop import Trainer
from hagi.train.ttt import TttRls, _restore_training_mode, _snapshot_training_modes

logger = logging.getLogger(__name__)

type _AdapterSnapshot = dict[int, tuple[torch.Tensor, bool]]
_POISONED_TRAINERS: weakref.WeakValueDictionary[int, object] = (
    weakref.WeakValueDictionary()
)


class _StatefulOptimizer(Protocol):
    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> object: ...


class _RollbackPoisonedTrainer(RuntimeError):
    """Raised when a trainer cannot be returned to a coherent transaction state."""


def _deepcopy_optimizer_state(state_dict: dict) -> dict:
    """Return an independent optimizer-state snapshot.

    ``Optimizer.state_dict()`` contains live tensor objects. A shallow mapping
    therefore keeps aliasing the optimizer as it continues to step; deepcopy is
    required for a rollback snapshot that can actually be restored later.
    """
    return copy.deepcopy(state_dict)


def _report_rollback_failure(message: str) -> None:
    """Log a rollback failure without ever raising.

    A logging handler can fail (custom handler, closed stream, test double), and
    this call sits on the failure path of an already-failing transaction. If it
    raised, the handler's exception would replace the error that actually caused
    the rollback, so the caller would debug the wrong failure. Diagnostics are
    best effort; the original exception must survive.
    """
    try:
        logger.exception(message)
    except BaseException:  # pragma: no cover - defensive
        pass


def _restore_rls_state_safely(ttt: TttRls, snapshot: tuple, context: str) -> None:
    """Best-effort RLS restore used only while another exception is primary.

    The fitter latches its own poison flag before a failed restore. Retrying the
    same invalid snapshot cannot make it usable and may raise a second error that
    hides the diagnostic or transaction exception already being propagated.
    """
    try:
        ttt.restore_state(snapshot)
    except BaseException:
        _report_rollback_failure(context)


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
    delta_rms_frac: float = 0.0
    """Applied delta RMS as a fraction of the residual stream (``rls`` mode only)."""


@dataclass
class SelfImproveStats:
    """Aggregated result returned by :func:`self_improve`."""

    iterations: list[SelfImproveResult] = field(default_factory=list)
    stopped: str = "max_iterations"
    best_ce: float | None = None
    accepted_updates: int = 0


def _snapshot_adapters(model: HAGI) -> _AdapterSnapshot:
    """Capture complete adaptive-parameter transaction state.

    Values are cloned by parameter identity. ``requires_grad`` is captured for
    every adaptive parameter as transaction state rather than used to filter the
    snapshot: a trainer that clears the flag after mutating a value must still
    be rolled back to the exact trainable/frozen partition the caller entered.
    """
    from hagi.model.adaptive import adaptive_parameters

    return {
        id(parameter): (parameter.detach().clone(), parameter.requires_grad)
        for parameter in adaptive_parameters(model)
    }


def _restore_adapters(model: HAGI, snapshot: _AdapterSnapshot) -> None:
    """Restore a complete adaptive-parameter snapshot by identity.

    The complete parameter set and every saved tensor's shape, dtype and device
    are validated before the first live write. Restore then applies both the
    captured value and ``requires_grad`` flag without consulting the parameter's
    current flag, so a trainer that cleared the flag mid-transaction cannot leave
    a mutated adapter permanently excluded from future updates.
    """
    from hagi.model.adaptive import adaptive_parameters

    current = tuple(adaptive_parameters(model))
    if set(snapshot) != {id(parameter) for parameter in current}:
        raise RuntimeError("_restore_adapters: adaptive parameter snapshot mismatch")
    for parameter in current:
        saved = snapshot[id(parameter)]
        if (
            not isinstance(saved, tuple)
            or len(saved) != 2
            or not isinstance(saved[0], torch.Tensor)
            or type(saved[1]) is not bool
        ):
            raise RuntimeError("_restore_adapters: invalid adaptive parameter state")
        value = saved[0]
        if (
            value.shape != parameter.shape
            or value.dtype != parameter.dtype
            or value.device != parameter.device
        ):
            raise RuntimeError("_restore_adapters: adaptive parameter geometry mismatch")
    with torch.no_grad():
        for parameter in current:
            value, requires_grad = snapshot[id(parameter)]
            parameter.data.copy_(value)
            parameter.requires_grad_(requires_grad)


def _restore_gradient_state(
    model: HAGI,
    optimizer: _StatefulOptimizer | None,
    param_snapshot: _AdapterSnapshot | None,
    opt_snapshot: dict | None,
    *,
    target: str,
) -> bool:
    """Restore one complete parameter/optimizer state pair.

    A present optimizer owns an optimizer-state half even when that half is
    temporarily absent from the live object. Treating a missing snapshot as a
    successful no-op would let the other half restore a value from a different
    transaction. Parameter and optimizer restore are attempted independently so
    one failure cannot skip the other, but the caller receives success only when
    the complete target state is present and both restores return normally.
    """
    if param_snapshot is None or (optimizer is not None and opt_snapshot is None):
        _report_rollback_failure(
            f"gradient {target}-state snapshot is incomplete"
        )
        return False

    restored = True
    if optimizer is not None:
        try:
            optimizer.load_state_dict(opt_snapshot)
        except BaseException:
            restored = False
            _report_rollback_failure(
                f"gradient {target}-state optimizer restore failed"
            )
    try:
        _restore_adapters(model, param_snapshot)
    except BaseException:
        restored = False
        _report_rollback_failure(
            f"gradient {target}-state adapter restore failed"
        )
    return restored


def _rollback_gradient_transaction(
    model: HAGI,
    trainer: object,
    optimizer: _StatefulOptimizer | None,
    param_snapshot: _AdapterSnapshot | None,
    opt_snapshot: dict | None,
    post_param_snapshot: _AdapterSnapshot | None,
    post_opt_snapshot: dict | None,
    *,
    pre_step: int | None,
    post_step: int | None,
) -> Literal["pre", "post", "poisoned"]:
    """Restore a complete state and align the resume horizon with it.

    ``pre`` is the only safe rollback result and leaves the trainer reusable. If
    either pre-state half cannot be restored, restore both halves of the captured
    post-state instead. That recovery is quarantined: it is paired with
    ``post_step`` and the trainer is poisoned. A missing or partial fallback, an
    unknown horizon, or a failed horizon/poison write returns ``poisoned``
    explicitly, because a post recovery nobody can invalidate is not a
    successful rollback.
    """
    if _restore_gradient_state(
        model,
        optimizer,
        param_snapshot,
        opt_snapshot,
        target="pre",
    ):
        if pre_step is None:
            _report_rollback_failure(
                "gradient pre-state step is unknown; trainer poisoned"
            )
            _poison_gradient_trainer(trainer)
            return "poisoned"
        try:
            setattr(trainer, "step", pre_step)
        except BaseException:
            _report_rollback_failure("gradient pre-state step restore failed")
            _poison_gradient_trainer(trainer)
            return "poisoned"
        return "pre"

    if _restore_gradient_state(
        model,
        optimizer,
        post_param_snapshot,
        post_opt_snapshot,
        target="post",
    ):
        if post_step is None:
            _report_rollback_failure(
                "gradient post-state step is unknown; trainer poisoned"
            )
            _poison_gradient_trainer(trainer)
            return "poisoned"
        try:
            setattr(trainer, "step", post_step)
        except BaseException:
            _report_rollback_failure(
                "gradient post-state step restore failed; trainer poisoned"
            )
            _poison_gradient_trainer(trainer)
            return "poisoned"
        return "post" if _poison_gradient_trainer(trainer) else "poisoned"

    _report_rollback_failure(
        "gradient post-state recovery failed; trainer poisoned"
    )
    _poison_gradient_trainer(trainer)
    return "poisoned"


def _is_gradient_trainer_poisoned(trainer: object) -> bool:
    """Return whether reuse of ``trainer`` must fail closed.

    The instance attribute is the normal marker. The weak external set is the
    fallback for injected trainers that reject attribute assignment, and an
    unreadable trainer is treated as unprovable and therefore quarantined.
    """
    try:
        if getattr(trainer, "_hagi_self_improve_rollback_poisoned", False) is True:
            return True
    except BaseException:
        return True
    try:
        return _POISONED_TRAINERS.get(id(trainer)) is trainer
    except BaseException:
        return True


def _poison_gradient_trainer(trainer: object) -> bool:
    """Mark a trainer discard-only and return whether quarantine is verifiable.

    Returning ``False`` means no durable marker could be written, so the caller
    must not report a successful ``post`` recovery: the entry gate would read the
    missing attribute as ``False`` and allow reuse of an unverifiable state.
    """
    try:
        setattr(trainer, "_hagi_self_improve_rollback_poisoned", True)
    except BaseException:
        pass
    try:
        # Keying by object id avoids invoking injected trainer ``__hash__`` or
        # equality. Weak values preserve the public trainer's lifetime instead
        # of keeping a discarded trainer alive through quarantine bookkeeping.
        _POISONED_TRAINERS[id(trainer)] = trainer
    except TypeError:
        # Non-weak-referenceable: the instance attribute above is then the only
        # available proof of quarantine.
        pass
    except BaseException:
        pass
    confirmed = _is_gradient_trainer_poisoned(trainer)
    if not confirmed:
        _report_rollback_failure("gradient trainer poison marker failed")
    return confirmed


def _validate_loop(
    cfg: Config,
    *,
    mode: str,
    n_new_tokens: int,
    max_iterations: int,
    patience: int,
    ce_min_improve: float,
    kl_max: float,
) -> None:
    """Validate the opt-in contract before allocating an optimizer."""
    if mode not in {"gradient", "rls"}:
        raise ValueError(f"mode must be 'gradient' or 'rls', got {mode!r}")
    if cfg.model.decision.enabled:
        raise ValueError(
            "self_improve does not support model.decision.enabled=True yet: "
            "generated trajectories carry no structured decision labels"
        )
    if not (cfg.model.adapters.enabled or cfg.model.cortex.enabled):
        raise ValueError(
            "model.adapters.enabled=True or model.cortex.enabled=True is required "
            "for self-improvement"
        )
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
    has_adapter_contour = cfg.model.adapters.pyramid.enabled or cfg.model.adapters.ttt_lora.enabled
    if not has_adapter_contour and not cfg.model.cortex.enabled:
        raise ValueError(
            "at least one adapter contour or model.cortex.enabled=True is required"
        )
    if cfg.model.adapters.pyramid.enabled and cfg.model.adapters.ttt_lora.enabled:
        raise ValueError(
            "pyramid and ttt_lora adapters are currently mutually exclusive: "
            "enable only one contour for self-improvement"
        )
    if mode == "rls" and not cfg.model.adapters.ttt_lora.enabled:
        raise ValueError(
            "mode='rls' fits the TTT-LoRA B matrix, so "
            "model.adapters.ttt_lora.enabled=True is required"
        )
    if mode == "rls" and cfg.model.cortex.enabled:
        raise ValueError(
            "mode='rls' updates TTT-LoRA only; disable model.cortex or use "
            "mode='gradient' for the Pyramidal Cortex"
        )


def _adapter_values(model: HAGI) -> list[float]:
    """Return compact summaries of all live adaptive components."""
    values: list[float] = []
    for module in model.modules():
        pyramid = getattr(module, "pyramid", None)
        if pyramid is not None and hasattr(pyramid, "scale"):
            values.append(float(pyramid.scale.detach().item()))
        lora = getattr(module, "lora_B", None)
        if lora is not None:
            values.append(float(lora.detach().float().mean().item()))
    if model.cortex is not None:
        values.extend(
            float(link.weight.detach().float().mean().item())
            for link in model.cortex.links
        )
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
    training_modes = _snapshot_training_modes(model)
    try:
        prompt = torch.tensor(
            [prompt_ids], dtype=torch.long, device=next(model.parameters()).device
        )
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
    finally:
        _restore_training_mode(model, training_modes)


def _score(
    model: HAGI,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Return exact CE and full logits for a generated trajectory."""
    training_modes = _snapshot_training_modes(model)
    try:
        model.eval()
        with torch.no_grad():
            output = model(input_ids, targets, return_logits=True)
        if output.ce is None or output.logits is None:
            raise RuntimeError("HAGI.forward did not return CE and logits")
        return float(output.ce.detach()), output.logits.detach()
    finally:
        _restore_training_mode(model, training_modes)


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
    mode: str = "gradient",
    ttt: TttRls | None = None,
    n_new_tokens: int = 16,
    max_iterations: int = 8,
    ce_min_improve: float = 0.0,
    patience: int = 3,
    kl_max: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    trainer: Trainer | None = None,
) -> SelfImproveStats:
    """Run bounded score → update → re-score iterations over a trajectory.

    The function is intentionally explicit: callers opt in by invoking it. It
    does not mutate ``cfg`` or the default generation path.

    Args:
        mode: ``"gradient"`` regenerates a trajectory every iteration and
            updates adapters through :meth:`Trainer.train_step`. ``"rls"``
            generates **once**, then reuses that trajectory for every
            iteration, updating ``lora_B`` in closed form from hidden-state
            features via :meth:`TttRls.step` — no optimizer, no per-iteration
            autoregression. The same KL/non-finite guards and the same
            transactional rollback apply to both modes.
        ttt: pre-configured fitter for ``mode="rls"``. Built with default
            hyperparameters when omitted. Caller-built fitters are the seam
            for tuning ``stream_frac``/``refit_rows``/caps, and their
            accumulators intentionally persist across calls (online learning).
            Ignored in ``"gradient"`` mode.
        n_new_tokens: generation budget per trajectory.
        max_iterations: iteration bound.
        ce_min_improve: minimum CE gain for an iteration to count as progress.
        patience: consecutive non-improving accepted updates before stopping.
        kl_max: KL(pre || post) bound; a breach rolls the update back.
        eos_token_id: overrides ``cfg.train.data.eos_token_id``.
        pad_token_id: overrides ``cfg.train.data.pad_token_id``.
        trainer: injected trainer for ``"gradient"`` mode. Not used in
            ``"rls"`` mode, where no optimizer is allocated at all.

    Returns:
        :class:`SelfImproveStats`.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    _validate_loop(
        cfg,
        mode=mode,
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
    if mode == "rls" and ttt is not None and ttt.model is not model:
        # The fitter's accumulators and its lora_B references are bound to the
        # model it was built from. Passing another model would write one
        # model's fitter state while scoring a different model, and a rollback
        # would restore the wrong lora_B.
        raise ValueError("ttt must be bound to the same model passed to self_improve")

    if mode == "gradient" and trainer is None:
        trainer = Trainer(model, cfg)
    if mode == "rls" and ttt is None:
        ttt = TttRls(model)
    if mode == "gradient":
        if trainer is None:  # defensive: construction above makes this unreachable
            raise RuntimeError("gradient mode requires a trainer")
        if _is_gradient_trainer_poisoned(trainer):
            raise RuntimeError(
                "gradient trainer is poisoned after an incomplete transaction rollback"
            )
    # ``generate``/``_score``/``_harvest`` each call ``model.eval()``. The loop
    # owns the caller's complete module-tree mode, so it restores it on every exit
    # path rather than leaving mixed train/eval state as a side effect.
    training_modes = _snapshot_training_modes(model)
    stats = SelfImproveStats()
    best_ce: float | None = None
    no_improve = 0
    cached: tuple[torch.Tensor, torch.Tensor, int, torch.Tensor] | None = None

    for iteration in range(max_iterations):
        if mode == "rls" and cached is not None:
            # Generation is hoisted out of the loop: the trajectory is the
            # feature source for every subsequent update, so it is paid once.
            sequence, input_ids, targets, n_generated = cached
        else:
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
                logger.warning(
                    "iteration %d: generation produced no new tokens", iteration
                )
                break

            input_ids = sequence[:-1].unsqueeze(0)
            targets = sequence[1:].unsqueeze(0)
            if mode == "rls":
                cached = (sequence, input_ids, targets, n_generated)
        pre_ce, pre_logits = _score(model, input_ids, targets)

        # Snapshot trainable adapter params AND optimizer state BEFORE the step,
        # so a KL/non-finite guard breach can roll the update back as a precise
        # transaction. The base is frozen and excluded from the optimizer, so
        # only adapter params + optimizer momentum move on a rejected step.
        # Injected lightweight trainers may not expose an optimizer; in that
        # case the adapter snapshot still protects the model state.
        param_snapshot = _snapshot_adapters(model) if mode == "gradient" else None
        # RLS never touches the optimizer, so in rls mode there is no optimizer
        # state to snapshot and no trainer.step to protect.
        optimizer = None if mode == "rls" else getattr(trainer, "optimizer", None)
        opt_snapshot = (
            # Deep-copy: optimizer.state_dict() returns live tensor objects
            # that keep mutating as the optimizer steps. A shallow snapshot
            # therefore restores nothing on rollback.
            _deepcopy_optimizer_state(optimizer.state_dict())
            if optimizer is not None
            else None
        )
        pre_step = None if mode == "rls" else getattr(trainer, "step", None)
        post_step = None

        ttt_state_snapshot = ttt.snapshot_state() if mode == "rls" else None
        delta_frac = 0.0
        if mode == "rls":
            try:
                ttt_stats = ttt.step(input_ids, targets)
                update_applied = ttt_stats.blocks_updated > 0
                delta_frac = ttt_stats.delta_rms_frac
                # Post-step scoring is part of the same transaction. An
                # exception here must not leave the accepted live state ahead
                # of the caller's pre-state snapshot.
                post_ce, post_logits = _score(model, input_ids, targets)
                kl_div = _kl_pre_post(
                    pre_logits,
                    post_logits,
                    n_generated_positions=n_generated,
                )
            except BaseException:
                _restore_rls_state_safely(
                    ttt,
                    ttt_state_snapshot,
                    "RLS exception rollback failed",
                )
                _restore_training_mode(model, training_modes)
                raise
        else:
            post_param_snapshot: _AdapterSnapshot | None = None
            post_opt_snapshot: dict | None = None
            try:
                # Keep the mode switch inside the transaction: an override can
                # partially recurse through submodules and then fail. The
                # transaction's finally must restore the caller's mode even
                # when no optimizer step has happened yet.
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
                if update_applied:
                    post_step = getattr(trainer, "step", None)

                # Diagnostics stay FAITHFUL: post_ce/kl_div are the real
                # post-update values regardless of whether the guard accepts or
                # rejects. Keep post-step states as recovery points if gradient
                # rollback itself fails.
                post_param_snapshot = (
                    _snapshot_adapters(model) if update_applied else None
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
            except BaseException:
                # Parameter and optimizer rollback are independent failure
                # domains. If the pre-state cannot be restored coherently, the
                # helper attempts the captured post-state and poisons the trainer;
                # the original exception still wins at the bare ``raise``.
                status = _rollback_gradient_transaction(
                    model,
                    trainer,
                    optimizer,
                    param_snapshot,
                    opt_snapshot,
                    post_param_snapshot,
                    post_opt_snapshot,
                    pre_step=pre_step,
                    post_step=post_step,
                )
                if status == "poisoned":
                    _poison_gradient_trainer(trainer)
                _restore_training_mode(model, training_modes)
                raise

        reject = False
        stop_reason: str | None = None
        if not (math.isfinite(pre_ce) and math.isfinite(post_ce) and math.isfinite(kl_div)):
            stats.stopped = "nonfinite_ce_or_kl"
            stop_reason = "nonfinite"
            reject = True
        elif kl_div > kl_max:
            stats.stopped = "kl_bound"
            stop_reason = "kl"
            reject = True

        # A skipped optimizer step (non-finite gradient) leaves the model
        # bit-identical, so it is not progress. Accepting it would advance the
        # CLI's step counter, persist an untouched model as a new checkpoint and
        # report success for a run that changed nothing. It takes the rollback
        # path: nothing to restore, but the resume horizon must not move.
        # RLS is exempt: a warmup step can legitimately write no lora_B while
        # still accumulating rows for a later refit.
        no_op = mode == "gradient" and not update_applied

        if reject or no_op:
            # Transactional rollback of the rejected step. A rejected guard
            # step must not count toward completed_steps for the resume
            # horizon. Restore optimizer state before touching model parameters
            # so that a parameter-only load cannot proceed from a corrupted
            # momentum state. Trainer.train_step() increments its counter
            # only after optimizer.step() succeeds.
            # Restore the complete transaction, including persistent RLS
            # accumulators and row buffers. A KL-rejected RLS step can mutate
            # those fields even when no lora_B value was accepted.
            try:
                if mode == "rls":
                    ttt.restore_state(ttt_state_snapshot)
                else:
                    # A trainer may mutate state and still report a false no-op.
                    # Never trust that flag to decide whether rollback is needed.
                    status = _rollback_gradient_transaction(
                        model,
                        trainer,
                        optimizer,
                        param_snapshot,
                        opt_snapshot,
                        post_param_snapshot,
                        post_opt_snapshot,
                        pre_step=pre_step,
                        post_step=post_step,
                    )
                    if status != "pre":
                        raise _RollbackPoisonedTrainer(
                            "gradient transaction rollback could not restore the pre-state"
                        )
                # The stop reason and its diagnostic are emitted only after the
                # rollback has run: a logger that raises (test doubles, a custom
                # handler) must not skip the restore.
            except _RollbackPoisonedTrainer:
                # Rollback could not restore a coherent state. The helper already
                # attempted post-state recovery and poisoned the trainer, so a
                # later call fails closed instead of resuming a mixed pair.
                _restore_training_mode(model, training_modes)
                raise
            except BaseException:
                # This exception came from the rollback attempt itself. Do not run
                # the same failing restore a second time and replace the first
                # error; restore mode and re-raise the original failure.
                _restore_training_mode(model, training_modes)
                raise
            update_applied = False
            try:
                if stop_reason == "nonfinite":
                    logger.warning("iteration %d: non-finite CE or KL; stopping", iteration)
                elif stop_reason == "kl":
                    logger.warning("iteration %d: KL %.4f exceeded bound %.4f", iteration, kl_div, kl_max)
                stats.iterations.append(
                    SelfImproveResult(
                        iteration=iteration,
                        generated_ids=sequence[len(prompt_ids) :].tolist(),
                        pre_ce=pre_ce,
                        post_ce=post_ce,
                        kl_div=kl_div,
                        update_applied=update_applied,
                        adapter_values_after=_adapter_values(model),
                        delta_rms_frac=delta_frac,
                    )
                )
            except BaseException:
                # Diagnostics are part of the transaction: a failing logger or
                # result construction must not leak the trainer's train mode.
                _restore_training_mode(model, training_modes)
                raise
        else:
            # An accepted update is not externally visible until this iteration
            # has been materialized into the returned statistics. Keep the whole
            # acceptance path inside one transaction boundary: failures from
            # adapter diagnostics, plateau bookkeeping or logging must not leave
            # live model/optimizer/fitter state ahead of ``pre_step``.
            try:
                if best_ce is None or post_ce < best_ce - ce_min_improve:
                    best_ce = post_ce
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        stats.stopped = "ce_plateau"

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
                        delta_rms_frac=delta_frac,
                    )
                )
                if stats.stopped == "ce_plateau":
                    logger.info("iteration %d: CE plateau; stopping", iteration)
                    break
            except BaseException:
                if mode == "rls":
                    _restore_rls_state_safely(
                        ttt,
                        ttt_state_snapshot,
                        "RLS accepted-diagnostic rollback failed",
                    )
                elif update_applied:
                    _rollback_gradient_transaction(
                        model,
                        trainer,
                        optimizer,
                        param_snapshot,
                        opt_snapshot,
                        post_param_snapshot,
                        post_opt_snapshot,
                        pre_step=pre_step,
                        post_step=post_step,
                    )
                _restore_training_mode(model, training_modes)
                raise

        # Drop the pre-transaction snapshot before the next iteration. Without
        # this, allocating the next snapshot briefly retains two independent
        # rollback copies plus the live fitter state.
        if mode == "rls":
            ttt_state_snapshot = None

        # A rejected KL/non-finite guard, or a skipped no-op step, ends the
        # search: the model is back at the pre-state, and retrying from it
        # would re-enter the same region.
        if reject or no_op:
            break

    stats.best_ce = best_ce
    _restore_training_mode(model, training_modes)
    return stats


__all__ = ["SelfImproveResult", "SelfImproveStats", "_restore_adapters", "_snapshot_adapters", "self_improve"]
