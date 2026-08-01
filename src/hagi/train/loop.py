"""Training loop.

One objective, computed once per microbatch. There is no loss aggregator, no
beta-anneal, no attention-mode curriculum and no distillation teacher — every one
of those existed in V28 and every one was disabled in both shipped configs.

The loop's real job is to make failure *visible*. A language model that is
diverging looks exactly like one that is training slowly, for thousands of steps,
unless you are watching the right observables. The three that matter here:

* ``ce`` in nats/token, against the measured unigram entropy of 8.06 nats. A
  model above that is worse than counting token frequencies.
* ``qk_gain`` — the mean QK-norm gain. Rising means the correlator is heading for
  saturation, which is the V30 failure (ce 2.32 at step 19k, 6.6 at step 53k).
* ``moe/entropy_ratio`` — the usable fraction of expert channels. Falling toward
  ``1/E`` means routing has collapsed and most parameters are dead.

Gradient accumulation weights each microbatch by its scored-token count, so
microbatches of unequal size (packed windows with different loss masks) average
correctly instead of over-weighting sparse ones.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import torch
from torch import nn

from hagi.config import Config
from hagi.train.optim import _muon_parameters, build_optimizer, set_learning_rate

logger = logging.getLogger(__name__)


def configure_runtime() -> None:
    """Set backend flags that materially change throughput."""
    import os

    # ROCm flash-attention kernels. Without this SDPA falls back to the math
    # backend (unfused bmm + softmax + bmm), which is 2-3x slower and allocates
    # the full attention matrix.
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def cast_model(model: nn.Module, precision: str) -> None:
    """Cast the model to ``precision``, keeping marked parameters in fp32.

    A parameter is kept in fp32 when its module sets ``keep_fp32 = True``. Three
    kinds carry the marker, for the same underlying reason — a small parameter
    whose meaningful updates are below bf16's local resolution:

    * **Normalization gains.** A gain at 1.0 receives gradients around 1e-4 to
      1e-5. The smallest bf16 step above 1.0 is ~0.0078, so those updates round
      to zero and every normalization layer stays frozen at its initialization
      for the entire run.
    * **The receiver gain.** One scalar controlling the whole output
      distribution's sharpness, starting at ``1/sqrt(H)`` ~ 0.022.
    * **The MoE router.** Top-k over E logits compares nearby numbers; at a 7-bit
      mantissa the ordering of two close experts becomes arbitrary, which turns
      routing into noise the balance controller then chases.

    Cost is a few times ``L * H`` parameters' worth of memory, which is
    negligible against the body, and it is not optional.
    """
    if precision == "fp32":
        return
    model.to(torch.bfloat16)
    for module in model.modules():
        if getattr(module, "keep_fp32", False):
            for param in module.parameters(recurse=False):
                param.data = param.data.float()


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip the global gradient norm; return the pre-clip value.

    The pre-clip norm is the diagnostic worth logging: once clipping is active the
    post-clip value is constant by construction and tells you nothing.
    """
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


def clip_gradients_by_group(model: nn.Module, max_norm: float) -> tuple[float, float]:
    """Clip Muon (body) and AdamW (codebook/conv/norms/gains) norms separately.

    A tied codebook is ``V*H`` parameters with dense token-frequency gradients,
    so its per-step gradient norm dwarfs the body's; a single global clip then
    spends most of its budget on the codebook and shrinks the body's update to a
    fraction of its step. Measured on the V31 1B run: codebook 8.92 + conv 5.32
    against a total of 10.59, leaving the 100 body matrices only ~19% of the
    clip budget at every step. Splitting the clip lets each group move at its own
    scale; the body's step is no longer a hostage of the codebook's.

    Returns ``(body_norm, rest_norm)`` pre-clip, or NaN for a group with no
    parameters.
    """
    body = _muon_parameters(model)
    rest = [p for p in model.parameters() if p not in set(body)]
    body_norm = float(torch.nn.utils.clip_grad_norm_(body, max_norm)) if body else float("nan")
    rest_norm = float(torch.nn.utils.clip_grad_norm_(rest, max_norm)) if rest else float("nan")
    return body_norm, rest_norm


class Trainer:
    """Owns the model, optimizer and step counter.

    Args:
        model: the model (already on device).
        cfg: top-level config.
        start_step: resume point.
    """

    def __init__(self, model: nn.Module, cfg: Config, start_step: int = 0) -> None:
        self.model = model
        self.cfg = cfg
        self.step = start_step
        cast_model(model, cfg.train.precision)
        self.optimizer = build_optimizer(model, cfg)

    def load_optimizer_state(self, state: dict) -> None:
        self.optimizer.load_state_dict(state)

    def train_step(self, microbatches: list[dict]) -> dict:
        """One optimizer step over a list of microbatches.

        Returns:
            Metrics for logging. ``update_applied`` is False when the step was
            skipped for non-finite gradients.

        Raises:
            ValueError: on an empty microbatch list.
        """
        if not microbatches:
            raise ValueError("train_step needs at least one microbatch")

        model, cfg = self.model, self.cfg
        model.train()
        device = next(model.parameters()).device
        self.optimizer.zero_grad(set_to_none=True)

        # Weight each microbatch by its scored-token count so unequal microbatches
        # average correctly rather than over-weighting sparse ones.
        token_counts: list[int] = []
        for batch in microbatches:
            mask = batch.get("loss_mask")
            token_counts.append(int(mask.sum()) if mask is not None else batch["targets"].numel())
        total_tokens = max(sum(token_counts), 1)

        ce_sum = 0.0
        loss_sum = 0.0
        z_sum = 0.0
        router_z_sum = 0.0
        for batch, count in zip(microbatches, token_counts, strict=True):
            # non_blocking=True on .to() is disabled: on this ROCm build the
            # async H2D transfer raced the compute stream and produced
            # intermittent "HIP error: unspecified launch failure" in CUDAEvent
            # (crash after ~10 steps, only in train_step, never in a manual
            # blocking loop). The transfer is small (a few tens of MB per
            # batch); blocking costs nothing measurable.
            output = model(
                batch["input_ids"].to(device),
                batch["targets"].to(device),
                doc_ids=batch["doc_ids"].to(device) if "doc_ids" in batch else None,
                loss_mask=batch["loss_mask"].to(device) if "loss_mask" in batch else None,
                images=batch["images"].to(device) if "images" in batch else None,
                spectrograms=batch["spectrograms"].to(device)
                if "spectrograms" in batch
                else None,
            )
            weight = count / total_tokens
            (output.loss * weight).backward()

            ce_sum += float(output.ce.detach()) * weight
            loss_sum += float(output.loss.detach()) * weight
            z_sum += float(output.z_loss.detach()) * weight if output.z_loss is not None else 0.0
            router_z_sum += (
                float(output.router_z_loss.detach()) * weight
                if output.router_z_loss is not None
                else 0.0
            )
            del output

        body_norm, rest_norm = clip_gradients_by_group(model, cfg.train.max_grad_norm)
        if not (math.isfinite(body_norm) and math.isfinite(rest_norm)):
            # Skip rather than raise: one bad microbatch in a long run should cost
            # one step, not the run. A persistent problem shows up as a run of
            # skipped steps in the log.
            logger.warning(
                "step %d: non-finite gradient norm (body=%s rest=%s), update skipped",
                self.step,
                body_norm,
                rest_norm,
            )
            self.optimizer.zero_grad(set_to_none=True)
            return {
                "step": self.step,
                "update_applied": False,
                "grad_norm": body_norm if math.isfinite(body_norm) else rest_norm,
                "body_grad_norm": body_norm,
                "rest_grad_norm": rest_norm,
            }

        adam_lr, muon_lr = set_learning_rate(self.optimizer, self.step, cfg)
        self.optimizer.step()

        # Controller updates go after the optimizer step, outside any
        # checkpointed region, so the forward stays pure and recomputation
        # reproduces the same expert selection.
        if hasattr(model, "commit_controller_updates"):
            model.commit_controller_updates()

        metrics = {
            "step": self.step,
            "loss": loss_sum,
            "ce": ce_sum,
            "bpt": ce_sum / math.log(2.0),
            "ppl": math.exp(min(ce_sum, 20.0)),
            "z_loss": z_sum,
            "router_z_loss": router_z_sum,
            "grad_norm": body_norm,
            "body_grad_norm": body_norm,
            "rest_grad_norm": rest_norm,
            "lr": adam_lr,
            "muon_lr": muon_lr,
            "tokens": total_tokens,
            "update_applied": True,
        }
        if self.step % max(1, cfg.train.logging.diag_interval) == 0 and hasattr(model, "diagnostics"):
            metrics.update(model.diagnostics())
        self.step += 1
        return metrics


def format_metrics(metrics: dict) -> str:
    """One-line log record. ``ce`` first: it is the only number that matters."""
    if not metrics.get("update_applied", True):
        return f"step {metrics['step']} | SKIPPED (grad_norm={metrics['grad_norm']})"
    parts = [
        f"step {metrics['step']}",
        f"ce={metrics['ce']:.4f}",
        f"bpt={metrics['bpt']:.3f}",
        f"ppl={metrics['ppl']:.1f}",
        f"grad={metrics['grad_norm']:.3f}",
        f"lr={metrics['lr']:.2e}",
    ]
    if "body_grad_norm" in metrics:
        parts.append(f"gb={metrics['body_grad_norm']:.3f}")
        parts.append(f"gr={metrics['rest_grad_norm']:.3f}")
    for key in ("qk_gain", "residual_gain", "logit_scale", "moe/entropy_ratio", "moe/max_load", "moe/bias_span"):
        if key in metrics:
            parts.append(f"{key.split('/')[-1]}={metrics[key]:.3f}")
    return " | ".join(parts)


def train(
    model: nn.Module,
    dataloader,
    cfg: Config,
    start_step: int = 0,
    optimizer_state: dict | None = None,
) -> Iterator[dict]:
    """Run training, yielding metrics at the configured interval.

    Checkpoints are written by step count; the caller only logs.
    """
    from hagi.train.checkpoint import save_checkpoint

    configure_runtime()
    trainer = Trainer(model, cfg, start_step)
    if optimizer_state is not None:
        trainer.load_optimizer_state(optimizer_state)

    accum = cfg.train.grad_accum_steps
    data_iter = iter(dataloader)

    while trainer.step < cfg.train.max_steps:
        microbatches = []
        for _ in range(accum):
            try:
                microbatches.append(next(data_iter))
            except StopIteration:
                data_iter = iter(dataloader)
                microbatches.append(next(data_iter))

        step_index = trainer.step
        metrics = trainer.train_step(microbatches)

        if step_index % max(1, cfg.train.logging.log_interval) == 0:
            yield metrics

        completed = step_index + 1
        if cfg.train.checkpoint_interval > 0 and completed % cfg.train.checkpoint_interval == 0:
            save_checkpoint(
                model,
                cfg,
                completed,
                cfg.train.checkpoint_dir,
                cfg.train.checkpoint_keep_last,
                optimizer=trainer.optimizer,
            )

    save_checkpoint(
        model,
        cfg,
        trainer.step,
        cfg.train.checkpoint_dir,
        cfg.train.checkpoint_keep_last,
        optimizer=trainer.optimizer,
    )
