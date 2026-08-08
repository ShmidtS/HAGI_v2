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
* ``logit_scale`` — receiver gain, which should rise as the channel learns.

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
from hagi.model.norms import HeadNorm, RMSNorm
from hagi.model.ternary import cache_ternary_weights, clear_ternary_weights
from hagi.train.optim import _muon_parameters, build_optimizer, set_learning_rate

logger = logging.getLogger(__name__)


def puncture_loss_mask(
    shape: tuple[int, ...],
    *,
    rate: float,
    mode: str,
    step: int,
    device: torch.device,
    base: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """Build / thin a boolean loss mask — erasure channel on supervision.

    Information-theoretically this is *puncturing* the coding objective: the
    body still processes every symbol (full channel use), but the receiver only
    scores a rate-``p`` subset. Under Bernoulli sampling the per-step CE is an
    unbiased estimator of the full-sequence CE; stride is a deterministic lattice
    with the same average rate and lower variance.

    Combined with an existing ``base`` mask via logical AND (packing pads,
    curriculum filters, etc. still apply).

    Returns:
        Boolean ``[B, T]`` mask, or None when ``rate >= 1`` and ``base is None``
        (caller scores everything).
    """
    if rate >= 1.0 and base is None:
        return None
    if rate >= 1.0:
        return base.bool() if base is not None else None

    if mode == "bernoulli":
        keep = torch.rand(shape, device=device, generator=generator) < float(rate)
    elif mode == "stride":
        # Keep every k-th position; phase rotates with the optimizer step so the
        # lattice covers the sequence over a short horizon.
        k = max(1, int(round(1.0 / float(rate))))
        phase = int(step) % k
        t = shape[-1]
        idx = torch.arange(t, device=device)
        keep_1d = ((idx % k) == phase)
        # Broadcast to [B, T] (or whatever leading dims ``shape`` carries).
        keep = keep_1d.expand(shape)
    else:
        raise ValueError(f"unknown ce_keep_mode {mode!r}")

    if base is not None:
        keep = keep & base.bool().to(device=device)
    return keep


def configure_runtime() -> None:
    """Set backend flags that materially change throughput."""
    import os

    # ROCm flash-attention kernels. Without this SDPA falls back to the math
    # backend (unfused bmm + softmax + bmm), which is 2-3x slower and allocates
    # the full attention matrix.
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")
    # Allow torch.compile to capture scalar outputs (loss values) without graph
    # breaks. Without this, every metrics.float() call splits the compiled graph.
    os.environ.setdefault("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1")
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
      distribution's sharpness, starting at ``1/sqrt(H)`` ~0.022.

    Cost is a few times ``L * H`` parameters' worth of memory, which is
    negligible against the body, and it is not optional.

    Norm variance precision follows ``precision`` when the module was built with
    ``fp32_variance=True`` (the default): under bf16 the fused kernel is 5x
    faster and numerically identical (verified max diff 0.0), so the variance
    accumulator is switched to the input dtype.
    """
    if precision == "fp32":
        return
    model.to(torch.bfloat16)
    for module in model.modules():
        if getattr(module, "keep_fp32", False):
            for param in module.parameters(recurse=False):
                param.data = param.data.float()
        if isinstance(module, (RMSNorm, HeadNorm)):
            module.fp32_variance = False


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip the global gradient norm; return the pre-clip value.

    The pre-clip norm is the diagnostic worth logging: once clipping is active the
    post-clip value is constant by construction and tells you nothing.
    """
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


def clip_gradients_by_group(
    model: nn.Module, max_norm: float
) -> tuple[torch.Tensor | float, torch.Tensor | float]:
    """Clip Muon and AdamW groups, leaving device norms unsynchronized.

    The caller decides when host visibility is required.  Returning the scalar
    tensors produced by ``clip_grad_norm_`` avoids two implicit GPU-to-CPU
    synchronizations in every optimizer step; CPU callers still receive tensors
    that compare and convert like scalars.
    """
    body = _muon_parameters(model)
    rest = [p for p in model.parameters() if p not in set(body)]
    body_norm = torch.nn.utils.clip_grad_norm_(body, max_norm) if body else float("nan")
    rest_norm = torch.nn.utils.clip_grad_norm_(rest, max_norm) if rest else float("nan")
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
        if getattr(cfg.train, "compile_model", False):
            # ROCm flash-attention backward breaks torch.compile (a fake/meta
            # kernel stride assertion in _scaled_dot_product_flash_attention_backward).
            # The mem-efficient SDPA backend compiles cleanly and is faster for
            # the short sequences here (measured 19.4 ms/step vs 20.4 flash-off
            # vs 29.9 baseline on the 8060S).
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            try:
                self.model = torch.compile(model, mode="default")
                logger.info("torch.compile enabled (mode=default, mem-efficient SDPA)")
            except Exception as exc:
                logger.warning("torch.compile failed (%s), continuing uncompiled", exc)
                self.model = model
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

        # OFDM coherence interval: the ternary map Q*s is constant across the
        # microbatches of one optimizer step (the master W only changes on
        # optimizer.step). Cache it once; clear after the step so the next
        # step recomputes against the updated master. Always on: the cached
        # path is also the host-bound fix (zero per-forward copies), so it is
        # a win even at grad_accum=1.
        use_ternary_cache = cfg.model.ternary.enabled and getattr(
            cfg.train, "ternary_step_cache", True
        )
        if use_ternary_cache:
            cache_ternary_weights(model)

        # Puncture the supervision stream (erasure channel on CE). Applied on
        # device after H2D so we do not rewrite host batches. Existing packing
        # masks are AND-ed in.
        keep_rate = float(getattr(cfg.train, "ce_keep_rate", 1.0))
        keep_mode = str(getattr(cfg.train, "ce_keep_mode", "bernoulli"))

        # Weight each microbatch by its scored-token count so unequal microbatches
        # average correctly rather than over-weighting sparse ones.
        prepared: list[tuple[dict, torch.Tensor | None]] = []
        token_counts: list[torch.Tensor] = []
        for batch in microbatches:
            ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            base_mask = batch["loss_mask"].to(device) if "loss_mask" in batch else None
            mask = puncture_loss_mask(
                tuple(targets.shape),
                rate=keep_rate,
                mode=keep_mode,
                step=self.step,
                device=device,
                base=base_mask,
            )
            # Keep counts on device.  Converting each mask sum with int() forced
            # one full HIP stream synchronization per microbatch.
            count = (
                mask.sum(dtype=torch.int64)
                if mask is not None
                else torch.tensor(targets.numel(), device=device, dtype=torch.int64)
            )
            token_counts.append(count.clamp_min(1))
            prepared.append(
                (
                    {
                        "input_ids": ids,
                        "targets": targets,
                        "doc_ids": batch["doc_ids"].to(device) if "doc_ids" in batch else None,
                        "loss_mask": mask,
                        "images": batch["images"].to(device) if "images" in batch else None,
                        "spectrograms": batch["spectrograms"].to(device)
                        if "spectrograms" in batch
                        else None,
                    },
                    mask,
                )
            )
        total_tokens = torch.stack(token_counts).sum().clamp_min(1)

        ce_sum = torch.zeros((), device=device, dtype=torch.float32)
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        z_sum = torch.zeros((), device=device, dtype=torch.float32)
        exact_ce_value: float | None = None
        exact_interval = int(cfg.train.logging.exact_ce_interval)
        for microbatch_index, ((batch, _mask), count) in enumerate(
            zip(prepared, token_counts, strict=True)
        ):
            # non_blocking=True on .to() is disabled: on this ROCm build the
            # async H2D transfer raced the compute stream and produced
            # intermittent "HIP error: unspecified launch failure" in CUDAEvent
            # (crash after ~10 steps, only in train_step, never in a manual
            # blocking loop). The transfer is small (a few tens of MB per
            # batch); blocking costs nothing measurable.
            #
            # Mark step boundary for CUDA graphs when the model is compiled:
            # without this, reduce-overhead mode reuses static output buffers
            # across microbatches and backward reads stale data.
            if getattr(cfg.train, "compile_model", False):
                torch.compiler.cudagraph_mark_step_begin()
            output = model(
                batch["input_ids"],
                batch["targets"],
                doc_ids=batch["doc_ids"],
                loss_mask=batch["loss_mask"],
                images=batch["images"],
                spectrograms=batch["spectrograms"],
            )
            if microbatch_index == 0 and exact_interval > 0 and self.step % exact_interval == 0:
                flat_hidden = output.hidden.detach().reshape(-1, output.hidden.shape[-1])
                flat_targets = batch["targets"].reshape(-1)
                rows = min(int(cfg.train.logging.exact_ce_rows), flat_targets.numel())
                # Use an independent RNG stream so receiver proposal sampling
                # cannot change the calibration rows. This keeps the estimate
                # random/unbiased while making architecture A/B reproducible.
                generator = torch.Generator(device=device)
                generator.manual_seed(int(cfg.train.logging.exact_ce_seed) + self.step)
                sample = torch.randperm(flat_targets.numel(), device=device, generator=generator)[:rows]
                with torch.no_grad():
                    exact_ce_value = float(
                        model.head.exact_loss(
                            flat_hidden.index_select(0, sample),
                            flat_targets.index_select(0, sample),
                        )
                    )
            weight = count.to(torch.float32) / total_tokens
            (output.loss * weight).backward()

            ce_sum = ce_sum + output.ce.detach().float() * weight
            loss_sum = loss_sum + output.loss.detach().float() * weight
            if output.z_loss is not None:
                z_sum = z_sum + output.z_loss.detach().float() * weight
            del output

        body_norm_raw, rest_norm_raw = clip_gradients_by_group(model, cfg.train.max_grad_norm)
        body_norm_tensor = torch.as_tensor(body_norm_raw, device=device, dtype=torch.float32)
        rest_norm_tensor = torch.as_tensor(rest_norm_raw, device=device, dtype=torch.float32)
        # One synchronization is necessary before mutating parameters: an invalid
        # norm must skip optimizer.step.  Transfer both decisions together.
        body_norm, rest_norm = torch.stack((body_norm_tensor, rest_norm_tensor)).tolist()
        if not (math.isfinite(body_norm) and math.isfinite(rest_norm)):
            # Skip rather than raise: one bad microbatch in a long run should cost
            # one step, not the run. A persistent problem shows up as a run of
            # skipped steps in the log. ``body_norm`` is NaN when Muon is disabled
            # (no body group); only ``rest_norm`` then gates the update.
            body_ok = math.isfinite(body_norm) or len(_muon_parameters(model)) == 0
            rest_ok = math.isfinite(rest_norm)
            if not (body_ok and rest_ok):
                logger.warning(
                    "step %d: non-finite gradient norm (body=%s rest=%s), update skipped",
                    self.step,
                    body_norm,
                    rest_norm,
                )
                self.optimizer.zero_grad(set_to_none=True)
                if use_ternary_cache:
                    clear_ternary_weights(model)
                return {
                    "step": self.step,
                    "update_applied": False,
                    "grad_norm": body_norm if body_ok else rest_norm,
                    "body_grad_norm": body_norm,
                    "rest_grad_norm": rest_norm,
                }
        adam_lr, muon_lr = set_learning_rate(self.optimizer, self.step, cfg)
        self.optimizer.step()

        if use_ternary_cache:
            clear_ternary_weights(model)

        if hasattr(model, "commit_controller_updates"):
            model.commit_controller_updates()

        # Metrics cross the device boundary once, after all scheduled GPU work.
        # Previously every float()/int() below synchronized the HIP stream.
        loss_value, ce_value, z_value, tokens_value = torch.stack(
            (loss_sum, ce_sum, z_sum, total_tokens.to(torch.float32))
        ).tolist()
        receiver = "conditional_nce" if cfg.model.head.sampled_softmax_k > 0 else "exact_ce"
        metrics = {
            "step": self.step,
            "loss": loss_value,
            "ce": ce_value,
            "bpt": ce_value / math.log(2.0),
            "ppl": math.exp(min(ce_value, 20.0)),
            "z_loss": z_value,
            "grad_norm": body_norm,
            "body_grad_norm": body_norm,
            "rest_grad_norm": rest_norm,
            "lr": adam_lr,
            "muon_lr": muon_lr,
            "tokens": int(tokens_value),
            "ce_keep_rate": keep_rate,
            "receiver": receiver,
            "update_applied": True,
        }
        if receiver == "conditional_nce":
            metrics["nce"] = ce_value
            metrics["nce_bits"] = ce_value / math.log(2.0)
        if exact_ce_value is not None:
            metrics["exact_ce"] = exact_ce_value
            metrics["exact_bpt"] = exact_ce_value / math.log(2.0)
            metrics["exact_ppl"] = math.exp(min(exact_ce_value, 20.0))
            metrics["kl"] = max(exact_ce_value - ce_sum, 0.0)
        if self.step % max(1, cfg.train.logging.diag_interval) == 0 and hasattr(model, "diagnostics"):
            metrics.update(model.diagnostics())
        self.step += 1
        return metrics


def format_metrics(metrics: dict) -> str:
    """One-line record naming local NCE separately from exact coding cost."""
    if not metrics.get("update_applied", True):
        return f"step {metrics['step']} | skipped"
    receiver = metrics.get("receiver", "exact_ce")
    objective_name = "nce" if receiver == "conditional_nce" else "ce"
    bits_name = "nce_bits" if receiver == "conditional_nce" else "bpt"
    parts = [
        f"step {metrics['step']}",
        f"{objective_name}={metrics['ce']:.4f}",
        f"{bits_name}={metrics['bpt']:.3f}",
    ]
    parts.append(f"ppl={metrics['ppl']:.1f}")
    if "exact_ce" in metrics:
        parts.append(f"exact_ce={metrics['exact_ce']:.4f}")
    for key in (
        "qk_gain",
        "residual_gain",
        "logit_scale",
    ):
        if key in metrics:
            parts.append(f"{key.split('/')[-1]}={metrics[key]:.3f}")
    parts.append(f"kl={metrics.get('kl', 0.0):.4f}")
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
