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
import os
from collections.abc import Iterator

import torch
from torch import nn
from pathlib import Path

from hagi.config import Config
from hagi.model.adaptive import freeze_base_in_place
from hagi.model.norms import BlockRMSNorm, HeadNorm, RMSNorm
from hagi.model.ternary import BitLinear, cache_ternary_weights, clear_ternary_weights
from hagi.train.optim import _muon_parameters, build_optimizer, set_learning_rate
from hagi.train.analytic_step import CurvatureProbe, select_step
from hagi.train.safeqp_controller import corpus_grad_gram

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


def cast_model(model: nn.Module, precision: str, *, ternary_fp32_master: bool = False) -> None:
    """Cast the model to ``precision`` and keep selected masters in FP32.

    ``keep_fp32`` always protects sensitive scalar/gain modules. When
    ``ternary_fp32_master`` is true, ``BitLinear.weight`` masters are also
    restored to FP32 after the BF16 cast; the effective ternary weight is still
    cast to the activation dtype in each forward. This is a training/master
    precision switch, not physical packed storage.

    Normalization gains and the receiver gain are kept in FP32 because their
    small updates are below BF16's local resolution at typical magnitudes.
    Norm variance follows the activation dtype under BF16 for the existing
    fused-kernel path.
    """
    if precision == "fp32":
        return
    protected_params: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    ternary_params: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    for module in model.modules():
        if getattr(module, "keep_fp32", False):
            protected_params.extend(
                (param, param.detach().clone())
                for param in module.parameters(recurse=False)
            )
        if ternary_fp32_master and isinstance(module, BitLinear):
            ternary_params.append((module.weight, module.weight.detach().clone()))
    model.to(torch.bfloat16)
    for module in model.modules():
        if isinstance(module, (RMSNorm, HeadNorm, BlockRMSNorm)):
            module.fp32_variance = False
    for param, snapshot in protected_params:
        param.data = param.data.float()
        param.data.copy_(snapshot)
    for param, snapshot in ternary_params:
        param.data = param.data.float()
        param.data.copy_(snapshot)


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
    rest = [p for p in model.parameters() if p.requires_grad and p not in set(body)]
    body_norm = torch.nn.utils.clip_grad_norm_(body, max_norm) if body else float("nan")
    rest_norm = torch.nn.utils.clip_grad_norm_(rest, max_norm) if rest else float("nan")
    return body_norm, rest_norm


def _freeze_base_for_adapters(model: nn.Module) -> None:
    """Backward-compatible alias for the explicit adaptive ownership helper."""
    freeze_base_in_place(model)


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
        cast_model(
            model,
            cfg.train.precision,
            ternary_fp32_master=cfg.train.ternary_fp32_master,
        )
        if (
            cfg.model.adapters.enabled
            or cfg.model.cortex.enabled
            or cfg.model.decision.enabled
        ) and cfg.train.adapt.freeze_base:
            _freeze_base_for_adapters(model)
        elif cfg.train.adapt.freeze_base:
            raise ValueError(
                "train.adapt.freeze_base=True requires model.adapters.enabled=True, "
                "model.cortex.enabled=True, or model.decision.enabled=True"
                " (no adaptive parameters exist to optimize when all are disabled)"
            )
        if getattr(cfg.train, "compile_model", False):
            # ROCm flash-attention backward breaks torch.compile (a fake/meta
            # kernel stride assertion in _scaled_dot_product_flash_attention_backward).
            # The mem-efficient SDPA backend compiles cleanly and is faster for
            # the short sequences here (measured 19.4 ms/step vs 20.4 flash-off
            # vs 29.9 baseline on the 8060S).
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            try:
                # max-autotune measured -3.7% vs default on the gen-2 LoRA
                # step (1349 vs 1401 ms, round 51c); autotune cache persists
                # so the ~80s warmup is one-time per shape/kernel set.
                self.model = torch.compile(model, mode="max-autotune-no-cudagraphs")
                logger.info(
                    "torch.compile enabled (mode=max-autotune-no-cudagraphs, mem-efficient SDPA)"
                )
            except Exception as exc:
                logger.warning("torch.compile failed (%s), continuing uncompiled", exc)
                self.model = model
        self.optimizer = build_optimizer(model, cfg)
        # Gradient-Gram scan state (log-only; see safeqp_controller docstring).
        self._gram_samples: list[tuple[str, dict]] | None = None
        self._gram_metrics_last: dict = {}
        # Analytic step (RecursiveGrowth.lean optimal_step_unconstrained):
        # measures the directional Lipschitz constant L so the step can be
        # computed from curvature instead of read off the schedule.
        self._curvature = CurvatureProbe(
            refresh=int(getattr(cfg.train, "analytic_step_probe_interval", 50))
        ) if getattr(cfg.train, "analytic_step", False) else None
        self._analytic_step_last: dict = {}

    def _apply_analytic_step(
        self, base_lr: float, probe_batch: dict | None
    ) -> dict:
        """Replace the schedule LR with the curvature-optimal step.

        ``RecursiveGrowth.lean`` ``optimal_step_unconstrained``: the guaranteed
        decrease ``g(eta) = eta*inner - L*||d||^2*eta^2/2`` is a concave
        quadratic whose maximum is analytic, ``eta* = inner/(L*||d||^2)``.

        ``L`` is measured by a two-point probe on ``probe_batch``: the
        smoothness identity ``|f(theta+d) - f(theta) - <g,d>| <= (L/2)||d||^2``
        solved for L. The probe displaces, measures, and restores -- the model
        is left exactly as found, so a probe can never corrupt the update.

        The direction is the momentum-weighted one the optimizer actually
        applies, whose alignment with the gradient is MEASURED rather than
        assumed: the raw gradient gives ``inner = -||g||^2 < 0``, which is not
        the descent direction the theorem assumes.

        The step is clipped to ``1/L`` (``optimal_step_ge_recip``) and to
        ``base_lr`` so it never exceeds the tuned baseline.

        Returns the metrics dict (empty when no estimate is available yet).
        """
        params = [
            p for g in self.optimizer.param_groups if not g.get("_muon")
            for p in g["params"] if p.grad is not None
        ]
        grads = [p.grad.detach() for p in params]
        if not params:
            return {}

        def loss_fn() -> float:
            """Loss on the probe batch at the CURRENT parameter values."""
            if probe_batch is None:
                return float("nan")
            with torch.no_grad():
                out = self.model(
                    probe_batch["input_ids"],
                    probe_batch["targets"],
                    doc_ids=probe_batch.get("doc_ids"),
                    loss_mask=probe_batch.get("loss_mask"),
                    images=probe_batch.get("images"),
                    spectrograms=probe_batch.get("spectrograms"),
                    decision_targets=probe_batch.get("decision_targets"),
                    decision_mask=probe_batch.get("decision_mask"),
                )
                loss = out.lm_loss if out.lm_loss is not None else out.ce
                return float(loss)

        try:
            smoothness = self._curvature.estimate(loss_fn, params, grads)
        except Exception as exc:  # a failed probe costs the step's LR, not the step
            logger.warning("step %d: curvature probe failed: %s", self.step, exc)
            return {}
        if smoothness is None or smoothness != smoothness:  # None or NaN
            return {}

        # Direction the Adam group applies: momentum/nesterov lookahead where
        # available, else the raw gradient. Its norm and alignment with g are
        # what the theorem's inner and dn2 refer to.
        direction = self._optimizer_direction(params, grads)
        dn2 = sum(float(d.double().pow(2).sum()) for d in direction)
        inner = sum(float((d * g).double().sum()) for d, g in zip(direction, grads))
        if dn2 <= 0.0:
            return {}

        # Ceiling on the analytic step. Taken from the CONFIGURED base LR,
        # never from the schedule's current value: during linear warmup the
        # schedule returns exactly 0, and a ceiling of 0 would be skipped,
        # letting the unclipped eta* through precisely when the step most
        # needs the guard. That is what took the first A/B run to CE 21.2.
        eta = select_step(
            inner=inner,
            dn2=dn2,
            smoothness=smoothness,
            base_lr=float(self.cfg.train.learning_rate),
            clip_to_lr=bool(getattr(self.cfg.train, "analytic_step_clip_lr", True)),
        )
        # Warmup still applies to the analytic step as a multiplier on the
        # ceiling, so the first steps cannot be large even if eta* would be.
        eta = min(eta, base_lr) if base_lr > 0.0 else 0.0
        for g in self.optimizer.param_groups:
            if not g.get("_muon"):
                g["lr"] = eta
        self._analytic_step_last = {
            "analytic_eta": eta,
            "analytic_L": smoothness,
            "analytic_inner": inner,
            "analytic_dn2": dn2,
        }
        return self._analytic_step_last

    def _optimizer_direction(
        self, params: list[torch.Tensor], grads: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """The direction the optimizer actually applies, per parameter.

        For AdamW that is the first-moment exponential moving average
        (``exp_avg``): the bias-corrected update is proportional to it, so
        it is the direction whose alignment with the current gradient the
        descent certificate is about. Before the first step there is no
        ``exp_avg`` and the direction is the raw gradient.
        """
        # HybridOptimizer is not an Optimizer: the per-parameter state for the
        # non-Muon groups lives on the wrapped AdamW.
        adam_state = self.optimizer.adamw.state
        direction: list[torch.Tensor] = []
        for p, g in zip(params, grads):
            buf = adam_state.get(p, {}).get("exp_avg")
            if buf is None:
                direction.append(g)
            else:
                direction.append(buf)
        return direction

    def _gram_scan_samples(self) -> list[tuple[str, dict]]:
        """Lazily read one tail calibration window per configured corpus.

        Windows are read from ``data/<name>.compact.bin`` tails (the same
        last-2M convention as the gate), cached for the run. Kept tiny on
        purpose: this is a measurement channel, not a data source.
        """
        if self._gram_samples is not None:
            return self._gram_samples
        import numpy as np

        from pathlib import Path

        tokens = int(self.cfg.train.gram_scan_tokens)
        samples: list[tuple[str, dict]] = []
        missing: list[str] = []
        for name in self.cfg.train.gram_scan_corpora:
            path = Path(self.cfg.train.data.data_dir) / f"{name}.compact.bin"
            if not path.is_file():
                missing.append(str(path))
                continue
            total = path.stat().st_size // 4
            with path.open("rb") as fh:
                fh.seek(max(total - 2_000_000, 0) * 4)
                raw = np.frombuffer(fh.read(tokens * 4 + 4), dtype=np.uint32).astype(np.int64)
            ids = torch.from_numpy(raw[: tokens + 1])
            samples.append(
                (name, {"input_ids": ids[None, :-1], "targets": ids[None, 1:]})
            )
        if missing:
            raise FileNotFoundError(
                "gram_scan_corpora: missing corpus files: " + ", ".join(missing)
            )
        self._gram_samples = samples
        return samples

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
        # Gradient-Gram scan (Dominate.lean/SafeQP.lean, log-only): every
        # gram_scan_interval steps, before the step's own gradients exist.
        # Failures cost one log line, never the step; grads are zeroed after
        # so the scan cannot leak into the update.
        gram_metrics: dict = {}
        interval = int(getattr(cfg.train, "gram_scan_interval", 0))
        if interval > 0 and self.step % interval == 0:
            try:
                scan = corpus_grad_gram(
                    model,
                    self._gram_scan_samples(),
                    device=next(model.parameters()).device,
                    eps_rel=float(cfg.train.gram_scan_eps_rel),
                )
                self._gram_metrics_last = scan
                gram_metrics = {
                    "gram_min_cos": scan["min_cos"],
                    "gram_conflicts": scan["n_conflicts"],
                    "gram_domination": scan["domination_share"],
                    "gram_active": int(scan.get("safe_qp_certificate", {}).get("active", 0)),
                }
                logger.info(
                    "step %d gram-scan: min_cos=%.4f conflicts=%d domination=%.3f active=%d",
                    self.step,
                    scan["min_cos"],
                    scan["n_conflicts"],
                    scan["domination_share"],
                    gram_metrics["gram_active"],
                )
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as exc:  # measurement channel must not kill training
                logger.warning("step %d gram-scan failed: %s", self.step, exc)

        model, cfg = self.model, self.cfg
        model.train()
        device = next(model.parameters()).device
        self.optimizer.zero_grad(set_to_none=True)
        # Decision-only is an explicit pre-registered experimental mode: the
        # caller supplies decision labels and omits LM targets. It is not a
        # fallback; ``receiver='decision_only'`` and the independent decision
        # denominator make that choice observable. Empty scored steps remain
        # fail-closed above.
        adaptive_only = cfg.train.adapt.freeze_base
        use_ternary_cache = (
            cfg.model.ternary.enabled
            and not adaptive_only
            and getattr(cfg.train, "ternary_step_cache", True)
        )
        if use_ternary_cache:
            cache_ternary_weights(model)

        # Puncture the supervision stream (erasure channel on CE). Applied on
        # device after H2D so we do not rewrite host batches. Existing packing
        # masks are AND-ed in.
        keep_rate = float(getattr(cfg.train, "ce_keep_rate", 1.0))
        keep_mode = str(getattr(cfg.train, "ce_keep_mode", "bernoulli"))

        # Weight each microbatch by its scored-token count so unequal microbatches
        # average correctly rather than over-weighting sparse ones. Decision rows
        # use an independent denominator: the auxiliary objective must not change
        # when a sequence contains more language tokens.
        prepared: list[tuple[dict, torch.Tensor, torch.Tensor]] = []
        token_counts: list[torch.Tensor] = []
        decision_counts: list[torch.Tensor] = []
        for batch in microbatches:
            ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device) if "targets" in batch else None
            base_mask = batch["loss_mask"].to(device) if "loss_mask" in batch else None
            # Optional vocab slicing (HAGI_VOCAB_SLICE="i:N" from train.py):
            # a sliced leaf only scores positions whose TARGET is inside its
            # slice, so the loss never rewards it for predicting tokens owned
            # by another leaf. Unlike masking the logits (which explodes the CE
            # whenever an out-of-slice target appears), this only narrows what
            # is learned, exactly like the pre-existing puncture mask. The
            # INPUT stays full-vocabulary: every leaf still sees the whole
            # context, which is what concat assembly requires.
            slice_bounds = getattr(self, "slice_bounds", None)
            if slice_bounds is not None and targets is not None:
                lo, hi = slice_bounds
                in_slice = (targets >= lo) & (targets < hi)
                base_mask = in_slice if base_mask is None else base_mask & in_slice
            if targets is not None:
                mask = puncture_loss_mask(
                    tuple(targets.shape),
                    rate=keep_rate,
                    mode=keep_mode,
                    step=self.step,
                    device=device,
                    base=base_mask,
                )
                count = (
                    mask.sum(dtype=torch.int64)
                    if mask is not None
                    else torch.tensor(targets.numel(), device=device, dtype=torch.int64)
                )
            else:
                mask = None
                count = torch.zeros((), device=device, dtype=torch.int64)
            decision_targets = batch.get("decision_targets")
            if decision_targets is not None:
                decision_targets = decision_targets.to(device)
            decision_mask = batch.get("decision_mask")
            if decision_mask is not None:
                decision_mask = decision_mask.to(device)
            if decision_targets is not None:
                decision_count = (
                    decision_mask.sum(dtype=torch.int64)
                    if decision_mask is not None
                    else torch.tensor(decision_targets.numel(), device=device, dtype=torch.int64)
                )
            else:
                decision_count = torch.zeros((), device=device, dtype=torch.int64)
            token_counts.append(count)
            decision_counts.append(decision_count)
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
                        "decision_targets": decision_targets,
                        "decision_mask": decision_mask,
                    },
                    count,
                    decision_count,
                )
            )
        total_tokens = torch.stack(token_counts).sum()
        total_decisions = torch.stack(decision_counts).sum()
        total_tokens_normalized = total_tokens.clamp_min(1)
        total_decisions_normalized = total_decisions.clamp_min(1)
        if int(total_tokens) == 0 and int(total_decisions) == 0:
            raise ValueError("train_step received no scored LM tokens or decision rows")

        ce_sum = torch.zeros((), device=device, dtype=torch.float32)
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        z_sum = torch.zeros((), device=device, dtype=torch.float32)
        decision_sum = torch.zeros((), device=device, dtype=torch.float32)
        exact_ce_value: float | None = None
        exact_interval = int(cfg.train.logging.exact_ce_interval)
        for microbatch_index, ((batch, count, decision_count),) in enumerate(
            zip(prepared, strict=True)
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
                decision_targets=batch["decision_targets"],
                decision_mask=batch["decision_mask"],
            )
            if (
                microbatch_index == 0
                and exact_interval > 0
                and self.step % exact_interval == 0
                and batch["targets"] is not None
            ):
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
            lm_weight = count.to(torch.float32) / total_tokens_normalized if output.lm_loss is not None else None
            # NCEExact anchor (round 30, anchor_drift_bound): the periodic
            # exact-CE gradient pins the sampled receiver's GLOBAL calibration
            # (the sampled loss leaves the out-of-sample logits free to
            # drift -- the round-28 collapse: exact 53 vs NCE 1.7). Computed
            # BEFORE the main backward and added to the objective so ONE
            # backward covers both.
            anchor_w = float(getattr(cfg.train, "nce_anchor_weight", 0.0) or 0.0)
            anchor_ce = None
            if (
                anchor_w > 0
                and microbatch_index == 0
                and exact_interval > 0
                and self.step % exact_interval == 0
                and batch["targets"] is not None
            ):
                flat_hidden = output.hidden.reshape(-1, output.hidden.shape[-1])
                flat_targets = batch["targets"].reshape(-1)
                rows = min(int(cfg.train.logging.exact_ce_rows), flat_targets.numel())
                generator = torch.Generator(device=device)
                generator.manual_seed(int(cfg.train.logging.exact_ce_seed) + self.step)
                sample = torch.randperm(flat_targets.numel(), device=device, generator=generator)[:rows]
                anchor_ce = model.head.exact_loss(
                    flat_hidden.index_select(0, sample),
                    flat_targets.index_select(0, sample),
                )
                self._anchor_ce_last = float(anchor_ce.detach())
            decision_weight = (
                decision_count.to(torch.float32) / total_decisions_normalized
                if output.decision_loss is not None
                else None
            )
            objective = None
            if output.lm_loss is not None:
                objective = output.lm_loss * lm_weight
            if output.decision_loss is not None:
                weighted = float(cfg.model.decision.loss_weight) * output.decision_loss * decision_weight
                objective = weighted if objective is None else objective + weighted
            if anchor_ce is not None:
                objective = (objective + anchor_w * anchor_ce) if objective is not None else anchor_w * anchor_ce
            if objective is not None:
                objective.backward()
                loss_sum = loss_sum + objective.detach().float()
                if output.ce is not None and lm_weight is not None:
                    ce_sum = ce_sum + output.ce.detach().float() * lm_weight
                if output.z_loss is not None and lm_weight is not None:
                    z_sum = z_sum + output.z_loss.detach().float() * lm_weight
                if output.decision_loss is not None and decision_weight is not None:
                    decision_sum = decision_sum + output.decision_loss.detach().float() * decision_weight
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
        # Analytic step (optimal_step_unconstrained): when enabled, replace the
        # schedule's Adam LR with eta* = <g,d>/(L*||d||^2) computed from the
        # measured curvature. ``lr`` stays a CEILING, so the first deployment
        # can never be more aggressive than the tuned baseline in one step.
        analytic_metrics: dict = {}
        if self._curvature is not None:
            analytic_metrics = self._apply_analytic_step(
                adam_lr, prepared[0][0] if prepared else None
            )
        self.optimizer.step()

        if use_ternary_cache:
            clear_ternary_weights(model)

        if not adaptive_only and hasattr(model, "commit_controller_updates"):
            model.commit_controller_updates()

        # Metrics cross the device boundary once, after all scheduled GPU work.
        # Previously every float()/int() below synchronized the HIP stream.
        loss_value, ce_value, z_value, tokens_value, decision_value, decisions_value = torch.stack(
            (loss_sum, ce_sum, z_sum, total_tokens.to(torch.float32), decision_sum, total_decisions.to(torch.float32))
        ).tolist()
        has_lm_objective = tokens_value > 0
        receiver = (
            "conditional_nce"
            if cfg.model.head.sampled_softmax_k > 0
            else "exact_ce"
        )
        if not has_lm_objective and decisions_value > 0:
            receiver = "decision_only"
        if not has_lm_objective:
            ce_value = None
        metrics = {
            "step": self.step,
            "loss": loss_value,
            "ce": ce_value,
            "bpt": ce_value / math.log(2.0) if ce_value is not None else None,
            "ppl": math.exp(min(ce_value, 20.0)) if ce_value is not None else None,
            "z_loss": z_value,
            "grad_norm": body_norm,
            "body_grad_norm": body_norm,
            "rest_grad_norm": rest_norm,
            "lr": adam_lr,
            "muon_lr": muon_lr,
            **analytic_metrics,
            "tokens": int(tokens_value),
            "n_decisions": int(decisions_value),
            "decision_loss": decision_value if decisions_value > 0 else None,
            "ce_keep_rate": keep_rate,
            "receiver": receiver,
            "update_applied": True,
        }
        if gram_metrics:
            metrics.update(gram_metrics)
        if receiver == "decision_only":
            metrics["lm_ce"] = None
        elif receiver == "conditional_nce":
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
    if receiver == "decision_only":
        return (
            f"step {metrics['step']} | decision_loss="
            f"{metrics.get('decision_loss', 0.0):.4f} | decisions={metrics.get('n_decisions', 0)}"
        )
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
    if "analytic_eta" in metrics:
        parts.append(f"eta={metrics['analytic_eta']:.3e}")
        parts.append(f"L={metrics['analytic_L']:.3e}")
    parts.append(f"kl={metrics.get('kl', 0.0):.4f}")
    return " | ".join(parts)


def train(
    model: nn.Module,
    dataloader,
    cfg: Config,
    start_step: int = 0,
    optimizer_state: dict | None = None,
    start_offset: int = 0,
) -> Iterator[dict]:
    """Run training, yielding metrics at the configured interval.

    Checkpoints are written by step count; the caller only logs.

    ``start_offset`` is the token offset this run began at (the previous
    expert's consumed tokens). It is added to the run's own consumed tokens
    when writing ``consumed.json`` so the file always holds the *cumulative*
    offset for the next expert, not just this slice's consumption.
    """
    from hagi.train.checkpoint import save_checkpoint

    configure_runtime()
    trainer = Trainer(model, cfg, start_step)
    # Vocab slicing (see scripts/train.py): only positions whose target is
    # inside [lo, hi) are scored, so each leaf learns its own token slice.
    slice_bounds = None
    slice_spec = os.environ.get("HAGI_VOCAB_SLICE", "")
    if slice_spec:
        index_str, count_str = slice_spec.split(":")
        si, sn = int(index_str), int(count_str)
        v = int(cfg.model.vocab_size)
        slice_bounds = (si * v // sn, (si + 1) * v // sn)
    trainer.slice_bounds = slice_bounds
    if optimizer_state is not None:
        trainer.load_optimizer_state(optimizer_state)

    accum = cfg.train.grad_accum_steps

    def _write_consumed() -> None:
        """Record how many tokens this expert consumed, for the next expert's
        start offset. consumed = steps * batch * seq * accum (tokens actually
        used for training). Written to ``consumed.json`` in the checkpoint dir.
        """
        import json
        from pathlib import Path
        consumed = start_offset + trainer.step * cfg.train.batch_size * cfg.train.data.seq_len * accum
        # The pipeline's ``_consumed`` reads the cumulative offset from the
        # corpus root (the parent of the slice dir), so that file must stay
        # where it is. It was, however, the ONLY record of what a run
        # consumed, and every run sharing a corpus root overwrote it -- the
        # M2 arms' consumption evidence was destroyed that way, leaving the
        # train/holdout disjointness claim with no artifact to check.
        # A per-run record next to the checkpoints fixes that without moving
        # the file the pipeline depends on.
        out = Path(cfg.train.checkpoint_dir).parent / "consumed.json"
        per_run = Path(cfg.train.checkpoint_dir) / "consumed.json"
        payload = json.dumps({"consumed_tokens": int(consumed), "step": int(trainer.step)})
        for path in (out, per_run):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        logger.info("consumed %d tokens (step %d) -> %s", consumed, trainer.step, out)
    data_iter = iter(dataloader)

    # Saturation early-stop: track the best exact_ce and stop once it has not
    # improved by more than ``saturation_tol`` over ``saturation_patience``
    # logged samples. This is the method's "saturate" step made concrete.
    patience = int(cfg.train.saturation_patience)
    tol = float(cfg.train.saturation_tol)
    min_steps = int(cfg.train.saturation_min_steps)
    best_exact_ce: float | None = None
    no_improve_count = 0

    # Divergence guard: stop once running ce rises ``divergence_delta`` above
    # its own running minimum for ``divergence_patience`` consecutive logged
    # samples. The round-54 boost-joint incident (ce 3.0 -> 4.0 at fixed lr)
    # was caught by a human reading the log; this makes it a controller rule.
    div_patience = int(cfg.train.divergence_patience)
    div_delta = float(cfg.train.divergence_delta)
    div_min_steps = int(cfg.train.divergence_min_steps)
    min_run_ce: float | None = None
    diverged_count = 0

    # Fixed-window gate evaluation (round 61): deterministic tail-window
    # batch per corpus, captured lazily on first use, weighted with the
    # canonical corpus weights. A STABLE control signal -- unlike the running
    # ce, which drifts with the data stream (round-54 false-alarm lesson).
    gate_interval = int(cfg.train.logging.gate_eval_interval)
    gate_batches: list | None = None

    def _gate_ce(model) -> float:
        import numpy as np  # local: only needed when the gate is enabled
        nonlocal gate_batches
        if gate_batches is None:
            gate_batches = []
            weights = []
            data_dir = Path(cfg.train.data.data_dir)
            for name, w in sorted((cfg.train.data.weights or {}).items()):
                p = data_dir / f"{name}.compact.bin"
                if not p.exists():
                    continue
                total = p.stat().st_size // 4
                start = max(0, total - 2_000_000)
                with p.open("rb") as fh:
                    fh.seek(start * 4)
                    T = np.frombuffer(fh.read(2048 * 4), dtype=np.uint32).astype(np.int64)
                ids = torch.from_numpy(T[:2048]).reshape(2, 1024)
                gate_batches.append((ids[:, :-1], ids[:, 1:]))
                weights.append(float(w))
            wsum = sum(weights) or 1.0
            weights = [w / wsum for w in weights]
            gate_batches.append(weights)
        weights = gate_batches[-1]
        model.eval()
        total = 0.0
        with torch.no_grad():
            for (x, y), w in zip(gate_batches[:-1], weights):
                x = x.to(next(model.parameters()).device)
                y = y.to(x.device)
                o = model(x, y)
                f = o.hidden.reshape(-1, o.hidden.shape[-1])
                total += w * float(model.head.exact_loss(f, y.reshape(-1)))
        model.train()
        return total

    while trainer.step < cfg.train.max_steps:
        microbatches = []
        for _ in range(accum):
            try:
                microbatches.append(next(data_iter))
            except StopIteration:
                # The corpus slice is exhausted (every source stream hit EOF).
                # This is a legitimate stop condition — the expert has consumed
                # all remaining data. Save the checkpoint and stop, so the
                # pipeline can start the next expert from the new offset.
                logger.info(
                    "data exhausted at step %d; saving checkpoint and stopping",
                    trainer.step,
                )
                _write_consumed()
                save_checkpoint(
                    model,
                    cfg,
                    trainer.step,
                    cfg.train.checkpoint_dir,
                    cfg.train.checkpoint_keep_last,
                    optimizer=trainer.optimizer,
                )
                return

        step_index = trainer.step
        metrics = trainer.train_step(microbatches)

        if step_index % max(1, cfg.train.logging.log_interval) == 0:
            # Log independently of the consumer: generator-driven callers
            # (scripts/lora/*) used to run silent for the whole run — the
            # empty-log incident of round 51. logger writes to stderr, which
            # launchers redirect alongside stdout, so progress is observable
            # in the log file from the first interval on.
            logger.info("%s", format_metrics(metrics))
            yield metrics

        # Fixed-window gate CE: the STABLE control signal for divergence and
        # saturation diagnosis (and for the T* fit -- traininfer_critical).
        if gate_interval > 0 and trainer.step % gate_interval == 0:
            gce = _gate_ce(model)
            logger.info("gate_ce step %d: %.4f", trainer.step, gce)
            metrics["gate_ce"] = gce

        # Saturation check on the exact_ce (the coding-cost SSOT). Only
        # evaluated when exact_ce is actually measured (exact_ce_interval>0).
        if patience > 0 and "exact_ce" in metrics and trainer.step >= min_steps:
            ce = float(metrics["exact_ce"])
            if best_exact_ce is None or ce < best_exact_ce - tol:
                best_exact_ce = ce
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(
                        "saturation: exact_ce %.4f not improved by %.4f over %d samples; stopping at step %d",
                        ce,
                        tol,
                        patience,
                        trainer.step,
                    )
                    break

        # Divergence guard on the running ce (logged every interval, unlike
        # exact_ce). Stops the run before divergence burns the remaining GPU
        # budget; the last good checkpoint stays from checkpoint_interval.
        div_key = "gate_ce" if gate_interval > 0 and "gate_ce" in metrics else "ce"
        if div_patience > 0 and div_key in metrics and trainer.step >= div_min_steps:
            run_ce = float(metrics[div_key])
            if min_run_ce is None or run_ce < min_run_ce:
                min_run_ce = run_ce
                diverged_count = 0
            elif run_ce > min_run_ce + div_delta:
                diverged_count += 1
                if diverged_count >= div_patience:
                    logger.warning(
                        "divergence: ce %.4f exceeds running min %.4f by >%.2f for %d samples; stopping at step %d",
                        run_ce,
                        min_run_ce,
                        div_delta,
                        div_patience,
                        trainer.step,
                    )
                    break
            else:
                diverged_count = 0

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

    _write_consumed()
    save_checkpoint(
        model,
        cfg,
        trainer.step,
        cfg.train.checkpoint_dir,
        cfg.train.checkpoint_keep_last,
        optimizer=trainer.optimizer,
    )
