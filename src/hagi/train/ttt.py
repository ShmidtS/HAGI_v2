"""Feature-driven test-time training: hidden-state features → LoRA delta.

``ARCHITECTURE.md`` names ``train/ttt.py`` as the online RLS/TTT adapter
contour. The algorithm reference is :meth:`scripts.qwen_ttt_lora.LowRankLoRA.rls_step`
(anchored ridge normal equations with an honest 1/5 holdout that never enters
the normal equations). That module lives in the experiments zone and is not
imported here, exactly as :mod:`hagi.model.adapters` cites it without
importing it.

Why features instead of generation
----------------------------------
The gradient-descent contour (:mod:`hagi.train.self_improve`) needs a
trajectory, and a trajectory costs one autoregressive forward per token
(``_generate_trajectory`` calls ``generate(..., use_cache=True)`` with
``min_new_tokens=n_new_tokens``). This contour never generates: a *single*
teacher-forced pass over already-known tokens yields, for every block, the
exact local error

    dL/d(adapter delta) == dL/d(block output)

:meth:`hagi.model.block.Block.forward` adds the adapter delta to the block
output (``x = x + adapter(mixer_input, x)``), so those two gradients are the
same tensor derivative — an identity, not an approximation. Feeding
``(mixer_input, target)`` rows to the ridge solve fits ``lora_B`` in closed
form, so one update is one forward, one backward and an ``r x r`` solve with
``r`` = LoRA rank. No sampling, no second scoring call, no optimizer state.

Scale of the target
-------------------
``LMHead`` reduces CE as a mean over scored positions (``ce =
-log_probs[:, 0].mean()``), so the raw per-position gradient is ~1/N of the
individual loss gradient and is tiny relative to the residual stream
(measured ~5.7e-5 of stream RMS on the tiny config). The target is therefore
expressed as a fraction of stream RMS (``stream_frac``), which removes the CE
reduction convention from the scale. It does **not** make the achieved step
independent of sequence length, and it is a *nominal* target rather than the
effective one:

* ``prior`` is an absolute pseudo-row count ported from ``TTT_ALPHA``, while
  the data it competes with is ``phi^T phi`` over the rows in the window. On an
  RMSNorm'd stream (measured ``rms(mixer_input) = 0.988``) a 52-row window
  contributes ~52 against a prior of 1000, so the solve is shrunk ~20x.
  Measured: ``delta_rms_frac`` lands at 1.04e-3 -> 1.87e-3 -> 2.56e-3 over
  three refits against ``stream_frac = 0.02``, and *rises* as rows accumulate.
  With ``prior = 1.0`` the same run gives 9.77e-3.
* Whether a step writes anything at all depends on rows-per-step versus
  ``refit_rows``. Measured at ``refit_rows = 64``: an 8-token window solves
  every ~5th step (2/10), a 32-token window every ~3rd (3/10), a >=128-token
  window every step (10/10). Between refits a step still pays a full
  forward+backward and writes nothing.

So ``stream_frac`` bounds the intent, not the result. The shrinkage is a
**warmup cost, not a mis-calibration**: ``prior`` is a steady-state anchor, and
at ``lam=0.9995``/``refit_rows=64`` the effective row count in ``G`` converges
to ~2032, against which a prior of 1000 is kappa ~= 0.49 -- the reference's
intent. Measured over 60 steps against a ``prior=1`` ceiling: frac climbs
3.29e-3 -> 5.35e-3 -> 6.80e-3 (ceiling 7.87e-3) and dCE recovers +9.01e-4 ->
+1.50e-3 -> +1.92e-3 (ceiling +2.23e-3), i.e. 40% of the signal at step 12 and
86% by step 60. A long window reaches steady state in a handful of steps
(~410 rows/step at seq=512 -> ~5 steps), so the default is kept: shortening
the warmup by weakening ``prior`` would trade away the steady-state anchor the
reference sized it for. What matters operationally is that a short-window
caller is in the warmup regime, and any benchmark run there (this repo's
real-dims sweep used T=32) measures the shrunk step, so it reads conservative
rather than optimistic.

Frozen base invariant
---------------------
Only ``lora_B`` is written. ``lora_A`` stays a frozen orthonormal buffer and no
base weight tensor is touched, so the base remains a pure function.

Step bound
----------
The ridge optimum is the *best fit* to the targets it was shown. On top of
that, ``max_delta_rms_frac`` caps the applied delta's RMS relative to the
residual stream it is added to, so a pathological gradient cannot let the
adapter dominate the base.

It is a backstop, not an active limiter at the shipped settings: measured
``delta_rms_frac`` stays in 3.3e-3 .. 8.2e-3 against the 0.10 cap, because the
absolute ``prior`` already shrinks the step far below ``stream_frac`` (see
"Scale of the target"). It binds only when ``stream_frac`` is raised or
``prior`` lowered -- which is exactly the configuration a caller tuning for a
bigger step would choose, so the guard is there for that caller rather than for
the default one. The non-finite case is handled earlier and louder: ``rls_step``
refuses non-finite rows outright rather than letting a cap scale them down.

Harvest is deliberately separated from fitting (:meth:`TttRls._harvest`
produces rows, :meth:`TttRls.rls_step` consumes them) because the
holdout-exclusion property cannot be tested through the model alone:
perturbing a holdout position's tokens changes the gradients of *earlier*
positions, since an earlier block output is read as KV by later positions.
Tests therefore also feed rows directly.
"""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, autograd, no_grad

from hagi.model.adapters import TttLoraAdapter
from hagi.model.model import HAGI


@dataclass
class _BlockRls:
    """Per-adapter anchored-RLS accumulators for one block.

    ``G`` holds ``phi^T phi`` plus the ``prior`` ridge on its diagonal; ``C``
    holds ``phi^T y`` for the *unscaled* target (``y = delta / scaling``), so a
    solved ``B`` reproduces the applied delta rather than an arbitrarily
    scaled product of it. ``phi_buf``/``y_buf`` are the capped training-row
    window the residual gate is measured on.
    """

    G: Tensor
    C: Tensor
    train_rows: int = 0
    refits: int = 0
    phi_buf: list[Tensor] = field(default_factory=list)
    y_buf: list[Tensor] = field(default_factory=list)
    buf_rows: int = 0
    holdout_rows: int = 0


@dataclass
class TttStats:
    """Outcome of one feature→delta adaptation step.

    The row counts are the observable proof of the holdout split: rows counted
    in ``holdout_rows`` never appear in ``train_rows``, so a caller can check
    that validation data stayed out of the normal equations. The
    no-autoregression invariant is asserted behaviourally (encoder call count)
    rather than through a field here -- a field defaulted to zero cannot fail,
    so it proves nothing.
    """

    ce: float
    blocks: int
    blocks_updated: int
    refits: int
    train_rows: int
    holdout_rows: int
    delta_rms_frac: float


def _snapshot_training_modes(
    model: HAGI,
) -> tuple[tuple[torch.nn.Module, bool], ...]:
    """Capture every module's own training flag before a recursive mode switch.

    ``nn.Module.train`` may be called selectively on a child, so a mixed
    train/eval module tree is a supported caller state. Saving only the root
    flag and later calling ``model.train(root_flag)`` would silently normalize
    that tree. Keep object identity plus the original flag for exact restore.
    """
    return tuple((module, module.training) for module in model.modules())


def _restore_training_mode(
    model: HAGI,
    training_modes: tuple[tuple[torch.nn.Module, bool], ...],
) -> None:
    """Restore a complete module-tree mode snapshot without masking failures.

    The public ``train()`` path remains the normal operation. Exact per-module
    assignment follows it so selective child modes survive even if the public
    recursive switch is overridden, partially fails, or normalizes the tree.
    """
    try:
        model.train(training_modes[0][1])
    except BaseException:
        pass
    for module, was_training in training_modes:
        module.training = was_training


def _remove_harvest_hooks(
    handles: Sequence[torch.utils.hooks.RemovableHandle],
) -> None:
    """Detach harvest hooks even when one ``remove()`` call fails.

    ``remove()`` is the public PyTorch contract and is tried first for every
    handle. A failure is contained so one bad handle cannot strand the remaining
    hooks or replace the primary harvest exception. For a failed removal, the
    handle's own weak dictionary reference and id provide the exact fallback
    deletion used by PyTorch; cleanup is best effort by design, so fallback
    failures are contained too.
    """
    for handle in handles:
        try:
            handle.remove()
            continue
        except BaseException:
            pass
        try:
            hooks_dict = handle.hooks_dict_ref()
            if hooks_dict is not None:
                hooks_dict.pop(handle.id, None)
            for ref in handle.extra_dict_ref:
                extra_dict = ref()
                if extra_dict is not None:
                    extra_dict.pop(handle.id, None)
        except BaseException:  # pragma: no cover - defensive
            pass


def _resid(B: Tensor, phi: Tensor, y: Tensor) -> float:
    """Relative squared residual of the fit ``phi @ B.T ~= y``."""
    pred = phi @ B.to(dtype=phi.dtype).T
    return ((pred - y).pow(2).sum() / y.pow(2).sum().clamp_min(1e-12)).item()


def _state_digest(
    g: Tensor,
    c: Tensor,
    lora_b: Tensor,
    phi_buf: Sequence[Tensor],
    y_buf: Sequence[Tensor],
    counters: tuple[int, int, int, int],
) -> str:
    """Hash every mutable value a snapshot carries, counters included.

    ``G``, ``C`` and ``lora_B`` come back as clones, and the row tensors are kept
    by reference because cloning a whole training window costs far more than the
    state the snapshot protects. Mixed representation means shape validation alone
    cannot tell an untouched snapshot from one that was modified after the fact,
    so bytes are hashed instead. The four plain-int counters need the same
    treatment: type and non-negative checks cannot distinguish an honest value
    from an edited one. Follows the byte-level digest already used by
    :func:`hagi.orchestrator.state.state_key_digest`.
    """
    digest = hashlib.sha256()
    digest.update(b"hagi-rls-state-v2\0")
    digest.update(repr(counters).encode("ascii"))
    digest.update(b"\0")
    for tensor in (g, c, lora_b, *phi_buf, *y_buf):
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()



_SNAPSHOT_HARD_MAX_BYTES = 256 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}")


class TttRls:
    """Online anchored-RLS fitter for a HAGI model's TTT-LoRA adapters.

    Args:
        model: a :class:`~hagi.model.model.HAGI` whose blocks carry a
            ``ttt_lora`` adapter.
        stream_frac: nominal target magnitude as a fraction of residual-stream
            RMS. Not the achieved step size -- the absolute ``prior`` shrinks it
            by ~20x at default ``refit_rows`` on an RMSNorm'd stream, and the
            low-rank projection captures only the reachable part. See "Scale of
            the target".
        reg: ridge relative to the mean diagonal of ``G`` (mirrors ``TTT_REG``).
        lam: forgetting factor, applied as ``lam ** n_rows`` per step.
        refit_rows: solve once this many training rows accumulate.
        prior: initial diagonal of ``G`` (mirrors ``TTT_ALPHA``).
        max_delta_rms_frac: cap on applied-delta RMS / residual-stream RMS.
        rows_max: cap on the retained training-row window.
        snapshot_max_bytes: hard cap for tensor data cloned for transactional
            rollback (``G``, ``C`` and ``lora_B``). Row buffers are retained by
            reference because the fitter only appends/removes immutable rows.

    Raises:
        ValueError: if a block lacks a ``ttt_lora`` adapter, or on a
            non-finite or out-of-range hyperparameter.
    """

    def __init__(
        self,
        model: HAGI,
        *,
        stream_frac: float = 0.02,
        reg: float = 1e-3,
        lam: float = 0.9995,
        refit_rows: int = 64,
        prior: float = 1000.0,
        max_delta_rms_frac: float = 0.10,
        rows_max: int = 2048,
        snapshot_max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        for name, val in (
            ("stream_frac", stream_frac),
            ("reg", reg),
            ("lam", lam),
            ("prior", prior),
            ("max_delta_rms_frac", max_delta_rms_frac),
        ):
            if not torch.isfinite(torch.tensor(float(val))):
                raise ValueError(f"{name} must be finite, got {val!r}")
        if stream_frac <= 0.0:
            raise ValueError(f"stream_frac must be > 0, got {stream_frac!r}")
        if not 0.0 < lam <= 1.0:
            raise ValueError(f"lam must be in (0, 1], got {lam!r}")
        if refit_rows < 1:
            raise ValueError(f"refit_rows must be >= 1, got {refit_rows!r}")
        if max_delta_rms_frac <= 0.0:
            raise ValueError(
                f"max_delta_rms_frac must be > 0, got {max_delta_rms_frac!r}"
            )
        if rows_max < 1:
            raise ValueError(f"rows_max must be >= 1, got {rows_max!r}")
        if type(snapshot_max_bytes) is not int or snapshot_max_bytes < 1:
            raise ValueError("snapshot_max_bytes must be a positive exact int")
        if snapshot_max_bytes > _SNAPSHOT_HARD_MAX_BYTES:
            raise ValueError(
                f"snapshot_max_bytes must not exceed {_SNAPSHOT_HARD_MAX_BYTES}"
            )
        # Both are amounts of anchoring, and zero is not "no regularisation" but
        # a different estimator: with prior=0 and fewer rows than rank, ``G`` is
        # singular and the solve throws mid-run. Reject it at construction like
        # the five hyperparameters above already do.
        if reg <= 0.0:
            raise ValueError(f"reg must be > 0, got {reg!r}")
        if prior <= 0.0:
            raise ValueError(f"prior must be > 0, got {prior!r}")

        self._model = model
        self.stream_frac = float(stream_frac)
        self.reg = float(reg)
        self.lam = float(lam)
        self.refit_rows = int(refit_rows)
        self.prior = float(prior)
        self.max_delta_rms_frac = float(max_delta_rms_frac)
        self.rows_max = int(rows_max)
        self._snapshot_max_bytes = int(snapshot_max_bytes)
        self._rollback_poisoned = False

        self._lora: dict[int, TttLoraAdapter] = {}
        self._state: dict[int, _BlockRls] = {}
        self._mixer_to_block: dict[int, int] = {}
        self._block_to_index: dict[int, int] = {}
        for i, block in enumerate(model.blocks):
            adapter = getattr(block, "adapters", None)
            lora = getattr(adapter, "ttt_lora", None)
            if lora is None:
                raise ValueError(
                    f"block {i} has no ttt_lora adapter; enable "
                    "model.adapters.enabled and model.adapters.ttt_lora.enabled"
                )
            r = lora.r
            device = lora.lora_B.device
            self._lora[i] = lora
            self._mixer_to_block[id(block.mixer)] = i
            self._block_to_index[id(block)] = i
            self._state[i] = _BlockRls(
                G=torch.eye(r, dtype=torch.float32, device=device) * self.prior,
                C=torch.zeros(
                    r, lora.hidden_size, dtype=torch.float32, device=device
                ),
            )

    @property
    def block_count(self) -> int:
        return len(self._lora)

    @property
    def model(self) -> HAGI:
        """Return the model this fitter is bound to."""
        return self._model

    @property
    def snapshot_max_bytes(self) -> int:
        """Effective transactional-snapshot cap in bytes.

        Read-only by design: the 256 MiB ceiling is a safety limit, so a
        caller that could raise it after construction could also bypass the
        memory guard that makes rollback safe.
        """
        return self._snapshot_max_bytes

    @no_grad()
    def snapshot_state(self) -> tuple:
        """Capture every mutable field used by one adaptation transaction.

        ``G``, ``C`` and ``lora_B`` are updated in place and therefore cloned.
        Row tensors are detached but never mutated by the fitter, so their list
        entries are retained instead of copied. The cloned byte count is checked
        before allocation against the effective cap, which is the smaller of the
        configured ``snapshot_max_bytes`` and the absolute ``_SNAPSHOT_HARD_MAX_BYTES``
        ceiling. This adapts the PyTorch clone/copy contract documented at
        https://docs.pytorch.org/docs/2.14/generated/torch.clone.html
        """
        self._ensure_usable()
        entries = sorted(self._state.items())
        copied_bytes = sum(
            tensor.numel() * tensor.element_size()
            for index, state in entries
            for tensor in (state.G, state.C, self._lora[index].lora_B)
        )
        cap = min(self._snapshot_max_bytes, _SNAPSHOT_HARD_MAX_BYTES)
        if copied_bytes > cap:
            raise MemoryError(
                f"RLS rollback snapshot needs {copied_bytes} bytes; "
                f"limit is {cap}"
            )
        return tuple(
            (
                index,
                state.G.detach().clone(),
                state.C.detach().clone(),
                self._lora[index].lora_B.detach().clone(),
                state.train_rows,
                state.refits,
                state.buf_rows,
                state.holdout_rows,
                tuple(state.phi_buf),
                tuple(state.y_buf),
                _state_digest(
                    state.G, state.C, self._lora[index].lora_B,
                    state.phi_buf, state.y_buf,
                    (state.train_rows, state.refits, state.buf_rows, state.holdout_rows),
                ),
            )
            for index, state in entries
        )

    @no_grad()
    def restore_state(self, snapshot: tuple) -> None:
        """Restore a complete snapshot or poison the fitter fail-closed.

        The whole snapshot is validated before the first live tensor is
        changed. Any later copy/allocation failure leaves ``_rollback_poisoned``
        set, preventing accidental reuse of a partially restored fitter.
        """
        self._rollback_poisoned = True
        entries = sorted(self._state.items())
        if not isinstance(snapshot, tuple) or len(snapshot) != len(entries):
            actual = len(snapshot) if isinstance(snapshot, tuple) else type(snapshot).__name__
            raise ValueError(
                f"RLS snapshot has {actual} blocks; model has {len(entries)}"
            )
        for position, (index, state) in enumerate(entries):
            saved = snapshot[position]
            if not isinstance(saved, tuple) or len(saved) != 11:
                raise ValueError(f"invalid RLS snapshot entry for block {index}")
            (
                saved_index,
                saved_g,
                saved_c,
                saved_b,
                train_rows,
                refits,
                buf_rows,
                holdout_rows,
                phi_buf,
                y_buf,
                state_digest,
            ) = saved
            if saved_index != index:
                raise ValueError(f"RLS snapshot block order changed at {index}")
            for saved_tensor, live_tensor, name in (
                (saved_g, state.G, "G"),
                (saved_c, state.C, "C"),
                (saved_b, self._lora[index].lora_B, "lora_B"),
            ):
                if (
                    not isinstance(saved_tensor, torch.Tensor)
                    or saved_tensor.shape != live_tensor.shape
                    or saved_tensor.dtype != live_tensor.dtype
                    or saved_tensor.device != live_tensor.device
                ):
                    raise ValueError(f"RLS snapshot {name} mismatch for block {index}")
            counters = (train_rows, refits, buf_rows, holdout_rows)
            if not all(type(value) is int and value >= 0 for value in counters):
                raise ValueError(f"invalid RLS snapshot counters for block {index}")
            if not isinstance(phi_buf, tuple) or not isinstance(y_buf, tuple):
                raise ValueError(f"invalid RLS snapshot buffers for block {index}")
            if len(phi_buf) != len(y_buf):
                raise ValueError(f"RLS row buffer length mismatch for block {index}")
            lora = self._lora[index]
            for phi_row, y_row in zip(phi_buf, y_buf):
                # Geometry, not just container type: a restored row that does not
                # match the block's LoRA shape/dtype would only fail later, in
                # ``torch.cat`` or the solve, i.e. after this restore has already
                # reset the poison flag and made the fitter look usable again.
                if (
                    not isinstance(phi_row, torch.Tensor)
                    or not isinstance(y_row, torch.Tensor)
                    or phi_row.ndim != 2
                    or y_row.ndim != 2
                    or phi_row.shape[1] != lora.r
                    or y_row.shape[1] != lora.hidden_size
                    or phi_row.shape[0] != y_row.shape[0]
                    or phi_row.dtype != state.G.dtype
                    or y_row.dtype != state.G.dtype
                    or phi_row.device != state.G.device
                    or y_row.device != state.G.device
                ):
                    raise ValueError(
                        f"invalid RLS snapshot row geometry for block {index}"
                    )
            if not isinstance(state_digest, str) or _HEX64.fullmatch(state_digest) is None:
                raise ValueError(f"invalid RLS snapshot digest for block {index}")
            if state_digest != _state_digest(
                saved_g, saved_c, saved_b, phi_buf, y_buf, counters
            ):
                # The snapshot mixes clones (``G``/``C``/``lora_B``) and shared
                # row references, so an edit made after the snapshot was taken is
                # invisible to shape checks -- the values are simply wrong.
                # Refuse the restore rather than presenting tampered state as a
                # rolled-back fitter with the poison flag cleared.
                raise ValueError(
                    f"RLS snapshot digest mismatch for block {index}"
                )
            if buf_rows != sum(row.shape[0] for row in phi_buf):
                raise ValueError(f"RLS buffered row count mismatch for block {index}")


        for position, (index, state) in enumerate(entries):
            saved = snapshot[position]
            state.G.copy_(saved[1])
            state.C.copy_(saved[2])
            self._lora[index].lora_B.copy_(saved[3])
            state.train_rows = saved[4]
            state.refits = saved[5]
            state.buf_rows = saved[6]
            state.holdout_rows = saved[7]
            state.phi_buf = list(saved[8])
            state.y_buf = list(saved[9])
        self._rollback_poisoned = False

    def _ensure_usable(self) -> None:
        if self._rollback_poisoned:
            raise RuntimeError("ttt: fitter is poisoned after incomplete rollback")

    def _harvest(
        self,
        input_ids: Tensor,
        targets: Tensor,
        loss_mask: Tensor | None,
    ) -> tuple[float, dict[int, list[tuple[Tensor, Tensor]]]]:
        """Teacher-forced pass; return CE and per-block (feature, gradient) pairs.

        Hooks are used rather than a modified ``Block.forward`` so the default
        path stays bit-for-bit identical. ``feats[i]`` and ``outs[i]`` are
        parallel lists: a block forward calls its mixer exactly once, so index
        ``k`` of both refers to the same firing even when ``loop_depth``
        repeats the block.
        """
        model = self._model
        feats: dict[int, list[Tensor]] = {}
        outs: dict[int, list[Tensor]] = {}

        def _mixer_pre(
            module: torch.nn.Module, args: tuple[Tensor, ...]
        ) -> None:
            feats.setdefault(self._mixer_to_block[id(module)], []).append(args[0])

        def _block_out(
            module: torch.nn.Module, inputs: tuple[Tensor, ...], output: Tensor
        ) -> None:
            outs.setdefault(self._block_to_index[id(module)], []).append(output)

        handles: list[torch.utils.hooks.RemovableHandle] = []
        training_modes = _snapshot_training_modes(model)
        try:
            for block in model.blocks:
                handles.append(block.mixer.register_forward_pre_hook(_mixer_pre))
                handles.append(block.register_forward_hook(_block_out))
            model.eval()
            with torch.enable_grad():
                output = model(input_ids, targets, loss_mask=loss_mask)
                if output.ce is None:
                    raise RuntimeError("HAGI.forward returned no CE to adapt on")
                ce = output.ce
                keys = [(i, k) for i in sorted(outs) for k in range(len(outs[i]))]
                flat = [outs[i][k] for i, k in keys]
                grads = autograd.grad(ce, flat)
        finally:
            _remove_harvest_hooks(handles)
            _restore_training_mode(model, training_modes)

        rows: dict[int, list[tuple[Tensor, Tensor]]] = {}
        for (i, k), grad in zip(keys, grads, strict=True):
            rows.setdefault(i, []).append((feats[i][k].detach(), grad.detach()))
        return float(ce.detach()), rows

    @staticmethod
    def _flatten(t: Tensor, loss_mask: Tensor | None) -> Tensor:
        flat = t.reshape(-1, t.shape[-1])
        if loss_mask is None:
            return flat
        keep = loss_mask.reshape(-1).to(dtype=torch.bool)
        if keep.shape[0] != flat.shape[0]:
            raise ValueError(
                f"loss_mask rows {keep.shape[0]} != feature rows {flat.shape[0]}"
            )
        return flat[keep]

    def _rows_for(
        self,
        i: int,
        per_firing: list[tuple[Tensor, Tensor]],
        loss_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Turn (feature, gradient) firings into ``(phi, y)`` fit rows.

        The target is ``-stream_frac * (rms_h / rms_g) * g``: the negative
        gradient rescaled to a fixed fraction of residual-stream RMS, then
        divided by ``scaling`` so the solved ``B`` emits exactly that delta.
        """
        lora = self._lora[i]
        feat = torch.cat(
            [self._flatten(f, loss_mask) for f, _ in per_firing], dim=0
        ).to(dtype=torch.float32)
        grad = torch.cat(
            [self._flatten(g, loss_mask) for _, g in per_firing], dim=0
        ).to(dtype=torch.float32)
        rms_stream = feat.pow(2).mean().sqrt().clamp_min(1e-12)
        rms_grad = grad.pow(2).mean().sqrt().clamp_min(1e-12)
        target = -self.stream_frac * (rms_stream / rms_grad) * grad
        phi = feat @ lora.lora_A.to(dtype=torch.float32)
        return phi, target / float(lora.scaling)

    def _record(self, st: _BlockRls, phi: Tensor, y: Tensor) -> None:
        """Append an owned, capped residual-gate window."""
        # The public rls_step API accepts caller-owned tensors. Clone before
        # retaining them so a later in-place mutation cannot silently rewrite
        # rollback history or the live residual-gate window.
        if phi.shape[0] > self.rows_max:
            phi = phi[-self.rows_max :]
            y = y[-self.rows_max :]
        st.phi_buf.append(phi.detach().clone())
        st.y_buf.append(y.detach().clone())
        st.buf_rows += phi.shape[0]
        while st.buf_rows > self.rows_max and len(st.phi_buf) > 1:
            dropped = st.phi_buf.pop(0)
            st.y_buf.pop(0)
            st.buf_rows -= dropped.shape[0]

    @no_grad()
    def rls_step(
        self, i: int, phi: Tensor, y: Tensor, holdout: bool = True
    ) -> tuple[bool, float]:
        """Record ``(phi, y)`` rows for block ``i``; refit if the gate allows.

        The last 1/5 rows become validation and never enter ``G``/``C``,
        mirroring :meth:`scripts.qwen_ttt_lora.LowRankLoRA.rls_step`. The solve
        is applied only when it lowers the training residual.

        Returns:
            ``(updated, delta_rms_frac)`` — whether ``lora_B`` was written and
            the applied delta's RMS as a fraction of the training feature RMS.

        Raises:
            ValueError: on shape mismatch against the block's LoRA geometry.
        """
        self._ensure_usable()
        if phi.ndim != 2 or y.ndim != 2:
            raise ValueError("phi and y must be 2D")
        if phi.shape[0] != y.shape[0]:
            # The reference checks this (qwen_ttt_lora.py:267); without it a
            # length mismatch surfaces as a broadcast error deep in the solve,
            # or -- worse for [N,r]@[r,H] shapes that happen to agree -- as a
            # silently wrong fit.
            raise ValueError(
                f"feature/target row count mismatch: {phi.shape[0]} vs {y.shape[0]}"
            )
        lora = self._lora[i]
        st = self._state[i]
        if phi.shape[1] != lora.r:
            raise ValueError(f"expected phi dim {lora.r}, got {phi.shape[1]}")
        if y.shape[1] != lora.hidden_size:
            raise ValueError(
                f"expected y dim {lora.hidden_size}, got {y.shape[1]}"
            )
        n = phi.shape[0]
        if n == 0:
            return False, 0.0

        # Refuse non-finite rows here rather than at the residual gate. ``G``
        # and ``C`` are written before the gate runs, so a single
        # NaN anywhere in the inputs poisons the accumulators for the life of
        # the fitter: every later refit then proposes a non-finite candidate
        # that the gate rejects, and the block is frozen forever without
        # raising. This is the public entry point that writes them, so the
        # guard belongs at this level and nowhere above it.
        if not (torch.isfinite(phi).all() and torch.isfinite(y).all()):
            raise RuntimeError(
                f"ttt: non-finite feature/target rows for block {i}"
            )

        n_va = n // 5 if (holdout and n >= 5) else (1 if (holdout and n > 1) else 0)
        split = n - n_va
        tr_phi, tr_y = phi[:split], y[:split]

        if n_va:
            # Counted, not stored. The slice above already keeps these rows out
            # of ``G``/``C``; retaining copies grew without bound (the training
            # window is capped by ``rows_max``, this was not) and made a
            # whole-history re-cat plus a full ``[N,r]@[r,H]`` matmul per block
            # per step, for a residual no caller read.
            st.holdout_rows += n_va

        nt = tr_phi.shape[0]
        decay = self.lam**nt
        # Check the products before touching persistent accumulators. Finite
        # inputs can still overflow in float32 (for example, squared features
        # around 1e20), and committing an inf/NaN here would permanently poison
        # the fitter without setting the rollback flag.
        gram = tr_phi.T @ tr_phi
        cross = tr_phi.T @ tr_y
        if not (torch.isfinite(gram).all() and torch.isfinite(cross).all()):
            raise RuntimeError(f"ttt: non-finite RLS normal equations for block {i}")
        st.G.mul_(decay).add_(gram)
        st.C.mul_(decay).add_(cross)
        st.train_rows += nt
        self._record(st, tr_phi, tr_y)

        if st.train_rows < self.refit_rows:
            return False, 0.0

        st.train_rows = 0
        st.refits += 1

        Hb = torch.cat(st.phi_buf)
        Yb = torch.cat(st.y_buf)
        ridge = st.G.diagonal().mean().clamp_min(1e-12) * self.reg
        eye = torch.eye(st.G.shape[0], dtype=st.G.dtype, device=st.G.device)
        B_cand = torch.linalg.solve(st.G + ridge * eye, st.C).T.contiguous()

        # The reference gates apply-positive (``candidate < current``,
        # scripts/qwen_ttt_lora.py:324). This port rewrote it as a reject
        # branch, and the two are NOT equivalent under NaN: ``nan >= x`` is
        # False, so a non-finite candidate skipped the reject and ``copy_``
        # wrote it into the weights. Keep the reference's form.
        current = _resid(lora.lora_B, Hb, Yb)
        candidate = _resid(B_cand, Hb, Yb)
        if not (candidate < current):
            return False, 0.0

        lora.lora_B.copy_(B_cand.to(dtype=lora.lora_B.dtype))
        B = lora.lora_B.to(dtype=torch.float32)
        stream = tr_phi.pow(2).mean().sqrt().clamp_min(1e-12)
        delta = (float(lora.scaling) * (tr_phi @ B.T)).pow(2).mean().sqrt()
        frac = float((delta / stream).item())
        if frac > self.max_delta_rms_frac:
            lora.lora_B.mul_(
                torch.full_like(lora.lora_B, self.max_delta_rms_frac / frac)
            )
            frac = self.max_delta_rms_frac
        return True, frac

    @no_grad()
    def step(
        self,
        input_ids: Tensor,
        targets: Tensor,
        loss_mask: Tensor | None = None,
        holdout: bool = True,
    ) -> TttStats:
        """One features→delta adaptation: harvest, then ridge-solve each block.

        Args:
            input_ids: ``[B, T]`` already-known tokens (never generated here).
            targets: ``[B, T]`` next-token targets, shifted by the caller.
            loss_mask: ``[B, T]`` positions to fit. None fits all of them.
            holdout: when True, the last 1/5 rows of each block are held out of
                ``G``/``C`` and counted in ``holdout_rows``.
        """
        self._ensure_usable()
        ce, rows = self._harvest(input_ids, targets, loss_mask)
        if not torch.isfinite(torch.tensor(ce)):
            raise RuntimeError(f"ttt: non-finite CE {ce}")

        updated = 0
        refits = 0
        train_rows = 0
        holdout_rows = 0
        frac_max = 0.0

        for i, per_firing in rows.items():
            phi, y = self._rows_for(i, per_firing, loss_mask)
            st = self._state[i]
            va_before = st.holdout_rows
            refits_before = st.refits
            did_update, frac = self.rls_step(i, phi, y, holdout=holdout)
            va_gained = st.holdout_rows - va_before
            holdout_rows += va_gained
            train_rows += phi.shape[0] - va_gained
            refits += st.refits - refits_before
            if did_update:
                updated += 1
                frac_max = max(frac_max, frac)

        return TttStats(
            ce=ce,
            blocks=len(self._lora),
            blocks_updated=updated,
            refits=refits,
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            delta_rms_frac=frac_max,
        )
