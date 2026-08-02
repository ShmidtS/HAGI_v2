"""Receiver: LM head with a source prior, z-loss, and chunked cross-entropy.

Four ideas, all about not wasting channel capacity.

**Source prior.** The unigram distribution is the zero-order source code and it
is free — it can be counted from the corpus before training starts. Adding a
fixed ``log p_unigram`` to the logits means the network's output is the
*conditional correction* to a known code, not the whole distribution. On this
corpus the unigram entropy is 8.06 nats against ``ln V = 12.48``, so the prior
hands over ~4.4 nats/token at initialization. Without it, the first thousands of
steps are spent learning token frequencies inside the weights — information that
was already available for free.

**Receiver gain.** The prior only *is* free if the learned part starts at zero.
With a tied codebook it does not: at initialization the residual stream is still
largely collinear with the input token's own code word, so the correlation
``<h, w_token>`` reaches ``H * init_std`` — a +41 logit at H=2048, against a
logit standard deviation of 0.9 over the rest of the alphabet. That single
outlier moves ``logsumexp`` to ~24 nats and the measured starting
cross-entropy to 31.5 against the 8.05 the prior should have delivered. One
learnable scalar gain on the correlation, initialized to ``1/sqrt(H)``, puts the
receiver exactly at the prior at step 0 and lets training raise the gain as the
conditional part becomes informative.

**z-loss.** ``log sum exp(logits)`` does not affect the distribution at all (the
softmax is shift-invariant), but it entirely determines the numerical
conditioning of the softmax and of bf16 gradients. Penalizing its square pins it
near zero, which removes the slow logit-scale drift that eventually turns a
healthy run into an unstable one.

**Chunked cross-entropy.** At V=262144 an ``[N, V]`` fp32 logit tensor is
~1 GB per 1024 rows and the backward pass needs it live. The head therefore
computes logits in row blocks and reduces each block to a scalar immediately;
backward recomputes the block. Peak memory becomes ``chunk * V`` instead of
``N * V``, which is what makes long context affordable at this vocabulary size.

Numerics: the projection runs in the parameter dtype, the softmax reduction in
fp32. The stable ``CE = logsumexp(z) - z[target]`` form is used directly; no
probability tensor is ever materialized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hagi.config import HeadConfig


def load_unigram_logprior(
    path: str, vocab_size: int, smoothing: float = 1.0
) -> torch.Tensor:
    """Load token counts and return a normalized ``log p`` vector.

    Args:
        path: ``.npy`` file of per-id integer counts (scripts/count_unigram.py).
        vocab_size: expected length V.
        smoothing: additive count so unreachable ids get a finite prior instead
            of ``-inf``. An ``-inf`` prior would permanently disable those logits
            and produce NaN gradients if the id ever appears.

    Returns:
        ``[V]`` float32 tensor of log-probabilities summing to 1 in probability.

    Raises:
        FileNotFoundError: missing file.
        ValueError: wrong length or non-positive total.
    """
    counts_path = Path(path)
    if not counts_path.exists():
        raise FileNotFoundError(f"unigram counts not found: {counts_path}")
    counts = np.load(counts_path).astype(np.float64)
    if counts.shape != (vocab_size,):
        raise ValueError(f"unigram counts have shape {counts.shape}, expected ({vocab_size},)")
    counts = counts + float(smoothing)
    total = counts.sum()
    if total <= 0:
        raise ValueError("unigram counts sum to zero")
    return torch.from_numpy(np.log(counts / total)).float()


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Cross-entropy over ``hidden @ weight.T`` without materializing all logits.

    Forward computes per-chunk logits, reduces to the loss sum, and — unless
    ``save_logits=True`` — discards them. Backward recomputes each chunk's logits
    and forms the gradient ``(softmax - onehot)`` in place, accumulating into
    ``grad_hidden`` and ``grad_weight``. The recompute costs one extra projection
    matmul and saves ``N * V`` floats of activation memory.

    With ``save_logits=True`` the forward stores the chunked logits on ``ctx``
    and backward reuses them, trading the recompute matmul (~40% of backward
    time) for ``N * V * 2`` bytes of VRAM. Worth it when VRAM is ample (the 1B
    config sits in 115GB with ~15GB used). Logits are held in a list, not saved
    through ``save_for_backward``: autograd's saved-tensor machinery would add
    them to the graph and keep them alive for the whole backward.

    ``z_loss_weight > 0`` adds ``w * logsumexp(z)^2`` per row, whose gradient is
    ``2 * w * lse * softmax`` — folded into the same pass.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None,
        chunk_rows: int,
        z_loss_weight: float,
        save_logits: bool = False,
    ):
        n = hidden.shape[0]
        # Matmul/softmax run in the parameter dtype (bf16 under bf16 training)
        # where the tensor cores are — fp32 measured 2.9x slower for a 0.23%
        # accuracy cost. fp64 inputs (gradient checks) keep fp64 throughout.
        work_dtype = hidden.dtype if hidden.dtype in (torch.float16, torch.bfloat16, torch.float64) else torch.float32
        acc_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        loss_sum = hidden.new_zeros((), dtype=acc_dtype)
        z_sum = hidden.new_zeros((), dtype=acc_dtype)
        saved_logits: list[torch.Tensor] = [] if save_logits else None
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            h_chunk = hidden[start:end]
            logits = F.linear(h_chunk, weight)
            if bias is not None:
                logits = logits + bias.to(work_dtype)
            if work_dtype in (torch.float16, torch.bfloat16):
                # Online-stable logsumexp in low precision: shift by the max so
                # the exponent does not overflow the bf16/fp16 exponent range.
                maxv = logits.max(dim=-1, keepdim=True).values
                lse = ((logits - maxv).exp().sum(dim=-1, keepdim=True).log() + maxv).to(acc_dtype)
                picked = logits.gather(-1, targets[start:end].unsqueeze(-1)).to(acc_dtype)
            else:
                logits = logits.to(acc_dtype)
                lse = torch.logsumexp(logits, dim=-1, keepdim=True)
                picked = logits.gather(-1, targets[start:end].unsqueeze(-1))
            loss_sum += (lse - picked).sum()
            if z_loss_weight > 0:
                z_sum += lse.squeeze(-1).pow(2).sum()
            if save_logits:
                saved_logits.append(logits.detach())
        ctx.save_for_backward(hidden, weight, targets, bias)
        ctx.chunk_rows = chunk_rows
        ctx.z_loss_weight = z_loss_weight
        ctx.n = n
        ctx.work_dtype = work_dtype
        ctx.saved_logits = saved_logits
        return loss_sum / max(n, 1), z_sum / max(n, 1)

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_z: torch.Tensor):  # type: ignore[override]
        hidden, weight, targets, bias = ctx.saved_tensors
        n = ctx.n
        work_dtype = ctx.work_dtype
        scale_ce = float(grad_loss) / max(n, 1)
        scale_z = float(grad_z) / max(n, 1) if ctx.z_loss_weight > 0 else 0.0

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None and bias.requires_grad else None

        saved = ctx.saved_logits
        for idx, start in enumerate(range(0, n, ctx.chunk_rows)):
            end = min(start + ctx.chunk_rows, n)
            h_chunk = hidden[start:end]
            if saved is not None:
                logits = saved[idx]
            else:
                logits = F.linear(h_chunk, weight)
                if bias is not None:
                    logits = logits + bias.to(work_dtype)
            if work_dtype in (torch.float16, torch.bfloat16):
                # bf16 softmax gradient is safe *in the current regime*: the
                # z-loss pins logsumexp near 0 and the logit scale keeps logits in
                # ~[-8, 8], so the max-shift exponent never overflows bf16's
                # exponent range. Measured vs fp32: grad_weight rel err 1e-4,
                # rare-token rows never zeroed (0/200). The old failure (grad
                # norm ~20x, NaN ~step 250) predates z_loss/prior and only
                # happened because logits drifted to +-100. Max-shift keeps the
                # softmax exact in bf16; keeping the whole gradient path in bf16
                # avoids the [N, V] fp32 cast, which was ~40ms/step.
                maxv = logits.max(dim=-1, keepdim=True).values
                lse = ((logits - maxv).exp().sum(dim=-1, keepdim=True).log() + maxv)
                probs = (logits - lse).exp()
                g = probs.clone()
                g.scatter_add_(-1, targets[start:end].unsqueeze(-1), torch.full_like(lse, -1.0))
                g.mul_(scale_ce)
                if scale_z != 0.0:
                    g.add_(probs * (2.0 * scale_z * lse))
            else:
                acc_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
                logits = logits.to(acc_dtype)
                lse = torch.logsumexp(logits, dim=-1, keepdim=True)
                probs = (logits - lse).exp()
                g = probs.clone()
                g.scatter_add_(-1, targets[start:end].unsqueeze(-1), torch.full_like(lse, -1.0))
                g.mul_(scale_ce)
                if scale_z != 0.0:
                    g.add_(probs * (2.0 * scale_z * lse))

            g = g.to(weight.dtype)
            grad_hidden[start:end] = g @ weight
            # ``(h^T @ g)^T`` computes the same ``g^T @ h`` but ~2x faster on
            # ROCm (measured 258ms vs 498ms at [30720,32768]@[32768,1152]).
            # The reduction axis (N) is identical; the tile layout of the first
            # operand differs, and this part's GEMM favors the [H, N] shape.
            grad_weight += (h_chunk.t() @ g).t()
            if grad_bias is not None:
                grad_bias += g.sum(dim=0)

        return grad_hidden, grad_weight, None, grad_bias, None, None, None


class LMHead(nn.Module):
    """Tied or untied output projection with a source prior and chunked CE.

    Args:
        hidden_size: H.
        vocab_size: V.
        cfg: head configuration.
        tied_weight: the source codebook to share, or None to own a projection.
        xm_cfg: optional :class:`~hagi.config.XMConfig` for best-of-K list
            decoding. When enabled, ``loss`` explores K mode-codes on
            high-entropy positions and keeps only the winner's gradient.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        cfg: HeadConfig,
        tied_weight: torch.Tensor | None = None,
        xm_cfg=None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.cfg = cfg
        self.chunk_rows = int(cfg.ce_chunk_rows)
        self.z_loss_weight = float(cfg.z_loss_weight)

        if tied_weight is not None:
            if tuple(tied_weight.shape) != (vocab_size, hidden_size):
                raise ValueError(
                    f"tied weight has shape {tuple(tied_weight.shape)}, expected ({vocab_size}, {hidden_size})"
                )
            # Held in a plain list, not as an attribute. Assigning an
            # ``nn.Parameter`` to a module attribute registers it, and the encoder
            # already owns this one: ``state_dict`` does not deduplicate, so the
            # checkpoint would carry the ``[V, H]`` codebook twice — 537M extra
            # entries at the 1B configuration. ``named_parameters`` *does*
            # deduplicate, which is why the parameter count would still look right.
            self._tied_ref = [tied_weight]
            self.projection = None
        else:
            self._tied_ref = []
            self.projection = nn.Linear(hidden_size, vocab_size, bias=False)
            nn.init.normal_(self.projection.weight, std=hidden_size**-0.5)

        # One learnable receiver gain. See the module docstring: without it a tied
        # codebook starts with a ``H * init_std`` self-correlation outlier, which
        # at H=2048 is a +41 logit and discards the unigram prior entirely.
        # ``keep_fp32``: a single scalar near 0.022 controlling the sharpness of
        # the whole output distribution must not be quantized to bf16 steps.
        self.keep_fp32 = True
        scale = float(cfg.logit_scale_init) if cfg.logit_scale_init > 0 else hidden_size**-0.5
        self.logit_scale = nn.Parameter(torch.tensor(scale))

        if cfg.unigram_prior:
            prior = load_unigram_logprior(cfg.unigram_path, vocab_size, cfg.unigram_smoothing)
            self.register_buffer("log_prior", prior, persistent=True)
        else:
            self.log_prior = None

        xm = xm_cfg
        if xm is not None and xm.enabled:
            self._xm = xm
            self._make_mode_codes(xm)
        else:
            self._xm = None
            self.mode_codes = None

    @property
    def is_tied(self) -> bool:
        """Whether the output projection is the shared source codebook."""
        return bool(self._tied_ref)

    @property
    def weight(self) -> torch.Tensor:
        """The effective ``[V, H]`` output projection."""
        return self._tied_ref[0] if self._tied_ref else self.projection.weight

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full logits for ``[..., H]`` — for generation and diagnostics only.

        Training must use :meth:`loss`, which never materializes ``[N, V]``.
        """
        out = F.linear(hidden * self.logit_scale.to(hidden.dtype), self.weight)
        if self.log_prior is not None:
            out = out + self.log_prior.to(out.dtype)
        return out

    def loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean cross-entropy and mean z-loss over ``[N, H]`` rows.

        With XM enabled (``cfg.xm.enabled``), this becomes best-of-K list
        decoding: for positions above the entropy gate, K mode-codes are added
        to ``hidden`` and the per-position CE is computed for each candidate;
        only the minimum (the candidate nearest the target, i.e. the maximum-
        likelihood hypothesis) contributes. See :class:`~hagi.config.XMConfig`.

        Args:
            hidden: ``[N, H]`` selected positions.
            targets: ``[N]`` target ids.

        Returns:
            ``(cross_entropy, z_loss)`` — both scalars, z_loss unweighted.
        """
        if hidden.ndim != 2 or targets.ndim != 1 or hidden.shape[0] != targets.shape[0]:
            raise ValueError(
                f"expected hidden [N, H] and targets [N], got {tuple(hidden.shape)} and {tuple(targets.shape)}"
            )
        if hidden.shape[0] == 0:
            zero = hidden.new_zeros((), dtype=torch.float32)
            return zero, zero
        bias = self.log_prior.to(hidden.dtype) if self.log_prior is not None else None
        # The gain is applied outside the custom Function: autograd then routes a
        # gradient to it through this multiply, with no extra pass over [N, V].
        scaled = hidden * self.logit_scale.to(hidden.dtype)
        if self.mode_codes is None:
            ce, z = _ChunkedCrossEntropy.apply(
                scaled,
                self.weight,
                targets,
                bias,
                self.chunk_rows,
                self.z_loss_weight,
                self.cfg.ce_save_logits,
            )
            return ce, z
        return self._loss_xm(scaled, targets, bias)

    def _loss_xm(self, scaled: torch.Tensor, targets: torch.Tensor, bias) -> tuple[torch.Tensor, torch.Tensor]:
        """Best-of-K list decoding over mode-codes (memory-saving mode).

        The paper's memory-saving variant (Appendix C): all K candidates are
        forwarded *without* gradients, the winner (minimum CE) is selected, and
        only it is re-forwarded with gradients to train on. This keeps the
        activation memory of standard training — backward goes through one
        candidate, not K — at the cost of one extra forward pass per explored
        row. On the bandwidth-bound Radeon the head forward is 141ms vs 551ms
        for head backward, so K=4 exploration costs ~4*141ms extra forward
        while backward stays a single 551ms pass.

        Rows below the entropy gate use the base prediction (candidate 0),
        i.e. ordinary CE.
        """
        n, h = scaled.shape
        xm = self._xm
        k = int(xm.num_modes)
        gate = float(xm.entropy_gate)
        weight = self.weight

        # Base logits in no-grad: log-partition, target logit and posterior
        # entropy in ONE [N,V] pass. The entropy gate uses the *posterior
        # entropy* H = <p, log p> = lse - <p, logits> (always <= ln V), not CE —
        # CE on an ambiguous target can far exceed ln V, which would make the
        # gate fire on every position.
        with torch.no_grad():
            log_z, log_pt, entropy = _head_stats_blocks(scaled, weight, targets, bias, self.chunk_rows)
            base_ce = log_z - log_pt  # [N]
            max_h = float(torch.log(torch.tensor(self.vocab_size, dtype=torch.float32)))
            explore = entropy > (gate * max_h)

        n_explore = int(explore.sum())
        if n_explore == 0:
            ce, z = _ChunkedCrossEntropy.apply(
                scaled, weight, targets, bias, self.chunk_rows, self.z_loss_weight, self.cfg.ce_save_logits
            )
            return ce, z

        # Candidate hidden states: mode_code[j] added to explored rows only.
        # Non-explored rows stay at the base prediction (candidate 0).
        mode_dt = self.mode_codes.to(scaled.dtype)  # [K, H]
        with torch.no_grad():
            exp_idx = explore.nonzero(as_tuple=True)[0]
            exp_hidden = scaled.index_select(0, exp_idx)  # [n_explore, H]
            exp_tgt = targets.index_select(0, exp_idx)
            # [K, n_explore, H] candidates via broadcasting.
            cand = exp_hidden.unsqueeze(0) + mode_dt.unsqueeze(1)  # [K, n_explore, H]
            flat = cand.reshape(k * n_explore, h)
            ce_rows = _ce_rows_blocks(
                flat, weight, exp_tgt.repeat(k), bias, self.chunk_rows
            ).reshape(k, n_explore)
            winner = ce_rows.argmin(dim=0)  # [n_explore] winner per explored row
            base_exp = base_ce.index_select(0, exp_idx)
            # Keep the base prediction where exploration found nothing better.
            better = ce_rows.min(dim=0).values < base_exp
            winner = winner.clone()
            winner[~better] = 0

        # Re-forward only the winner with gradients (memory-saving: one
        # backward, not K). Winner hidden = base + mode_code[winner] on explored
        # rows; base on non-explored rows and on rows where the base won.
        win_offset = torch.zeros_like(scaled)
        win_offset[exp_idx] = mode_dt[winner]
        win_hidden = scaled + win_offset
        ce, z = _ChunkedCrossEntropy.apply(
            win_hidden, weight, targets, bias, self.chunk_rows, self.z_loss_weight, self.cfg.ce_save_logits
        )
        return ce, z

    def _make_mode_codes(self, xm) -> None:
        """Create the learnable K mode-code embeddings (best-of-K hypotheses).

        Small init std: a mode is a perturbation of the base prediction toward
        a distinct hypothesis, not a new prediction. ``keep_fp32`` so the tiny
        codes do not round to zero under bf16.
        """
        k, h = int(xm.num_modes), self.hidden_size
        self.mode_codes = nn.Parameter(torch.randn(k, h) * float(xm.mode_std))
        self.mode_codes.keep_fp32 = True

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, tied={self.is_tied}, "
            f"unigram_prior={self.log_prior is not None}, chunk_rows={self.chunk_rows}"
        )


# ---------------------------------------------------------------------------
# XM helpers: chunked per-row quantities over V without materializing [N, V].
# Each returns a [N] tensor; the chunk loop keeps peak memory at chunk*V.
# ``weight`` is the tied/untied output projection ([V, H]) resolved by the
# calling LMHead.
# ---------------------------------------------------------------------------


def _head_stats_blocks(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, bias, chunk_rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row (log_partition, log_p_target, posterior_entropy) in ONE [N,V] pass.

    All three are derived from the same per-row logits, so computing them
    separately would do three [N,V] matmuls on a bandwidth-bound part. This
    does one, keeping the gate and the base CE essentially free over the
    winner re-forward.
    """
    n = hidden.shape[0]
    log_z = torch.empty(n, device=hidden.device, dtype=torch.float32)
    log_pt = torch.empty(n, device=hidden.device, dtype=torch.float32)
    entropy = torch.empty(n, device=hidden.device, dtype=torch.float32)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        logits = F.linear(hidden[start:end], weight)
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
        z = logits.float()
        lse = z.logsumexp(dim=-1, keepdim=True)
        p = (z - lse).exp()
        log_z[start:end] = lse.squeeze(-1)
        log_pt[start:end] = z.gather(-1, targets[start:end].unsqueeze(-1)).squeeze(-1)
        entropy[start:end] = lse.squeeze(-1) - (p * z).sum(dim=-1)
    return log_z, log_pt, entropy


def _log_partition_blocks(
    hidden: torch.Tensor, weight: torch.Tensor, bias, chunk_rows: int
) -> torch.Tensor:
    """Per-row logsumexp(logits) = log sum_v exp(<h, w_v> + b_v)."""
    n = hidden.shape[0]
    out = torch.empty(n, device=hidden.device, dtype=torch.float32)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        logits = F.linear(hidden[start:end], weight)
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
        out[start:end] = logits.float().logsumexp(dim=-1)
    return out


def _log_p_target(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, bias, chunk_rows: int
) -> torch.Tensor:
    """Per-row log p(target) = <h, w_target> + b_target."""
    n = hidden.shape[0]
    out = torch.empty(n, device=hidden.device, dtype=torch.float32)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        logits = F.linear(hidden[start:end], weight)
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
        out[start:end] = logits.gather(-1, targets[start:end].unsqueeze(-1)).squeeze(-1).float()
    return out


def _ce_rows_blocks(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, bias, chunk_rows: int
) -> torch.Tensor:
    """Per-row CE = logsumexp - log p(target)."""
    return _log_partition_blocks(hidden, weight, bias, chunk_rows) - _log_p_target(
        hidden, weight, targets, bias, chunk_rows
    )


