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

    Forward computes per-chunk logits, reduces to the loss sum, and discards
    them. Backward recomputes each chunk's logits and forms the gradient
    ``(softmax - onehot)`` in place, accumulating into ``grad_hidden`` and
    ``grad_weight``. The recompute costs one extra projection matmul and saves
    ``N * V`` floats of activation memory.

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
    ):
        n = hidden.shape[0]
        # Matmul/softmax run in the parameter dtype (bf16 under bf16 training)
        # where the tensor cores are — fp32 measured 2.9x slower for a 0.23%
        # accuracy cost. fp64 inputs (gradient checks) keep fp64 throughout.
        work_dtype = hidden.dtype if hidden.dtype in (torch.float16, torch.bfloat16, torch.float64) else torch.float32
        acc_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        loss_sum = hidden.new_zeros((), dtype=acc_dtype)
        z_sum = hidden.new_zeros((), dtype=acc_dtype)
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
        ctx.save_for_backward(hidden, weight, targets, bias)
        ctx.chunk_rows = chunk_rows
        ctx.z_loss_weight = z_loss_weight
        ctx.n = n
        ctx.work_dtype = work_dtype
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

        for start in range(0, n, ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, n)
            h_chunk = hidden[start:end]
            logits = F.linear(h_chunk, weight)
            if bias is not None:
                logits = logits + bias.to(work_dtype)
            if work_dtype in (torch.float16, torch.bfloat16):
                maxv = logits.max(dim=-1, keepdim=True).values
                lse = ((logits - maxv).exp().sum(dim=-1, keepdim=True).log() + maxv)
                probs = (logits - maxv).exp().to(work_dtype)  # probs in work dtype
                g = probs.clone()
                g.scatter_add_(-1, targets[start:end].unsqueeze(-1), torch.full_like(lse, -1.0))
                # scale_ce and z-term applied in work dtype
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
            grad_weight += g.t() @ h_chunk
            if grad_bias is not None:
                grad_bias += g.sum(dim=0)

        return grad_hidden, grad_weight, None, grad_bias, None, None


class LMHead(nn.Module):
    """Tied or untied output projection with a source prior and chunked CE.

    Args:
        hidden_size: H.
        vocab_size: V.
        cfg: head configuration.
        tied_weight: the source codebook to share, or None to own a projection.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        cfg: HeadConfig,
        tied_weight: torch.Tensor | None = None,
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
        ce, z = _ChunkedCrossEntropy.apply(
            scaled, self.weight, targets, bias, self.chunk_rows, self.z_loss_weight
        )
        return ce, z

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, tied={self.is_tied}, "
            f"unigram_prior={self.log_prior is not None}, chunk_rows={self.chunk_rows}"
        )
