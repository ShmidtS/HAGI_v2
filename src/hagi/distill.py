"""Distillation for orthogonal-expert merging.

The measured fact (DeepSeek-V4, see scripts/dsv4_spectrum_gap.py and
scripts/dsv4_full_activation_analysis.py): the 256 routed experts per layer are
mutually ORTHOGONAL — pairwise output cosine ~1/sqrt(257), flat SVD spectrum,
white-noise 2D spectrum, cross-layer cosine ~0.0006. Orthogonal weights cannot
be merged by any linear re-mixing: a rotation (Hadamard / DFT-3 / Procrustes)
preserves the Gram matrix, so a Hadamard "sum channel" carries exactly as much
shared signal as a random direction (cos ~1/sqrt(N)). Re-mixing is a pure
permutation of the expert axis — it adds no information.

The only operation that transfers the BEHAVIOR of orthogonal experts into a
compact model is distillation: match the teacher's OUTPUTS (logits or hidden
states), never its weights. This module provides the two losses and a small
training primitive for that.

  * :func:`logit_kl`   — soft-label KD (teacher and student share a vocabulary).
  * :func:`feature_mse` — hidden-state distillation (any vocabulary/dimension:
    a learnable linear projection aligns the teacher's hidden width to the
    student's).
  * :func:`distill_loss` — dispatch on ``mode`` with a temperature and the
    standard ``alpha`` blend against the student's own task loss.
  * :func:`distill_experts` — merge N orthogonal expert functions into one
    compact student by matching the teacher's combined output on a frozen
    activation sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def logit_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Knowledge-distillation loss (KL of the soft distributions).

    ``student_logits``/``teacher_logits`` are ``[..., V]`` with an identical
    vocabulary. The teacher's distribution is the target and the student's is
    the prediction, so the gradient flows only into the student. Both are
    temperature-scaled and the result is rescaled by ``T**2`` (Hinton et al.)
    so the gradient magnitude is independent of the temperature.

    ``reduction`` follows :func:`torch.nn.functional.kl_div` (``"batchmean"``
    is the standard KD choice: the KL is averaged over the batch dimension and
    summed over the class dimension).
    """
    t = float(temperature)
    if t <= 0:
        raise ValueError("temperature must be positive")
    s = F.log_softmax(student_logits / t, dim=-1)
    q = F.softmax(teacher_logits / t, dim=-1)
    return (t * t) * F.kl_div(s, q, reduction=reduction)


def feature_mse(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
    """Hidden-state (feature) distillation: mean squared error.

    Used when teacher and student have different vocabularies, or as a richer
    signal than logits alone. The two tensors must already be aligned in shape
    (see :class:`FeatureAdapter` for the projection that does the alignment).
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_hidden.shape)} vs "
            f"teacher {tuple(teacher_hidden.shape)}"
        )
    return F.mse_loss(student_hidden.float(), teacher_hidden.float())


class FeatureAdapter(nn.Module):
    """Learnable linear projection that aligns a teacher's hidden width to a
    student's.

    ``y = norm(W x + b)`` where ``x`` is ``[..., teacher_dim]`` and ``y`` is
    ``[..., student_dim]``. The optional LayerNorm makes the loss insensitive
    to the absolute scale of the teacher stream (the student only has to match
    the *shape* of the representation, not its magnitude), which is the right
    objective for feature distillation across architectures.
    """

    def __init__(self, teacher_dim: int, student_dim: int, norm: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=(teacher_dim**-0.5))
        self.norm = nn.LayerNorm(student_dim) if norm else nn.Identity()

    def forward(self, teacher_hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(teacher_hidden))


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    mode: str = "logit",
    temperature: float = 1.0,
    student_hidden: torch.Tensor | None = None,
    teacher_hidden: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch distillation loss.

    Args:
        mode: ``"logit"`` (KL on logits, shared vocab) or ``"feature"``
            (MSE on hidden states, already projected/shape-aligned). ``"both"``
            returns their sum (feature is doubled to keep the scale comparable).
    """
    if mode == "logit":
        return logit_kl(student_logits, teacher_logits, temperature)
    if mode == "feature":
        if student_hidden is None or teacher_hidden is None:
            raise ValueError("mode='feature' requires student_hidden and teacher_hidden")
        return feature_mse(student_hidden, teacher_hidden)
    if mode == "both":
        if student_hidden is None or teacher_hidden is None:
            raise ValueError("mode='both' requires student_hidden and teacher_hidden")
        return logit_kl(student_logits, teacher_logits, temperature) + 2.0 * feature_mse(
            student_hidden, teacher_hidden
        )
    raise ValueError(f"unknown distill mode {mode!r} (expected 'logit', 'feature', 'both')")


def distill_experts(
    student: nn.Module,
    teacher: nn.Module,
    inputs: torch.Tensor,
    *,
    steps: int,
    lr: float = 1e-3,
    batch_size: int = 4096,
    temperature: float = 1.0,
    mode: str = "feature",
    teacher_hidden_fn=None,
    student_hidden_fn=None,
    log_interval: int = 0,
) -> list[float]:
    """Train ``student`` to reproduce ``teacher``'s output on a frozen sample.

    The distillation loop for the orthogonal-expert merge: the teacher is a
    router + orthogonal experts (e.g. a DeepSeek MoE layer) and the student is
    a compact model (e.g. a single HAGI-style SwiGLU FFN). ``inputs`` is a
    frozen activation sample ``[N, D]`` collected from the teacher's own
    forward pass; ``teacher`` maps it to the target output ``[N, D']``.

    Args:
        student: compact model ``[N, D] -> [N, D']``.
        teacher: frozen teacher ``[N, D] -> [N, D']``.
        inputs: frozen ``[N, D]`` activation sample.
        steps: optimizer steps (full-batch over minibatches of ``inputs``).
        mode: ``"feature"`` (MSE) or ``"logit"`` (KL) — see :func:`distill_loss`.
        teacher_hidden_fn / student_hidden_fn: optional hooks returning the
            hidden states to compare in ``"feature"`` mode; when None the
            model outputs themselves are compared.
        log_interval: print the loss every N steps (0 = silent).

    Returns:
        The loss history.
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    history: list[float] = []
    n = inputs.shape[0]
    student.train()
    teacher.eval()
    with torch.no_grad():
        target = teacher(inputs)
        if teacher_hidden_fn is not None:
            target = teacher_hidden_fn(target, inputs)
    target = target.detach()

    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=inputs.device)
        xb = inputs[idx]
        yb = target[idx]
        out = student(xb)
        if student_hidden_fn is not None:
            out = student_hidden_fn(out, xb)
        loss = distill_loss(
            out,
            yb,
            mode=mode,
            temperature=temperature,
            student_hidden=out if mode in ("feature", "both") else None,
            teacher_hidden=yb if mode in ("feature", "both") else None,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if log_interval and (step + 1) % log_interval == 0:
            print(f"  distill step {step + 1}/{steps}: loss={history[-1]:.6f}", flush=True)
    return history
