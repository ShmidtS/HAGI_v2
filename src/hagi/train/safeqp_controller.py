"""SafeQP.lean / Dominate.lean prescription (reviewer round-5 priority 2):
measure gradient geometry BEFORE altering updates — log-only first deployment.

Dominate.lean: when one corpus holds ~95% of the weighted gradient-norm share,
the mixed step ``g = sum w_i g_i`` is nearly ``g_s`` and any corpus with
``<g_i, g_s> < 0`` is damaged in first order — no learning rate fixes that.
SafeQP.lean (safeQP_descent): the projection ``d* = argmin 0.5||d-g||^2`` s.t.
``<g_i, d> >= -eps_i`` satisfies ``<g, d*> >= ||d*||^2`` (guaranteed descent).

This module measures, it does not act. The scan computes per-corpus gradients
on small calibration windows, their Gram matrix, cos geometry and — when a
conflict exists — the SafeQP dual solution with its certificate. Updates are
never modified: the first deployment is a prevalence measurement (how often
do real conflicts occur at the current mix?), exactly the
theorem -> measure -> predict -> run discipline the project runs on.
"""

from __future__ import annotations

import logging

import torch

from hagi.model.formal import safe_qp_solve

logger = logging.getLogger(__name__)


def _flat_grad(model: torch.nn.Module, device: torch.device) -> torch.Tensor | None:
    """Concatenate all current parameter gradients into one fp32 vector."""
    parts = [p.grad.detach().float().reshape(-1) for p in model.parameters() if p.grad is not None]
    if not parts:
        return None
    return torch.cat(parts)


def corpus_grad_gram(
    model: torch.nn.Module,
    samples: list[tuple[str, dict]],
    device: torch.device | str = "cpu",
    eps_rel: float = 0.0,
    corpus_weights: list[float] | None = None,
) -> dict:
    """Per-corpus gradient Gram scan + conflict certificate (read-only for training).

    Args:
        model: any HAGI model exposing ``model(input_ids, targets) -> output``
            with ``output.lm_loss`` (the Trainer's forward signature).
        samples: list of ``(corpus_name, {"input_ids": [1,T], "targets": [1,T]})``
            calibration windows (small — this is a measurement, not training).
        device: compute device.
        eps_rel: relative safety budget; corpus i is in conflict when
            ``<g_i, g> < -eps_rel * ||g_i|| * ||g||`` with ``g`` the weighted
            mean direction (eps_rel=0 → any negative alignment is a conflict).
        corpus_weights: mix weights per corpus (normalized internally);
            uniform when omitted.

    Returns:
        Dict with norms, cos matrix, min off-diagonal cos, conflict flags and,
        when at least one conflict exists, the ``safe_qp_solve`` certificate.
        Model gradients are restored to None (``zero_grad(set_to_none=True)``).
    """
    device = torch.device(device)
    names = [n for n, _ in samples]
    k = len(samples)
    if k == 0:
        raise ValueError("corpus_grad_gram needs at least one sample")
    w = (
        torch.tensor(corpus_weights, dtype=torch.float64)
        if corpus_weights is not None
        else torch.full((k,), 1.0 / k, dtype=torch.float64)
    )
    if w.shape != (k,) or (w < 0).any() or float(w.sum()) <= 0:
        raise ValueError("corpus_weights must be nonnegative and sum > 0")
    w = w / w.sum()

    grads: list[torch.Tensor] = []
    for name, batch in samples:
        model.zero_grad(set_to_none=True)
        output = model(
            batch["input_ids"].to(device),
            batch["targets"].to(device),
        )
        loss = output.lm_loss if output.lm_loss is not None else output.ce
        if loss is None:
            raise ValueError(f"corpus {name}: forward produced no lm loss")
        loss.backward()
        g = _flat_grad(model, device)
        if g is None:
            raise ValueError(f"corpus {name}: backward produced no gradients")
        grads.append(g.to("cpu", torch.float64))
        del output, loss, g
    model.zero_grad(set_to_none=True)

    gmat = torch.stack(grads)                      # [K, P]
    norms = gmat.norm(dim=1)                       # [K]
    gram = gmat @ gmat.T                           # [K, K]
    denom = torch.clamp(norms[:, None] * norms[None, :], min=1e-30)
    cos = gram / denom
    # min OFF-DIAGONAL cos (diagonal is trivially 1; cos - I leaves float
    # dust like -1e-15 that would win the min)
    if k > 1:
        off_mask = ~torch.eye(k, dtype=torch.bool)
        min_cos = float(cos[off_mask].min())
    else:
        min_cos = 1.0
    mean_dir = (w[:, None] * gmat).sum(0)          # [P]
    mean_norm = float(mean_dir.norm())
    b = gmat @ mean_dir                            # [K] <g_i, g>
    eps = eps_rel * norms * mean_norm
    conflicts = b < -eps - 1e-12

    result: dict = {
        "corpora": names,
        "norms": norms.tolist(),
        "cos": cos.tolist(),
        "min_cos": min_cos,
        "conflicts": conflicts.tolist(),
        "n_conflicts": int(conflicts.sum()),
        "domination_share": float((norms * norms * w * k).max() / max(float((norms * norms * w * k).sum()), 1e-30)),
    }
    if bool(conflicts.any()):
        lam, cert = safe_qp_solve(gram, b, eps)
        result["safe_qp_lambda"] = lam.tolist()
        result["safe_qp_certificate"] = cert
    for g in grads:
        del g
    return result
