"""RLTL;DR-style insight channel: failure → compressed insight → SFT.

Measured motivation (arXiv 2609.37633): when the reward signal is absent
(all rollouts fail), a short self-generated correction — an *insight*, ~17
tokens — internalized via SFT carries most of the self-improvement effect
at a fraction of the backward tokens (12M insight-backward vs 720M
forward in their RL loop). The costly part is experience generation, not
backprop.

HAGI mapping (no verifier in the corpus regime): the "failure" signal is
the model's own per-token CE. A hard span (top-quantile CE above the
absolute ``min_ce`` floor) is a failure; the ground-truth continuation of
that span, truncated to ``max_tokens``, is the compressed correction.
Internalization is a CE step with ``lambda_insight`` weight on the
insight target positions — the model learns to *produce and use* the
correction, so the behavior survives when the runtime prefix is gone
(paper: the insight-only SFT arm recovers most of the full effect).

    Safety follows the :mod:`hagi.train.self_improve` contract plus R81
    ``tldr_drift_null``: with ``cfg.adapter_only=True`` (default) the
    update touches ONLY adapter parameters (requires_grad=True — base
    frozen upstream via ``train.adapt.freeze_base``); if the base is
    NOT frozen the cycle refuses rather than drifting the cortex —
    the insight channel is the hippocampus, and the hippocampus may
    never write to the cortex directly. The KL guard uses exact
    full-vocabulary KL(p_post || p_pre); the verifier rule applies:
    the update is KEPT only when post-window CE measurably improved
    (``certified``), otherwise rolled back — never improve on
    uncertified deltas.
    """

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hagi.model.model import HAGI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Insight:
    """One compressed correction.

    Attributes:
        context_ids: ``[C]`` input tokens up to and including the hard
            span's first token (the runtime prefix).
        insight_ids: ``[K <= max_tokens]`` ground-truth correction targets.
        span_ce: mean CE of the hard span — the sort key (worst first).
    """

    context_ids: torch.Tensor
    insight_ids: torch.Tensor
    span_ce: float


def _log_probs(model: HAGI, input_ids: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary log-softmax ``[B, T, V]`` (float32)."""
    logits = model(input_ids, return_logits=True).logits
    return F.log_softmax(logits.float(), dim=-1)


@torch.no_grad()
def extract_insights(
    model: HAGI,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    cfg,
    max_rows: int = 32,
) -> list[Insight]:
    """Select hard spans (top-quantile CE above ``min_ce``) as insights.

    A span is a run of consecutive hard token positions; the insight is
    the ground-truth continuation after the run's first token, truncated
    to ``max_tokens``.     The ``min_ce`` floor makes a quantile of an
    already-easy window invent nothing.

    Args:
        max_rows: hard cap on the returned insights. Without it the row
            count is unbounded and ``insight_sft_loss`` materialises a
            ``[rows, len, V]`` logit tensor -- on a fresh model every
            position exceeds ``min_ce`` (ce = ln V), so a 32x1024 window
            yields 1024 rows and 139 GB of logits, which fails a HIP
            launch on any GPU. The cap keeps the HOTTEST spans (the list
            is sorted by span CE, descending), so the signal is preserved
            and only the tail is dropped.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    with torch.no_grad():
        logp = _log_probs(model, input_ids)
    ce = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T]
    bsz, seq = ce.shape
    threshold = torch.quantile(ce.float(), float(cfg.fail_quantile))
    max_tokens = int(cfg.max_tokens)
    min_ce = float(cfg.min_ce)

    insights: list[Insight] = []
    for b in range(bsz):
        hard = (ce[b] >= threshold) & (ce[b] >= min_ce)
        run_start: int | None = None
        for t in range(seq + 1):
            active = t < seq and bool(hard[t])
            if active and run_start is None:
                run_start = t
            elif not active and run_start is not None:
                end = min(seq, run_start + max_tokens)
                if end > run_start:
                    insights.append(
                        Insight(
                            context_ids=input_ids[b, : run_start + 1].clone(),
                            insight_ids=targets[b, run_start:end].clone(),
                            span_ce=float(ce[b, run_start:end].mean()),
                        )
                    )
                run_start = None
    insights.sort(key=lambda i: i.span_ce, reverse=True)
    return insights[: max_rows]


def insight_sft_loss(
    model: HAGI,
    insights: list[Insight],
    cfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internalization loss: exact weighted CE, insight positions upweighted.

    Each row is ``context + insight``; targets are the shifted tokens.
    The insight target positions (the correction itself) carry weight
    ``lambda_insight``, the context targets weight 1 — learning to emit
    the correction from context is the internalization.

    Returns:
        ``(loss, n_scored)`` — scalar mean over scored positions and the
        number of them (for guards and logging).
    """
    if not insights:
        raise ValueError("insight_sft_loss needs at least one insight")
    lam = float(cfg.lambda_insight)
    device = insights[0].context_ids.device

    rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for ins in insights:
        seq = torch.cat([ins.context_ids, ins.insight_ids])  # [C+K]
        inp, tgt = seq[:-1], seq[1:]
        k = ins.insight_ids.numel()
        w = torch.ones(inp.shape[0], device=device)
        w[-(k - 1) :] = lam if k > 1 else w[-1:]
        rows.append((inp, tgt, w))
    max_len = max(r[0].shape[0] for r in rows)

    input_ids = torch.zeros(len(rows), max_len, dtype=torch.long, device=device)
    targets = torch.full((len(rows), max_len), -1, dtype=torch.long, device=device)
    weights = torch.zeros(len(rows), max_len, device=device)
    for i, (inp, tgt, w) in enumerate(rows):
        input_ids[i, : inp.shape[0]] = inp
        targets[i, : tgt.shape[0]] = tgt
        weights[i, : w.shape[0]] = w
    valid = targets >= 0

    logp = _log_probs(model, input_ids)
    ce = -logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    w = weights * valid
    loss = (ce * w).sum() / w.sum().clamp(min=1e-8)
    return loss, int(valid.sum())


def run_insight_cycle(
    model: HAGI,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    cfg,
    *,
    lr: float = 1e-3,
    old_windows: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict:
    """One certified insight step: measure → SFT → re-score → guard.

    Adapter-only by construction: only parameters with
    ``requires_grad=True`` (base frozen upstream, cf.
    ``train.adapt.freeze_base``) receive the update. The KL guard uses
    exact full-vocabulary KL(p_post || p_pre) on the scored window; a
    trip or a non-finite loss rolls the parameters back to the pre-step
    state (transactional, cf. ``self_improve.py``).

    With ``cfg.safeqp`` and ``old_windows`` the update is SafeQP-filtered
    first (R78 ``insight_consolidation_safe``): the insight gradient is
    projected onto ``{d : <g_old_i, d> >= -eps_i}`` via
    :func:`hagi.model.formal.safe_qp_solve`, so old-domain drift is
    bounded BEFORE the step, not merely observed after it. The
    certificate (feasibility / descent / contraction) is reported — the
    certified alignment ``||d*||^2 <= <g_I, d*>`` from the Lean core.

    Returns:
        Report dict: ``pre_ce``, ``post_ce``, ``kl``, ``n_insights``,
        ``n_scored``, ``applied`` (False when rolled back), ``reason``,
        and with SafeQP on: ``safe_qp_lambda``, ``safe_qp_certificate``.
    """
    model.eval()
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    # The caller may hand over windows read from disk (CPU) while the model
    # lives on the GPU; index_select inside the model then fails on a device
    # mismatch. Coerce here so every entry point works, not just the ones
    # that happen to pass device-resident tensors.
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    if old_windows:
        old_windows = [
            (ids.to(device), tgt.to(device)) for ids, tgt in old_windows
        ]
    with torch.no_grad():
        pre_logp = _log_probs(model, input_ids)
        pre_ce = float(
            (-pre_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).mean()
        )
    insights = extract_insights(
        model, input_ids, targets, cfg,
        max_rows=int(getattr(cfg, "max_rows", 32)),
    )
    report = {
        "pre_ce": pre_ce,
        "post_ce": None,
        "kl": None,
        "n_insights": len(insights),
        "n_scored": 0,
        "applied": False,
        "reason": "no_insights",
    }
    if not insights:
        return report

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        report["reason"] = "nothing_trainable"
        return report
    trainable_names = {
        n for n, p in model.named_parameters() if p.requires_grad
    }
    if getattr(cfg, "adapter_only", True):
        # R81 tldr_drift_null applied at code level: the insight channel
        # may not touch the cortex. Adapter params are those the base-freeze
        # left trainable (BlockAdapter/LoRA/cortex scales). A warm base is
        # a hard refusal, not a silent full-model SFT.
        base_leaking = [
            n
            for n in trainable_names
            if not ("adapter" in n or "cortex" in n or "lora" in n)
        ]
        if base_leaking:
            report["reason"] = "base_not_frozen"
            logger.warning(
                "insight cycle refused: %d base params trainable (e.g. %s); "
                "freeze the base (train.adapt.freeze_base) first",
                len(base_leaking),
                base_leaking[0],
            )
            return report
    snapshot = copy.deepcopy(
        {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    )
    opt: torch.optim.AdamW | None = None
    if not (getattr(cfg, "safeqp", False) and old_windows):
        opt = torch.optim.AdamW(params, lr=lr)
    reason = "kl_bound"
    try:
        loss, n_scored = insight_sft_loss(model, insights, cfg)
        report["n_scored"] = n_scored
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite insight loss {loss}")
        model.zero_grad(set_to_none=True)
        loss.backward()
        if opt is not None:
            opt.step()
        else:
            # SafeQP consolidation (R78): project the insight gradient onto
            # the per-domain drift-feasible cone and step along d* directly.
            from hagi.model.formal import safe_qp_solve

            grads = [p.grad.detach().clone().flatten() for p in params]
            g_I = torch.cat(grads)
            dom_grads: list[torch.Tensor] = []
            for w_ids, w_tgt in old_windows or []:
                model.zero_grad(set_to_none=True)
                dom_loss = model(w_ids, w_tgt).lm_loss
                dom_loss.backward()
                dom_grads.append(
                    torch.cat(
                        [p.grad.detach().clone().flatten() for p in params]
                    )
                )
            model.zero_grad(set_to_none=True)
            K = len(dom_grads)
            G = torch.stack(
                [
                    torch.stack([torch.dot(a, b) for b in dom_grads])
                    for a in dom_grads
                ]
            )
            b = torch.stack([torch.dot(a, g_I) for a in dom_grads])
            eps = torch.full((K,), float(cfg.drift_eps))
            lam, cert = safe_qp_solve(G, b, eps, g_norm_sq=float(torch.dot(g_I, g_I)))
            # KKT: d* = g_I + sum lam_i g_i (NOT minus — the projection pulls
            # g back TOWARD the old-domain gradients to restore feasibility)
            d_star = g_I + sum(
                float(lam[i]) * dom_grads[i] for i in range(K)
            )
            report["safe_qp_lambda"] = [float(x) for x in lam]
            report["safe_qp_certificate"] = cert
            # certified descent direction; lr remains the step cap until a
            # measured Lipschitz constant upgrades eta to the analytic form
            offset = 0
            with torch.no_grad():
                for p in params:
                    n = p.numel()
                    p.add_(d_star[offset : offset + n].view_as(p), alpha=-lr)
                    offset += n

        model.eval()
        with torch.no_grad():
            post_logp = _log_probs(model, input_ids)
            post_ce = float(
                (-post_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).mean()
            )
            kl = float(
                (post_logp.exp() * (post_logp - pre_logp)).sum(-1).mean()
            )
        report["post_ce"] = post_ce
        report["kl"] = kl
        if kl <= float(cfg.kl_bound) and post_ce <= pre_ce + 1e-6:
            # verifier rule (certified-improvement gate): keep the update
            # only when the scored window's exact CE measurably improved
            report["applied"] = True
            report["reason"] = "ok"
        else:
            reason = "kl_bound" if kl > float(cfg.kl_bound) else "no_certified_gain"
            logger.info(
                "insight gate: kl=%.4f (bound %.4f), ce %.4f -> %.4f -> %s",
                kl,
                float(cfg.kl_bound),
                pre_ce,
                post_ce,
                reason,
            )
    except FloatingPointError as exc:
        reason = "non_finite"
        logger.warning("insight cycle failed: %s", exc)
    finally:
        if not report["applied"]:
            if report["kl"] is not None or report["reason"] == "no_insights":
                report["reason"] = reason
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in snapshot:
                        p.copy_(snapshot[n])
        model.zero_grad(set_to_none=True)
        model.train(False)
    return report
