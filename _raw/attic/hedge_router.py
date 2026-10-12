"""§6: the Hedge router at inference (optional, default OFF).

ALGORITHMS.md §6:

    routing: top-k by |c_i| with an error budget
        min k : sum_{i>k} c_i^2 <= eps_route      [gating_tail_bound]
    weights: Hedge, eta = sqrt(2 ln K / T)         [router_regret_bound]

Both primitives already exist and are tested:
``hagi.train.hedge.optimal_hedge_eta`` / ``hedge_weights`` (the closed
regret-optimal rate) and ``hagi.train.hedge.minimal_topk`` (the exact
tail-identity top-k budget). This module only COMBINES them into a
router step over per-leaf logits, plus the honest skip.

HONEST SKIP NOTE (why this is not wired into ``generate``'s hot path)
---------------------------------------------------------------------
The merged model (``hagi.model.merge.MergedHAGI``) does not expose
per-leaf logits: the N block-diagonal expert bodies feed ONE wide head
over the concatenated hidden stream, so per-leaf head info simply does
not exist after the merge. Producing it would require either K extra
tail+head forwards per token or a structural forward change -- exactly
the "costly forward change" this integration was told to refuse.

Therefore :func:`hedge_route` is the consumer for the day a model DOES
expose per-leaf logits (any ``[K, B, V]`` tensor), and
:func:`maybe_hedge_route` is the generation-path seam: it checks the
capability, and when it is absent it returns the merged logits
UNCHANGED together with the skip reason, so enabling
``router.hedge: true`` on today's merged checkpoints is a no-op with a
logged explanation rather than a crash or a silent behavior change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from hagi.train.hedge import (
    hedge_weights,
    minimal_topk,
    optimal_hedge_eta,
)

__all__ = [
    "SKIP_NO_PER_LEAF_LOGITS",
    "HedgeRouteResult",
    "leaf_disagreement_losses",
    "hedge_route",
    "maybe_hedge_route",
]

# The documented skip: per-leaf logits are not exposed by MergedHAGI.
SKIP_NO_PER_LEAF_LOGITS = (
    "hedge router skipped: the merged model exposes no per-leaf logits "
    "(one wide head over the block-diagonal body); computing them would "
    "need K extra head forwards per token -- refused by design "
    "(ALGORITHMS.md §6 integration note)"
)


@dataclass(frozen=True)
class HedgeRouteResult:
    """One routing step's output.

    Attributes:
        logits: the mixed ``[B, V]`` logits to sample from.
        k: the top-k budget actually kept (``min k : tail <= eps_route``).
        weights: the Hedge weights ``[K]`` used for the mix.
        eta: the regret-optimal Hedge rate for this round count.
        skipped: the skip reason when routing did not happen, else "".
    """

    logits: torch.Tensor
    k: int
    weights: torch.Tensor | None
    eta: float
    skipped: str = ""


def leaf_disagreement_losses(per_leaf_logits: torch.Tensor) -> torch.Tensor:
    """Per-leaf loss = mean over tokens of KL(consensus || leaf_i).

    The §6 signal: a leaf that disagrees with the pooled ensemble is the
    one whose contribution the Hedge weights should decay. The consensus
    is the uniform log-probability mixture (geometric pool in log
    space), so the loss is cheap: one log_softmax per leaf and a mean.

    Args:
        per_leaf_logits: ``[K, B, V]`` leaf logits (any leading shape
            flattened to rows works; the last dim is the vocab).

    Returns:
        ``[K]`` non-negative mean disagreement losses.
    """
    if per_leaf_logits.ndim < 2:
        raise ValueError("per_leaf_logits must be [K, rows, V] (or [K, V])")
    logp = torch.log_softmax(per_leaf_logits.float().reshape(
        per_leaf_logits.shape[0], -1, per_leaf_logits.shape[-1]), dim=-1)
    # consensus: uniform geometric pool in log space (log-sum-exp mean)
    consensus = torch.logsumexp(logp, dim=0) - math.log(logp.shape[0])
    # KL(consensus || leaf_i) per row, averaged over rows
    p = consensus.exp()
    kl = (p * (consensus.unsqueeze(0) - logp)).sum(-1)
    return kl.mean(-1)


def hedge_route(
    per_leaf_logits: torch.Tensor,
    eps_route: float,
    round_index: int = 1,
    weights: torch.Tensor | None = None,
) -> HedgeRouteResult:
    """One §6 routing step: Hedge weights + exact tail-budget top-k.

    Args:
        per_leaf_logits: ``[K, *, V]`` per-leaf logits.
        eps_route: the routing error budget (tail of squared importance).
        round_index: how many routing rounds have happened (Hedge T).
        weights: previous Hedge weights, uniform when omitted.

    Returns:
        The mixed logits over the top-k leaves, with the routing record.
    """
    if per_leaf_logits.ndim < 2:
        raise ValueError("per_leaf_logits must be [K, rows, V] (or [K, V])")
    k_leaves = per_leaf_logits.shape[0]
    if k_leaves < 2:
        raise ValueError("hedge routing needs >= 2 leaves")
    losses = leaf_disagreement_losses(per_leaf_logits)
    eta = optimal_hedge_eta(k_leaves, round_index)
    w = hedge_weights(losses, eta, weights=weights)
    # importance for the tail budget: agreement score, non-negative,
    # scaled so a confident agreeing leaf dominates the budget decision.
    importance = torch.softmax(-losses, dim=0)
    k = minimal_topk(importance, eps_route)
    keep = torch.argsort(losses)[:k]           # the k least-disagreeing
    flat = per_leaf_logits.reshape(k_leaves, -1, per_leaf_logits.shape[-1])
    # renormalize over the kept subset: the mix must stay a convex
    # combination of KEPT leaves (dropping the tail re-weights, not
    # rescales, the survivors).
    w_keep = w[keep] / w[keep].sum().clamp_min(1e-30)
    mixed = torch.einsum("k,krv->rv", w_keep.to(flat.dtype), flat[keep])
    return HedgeRouteResult(
        logits=mixed.reshape(per_leaf_logits.shape[1:]),
        k=k, weights=w, eta=eta,
    )


def maybe_hedge_route(
    per_leaf_logits: torch.Tensor | None,
    eps_route: float,
    round_index: int = 1,
    weights: torch.Tensor | None = None,
) -> HedgeRouteResult:
    """The generation-path seam: route when possible, skip when not.

    ``per_leaf_logits is None`` is the "model does not expose per-leaf
    head info" case (today's ``MergedHAGI``): the caller's merged logits
    must be passed through unchanged by the caller -- this function has
    nothing to return but the skip record, so ``logits`` is an empty
    tensor and ``skipped`` carries the documented reason.
    """
    if per_leaf_logits is None:
        return HedgeRouteResult(
            logits=torch.empty(0), k=0, weights=None, eta=0.0,
            skipped=SKIP_NO_PER_LEAF_LOGITS,
        )
    return hedge_route(per_leaf_logits, eps_route, round_index, weights)
