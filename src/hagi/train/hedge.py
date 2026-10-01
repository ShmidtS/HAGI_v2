"""Hedge for expert routing: the regret-optimal learning rate.

``Hagi/Autonomy/Hedge.lean``:

``exp_neg_le_quad`` / ``hedge_step``
    ``sum_i p_i * exp(-eta * l_i) <= 1 - eta * sum_i p_i * l_i + eta^2/2``
    -- the per-step Hedge potential decreases at least linearly in
    the weighted loss, plus a ``eta^2/2`` slack.

``hedge_telescope``
    ``prod_t (1 + u_t) <= exp(sum_t u_t)`` -- the product form of the
    same argument over T rounds.

``router_regret_bound``
    ``A - Lstar <= log K / eta + eta * T / 2``

    The bound is a convex function of ``eta``, so its MINIMUM is
    analytic: differentiating ``ln K/eta + eta*T/2`` gives
    ``-ln K/eta^2 + T/2 = 0``, i.e.

        eta* = sqrt(2 * ln K / T)

    and the minimal regret is ``sqrt(2 * T * ln K)`` -- the standard
    Hedge/exponential-weights rate, here machine-checked. As with
    ``optimal_step_unconstrained`` the learning rate is COMPUTED from
    (K, T), not tuned.

``eta`` is clamped to ``(0, 1]`` because ``hedge_step`` assumes
``0 <= eta <= 1``; ``T`` below 1 makes ``eta*`` exceed 1, where the
bound's own hypothesis fails and the uniform mix is optimal anyway.

This module provides the closed forms plus a vectorised weight update;
the caller supplies the measured per-expert losses.
"""

from __future__ import annotations

import math

import torch


def optimal_hedge_eta(n_experts: int, rounds: int, cap: float = 1.0) -> float:
    """``eta* = sqrt(2 * ln K / T)`` -- minimizes ``router_regret_bound``.

    Args:
        n_experts: ``K``, strictly positive.
        rounds: ``T``, the number of routing decisions so far.
        cap: upper clamp for ``eta`` (``hedge_step`` needs ``eta <= 1``).

    Returns:
        The regret-optimal learning rate in ``(0, cap]``.

    Raises:
        ValueError: on non-positive ``n_experts`` or ``rounds``.
    """
    if n_experts <= 0:
        raise ValueError("optimal_hedge_eta needs a positive expert count")
    if rounds < 1:
        rounds = 1
    eta = math.sqrt(2.0 * math.log(n_experts) / rounds)
    return min(max(eta, 1e-6), cap)


def regret_bound(n_experts: int, rounds: int, eta: float) -> float:
    """``A - Lstar <= ln K / eta + eta * T / 2`` (``router_regret_bound``)."""
    if n_experts <= 0 or eta <= 0:
        raise ValueError("regret_bound needs positive expert count and eta")
    return math.log(n_experts) / eta + eta * rounds / 2.0


def min_regret(n_experts: int, rounds: int) -> float:
    """``sqrt(2 * T * ln K)`` -- the bound at ``eta*``, for comparison."""
    return math.sqrt(2.0 * rounds * math.log(max(n_experts, 1)))


def hedge_weights(
    losses: torch.Tensor,
    eta: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """One exponential-weights update, normalized to a distribution.

    ``p_i <- p_i * exp(-eta * l_i)``, renormalized. Computed in log
    space (subtract the max) so a large ``eta`` cannot overflow.

    Args:
        losses: ``[K]`` per-expert losses this round.
        eta: the learning rate from :func:`optimal_hedge_eta`.
        weights: optional ``[K]`` previous weights; a uniform mix when
            omitted.

    Returns:
        The updated ``[K]`` weight vector, summing to 1.
    """
    if eta <= 0.0:
        raise ValueError("hedge_weights needs a positive eta")
    logp = (
        torch.log(weights.clamp_min(1e-30))
        if weights is not None
        else torch.zeros_like(losses)
    )
    logp = logp - eta * losses
    logp = logp - logp.max()
    p = torch.softmax(logp, dim=0)
    return p


def hedge_potential(
    weights: torch.Tensor, losses: torch.Tensor, eta: float
) -> tuple[float, float]:
    """``(potential, bound)`` for ``hedge_step``, for logging.

    The Lean bound is ``sum_i p_i*exp(-eta*l_i) <= 1 - eta*Lbar +
    eta^2/2`` where ``Lbar = sum_i p_i*l_i``. Returns both sides so a
    caller can see the slack; the bound holds only for ``eta <= 1``.
    """
    lbar = float((weights * losses).sum())
    potential = float((weights * torch.exp(-eta * losses)).sum())
    return potential, 1.0 - eta * lbar + eta * eta / 2.0


def minimal_topk(importance: torch.Tensor, eps_route: float) -> int:
    """``min k : sum_{i>k} c_i^2 <= eps`` (``gating_tail_bound``).

    The top-k truncation error is EXACTLY the squared tail of the
    importance scores, so k is read off the sorted cumulative tail
    rather than swept: walk down the sorted scores until the tail
    budget is met. O(K log K) for the sort, O(K) for the walk -- no
    search over k.

    Args:
        importance: ``[K]`` non-negative importance scores ``c_i``.
        eps_route: the allowed truncation error.

    Returns:
        The smallest ``k`` meeting the budget, or ``K`` when even the
        full set exceeds it (the caller then keeps everything rather
        than violating the certified bound).
    """
    if eps_route <= 0.0:
        return int(importance.numel())
    c2 = torch.sort(importance.double(), descending=True).values ** 2
    tail = torch.flip(torch.cumsum(torch.flip(c2, [0]), 0), [0])  # tail[k] = sum_{i>=k}
    ok = torch.nonzero(tail[1:] <= eps_route, as_tuple=False)
    if ok.numel() == 0:
        return int(importance.numel())
    return int(ok[0].item()) + 1