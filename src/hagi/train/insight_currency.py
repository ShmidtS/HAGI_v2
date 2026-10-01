"""The insight currency: one Lyapunov value for every action.

``Hagi/Autonomy/Insight.lean``.

``ce_gap_kl_identity`` / ``insight_kl_descent``
    ``crossEntropy q pTheta' - crossEntropy q pI
       = klDiv q pTheta' - klDiv q pI``

    The cross-entropy improvement of internalizing an insight is
    EXACTLY the KL improvement. One number, not two: the insight
    channel's gain is the KL gap, which is why ``ALGORITHMS.md`` §1
    lists ``G_insight`` (KL-gap) as the certified gain for the
    ``internalize`` action and why the drift-null property below can be
    stated in the same currency as every other action.

``experience_cycle_bound``
    One cycle obeys

        E_next <= E_t - (G_insight + G_merge + eta*||d*||^2/2
                          - C_exp - kappa*sqrt(n)*s/2)

    so growth / internalize / stop are all decided by the SIGN of one
    bracket -- a single Lyapunov currency rather than three
    incomparable heuristics. This is the quantity
    :func:`cycle_bracket` computes.

``tldr_drift_null``
    Outside the adapter's carrier the drift is EXACTLY zero, so an
    insight that lands off-carrier costs nothing and gains nothing --
    the safe default, encoded by :func:`drift_null_ok`.

The module also carries the measured cost asymmetry the theorem
comment cites: dedup-internalization was ~100x cheaper than rollout
exploration (2.9M/292k vs 720M/12M tokens), which is why
``internalize`` dominates ``explore`` in the action list.
"""

from __future__ import annotations

import math

import torch


def kl_insight_gain(
    q: torch.Tensor, p_before: torch.Tensor, p_after: torch.Tensor
) -> float:
    """``KL(q||p_after) - KL(q||p_before)`` -- the certified insight gain.

    By ``insight_kl_descent`` this EQUALS
    ``CE(q, p_after) - CE(q, p_before)``, so a caller holding
    distributions can price the insight in KL and know the CE change
    exactly, with no extra forward.

    Args:
        q: ``[V]`` the target distribution (the insight), positive.
        p_before: ``[V]`` the model distribution before internalizing.
        p_after: ``[V]`` the model distribution after.

    Returns:
        The signed gain. Positive means the insight was internalized
        (KL fell); negative means it was not.
    """
    kl_before = float((q * (q.clamp_min(1e-30).log() - p_before.clamp_min(1e-30).log())).sum())
    kl_after = float((q * (q.clamp_min(1e-30).log() - p_after.clamp_min(1e-30).log())).sum())
    return kl_before - kl_after


def cycle_bracket(
    g_insight: float,
    g_merge: float,
    dn2: float,
    eta: float,
    c_explore: float,
    c_quant: float,
) -> float:
    """The experience-cycle bracket (``experience_cycle_bound``).

    ``E_next <= E_t - bracket``. The SIGN alone decides: positive means
    the cycle certifies a net decrease in external cost and may be
    kept; negative means it must be rejected.

    Args:
        g_insight: ``G_insight`` -- the KL-gap gain of the insight.
        g_merge: ``G_merge`` -- the Jensen-gap gain of the merge.
        dn2: ``||d*||^2`` for the safe joint step.
        eta: the step size applied to ``d*``.
        c_explore: ``C_exp`` -- the exploration cost (rollouts).
        c_quant: ``kappa*sqrt(n)*s/2`` -- the compression cost.
    """
    return g_insight + g_merge + eta * dn2 / 2.0 - c_explore - c_quant


def drift_null_ok(
    b: torch.Tensor, x: torch.Tensor, a: torch.Tensor | None = None
) -> bool:
    """``tldr_drift_null``: ``B x = 0`` implies ``(A B) x = 0``, EXACTLY.

    The LoRA update is ``delta W = A @ B`` with ``A : [r, m]`` and
    ``B : [m, n]`` acting on an input ``x : [n]``. The theorem proves
    ``B x = 0`` forces ``(A B) x = A (B x) = 0`` -- zero drift, not an
    epsilon bound. An insight carried outside the adapter's support
    therefore leaves base skills provably untouched and can be dropped
    at zero cost instead of paying for a gate.

    Args:
        b: the ``B`` factor, ``[m, n]``.
        x: the input, ``[n]``.
        a: the ``A`` factor, ``[r, m]``. When given, the returned
            predicate checks the COMPOSITE drift, which is the
            statement that actually matters; when omitted only the
            hypothesis ``B x = 0`` is checked.

    Returns:
        True iff the relevant drift vanishes to numerical tolerance.
    """
    bx = b.to(x.dtype) @ x
    if a is None:
        return bool(bx.abs().max() <= 1e-6)
    composite = a.to(x.dtype) @ bx
    return bool(composite.abs().max() <= 1e-6)


def internalize_cost(tokens_insight: int, tokens_rollout: int) -> dict:
    """The measured cost asymmetry the theorem comment records.

    Deduplicating an insight (internalizing) costs orders of magnitude
    fewer tokens than producing one by rollout exploration. This is why
    ``internalize`` precedes ``grow_leaf`` in the action list: the same
    gain is available at ~1/100 of the price.

    Returns the ratio so a caller can log it against the measured
    numbers of its own run rather than the ones quoted here.
    """
    if tokens_insight <= 0 or tokens_rollout <= 0:
        raise ValueError("token counts must be positive")
    return {
        "insight_tokens": tokens_insight,
        "rollout_tokens": tokens_rollout,
        "ratio": tokens_rollout / tokens_insight,
    }
