"""PPT discovery: a power target with a certified search layer.

``Hagi/Discovery/PPT.lean`` (R98). The discovery layer needs a target
distribution over candidate trajectories that favours BETTER ones
without becoming greedy, and it needs to be able to SAY how likely a
useful discovery is before spending the compute.

``pptTarget_isProbability`` / ``_pos``
    ``pi_alpha(x) = p(x)^alpha / Z_alpha`` is a proper distribution on a
    finite trajectory space for ``alpha > 0``.

    A boundary worth stating explicitly, because it is easy to conflate:
    this is a power of the JOINT SEQUENCE probability, NOT a tokenwise
    temperature. The Concat law ``softmax(c z) != softmax(z)`` proven
    earlier in this project does NOT transfer here, and no equivalence
    is claimed. Per-sequence powering and per-token scaling are
    different operators and only the former is what is proved.

``mh_stationary``
    A Metropolis-Hastings kernel with detailed balance against
    ``pi_alpha`` has ``pi_alpha`` as its stationary distribution. So a
    refinement step does not drift the target -- this is what makes a
    long search trustworthy.

``pptSwap_stationary``
    A swap kernel between adjacent levels of the ``alpha`` ladder
    preserves the PRODUCT of the two stationary distributions; the
    ``min(1, ratio)`` acceptance compensates exactly.

``pptMixing`` / ``ppt_tvContraction``
    Under a Doeblin condition (every state sees some input of mass at
    least ``eps``), total-variation distance contracts GEOMETRICALLY.
    This is the mixing-speed claim, and it is what makes the stationary
    mass usable as a prediction rather than a wish.

``truncation_bias``
    A kernel confined to a subset ``S`` CANNOT preserve a target with
    mass outside ``S``. One-sided truncation is structurally biased.
    Practical consequence for this project: replay buffers and experience
    compression may not silently drop states and then claim the data
    distribution is unchanged.

``pptSuccess`` / ``pptSuccess_lower``
    A composite success indicator -- PPT found a gamma-good candidate
    AND the verifier accepted it AND the budget was respected -- with an
    explicit union-bound lower bound. This is the ``S_t`` that feeds the
    already-proved R95 concentration and R96 takeoff, so ``p0`` is a
    measured quantity rather than a postulate: under the stationary
    law, ``Pr[find a gamma-good candidate]`` IS the stationary mass of
    the good set, computable by a forward pass over the current model
    BEFORE the expensive sampling run.
"""

from __future__ import annotations

import math

import torch


def ppt_target(probs: torch.Tensor, alpha: float) -> torch.Tensor:
    """``pi_alpha(x) = p(x)^alpha / Z_alpha`` -- the power target.

    Args:
        probs: ``[N]`` non-negative weights over candidates.
        alpha: the sharpness exponent, strictly positive. ``alpha = 1``
            returns the input normalized; larger alpha concentrates on
            the heavy tail.

    Returns:
        ``[N]`` the normalized target.

    Raises:
        ValueError: on non-positive ``alpha`` or an all-zero input.
    """
    if alpha <= 0.0:
        raise ValueError("ppt_target needs a positive alpha")
    w = probs.double().clamp_min(0.0)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("ppt_target needs a non-zero weight vector")
    powered = torch.pow(w / total, alpha)
    norm = float(powered.sum())
    if norm <= 0.0:
        raise ValueError("ppt_target: the powered weights vanished")
    return powered / norm


def stationary_good_mass(
    probs: torch.Tensor, good: torch.Tensor, alpha: float
) -> float:
    """``Pr[PPT finds a gamma-good candidate]`` under the target.

    THE point of the layer: this is the discovery probability
    ``p0`` that R95's concentration needs, and it is COMPUTABLE from the
    current model's own scores -- a forward pass, before any sampling.
    So the expensive run can be launched only when the predicted
    discovery probability justifies it.

    Args:
        probs: ``[N]`` candidate weights.
        good: ``[N]`` boolean mask of gamma-good candidates.
        alpha: the power-target exponent.

    Returns:
        The stationary mass of the good set, in ``[0, 1]``.
    """
    target = ppt_target(probs, alpha)
    mask = good.bool()
    if not mask.any():
        return 0.0
    return float(target[mask].sum())


def mh_acceptance_ratio(
    log_pi_new: float, log_pi_old: float, log_q_back: float, log_q_fwd: float
) -> float:
    """``min(1, pi(y)q(x|y) / (pi(x)q(y|x)))`` -- the MH acceptance.

    ``mh_stationary``: a kernel accepting with this probability has
    ``pi`` as its stationary distribution (detailed balance). Computed
    in log space so a large ratio cannot overflow.

    Args:
        log_pi_new: ``log pi(y)``.
        log_pi_old: ``log pi(x)``.
        log_q_back: ``log q(x|y)``.
        log_q_fwd: ``log q(y|x)``.

    Returns:
        The acceptance probability in ``[0, 1]``.
    """
    log_ratio = log_pi_new + log_q_back - log_pi_old - log_q_fwd
    if log_ratio >= 0.0:
        return 1.0
    return float(math.exp(log_ratio))


def tv_contraction_bound(
    min_input_mass: float, gap: float, steps: int = 1
) -> float:
    """``(1 - gap)^t * dTV`` -- the geometric Doeblin contraction.

    ``pptMixing`` / ``ppt_tvContraction``: with every state seeing some
    input of mass at least ``eps = min_input_mass``, total variation
    contracts by at least ``gap`` (e.g. the big-jump gap ``1-eps``) per
    step. The bound is uniform in the starting distribution, which is
    what makes the stationary mass a usable prediction.

    Args:
        min_input_mass: ``eps`` in ``(0, 1]``; accepted for symmetry with
            the theorem, the per-step factor is ``gap`` itself.
        gap: per-step contraction factor in ``(0, 1]``.
        steps: ``t``, non-negative.

    Returns:
        The remaining distance multiplier after ``t`` steps.

    Raises:
        ValueError: on out-of-range inputs.
    """
    if not 0.0 < min_input_mass <= 1.0:
        raise ValueError("min_input_mass must lie in (0, 1]")
    if not 0.0 < gap <= 1.0:
        raise ValueError("gap must lie in (0, 1]")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return float((1.0 - gap) ** steps)


def steps_to_mix(min_input_mass: float, gap: float, tolerance: float) -> int:
    """Smallest ``t`` with the TV bound below ``tolerance``.

    Inverting the geometric contraction, so "how long until this
    stationary mass is trustworthy" is arithmetic.

    Args:
        min_input_mass: ``eps``.
        gap: per-step contraction.
        tolerance: target distance, in ``(0, 1)``.

    Returns:
        The number of steps (at least 1).

    Raises:
        ValueError: on bad inputs.
    """
    if not 0.0 < min_input_mass <= 1.0:
        raise ValueError("min_input_mass must lie in (0, 1]")
    if not 0.0 < gap <= 1.0:
        raise ValueError("gap must lie in (0, 1]")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie in (0, 1)")
    if gap >= 1.0:
        return 1
    n = math.ceil(math.log(tolerance) / math.log1p(-gap))
    return max(1, n)


def truncation_bias(probs: torch.Tensor, keep: torch.Tensor) -> float:
    """Mass LOST by confining the kernel to a subset -- ``truncation_bias``.

    A kernel confined to ``keep`` cannot preserve a target with mass
    outside it. The dropped mass IS the bias: the retained target is
    renormalised, so the distribution is not merely missing states, it
    is a DIFFERENT distribution.

    Args:
        probs: ``[N]`` the original target weights (unnormalized is fine).
        keep: ``[N]`` boolean mask of retained states.

    Returns:
        The discarded total mass, in ``[0, 1]``. Zero means the
        truncation is lossless and the bias claim does not apply.
    """
    w = probs.double().clamp_min(0.0)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("truncation_bias needs a non-zero weight vector")
    kept = float(w[keep.bool()].sum())
    return max(0.0, min(1.0, 1.0 - kept / total))


def ppt_success_lower(
    find_good: float, verify_accept: float, budget_ok: float
) -> float:
    """``Pr[find AND verify AND budget] >= find + verify + budget - 2``.

    ``pptSuccess_lower``: the composite indicator's lower bound by the
    union bound on the three failure events. It is a FLOOR, not an
    estimate -- tight when the three events nearly partition the
    failure mass, loose when they overlap.

    This is the ``p0`` R95 consumes, and being a floor means a takeoff
    computed from it is a takeoff of the WORST case.

    Args:
        find_good: probability the search finds a good candidate.
        verify_accept: probability the verifier accepts.
        budget_ok: probability the budget is respected.

    Returns:
        The lower bound, clipped to ``[0, 1]``.
    """
    bound = find_good + verify_accept + budget_ok - 2.0
    return max(0.0, min(1.0, bound))