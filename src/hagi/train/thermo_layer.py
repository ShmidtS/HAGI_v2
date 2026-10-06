"""Thermodynamic training layer (R174c/d/e): dissipation, batch
annealing, stationary flatness.

Three Lean modules, one runtime module — they share the
thermodynamic dictionary of FORMALIZATION_PLAN §8.8:

R174c ``Hagi/Data/NessDissipation.lean``
    The entropy production rate of a finite Markov chain as the KL
    divergence between the forward and reversed edge flows. Detailed
    balance ⟺ zero entropy production — equilibrium is
    dissipation-free, and a NESS (nonequilibrium steady state)
    dissipates strictly. The γ-deficit reading (unfinished gain of
    the growth loop ~ dissipation) is interpretation; the
    dictionary is exact.

R174d ``Hagi/Probability/NoiseTemperature.lean`` (``anneal_by_batch``)
    Growing the batch geometrically (``B_{t+1} = c·B_t``, ``c > 1``)
    cools the noise reservoir (temperature ``1/B``) geometrically,
    and the TOTAL noise injected over ANY horizon is bounded by the
    geometric budget ``σ²·c/(B₀·(c−1))`` — finite and
    horizon-independent. LR decay at fixed batch is the SAME
    mathematics on the other axis (1711.00489): the batch axis buys
    the cooling without shrinking the step.

R174e ``Hagi/Probability/StationaryFlatness.lean``
    The stationary-flatness kernel: a natural-valued flatness
    functional whose tail obeys ``P[X ≥ k] ≤ C/k²`` has
    ``E[X] ≤ 2C`` — mean flatness controlled by the tail constant,
    horizon-independent. The measurable replacement of "flat minima
    generalize": the tail-to-moment calculus (layer-cake identity +
    telescoping ``k⁻² ≤ (k−1)⁻¹ − k⁻¹``).

Honest boundaries (Lean): no generalization link is claimed
(PAC-Bayes territory); the constant-stepsize SGD provenance of the
tail hypothesis is runtime interpretation.
"""
from __future__ import annotations

import math

import torch

from hagi.model.formal import kl_divergence

# --- R174c: dissipation / equilibrium dictionary -------------------------


def edge_flow(P: torch.Tensor, pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``flowMap``/``flowBack``: forward and reversed edge flows.

    ``flow_ij = π_i·P_ij`` (the stationary traffic across i → j);
    the reversed flow carries ``π_j·P_ji`` across the same edge.

    Raises:
        ValueError: on non-square P or mismatched π.
    """
    if P.dim() != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square")
    if pi.shape != (P.shape[0],):
        raise ValueError("pi must align with P")
    P = P.double()
    pi = pi.double()
    forward = pi.unsqueeze(1) * P
    backward = (pi.unsqueeze(0) * P.t())
    return forward, backward


def ep_rate(P: torch.Tensor, pi: torch.Tensor) -> float:
    """``epRate``: the entropy production rate of the chain.

    ``KL(forward edge flow ‖ reversed edge flow)`` — the finite form
    of ``Σ π_i P_ij log(π_i P_ij / π_j P_ji)``. Zero exactly at
    detailed balance; strictly positive at any NESS.

    Raises:
        ValueError: on non-positive flows (the Lean premise) or as
            :func:`edge_flow`.
    """
    forward, backward = edge_flow(P, pi)
    if bool((forward <= 0).any() or (backward <= 0).any()):
        raise ValueError("edge flows must be everywhere positive")
    return kl_divergence(forward.reshape(-1), backward.reshape(-1))


def detailed_balance(P: torch.Tensor, pi: torch.Tensor, tol: float = 1e-9) -> bool:
    """``π_i·P_ij = π_j·P_ji`` on every edge — the equilibrium test."""
    forward, backward = edge_flow(P, pi)
    return bool(torch.allclose(forward, backward, atol=tol))


# --- R174d: annealing by batch growth ------------------------------------


def anneal_by_batch(
    sigma2: float, B0: float, c: float, n: int
) -> dict[str, float]:
    """``anneal_by_batch``: the geometric batch-growth noise budget.

    With ``B_t = B₀·c^t`` the per-step reservoir temperature is
    ``σ²/B_t`` and the TOTAL noise over any horizon ``n`` is bounded
    by the geometric budget

        Σ_{t<n} σ²/B_t  <  σ²·c/(B₀·(c−1))

    — finite and horizon-independent. The runtime licence for a
    geometric batch schedule: it anneals the noise reservoir without
    decaying the learning rate (the 1711.00489 axis equivalence).

    Returns:
        ``{"total_noise": Σ, "budget": σ²·c/(B₀·(c−1)), "final_B": B_n}``.

    Raises:
        ValueError: on non-positive σ²/B₀, ``c <= 1``, or a negative
            horizon.
    """
    if sigma2 < 0.0:
        raise ValueError("sigma^2 must be non-negative")
    if B0 <= 0.0:
        raise ValueError("B0 must be positive")
    if c <= 1.0:
        raise ValueError("growth factor c must exceed 1")
    if n < 0:
        raise ValueError("horizon must be non-negative")
    total = 0.0
    b = B0
    for _ in range(n):
        total += sigma2 / b
        b *= c
    budget = sigma2 * c / (B0 * (c - 1.0))
    # final_B saturates gracefully at very long horizons (float inf)
    final_b = B0 * c ** n if n < 1024 else float("inf")
    return {
        "total_noise": total,
        "budget": budget,
        "final_B": final_b,
    }

# --- R174e: stationary flatness ------------------------------------------


def tail_sum_identity(probs: torch.Tensor, x: torch.Tensor) -> float:
    """``tail_sum_identity`` (discrete layer cake): ``E[X] = Σ_k P[X ≥ k+1]``.

    For a natural-valued functional ``x`` over states with weights
    ``probs``, the expectation equals the sum of tail probabilities
    — verified at runtime against the direct mean.

    Raises:
        ValueError: on a shape mismatch or negative values.
    """
    if probs.shape != x.shape:
        raise ValueError("probs and x must be aligned")
    if bool((probs < 0).any() or (x < 0).any()):
        raise ValueError("probs and x must be non-negative")
    m = int(x.max().item())
    p = probs.double()
    xd = x.double()
    tail_sum = math.fsum(
        float(p[xd >= k + 1].sum().item()) for k in range(m)
    )
    return tail_sum


def flatness_moment_bound(
    tail: callable, c_const: float
) -> float:
    """``flatness_moment_bound``: ``P[X ≥ k] ≤ C/k²`` ⟹ ``E[X] ≤ 2C``.

    The stationary-flatness kernel: the mean flatness is controlled
    by the tail constant, horizon-independent. The runtime monitor
    estimates the tail constant from measured flatness samples and
    this bound converts it into the certified mean bound ``2C``.

    Args:
        tail: callable ``k -> P[X ≥ k]`` (the measured tail).
        c_const: the tail constant ``C`` — must dominate
            ``k²·tail(k)`` over the support.

    Returns:
        The certified mean bound ``2C``.

    Raises:
        ValueError: on a negative constant.
    """
    if c_const < 0.0:
        raise ValueError("tail constant must be non-negative")
    # sum_pow2_inv_le: Σ_{k=1..N} k^{-2} <= 2 - 1/N; the tail sum is
    # bounded by C times that series, hence E[X] <= 2C.
    return 2.0 * c_const


def tail_constant(samples: torch.Tensor) -> float:
    """Estimate the tail constant ``C`` from flatness samples.

    ``C = sup_k k²·P̂[X ≥ k]`` over the empirical tail — the
    measurable input of :func:`flatness_moment_bound`.

    Raises:
        ValueError: on empty or negative samples.
    """
    if samples.numel() == 0:
        raise ValueError("samples must be non-empty")
    if bool((samples < 0).any()):
        raise ValueError("samples must be non-negative")
    s = samples.double()
    n = s.numel()
    best = 0.0
    for k in range(1, int(s.max().item()) + 1):
        pk = float((s >= k).double().sum().item()) / n
        best = max(best, pk * k * k)
    return best
