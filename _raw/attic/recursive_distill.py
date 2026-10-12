"""RecursiveDistill (primes 710094c): the grow-compress loop at fixed width.

``Hagi/Ensemble/RecursiveDistill.lean``. The reverse-recursion program
(§6b.1, gen8 line): the model SPIKES at fixed width H and grows DENSITY —
``gen_k: 3 sibs (H) -> merged (3H) -> joint -> distill into a student (H)``,
``gen_{k+1}`` sibs seeded from the distillate. The Lean file proves the
bookkeeping that must hold for the loop to be worth running; this port
makes each statement a runtime decision:

``distill_chain_telescoping``
    The single-cycle bridge slack is ADDITIVE over generations, never
    multiplicative: after n cycles the distillate sits at most
    ``sum(delta_k)`` above the FIRST ensemble. Runtime: the accumulated
    slack is the running sum of per-cycle deltas -- report it, never
    assume it shrinks.

``ensemble_linear_descent`` / ``distill_compound_gain``
    If every cycle's certified growth gain ``c_k`` exceeds its
    distillation slack ``delta_k`` by ``g > 0``, the improvement
    accumulates LINEARLY in the number of cycles. The honest target is
    only "distillate gen_k beats distillate gen_{k-1}" -- the distillate
    stays worse than the uncompressed joint, accepted by design.

``distill_leak_gate``
    A cycle with ``delta >= c`` has nonpositive net -- the chain cannot
    improve through it; the loop must STOP. Runtime: the per-cycle gate,
    ``leak_gate`` below.

``harvest_budget`` / ``exhausted_when``
    The disagreement field D decays geometrically (``D_{t+1} <= rho*D_t``);
    the total harvestable gain is at most ``gamma*D_0/(1-rho)`` (the plan's
    ``G* = D/(1-rho)``), and once the remaining field ``rho^n*D_0`` drops
    below ``eps/gamma`` no future cycle can certify ``eps`` -- EXHAUSTED:
    switch corpus, not more cycles.

All functions take measured scalars (nats) and return decisions or
bounds -- no ``h_emp_`` premise hides inside.
"""

from __future__ import annotations

import math


def distill_chain_telescoping(deltas: list[float]) -> float:
    """Sum of the per-cycle bridge slacks: the additive distance to E_0.

    ``S n <= E 0 + sum_k delta k`` -- the guarantee the telescoping
    bridge leaves after n cycles. Each ``delta k`` is the single-cycle
    R134 bridge ``KL_k + M_k*||q - p_E||_1`` measured on the gate window.

    Raises:
        ValueError: on any negative delta (a slack cannot be negative).
    """
    if any(d < 0.0 for d in deltas):
        raise ValueError("deltas must be non-negative (bridge slacks)")
    return math.fsum(deltas)


def cycle_net(gain: float, delta: float) -> float:
    """``c_k - delta_k``: the honest per-cycle net (may be negative)."""
    return gain - delta


def distill_compound_gain(deltas: list[float], gains: list[float]) -> dict[str, float]:
    """Linear accumulation while the net is positive (``genMean_compound``).

    With ``hdist``/``hgrow``/``hnet`` per cycle, ``S n + n*g <= E 0 +
    delta n``. The runtime reading: the WORST per-cycle net is the rate
    the chain can certify -- ``g_min = min_k (c_k - delta_k)`` -- and the
    accumulated certified improvement after n cycles is ``n * g_min``.

    Returns:
        Dict with ``g_min`` (the certified rate; <= 0 means no compound
        guarantee), ``net_total`` (sum of nets), ``slack_total`` (the
        telescoping distance to E 0), ``n_cycles``.

    Raises:
        ValueError: on length mismatch or negative deltas.
    """
    if len(deltas) != len(gains):
        raise ValueError("deltas and gains must have the same length")
    nets = [cycle_net(c, d) for c, d in zip(gains, deltas)]
    return {
        "g_min": min(nets) if nets else 0.0,
        "net_total": math.fsum(nets),
        "slack_total": distill_chain_telescoping(deltas),
        "n_cycles": float(len(nets)),
    }


def leak_gate(delta: float, gain: float) -> dict[str, object]:
    """``distill_leak_gate``: stop the loop when the slack eats the gain.

    ``delta <= c`` means the next distillate does not regress against the
    current one (``S(n+1) <= S n``). When the slack EXCEEDS the gain the
    guarantee is void -- the chain may degrade, and the loop must STOP.

    Returns:
        Dict with ``net`` (``c - delta``), ``stop`` (True when the net is
        negative: distillation leaks more than growth certified).
    """
    net = cycle_net(gain, delta)
    return {"net": net, "stop": bool(net < 0.0)}


def harvest_budget(gamma: float, d0: float, rho: float) -> float:
    """``gamma*D_0/(1-rho)``: the total harvestable gain over any horizon.

    Free growth is finite, however long the loop runs (the plan's
    ``G* = D/(1-rho)``).

    Raises:
        ValueError: on ``rho`` outside [0, 1), non-positive ``gamma`` or
            negative ``d0``.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive (eps/gamma divides by it)")
    if d0 < 0.0:
        raise ValueError("D_0 must be non-negative")
    return gamma * d0 / (1.0 - rho)


def remaining_field(d0: float, rho: float, cycles: int) -> float:
    """``rho^n * D_0``: the disagreement field left after n cycles.

    Raises:
        ValueError: as :func:`harvest_budget` (rho range).
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    return (rho ** cycles) * d0


def exhausted_when(
    gamma: float, eps: float, d0: float, rho: float, cycles: int
) -> dict[str, object]:
    """``exhausted_when``: no future cycle can certify ``eps``.

    Once ``rho^n * D_0 < eps/gamma`` the recursion is EXHAUSTED; the next
    meaningful move is a NEW CORPUS (fresh disagreement), not more
    cycles.

    Returns:
        Dict with ``field`` (the remaining disagreement), ``threshold``
        (``eps/gamma``), ``exhausted`` (bool).

    Raises:
        ValueError: as :func:`harvest_budget``; on non-positive ``eps``.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    field = remaining_field(d0, rho, cycles)
    threshold = eps / gamma
    return {
        "field": field,
        "threshold": threshold,
        "exhausted": bool(field < threshold),
    }


def cycle_report(
    deltas: list[float],
    gains: list[float],
    *,
    gamma: float,
    eps: float,
    d0: float,
    rho: float,
) -> dict[str, object]:
    """Aggregate one recursion's ledger: compound gain, leak, exhaustion.

    The single call the distill pipeline makes after each cycle: feeds
    the measured per-cycle slacks (gate-window ``CE_student -
    CE_teacher``) and certified gains (deep-200 vs the previous
    distillate), gets back the stop/continue verdict and the exhaustion
    check that decides corpus vs more cycles.
    """
    compound = distill_compound_gain(deltas, gains)
    last_delta = deltas[-1] if deltas else 0.0
    last_gain = gains[-1] if gains else 0.0
    gate = leak_gate(last_delta, last_gain)
    exhaustion = exhausted_when(gamma, eps, d0, rho, len(deltas))
    budget = harvest_budget(gamma, d0, rho)
    return {
        **compound,
        **gate,
        "harvest_budget": budget,
        **exhaustion,
        # A cycle is only worth running when BOTH hold: the leak gate
        # (this cycle's net is positive) and the field (enough
        # disagreement left to certify eps).
        "verdict": (
            "EXHAUSTED"
            if exhaustion["exhausted"]
            else ("STOP_LEAK" if gate["stop"] else "CONTINUE")
        ),
    }


def harvest_accounting_identity(
    harvests: list[float], d0: float, d_n: float
) -> bool:
    """``harvest_accounting_identity`` (addendum): sum h_t == D_0 - D_n.

    The field pays unit for unit (``D_{t+1} = D_t - h_t``): the total
    harvested gain telescopes to the field's total drop. ALL certified
    gain comes from the field, nothing else. Runtime check: the measured
    harvests and the measured field endpoints must satisfy the identity
    to tolerance -- a violation means the ledger's "gain" is not coming
    from the field (two clocks disagreeing).
    """
    return abs(math.fsum(harvests) - (d0 - d_n)) <= 1e-9 * max(1.0, abs(d0))


def greedy_horizon_optimal(gamma: float, d0: float, n: int) -> dict[str, float]:
    """``greedy_horizon_optimal`` (addendum): the certified harvest ceiling.

    ANY schedule respecting ``h_t <= gamma*D_t`` harvests at most
    ``D_0*(1-(1-gamma)^n)`` by step n; the GREEDY schedule (extract the
    max every cycle) achieves it, and its residual field is exactly
    ``(1-gamma)^n * D_0`` -- which meets the ``exhausted_when``
    criterion. Runtime reading: this is the CEILING the ledger compares
    the measured cumulative gain against. Reaching it means the
    recursion runs greedy-optimal; staying far below while the field
    stays high means the cycles underharvest (distill slack eating the
    extractable gain, corpus too narrow).

    Raises:
        ValueError: on ``gamma`` outside (0, 1], negative ``d0`` or
            negative ``n``.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if d0 < 0.0:
        raise ValueError("D_0 must be non-negative")
    if n < 0:
        raise ValueError("n must be non-negative")
    residual = (1.0 - gamma) ** n * d0
    return {"ceiling": d0 - residual, "residual_field": residual}
