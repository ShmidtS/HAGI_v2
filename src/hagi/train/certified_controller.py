"""Certified growth controller core (ALGORITHMS.md §0, §1, §9, §17).

Pure, dependency-free decision core for the certified growth loop:

    while True:
        1. MEASURE       twoGap, d*, G, kappa, s, inj, xi
        2. CERTIFY       Gamma_i / K_i for each action i        [R86]
        3. SELECT        argmax Gamma_i / K_i (budget B)        [R86]
        4. EXECUTE       merge/joint/internalize/prune          [R71-R82]
        5. CERTIFICATE   CE went down, old domains untouched    [R72, R81]
        6. STOP iff      consensus AND inj <= xi                [R80]

Everything here is testable without subprocess, GPU, or torch: the only
imports are ``math`` and ``dataclasses``. Torch-dependent helpers (the
merge_price twoGap gate) are imported lazily inside the function that
delegates to them, so importing this module stays cheap and pure.

Section references are to ``C:/primes/ALGORITHMS.md`` (the SSOT) and the
Lean sources under ``C:/primes/Hagi``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "Action",
    "CertifiedVerdict",
    "GrowthPhase",
    "estimate_actions",
    "select_action",
    "certified_ab",
    "hoeffding_n",
    "stop_condition",
    "leak_gate",
    "exhausted_check",
    "merge_gate_admission",
    "phase_log_line",
]

# Action names of the §1 loop.
ACTION_MERGE = "merge"
ACTION_JOINT = "joint"
ACTION_INTERNALIZE = "internalize"
ACTION_PRUNE = "prune"
ACTION_GROW_LEAF = "grow_leaf"
ACTION_DISTILL = "distill"


@dataclass(frozen=True)
class Action:
    """One certified candidate action: predicted gain Gamma, measured cost K.

    Gamma is the certified predicted gain from ALGORITHMS.md §1 (in nats
    of CE, or the action's native energy unit); K is the MEASURED cost
    (wall-clock or FLOPs) of executing it. ``ratio_dominance`` (R86)
    ranks actions by Gamma/K, so both fields are required, never
    estimated from the other.
    """

    name: str
    gamma: float
    cost: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        """Gamma_i / K_i — the certified efficiency ratio [R86]."""
        return self.gamma / self.cost


def estimate_actions(measurements: Mapping[str, Any]) -> list[Action]:
    """Compute Gamma_i and K_i for the §1 action set from measurements.

    Expected keys (missing ones simply omit that action):

    - ``two_gap`` (GapLaw) and ``merge_cost``:
        merge:        Gamma = twoGap                      [§1]
    - ``eta``, ``d_star_norm_sq`` and ``joint_cost``:
        joint:        Gamma = eta * ||d*||^2 / 2          [SafeQP_descent]
    - ``g_insight`` (KL-gap) and ``internalize_cost``:
        internalize:  Gamma = G_insight                   [insight_kl_descent]
    - ``savings``, ``delta_prune`` and ``prune_cost``:
        prune:        Gamma = savings - delta^2 / 4       [prune_certificate R72]
    - ``g_new``, ``grow_cost`` (+ ``grow_exec_cost`` for K):
        grow_leaf:    Gamma = G_new - cost                [liveness_merge]
    - ``c_k``, ``delta_k`` and ``distill_cost``:
        distill:      Gamma = c_k - delta_k               [distill_leak_gate R134]

    Every Gamma formula is a direct transcription of §1; nothing here is
    tuned. Costs default to 1.0 when the caller does not measure them,
    which degenerates the ratio ranking to a Gamma ranking -- documented
    rather than forbidden, so a measurement-poor caller still gets the
    certified gains computed honestly.
    """
    actions: list[Action] = []

    if (g := measurements.get("two_gap")) is not None:
        actions.append(Action(
            ACTION_MERGE, float(g), float(measurements.get("merge_cost", 1.0)),
            {"formula": "twoGap [GapLaw]"},
        ))
    if (eta := measurements.get("eta")) is not None and \
            (dn2 := measurements.get("d_star_norm_sq")) is not None:
        actions.append(Action(
            ACTION_JOINT, float(eta) * float(dn2) / 2.0,
            float(measurements.get("joint_cost", 1.0)),
            {"formula": "eta*||d*||^2/2 [SafeQP_descent]"},
        ))
    if (gi := measurements.get("g_insight")) is not None:
        actions.append(Action(
            ACTION_INTERNALIZE, float(gi),
            float(measurements.get("internalize_cost", 1.0)),
            {"formula": "G_insight KL-gap [insight_kl_descent]"},
        ))
    if (sv := measurements.get("savings")) is not None and \
            (dp := measurements.get("delta_prune")) is not None:
        actions.append(Action(
            ACTION_PRUNE, float(sv) - float(dp) ** 2 / 4.0,
            float(measurements.get("prune_cost", 1.0)),
            {"formula": "savings - delta^2/4 [prune_certificate R72]"},
        ))
    if (gn := measurements.get("g_new")) is not None:
        # grow: Gamma = G_new - cost [liveness_merge]; the cost enters the
        # gain itself here per §1 ("G_new - стоимость"), while K is the
        # measured execution cost.
        cost = float(measurements.get("grow_cost", 0.0))
        actions.append(Action(
            ACTION_GROW_LEAF, float(gn) - cost,
            float(measurements.get("grow_exec_cost", 1.0)),
            {"formula": "G_new - cost [liveness_merge]"},
        ))
    if (ck := measurements.get("c_k")) is not None and \
            (dk := measurements.get("delta_k")) is not None:
        actions.append(Action(
            ACTION_DISTILL, float(ck) - float(dk),
            float(measurements.get("distill_cost", 1.0)),
            {"formula": "c_k - delta_k [distill_leak_gate R134]"},
        ))
    return actions


def select_action(actions: list[Action], budget: float | None = None) -> Action | None:
    """Argmax Gamma_i / K_i under the budget [ratio_dominance R86].

    ``budget_allocation_dominance`` (R86): concentrating the budget on
    the argmax Gamma_certified/K dominates ANY split of the budget, so
    the selection is a single argmax, not an allocation.

    Actions whose cost exceeds ``budget`` (when one is given) are not
    executable this cycle and are excluded. Actions with Gamma <= 0 have
    no certified gain and are excluded. Returns None when nothing is
    both affordable and certified-positive. Ties break on the FIRST
    action in list order, so the caller controls tie-breaking by
    controlling the order (deterministic, testable).
    """
    best: Action | None = None
    for a in actions:
        if a.gamma <= 0.0 or a.cost <= 0.0:
            continue
        if budget is not None and a.cost > budget:
            continue
        if best is None or a.ratio > best.ratio:
            best = a
    return best


# --- §9: certified A/B comparison (Hoeffding premise, R114/R118) ----------


@dataclass(frozen=True)
class CertifiedVerdict:
    """Outcome of a certified A/B comparison (§9).

    ``accepted``: B is certified better than A (CE_A - CE_B > 2*eps with
        n at or above the Hoeffding threshold).
    ``rejected``: B is certified worse (CE_B - CE_A > reject_margin with
        n sufficient). The default reject margin is the same 2*eps; a
        caller may pass a narrower one (deep200_certify uses eps).
    ``undecided``: neither margin cleared at this sample size -- the
        comparison lacks the evidence to move in EITHER direction.
    """

    verdict: str            # "ACCEPT" | "REJECT" | "UNDECIDED"
    delta: float            # ce_a - ce_b: positive = B better
    n: int
    n_required: int
    reason: str

    @property
    def accepted(self) -> bool:
        return self.verdict == "ACCEPT"

    @property
    def rejected(self) -> bool:
        return self.verdict == "REJECT"

    @property
    def undecided(self) -> bool:
        return self.verdict == "UNDECIDED"


def hoeffding_n(eps: float, delta: float) -> int:
    """n >= log(2/delta) / (2 eps^2) -- the §9 sample threshold.

    With this many samples, an observed mean margin > 2*eps certifies
    mu_A <= mu_B with probability >= 1 - delta (certified_premise,
    R114). Pure ceiling-free form: callers compare ``n >= hoeffding_n``.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return math.ceil(math.log(2.0 / delta) / (2.0 * eps * eps))


def certified_ab(
    ce_a: float,
    n_a: int,
    ce_b: float,
    n_b: int,
    eps: float,
    delta: float,
    reject_margin: float | None = None,
) -> CertifiedVerdict:
    """§9: accept B iff CE_A - CE_B > 2*eps with n >= log(2/delta)/(2 eps^2).

    ``n`` is the effective sample size ``min(n_a, n_b)``: the weaker of
    the two estimates bounds the Hoeffding premise. ``reject_margin`` is
    the certified-regression margin for B (default the same 2*eps;
    deep200_certify passes ``eps`` to keep its historical asymmetry).
    """
    rej = 2.0 * eps if reject_margin is None else reject_margin
    # n may be inf (callers that make no sample-size claim); compare in
    # float and only round for reporting.
    n = min(n_a, n_b)
    n_req = hoeffding_n(eps, delta)
    d = float(ce_a) - float(ce_b)
    if n < n_req:
        return CertifiedVerdict(
            "UNDECIDED", d, n, n_req,
            f"n={n} < n_required={n_req}: the margin {d:+.4f} cannot be "
            f"certified at eps={eps:g}, delta={delta:g}",
        )
    if d > 2.0 * eps:
        return CertifiedVerdict(
            "ACCEPT", d, n, n_req,
            f"CE_A - CE_B = {d:+.4f} > 2*eps = {2 * eps:.4f} with n={n} "
            f">= {n_req}: false-upgrade probability <= delta",
        )
    if d < -rej:
        return CertifiedVerdict(
            "REJECT", d, n, n_req,
            f"CE_B - CE_A = {-d:+.4f} > margin {rej:.4f} with n={n} >= {n_req}",
        )
    return CertifiedVerdict(
        "UNDECIDED", d, n, n_req,
        f"margin {d:+.4f} inside the certification band "
        f"(-{rej:.4f}, {2 * eps:.4f}) at n={n}",
    )


# --- §0/§4: stop conditions ----------------------------------------------


def stop_condition(consensus: bool, inj: float, xi: float) -> bool:
    """liveness_two_axis (R80): STOP iff consensus AND inj <= xi.

    The loop cannot freeze while either axis is alive: disagreement
    (consensus = False) or a live data axis (inj > xi). A stop under
    this predicate is an honest stop, not a stall.
    """
    return bool(consensus) and inj <= xi


def leak_gate(cycle_gain: float, distill_slack: float) -> bool:
    """distill_leak_gate (R134, §17): continue the distill recursion?

    The recursion pays while ``c_k - delta_k >= g``; when the leak
    ``delta`` reaches the cycle gain ``c`` the recursion only distills
    its own leak, so it must STOP, not be forced. Returns True when the
    gate ALLOWS another cycle (slack below the gain), False when the
    recursion must stop.
    """
    return distill_slack < cycle_gain


def exhausted_check(
    rho: float, D0: float, eps: float, gamma: float, n: int = 1
) -> bool:
    """exhausted_when (§17): rho^n * D_0 < eps/gamma => EXHAUSTED.

    The greedy harvest decays the frontier geometrically; once the
    residual frontier drops below ``eps/gamma`` no further harvest step
    can clear ``eps``. EXHAUSTED means: switch corpus (inject fresh
    data), do not keep cycling. ``n`` is the number of decay steps
    already applied; default 1 (the per-cycle check).
    """
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie in (0, 1)")
    if eps <= 0.0 or gamma <= 0.0:
        raise ValueError("eps and gamma must be positive")
    if n < 1:
        raise ValueError("n must be >= 1")
    return (rho ** n) * D0 < eps / gamma


def merge_gate_admission(
    two_gap: float,
    prices: list[float],
    weights: list[float] | None = None,
    kappa: float = 0.05,
    s: float = 1.0,
    n: int = 200,
) -> bool:
    """Delegate the merge admission to the twoGap gate (merge_price, Th.1).

    ALGORITHMS.md §3: MERGE iff twoGap > sum_i w_i price_i + kappa
    sqrt(n) s / 2. The certified formula lives in
    ``hagi.train.merge_price.merge_gate``; this seam delegates to it so
    there is ONE implementation of the gate (torch import is lazy to
    keep this module importable without a GPU stack).
    """
    from hagi.train.merge_price import merge_gate

    return merge_gate(
        two_gap, prices, weights=weights, kappa=kappa, s=s, n=n
    )


# --- §23: phase annotations ----------------------------------------------


class GrowthPhase:
    """The §23 single-training-pipeline phases, as loggable names.

    Log-only: mapping a supervisor stage to its theory phase annotates
    the ledger with which part of the certified loop produced each row.
    """

    DISCOVER = "DISCOVER"
    MEASURE = "MEASURE"
    SELECT = "SELECT"
    GATE = "GATE"
    DISTILL = "DISTILL"
    CONSOLIDATE = "CONSOLIDATE"
    IGNITE = "IGNITE"
    SAFE_UPDATE = "SAFE_UPDATE"
    LOOP = "LOOP"
    STOP_CONTINUE = "STOP/CONTINUE"
    COMPRESS = "COMPRESS"
    ROUTE = "ROUTE"

    ALL = (
        DISCOVER, MEASURE, SELECT, GATE, DISTILL, CONSOLIDATE, IGNITE,
        SAFE_UPDATE, LOOP, STOP_CONTINUE, COMPRESS, ROUTE,
    )


def phase_log_line(phase: str, lane: str, detail: str = "") -> str:
    """One canonical ledger/log line naming the theory phase (§23)."""
    if phase not in GrowthPhase.ALL:
        raise ValueError(f"unknown phase {phase!r}")
    line = f"[phase {phase}] lane={lane}"
    if detail:
        line += f" {detail}"
    return line
