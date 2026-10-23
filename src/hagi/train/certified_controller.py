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
    "anytime_safe_threshold",
    "anytime_accept_gate",
    "anytime_freedman_term",
    "anytime_freedman_thresholds",
    "anytime_freedman_thresholds_two_sided",
    "anytime_valid_budget",
    "schedule_from_budget",
    "anytime_safety_check",
    "stochastic_noise_schedule",
    "noise_eps_like",
    "gamma_min_for_target",
    "gamma_ceiling",
    "gamma_deficit_no_takeoff",
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


def certified_argmax(
    actions: list[Action], eps_gamma: float, budget: float | None = None
) -> tuple[Action | None, dict]:
    """§AW argmax Gamma/K with the 2*eps_Gamma certification bracket.

    ``CertifiedArgmax`` + ``tie_ambiguity`` (Lean Budget/CertifiedArgmax):
    selecting the argmax over measured Gammas is legitimate ONLY when
    the top1-top2 ratio gap exceeds 2*eps_Gamma, where eps_Gamma is the
    measurement uncertainty of Gamma (split-half bracket). Inside the
    bracket the choice is a tie in disguise -- acting on it would be
    noise-chasing, so the honest verdict is AMBIGUOUS (return None).

    Returns (chosen, meta): chosen is None when nothing is
    certified-positive or the bracket is not cleared; meta records the
    bracket audit trail (ratios, gap, 2*eps_gamma, verdict).
    """
    meta: dict = {"eps_gamma": eps_gamma}
    ranked = sorted(
        (a for a in actions if a.gamma > 0.0 and a.cost > 0.0
         and (budget is None or a.cost <= budget)),
        key=lambda a: a.ratio,
        reverse=True,
    )
    meta["ratios"] = [round(a.ratio, 6) for a in ranked]
    if not ranked:
        meta["verdict"] = "NO_POSITIVE_ACTION"
        return None, meta
    if len(ranked) == 1:
        meta["verdict"] = "CERTIFIED"  # single positive action: no tie
        return ranked[0], meta
    gap = ranked[0].ratio - ranked[1].ratio
    meta["gap"] = round(gap, 6)
    meta["threshold"] = 2.0 * eps_gamma
    if gap > 2.0 * eps_gamma:
        meta["verdict"] = "CERTIFIED"
        return ranked[0], meta
    meta["verdict"] = "AMBIGUOUS"  # tie_ambiguity: inside the bracket
    return None, meta


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
    raw = math.log(2.0 / delta) / (2.0 * eps * eps)
    # Relative-tolerance ceil: when eps was derived from n exactly
    # (eps = sqrt(log(2/delta)/(2n))), floating noise in raw must not
    # push the requirement to n+1 — that froze the ladder at gen51
    # (accept_n=14400 vs required 14401 on every generation 52-86).
    return math.ceil(raw - 1e-9 * max(1.0, abs(raw)))


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


# --- Anytime-valid accept band (R523–R525, R528) --------------------------
# Port 2026-10-16: Probability/CertifiedEstimator.lean (anytime_safe_threshold)
# + Budget/AnytimeCondensation.lean. Closes the THEORY_MAP open item
# "anytime-validity in the accept gate": a gate evaluated at EVERY round t
# with a fixed threshold alpha-inflates; the uniform level alpha over T
# rounds requires the anytime band u = sqrt(2 sigma^2 ln(T/alpha)).


def anytime_safe_threshold(sigma2: float, T: int, alpha: float) -> float:
    """u = sqrt(2 sigma^2 ln(T/alpha)) — the anytime-safe band [R525].

    ``anytime_safe_threshold`` (CertifiedEstimator.lean): for a
    sub-Gaussian (CGF-bounded, parameter sigma2) statistic checked at
    every t < T, the uniform failure level alpha is controlled by the
    threshold u — versus the naive per-round band which inflates the
    total failure mass T-fold. Premises: sigma2 > 0 is the MEASURED
    (h_emp_) variance proxy of the gated statistic, T >= 1 the number of
    checks the gate will make, alpha in (0, 1) the total level.
    """
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be positive")
    if T < 1:
        raise ValueError("T must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return math.sqrt(2.0 * sigma2 * math.log(T / alpha))


def anytime_accept_gate(
    delta: float, sigma2: float, T: int, alpha: float
) -> tuple[bool, float]:
    """Anytime accept gate: |delta| must clear the R525 band.

    Composes ``anytime_safe_threshold`` with the accept decision: a
    measured improvement delta is certified only when delta > u, where u
    is the uniform band over T upcoming gate evaluations. Returns
    (accept, u) so callers log the physical threshold they cleared.
    """
    u = anytime_safe_threshold(sigma2, T, alpha)
    return delta > u, u


# --- Anytime-Freedman crossing ladder (R541-R542, FreedmanTwoSided.lean) ----
# Port: closes the THEORY_MAP open item "anytime Freedman M+V". Theorem
# ``anytime_freedman_thresholds``: for a martingale increment sequence with
# |g| <= c, zero mean and conditional variance <= v_k, the probability that
# ANY threshold rung j <= T is crossed while the running variance proxy
# stays <= V_j is bounded by the SUM of per-rung Freedman terms
# exp(-r_j^2 / (4 (V_j + c r_j))). Two-sided form (R542): same with |S|,
# factor 2 (each arm S and -S is its own threshold ladder).


def anytime_freedman_term(r: float, V: float, c: float) -> float:
    """One rung of the anytime-Freedman ladder [R541].

    ``history_additive_freedman`` (FreedmanTwoSided.lean): per-rung bound
    exp(-r^2 / (4 (V + c r))). r > 0 is the threshold rung, V >= 0 the
    variance budget, c > 0 the increment range cap.
    """
    if r <= 0.0:
        raise ValueError("r must be positive")
    if V < 0.0:
        raise ValueError("V must be nonnegative")
    if c <= 0.0:
        raise ValueError("c must be positive")
    return math.exp(-(r * r) / (4.0 * (V + c * r)))


def anytime_freedman_thresholds(
    thresholds: list[float], variances: list[float], c: float
) -> float:
    """P[exists j <= T: S_j >= r_j with Var running <= V_j] bound [R541].

    Union bound over rungs j of per-rung Freedman terms. Premises (h_emp_):
    increments bounded by c, zero conditional mean, conditional variance
    <= v_j. thresholds/variances must be nonempty, equal length (T+1).
    """
    if not thresholds or len(thresholds) != len(variances):
        raise ValueError("thresholds and variances must be nonempty, equal length")
    return sum(
        anytime_freedman_term(r, v, c) for r, v in zip(thresholds, variances)
    )


def anytime_freedman_thresholds_two_sided(
    thresholds: list[float], variances: list[float], c: float
) -> float:
    """Two-sided anytime-Freedman bound [R542]: P[exists j <= T:
    |S_j| >= r_j with Var <= V_j] <= 2 * sum of per-rung terms (each arm
    S and -S is its own ladder with the same premises). Use for symmetric
    drift monitors (two-sided CE A/B)."""
    return 2.0 * anytime_freedman_thresholds(thresholds, variances, c)


# --- Ville / anytime-safety stack (Unified/AnytimeMartingale.lean) ----------
# Port: closes THEORY_MAP open item "anytime-valid control (Ville / geometric
# schedule) on top of SafeQP". Per-step failure budgets that DECAY
# geometrically (delta0*rho^t) sum to a CONSTANT total budget delta0/(1-rho)
# over ANY horizon - valid at arbitrary stopping (union bound + geometric
# telescoping), no peeking penalty.


def anytime_valid_budget(delta0: float, rho: float) -> float:
    """Total failure budget over ANY horizon [``anytime_valid_budget``]:
    sum_t delta0*rho^t <= delta0/(1-rho) for 0 <= rho < 1. Choose the
    per-step schedule so the total budget equals the affordable alpha."""
    if delta0 < 0.0:
        raise ValueError("delta0 must be nonnegative")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    return delta0 / (1.0 - rho)


def schedule_from_budget(delta: float, rho: float) -> float:
    """Inverse of ``anytime_valid_budget``: per-step initial budget delta0
    whose geometric schedule sums to a total level ``delta``
    (premise of ``anytime_safety``: delta0/(1-rho) <= delta)."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    return delta * (1.0 - rho)


def anytime_safety_check(
    step_failures: list[float], delta0: float, rho: float, delta: float
) -> bool:
    """Honest verification of the ``anytime_safety`` premises: every
    MEASURED per-step bad-event mass must fit under the schedule
    delta0*rho^t, and the schedule total must fit the level
    (delta0/(1-rho) <= delta). True only when the certificate chain holds."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    budget = anytime_valid_budget(delta0, rho)
    # theorem premise delta0/(1-rho) <= delta; float tolerance on the
    # boundary (exact-arithmetic identity, 1 ulp of scheduling noise)
    if budget > delta and not math.isclose(budget, delta, rel_tol=1e-12):
        return False
    for t, bad in enumerate(step_failures):
        if bad > delta0 * rho**t:
            return False
    return True


def stochastic_noise_schedule(
    sigma: float, d_norm: float, n_domains: int, m_t: int,
    delta0: float, rho: float, t: int,
) -> float:
    """``anytime_safety_stochastic`` runtime form: SafeQP epsilon inflation
    at step t of the geometric schedule - noiseEps evaluated at the DECAYING
    per-step level delta_t = delta0*rho^t, so uniformity over any horizon is
    certified by the constant budget delta0/(1-rho). Premises (h_emp_):
    bounded zero-mean per-sample noise, m_t >= 1."""
    if m_t < 1:
        raise ValueError("m_t must be >= 1")
    if t < 0:
        raise ValueError("t must be nonnegative")
    return noise_eps_like(sigma, d_norm, n_domains, m_t, delta0 * rho**t)


def noise_eps_like(
    sigma: float, d_norm: float, n_domains: int, m: int, delta: float
) -> float:
    """noiseEps form (identical to ``safeqp_step.noise_eps``; lazy import,
    single SSOT): sigma * ||d|| * sqrt(2 ln(|K|/delta) / m)."""
    from hagi.train.safeqp_step import noise_eps  # lazy, single SSOT
    return noise_eps(n_domains, sigma, d_norm, m, delta)


# --- Gamma-budget of the mixer (R530, Growth/GammaDeficit.lean) -----------


def gamma_min_for_target(alpha: float, c_star: float, d_bar: float) -> float:
    """gamma >= alpha * C* / Dbar — the minimal harvest coefficient
    [R530 ``gamma_min_for_target``].

    If the frontier gate (alpha * C_t <= G_t) is to drive capability to
    the target C* while diversity stays bounded by Dbar, then gamma >=
    alpha * C* / Dbar is FORCED. Below it the ceiling gamma*Dbar/alpha
    is strictly under C* and no schedule can reach the target.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if c_star <= 0.0:
        raise ValueError("c_star must be positive")
    if d_bar <= 0.0:
        raise ValueError("d_bar must be positive")
    return alpha * c_star / d_bar


def gamma_ceiling(gamma: float, d_bar: float, alpha: float) -> float:
    """gamma * Dbar / alpha — the capability ceiling under the gate
    [R530 ``gamma_deficit_no_takeoff``].

    Under G_t <= gamma * D_t and D_t <= Dbar, capability NEVER exceeds
    gamma * Dbar / alpha, at any time step — the certified no-takeoff
    ceiling used as the honest upper bound when the gamma budget is in
    deficit.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if d_bar <= 0.0:
        raise ValueError("d_bar must be positive")
    return gamma * d_bar / alpha


def gamma_deficit_no_takeoff(
    gamma: float, alpha: float, c_star: float, d_bar: float
) -> tuple[bool, float]:
    """gamma-deficit verdict: honest takeoff requires a gamma budget
    [R530]. Returns (deficit, required_gamma); deficit is True when
    gamma * Dbar < alpha * C*, i.e. the ceiling sits strictly below the
    target — no schedule, LR or batch trick recovers it; only raising
    gamma (disagreement distill / fresh diversity) does.
    """
    req = gamma_min_for_target(alpha, c_star, d_bar)
    return gamma < req, req


# --- Plasticity radius (Growth/PlasticityLedger.lean) ------------------------


def plasticity_radius(p_rows: list[list[float]]) -> tuple[float, float]:
    """Information radius of the family as plasticity certificate
    [``plasticityRadius`` / ``kl_le_log_card`` / ``radius_zero_of_eq``
    (Growth/PlasticityLedger.lean)]:

        radius = sum_i KL(P_i || famMix),  famMix = uniform mixture

    Certificates: each KL(P_i||famMix) <= log|I|, hence
    radius <= |I| * log|I|; radius == 0 exactly when all P_i are equal
    (the family carries no plastic diversity - nothing to harvest).
    Runtime: diagnostic for the disagreement/merger channels - radius is
    the measurable (h_emp_) E_dev-proxy on the distribution level.
    Returns (radius, upper_bound |I|*log|I|).
    """
    import math as _m
    if not p_rows:
        raise ValueError("empty family")
    n_v = len(p_rows[0])
    if any(len(r) != n_v for r in p_rows):
        raise ValueError("rows must have equal length")
    for r in p_rows:
        if any(x < 0 for x in r):
            raise ValueError("probabilities must be nonnegative")
        tot = sum(r)
        if abs(tot - 1.0) > 1e-6:
            raise ValueError("rows must sum to 1")
    m = len(p_rows)
    mix = [sum(r[v] for r in p_rows) / m for v in range(n_v)]
    rad = 0.0
    for r in p_rows:
        for pv, mv in zip(r, mix):
            if pv > 0.0 and mv > 0.0:
                rad += pv * _m.log(pv / mv) / m  # mean KL over family
    # mean-KL radius: sum_i KL / |I| == KL against the mixture average;
    # the theorem's upper bound on the SUM form scales by |I|
    return rad * m, m * _m.log(m)
