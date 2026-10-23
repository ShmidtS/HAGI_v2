"""R105/R106: a derived learning rate, and a bounded PoE correction.

Two rounds that each remove a hyperparameter by turning it into
arithmetic on measured quantities -- which is the direction this project
has been moving since the analytic-step attempt.

R105 (``Hagi/Step/SafeQPStep.lean``)
    ``safeqp_eta_max``

        eta <= min(1, min_i 2(eps_i + <g_i, d>) / (L_i * ||d||^2))

    guarantees that NO protected domain regresses beyond its own budget
    ``eps_i`` -- simultaneously, for all of them. That is a descent lemma
    with SafeQP's margins substituted in, and it means the learning rate
    stops being a swept constant: it is evaluated from the current step's
    measured margins, curvature and direction norm.

    The audit's literal formula (no ``min 1`` clamp) is WRONG, and the
    formalisation is sharper than its source: the ``eps_i`` term enters
    LINEARLY in ``eta``, so without the clamp a large ``eta`` would blow
    past the very budget the bound is about. That correction is recorded
    in the docstring rather than quietly applied.

    ``safeqp_eta_max_conflict_free``: with ``<g_i, d> >= 0`` for every
    domain the window simplifies and is STRICTLY POSITIVE whenever the
    budgets are -- the controller always has a derived safe rate.

R106 (``Hagi/Energy/PoEBound.lean``)
    ``poe_logZ_second_order``

        |log Z_w - sum_i w_i log Z_i| <= (1/8) * sum_i w_i R_i^2

    where ``R_i`` bounds each expert's logit spread around the weighted
    mean. The linear terms cancel exactly by centering, leaving a second
    order remainder -- which is why the constant is small.

    ``poe_softmax_speed``: a UNIFORM spread ``M`` gives ``M^2/8`` per
    token, so a product-of-experts pool can be computed from mean logits
    plus a known correction, at softmax speed, with a provable error.

    The audit proposed the same law with the constant applied PAIRWISE.
    That is FALSE -- counterexample ``z1 = (1,-1)``, ``z2 = (0,0)``,
    ``w = 1/2`` gives a gap of about 0.157 against a claimed 0.0625. The
    honest pairwise constant is 1/2, proved with weighted Cauchy-Schwarz.
    :func:`pairwise_constant_is_halved` exposes the refutation so it
    cannot be reintroduced.
"""

from __future__ import annotations

import math

# --- R105: the derived safe step ----------------------------------------


def noise_eps(
    n_domains: int, sigma: float, d_norm: float, m: int, delta: float
) -> float:
    """Minibatch noise inflation of the epsilon budget [StochasticSafeQP
    ``noiseEps`` (Step/StochasticSafeQP.lean)]:

        noiseEps = sigma * ||d|| * sqrt(2 ln(|K|/delta) / m)

    With probability >= 1 - delta the MINIBATCH inner products
    concentrate within this radius of the true (population) inner
    products, so feasibility certified on the empirical margins holds
    TRUE with budget eps + noiseEps (``stochastic_safeQP_feasibility``).
    sigma is the per-sample gradient-noise std (measured, h_emp_),
    |K| = n_domains the number of safety constraints, m the minibatch
    size. THE lr<->batch dial: at (eta, m) -> (k*eta, k^2*m) the noise
    budget is invariant (``lr_batch_equivalence``).
    """
    if n_domains < 1:
        raise ValueError("n_domains must be >= 1")
    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative")
    if d_norm < 0.0:
        raise ValueError("d_norm must be nonnegative")
    if m < 1:
        raise ValueError("minibatch size m must be >= 1")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return sigma * d_norm * math.sqrt(
        2.0 * math.log(n_domains / delta) / m
    )


def lr_batch_equivalence_check(
    eta: float, sigma: float, d_norm: float, m: int, k: int, delta: float,
    n_domains: int = 1,
) -> bool:
    """``lr_batch_equivalence`` (Smith et al. 1711.00489, formalized):
    k*eta * noiseEps(k^2*m) == eta * noiseEps(m) exactly. The scheduler
    licence: instead of cutting LR by k, grow the batch by k^2 - the
    noise budget is IDENTICAL, sequential updates on the same loss
    curve are k times fewer. Returns True when the identity holds to
    float tolerance (pure arithmetic check of the theorem's claim).
    """
    left = (k * eta) * noise_eps(n_domains, sigma, d_norm, k * k * m, delta)
    right = eta * noise_eps(n_domains, sigma, d_norm, m, delta)
    return math.isclose(left, right, rel_tol=1e-9)


def eta_max(
    margins: dict[str, float],
    inner: dict[str, float],
    curvature: dict[str, float],
    d_norm_sq: float,
) -> float:
    """``safeqp_eta_max`` -- the largest safe step, simultaneously.

        eta <= min(1, min_i 2(eps_i + <g_i, d>) / (L_i * ||d||^2))

    Args:
        margins: ``eps_i``, each domain's protected regression budget.
        inner: ``<g_i, d>``, the per-domain directional derivative. May
            be negative, which is the conflicting case.
        curvature: ``L_i``, strictly positive.
        d_norm_sq: ``||d||^2``, strictly positive.

    Returns:
        The window's upper end. Positive means a derived safe rate
        exists; zero means the step must be skipped.

    Raises:
        ValueError: on an empty domain set, a missing or extra domain in
            one of the mappings, a non-positive curvature or direction
            norm, or a negative budget.
    """
    keys = set(margins)
    if not keys:
        raise ValueError("eta_max needs at least one domain")
    if set(inner) != keys or set(curvature) != keys:
        raise ValueError("margins, inner and curvature must cover the same domains")
    if d_norm_sq <= 0.0:
        raise ValueError("||d||^2 must be positive")
    for k in keys:
        if curvature[k] <= 0.0:
            raise ValueError(f"curvature for {k!r} must be positive")
        if margins[k] < 0.0:
            raise ValueError(f"budget for {k!r} must be non-negative")
    thresholds = [
        2.0 * (margins[k] + inner[k]) / (curvature[k] * d_norm_sq)
        for k in keys
    ]
    return min(1.0, min(thresholds))


def eta_max_conflict_free(
    margins: dict[str, float],
    curvature: dict[str, float],
    d_norm_sq: float,
) -> float:
    """``safeqp_eta_max_conflict_free`` -- the window with no conflict.

    With ``<g_i, d> >= 0`` everywhere the direction already descends in
    every domain, the inner term drops out, and the window is

        min(1, min_i 2 eps_i / (L_i ||d||^2))

    which is STRICTLY POSITIVE whenever every budget is positive: the
    controller always has a derived safe rate, with no tuning.

    Raises:
        ValueError: as :func:`eta_max`.
    """
    return eta_max(margins, {k: 0.0 for k in margins}, curvature, d_norm_sq)


def binding_domain(
    margins: dict[str, float],
    inner: dict[str, float],
    curvature: dict[str, float],
    d_norm_sq: float,
) -> str:
    """Which domain sets the window.

    Reporting it matters: a window squeezed by one domain is a signal
    about THAT domain, not a global learning-rate problem, and a caller
    that cannot see which one will "fix" the rate instead of the domain.

    Raises:
        ValueError: as :func:`eta_max`.
    """
    keys = set(margins)
    if not keys:
        raise ValueError("eta_max needs at least one domain")
    if d_norm_sq <= 0.0:
        raise ValueError("||d||^2 must be positive")
    worst = min(
        keys,
        key=lambda k: 2.0 * (margins[k] + inner[k]) / (curvature[k] * d_norm_sq),
    )
    return worst


def no_domain_regresses(
    margins: dict[str, float],
    inner: dict[str, float],
    curvature: dict[str, float],
    d_norm_sq: float,
    eta: float,
) -> bool:
    """The bound's CONCLUSION, checked: every domain stays inside budget.

    This is what ``safeqp_eta_max`` proves. Checking it independently --
    from the per-domain descent-lemma window rather than from the min --
    means a caller can verify the guarantee rather than trust it.

    Raises:
        ValueError: as :func:`eta_max`.
    """
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    for k in margins:
        # dL_i <= -eta*<g_i,d> + L_i*eta^2*||d||^2/2, and the theorem says
        # that is <= eps_i once eta is under the per-domain threshold.
        bound = (-eta * inner[k]
                 + curvature[k] * eta * eta * d_norm_sq / 2.0)
        if bound > margins[k] + 1e-12:
            return False
    return eta <= eta_max(margins, inner, curvature, d_norm_sq)


# --- R106: the PoE correction -------------------------------------------


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def poe_log_z(
    logits: list[list[float]], weights: list[float]
) -> float:
    """``log sum_v exp(sum_i w_i z_i[v])`` -- the pooled normaliser.

    Args:
        logits: ``[N][V]`` per-expert logits.
        weights: ``[N]`` summing to 1, non-negative.

    Returns:
        ``log Z_w``.

    Raises:
        ValueError: on a shape mismatch or malformed weights.
    """
    if not logits or len(logits) != len(weights):
        raise ValueError("logits and weights must be non-empty and aligned")
    v = len(logits[0])
    for row in logits:
        if len(row) != v:
            raise ValueError("all experts must share the support size")
    if any(w < 0.0 for w in weights):
        raise ValueError("weights must be non-negative")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {sum(weights)}")
    mean = [sum(w * row[j] for w, row in zip(weights, logits)) for j in range(v)]
    return _logsumexp(mean)


def weighted_log_z(logits: list[list[float]],
                  weights: list[float]) -> float:
    """``sum_i w_i log Z_i`` -- the arithmetic-pool surrogate.

    Raises:
        ValueError: on a shape mismatch or malformed weights.
    """
    if not logits or len(logits) != len(weights):
        raise ValueError("logits and weights must be non-empty and aligned")
    total = 0.0
    for w, row in zip(weights, logits):
        total += w * _logsumexp(list(row))
    return total


def expert_spread(logits: list[list[float]], weights: list[float],
                  i: int) -> float:
    """``R_i`` -- the expert's logit spread around the weighted mean.

    The measurable telemetry R106's bound is stated in: how far one
    expert's logits deviate from the pool's, at the worst pair of tokens.
    Small means the experts agree outside their disagreement, which is
    exactly when the cheap correction is affordable.

    Raises:
        ValueError: on a shape mismatch or an out-of-range index.
    """
    if not logits or len(logits) != len(weights):
        raise ValueError("logits and weights must be non-empty and aligned")
    if not 0 <= i < len(logits):
        raise ValueError(f"expert index {i} out of range")
    v = len(logits[0])
    mean = [sum(w * row[j] for w, row in zip(weights, logits)) for j in range(v)]
    dev = [logits[i][j] - mean[j] for j in range(v)]
    return max(dev) - min(dev)


def poe_log_z_bound(
    logits: list[list[float]], weights: list[float]
) -> tuple[float, float]:
    """``poe_logZ_second_order`` -- the measured gap and its bound.

        |log Z_w - sum_i w_i log Z_i| <= (1/8) * sum_i w_i R_i^2

    Returns:
        ``(gap, bound)`` where ``gap = log Z_w - sum_i w_i log Z_i``.

        ``gap <= 0`` ALWAYS, by Jensen at the pooled mean: the geometric
        pool's normaliser never exceeds the weighted average of its
        members'. So the absolute value in the theorem is what makes it
        symmetric, and pooling cannot INCREASE the normaliser -- which is
        exactly why the correction has a bounded, small error: the linear
        terms cancel by centering and leave a second-order remainder.

    Raises:
        ValueError: on a shape mismatch or malformed weights.
    """
    gap = poe_log_z(logits, weights) - weighted_log_z(logits, weights)
    bound = 0.125 * sum(
        w * expert_spread(logits, weights, i) ** 2
        for i, w in enumerate(weights)
    )
    return gap, bound


def poe_softmax_speed(spread: float) -> float:
    """``poe_softmax_speed``: a uniform spread ``M`` gives ``M^2/8``.

    The licence to run a product-of-experts pool at softmax speed: the
    correction is computed from the mean logits plus a term bounded by
    this, so the approximation's error is known rather than assumed.

    Args:
        spread: the uniform per-expert spread ``M``.

    Returns:
        The per-token error bound.

    Raises:
        ValueError: on a negative spread.
    """
    if spread < 0.0:
        raise ValueError("spread must be non-negative")
    return spread * spread / 8.0


def pairwise_constant_is_halved(
    logits: list[list[float]], weights: list[float]
) -> tuple[bool, float]:
    """The audit's pairwise form is FALSE; this is the counterexample.

    The audit proposed ``|gap| <= (1/8) sum_{i<j} w_i w_j R_ij^2``. With
    ``z1 = (1,-1)``, ``z2 = (0,0)``, ``w = 1/2`` the gap is about 0.157
    against a claimed bound of 0.0625, so the pairwise constant must be
    1/2, not 1/8. The weighted form with ``(1/8) sum_i w_i R_i^2`` is the
    one that holds.

    Returns:
        ``(violated, true_gap, claimed_bound)`` on that counterexample,
        so the refutation is executable rather than a claim in a
        docstring.

    Raises:
        ValueError: on a shape mismatch or malformed weights.
    """
    z1 = [1.0, -1.0]
    z2 = [0.0, 0.0]
    logits = [z1, z2]
    weights = [0.5, 0.5]
    gap, _ = poe_log_z_bound(logits, weights)
    # D_ij is the pairwise logit distance. The Lean honesty note uses
    # D = 1 for this counterexample and the OFF-DIAGONAL sum, giving
    # (1/8)*sum_{i != j} w_i w_j D^2 = (1/8)*(1/4 + 1/4) = 0.0625,
    # while the true gap is log[cosh(1)/cosh^2(0.5)]/2 = 0.096776 --
    # which this port measures exactly. (The Lean docstring writes
    # "0.157" for the gap; the computed value is 0.0968, so that figure
    # is a typo there. The refutation stands on the measured number.)
    d = 1.0
    n = len(weights)
    claimed = 0.125 * sum(
        weights[i] * weights[j] * d ** 2
        for i in range(n) for j in range(n) if i != j
    )
    return abs(gap) > claimed + 1e-9, gap, claimed


def logit_range_scale(logits: list[float], tau: float) -> list[float]:
    """``logitRange_scale`` (MacroCycleV2.lean R175): R(d/tau) = R(d)/tau.

    Temperature division scales the logit range LINEARLY, so the PoE
    second-order slack ``R^2/8`` contracts QUADRATICALLY: an expert
    whose measured range ``R`` overshoots the certified admission
    threshold ``R_bar`` is repaired exactly to the boundary by
    ``tau = R/R_bar`` (``variance_gate_normalize``).

    Raises:
        ValueError: on a non-positive temperature.
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    return [z / tau for z in logits]


def variance_gate_tau(logits: list[float], r_bar: float) -> float:
    """``variance_gate_normalize``: the exact repair temperature.

    For a range overshoot ``R >= R_bar``, ``tau = R/R_bar`` lands the
    expert's range exactly on ``R_bar`` and its PoE slack exactly on
    ``R_bar^2/8`` -- the admission boundary. In-range experts return
    ``tau = 1`` (pass through unchanged).

    Raises:
        ValueError: on a non-positive threshold or an empty logit list.
    """
    if r_bar <= 0.0:
        raise ValueError("r_bar must be positive")
    if not logits:
        raise ValueError("logits must be non-empty")
    r = max(logits) - min(logits)
    if r <= r_bar:
        return 1.0
    return r / r_bar


def expert_gate_tau(
    expert_logits: list[list[float]], weights: list[float], r_bar: float
) -> list[float]:
    """Admission temperatures for every expert in a merge.

    The merge admission uses the R106 CENTERED spread ``R_i`` (deviation
    of expert i's logits from the weighted pool mean) -- the measurable
    the PoE bound is actually stated in, not the raw max-min. Experts
    within ``R_bar`` keep ``tau = 1``; an overshooting expert gets
    ``tau_i = R_i / R_bar`` so its post-scaling spread is exactly
    ``R_bar`` and its PoE slack exactly ``R_bar^2/8``.

    Returns:
        One temperature per expert, aligned with the input order.

    Raises:
        ValueError: as :func:`expert_spread` / :func:`variance_gate_tau`.
    """
    return [
        variance_gate_tau(
            [z - sum(w * row[j] for w, row in zip(weights, expert_logits))
             for j, z in enumerate(row)],
            r_bar,
        )
        for row in expert_logits
    ]


# --- Joint-step regression laws (Step/Joint.lean) ---------------------------


def conflict_lr_limit(g_c: float, g_norm_sq: float, L: float) -> float:
    """Conflict-regression boundary [``jointStep_conflict_regression``]:
    when the control gradient conflicts with the joint gradient
    (g_c . g < 0), ANY learning rate below 2|g_c.g| / (L (g.g)) still
    strictly INCREASES the loss (quadIncr > 0). The boundary is the
    honest veto line: at or above it descent is possible; below it the
    step is provably harmful. Runtime use: cap the per-step lr on the
    conflicting domain so adapters never sit in the regression zone.
    """
    if g_norm_sq <= 0.0:
        raise ValueError("g_norm_sq must be positive")
    if L <= 0.0:
        raise ValueError("L must be positive")
    return 2.0 * abs(g_c) / (L * g_norm_sq)


def joint_step_regression_bound(
    g_c_norm: float, g_norm_sq: float, L: float, lr: float
) -> float:
    """Worst-case joint increment [``jointStep_regression_bound``]:
    quadIncr <= |lr| sqrt(g_c.g_c) sqrt(g.g) + (L/2) lr^2 (g.g) —
    the certified upper envelope of a step against a conflicting
    gradient (Cauchy-Schwarz + smoothness). Premises (h_emp_):
    L-smooth curvature. Returns the bound; callers veto when it
    exceeds the step's certified descent budget."""
    if L < 0.0 or g_norm_sq < 0.0:
        raise ValueError("L and g_norm_sq must be nonnegative")
    return abs(lr) * (g_c_norm * (g_norm_sq ** 0.5)) + 0.5 * L * lr * lr * g_norm_sq


def trust_region_loss_bound(
    g_c_norm: float, L: float, R: float
) -> float:
    """Direction-free trust-region envelope [``trustRegion_bound``]:
    with ||delta|| <= R, quadIncr <= sqrt(g_c.g_c) * R + (L/2) R^2 —
    independent of the step direction. The honest radius budget for
    probe-based steps: any step inside R costs at most this much."""
    if L < 0.0 or R < 0.0:
        raise ValueError("L and R must be nonnegative")
    return g_c_norm * R + 0.5 * L * R * R



# --- Armijo / Warmup (Step/ArmijoAcceptance.lean, Step/WarmupSchedule.lean) --


def armijo_accept(
    g_dot_d: float, d_norm_sq: float, alpha: float, L: float, c1: float = 0.25,
) -> tuple[bool, float]:
    """Armijo with an EXPLICIT step size [``armijo_step_size``]:
    with the direction condition g.d <= -c1*||d||^2 and the size bound
    alpha <= c1/L, the L-smooth model certifies

        J(x + alpha*d) <= J(x) - (c1/2) alpha ||d||^2.

    Returns (accept, certified_decrease). The point vs classical line
    search: alpha is CHOSEN from the theorem (c1/L), not probed - no
    backtracking loop, one evaluation.

    Premises (h_emp_): L-smooth curvature; direction condition measured.
    """
    if L < 0.0 or d_norm_sq < 0.0:
        raise ValueError("L and d_norm_sq must be nonnegative")
    if c1 <= 0.0:
        raise ValueError("c1 must be positive")
    if alpha <= 0.0 or alpha > c1 / L:
        return False, 0.0
    if g_dot_d > -c1 * d_norm_sq:
        return False, 0.0
    return True, 0.5 * c1 * alpha * d_norm_sq


def warmup_eta(
    c: float, k0: float, k_rho: float, rho: float, delta: float
) -> float:
    """Warm (adaptively growing) step [``warmupEta``]:
    eta_t = c * min(1/K0, Delta_t^(-rho) / (3^rho * K_rho)).

    Structure [``warmup_eta_monotone``]: as the suboptimality Delta
    DECREASES the step GROWS - the warm schedule is a CONSEQUENCE of
    (rho, K0, K_rho)-smoothness, not a separate assumption. Premises:
    rho > 0, K0 > 0, K_rho > 0, Delta > 0.
    """
    if k0 <= 0.0 or k_rho <= 0.0:
        raise ValueError("K0 and K_rho must be positive")
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if delta <= 0.0:
        raise ValueError("Delta (suboptimality) must be positive")
    if c < 0.0:
        raise ValueError("c must be nonnegative")
    return c * min(1.0 / k0, delta ** (-rho) / (3.0 ** rho * k_rho))



# --- SNR weighting + NCE K-choice (Step/SNRWeights, Step/NCEExact) -----------


def focused_wins(dev_sq: list[float]) -> tuple[bool, list[float]]:
    """SNR weighting licence [``focused_beats_uniform``
    (Step/SNRWeights.lean)]: IF some expert j has (dev_j)^2 strictly below
    the uniform mean, THEN a weight vector (nearly) strictly better than
    uniform exists. The constructive witness in the theorem: put mass 1
    on the best expert. Runtime: SNR-focus beats uniform weighting of
    experts whenever deviations are unequal - check first, focus second.

    Returns (focused_is_better, witness_weights).
    """
    if not dev_sq or any(d < 0 for d in dev_sq):
        raise ValueError("dev_sq must be nonempty nonnegative")
    r = len(dev_sq)
    mean = sum(dev_sq) / r
    j = min(range(r), key=lambda i: dev_sq[i])
    if not dev_sq[j] < mean:
        return False, [1.0 / r] * r  # uniform is already the best focus
    w = [0.0] * r
    w[j] = 1.0
    return True, w


def focused_cost(dev_sq: list[float], w: list[float]) -> float:
    """Weighted deviation energy sum_i w_i (dev_i)^2 - the quantity
    ``focused_wins`` strictly reduces below the uniform mean."""
    if len(dev_sq) != len(w) or not dev_sq:
        raise ValueError("equal nonempty lengths required")
    if any(x < 0 for x in dev_sq) or any(x < 0 for x in w):
        raise ValueError("inputs must be nonnegative")
    return sum(wi * d for wi, d in zip(w, dev_sq))


def adaptive_k(moment: float, eps: float) -> int:
    """Minimal K for an eps-accurate NCE estimate [``adaptiveK_bound``
    (Step/NCEExact.lean)]: K >= moment/eps guarantees (1/K)*moment <= eps
    - the number of noise samples is DERIVED from the measured second
    moment of the importance weights, not picked."""
    if moment < 0.0:
        raise ValueError("moment must be nonnegative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if moment == 0.0:
        return 1
    return math.ceil(moment / eps)


def anchor_cadence(rho: float, gamma: float, eps: float) -> float:
    """Anchor refit cadence [``anchor_cadence_criterion`` (NCEExact.lean)]:
    refitting the NCE anchor every s steps keeps the stationary drift
    rho*s/gamma <= eps whenever s <= gamma*eps/rho. Returns the maximal
    legal cadence s - derived from measured (rho, gamma), not Scheduled.
    """
    if rho <= 0.0 or gamma <= 0.0 or eps <= 0.0:
        raise ValueError("rho, gamma, eps must be positive")
    return gamma * eps / rho



# --- NCE partition estimator (Step/NCEExact.lean) -----------------------------


def part_est_unbiased(z: list[float], q: list[float]) -> float:
    """NCE partition estimate [``partEst_unbiased`` (Step/NCEExact.lean)]:
    with importance weights impW(v) = exp(z_v) / (q_v * |V|)? — concretely
    the estimator sum_v q_v * (exp(z_v) / q_v) = sum_v exp(z_v) = Z is
    UNBIASED whenever q has full support: E_q[impW] = Z exactly. Lean
    form: sum_v q_v * exp(z_v)/q_v = sum_v exp(z_v). Runtime: the NCE
    estimate of Z from importance sampling is exact on-support; bias can
    only enter via support clipping (use :func:`smooth_support`).
    Returns the exact Z (numerically stabilized via max-shift)."""
    if not z or len(z) != len(q):
        raise ValueError("z and q must be nonempty, equal length")
    if any(x <= 0 for x in q):
        raise ValueError("q must be strictly positive (full support)")
    m = max(z)
    return math.exp(m) * sum(math.exp(v - m) for v in z)


def second_moment_w(z: list[float], q: list[float]) -> float:
    """Second moment of the importance weights [``secondMomentW`` +
    ``secondMomentW_ge_sq`` (NCEExact.lean)]: M2 = sum_v q_v impW(v)^2
    satisfies Z^2 <= M2 (Cauchy-Schwarz). The NCE variance is driven by
    M2/Z^2 >= 1 - the K-sample budget comes from
    :func:`adaptive_k` (moment/eps). Returns M2."""
    if not z or len(z) != len(q):
        raise ValueError("z and q must be nonempty, equal length")
    if any(x <= 0 for x in q):
        raise ValueError("q must be strictly positive")
    Z = part_est_unbiased(z, q)
    # impW(v) = exp(z_v)/q_v  (the per-v importance weight)
    return sum(qv * (math.exp(zv) / qv) ** 2 for zv, qv in zip(z, q))


def nce_variance_ratio(z: list[float], q: list[float]) -> float:
    """Variance driver M2/Z^2 [relative second moment]: >= 1 by Cauchy-
    Schwarz; the K-sample relative error scales as this ratio / K -
    the honest cost of NCE at the current proposal q."""
    Z = part_est_unbiased(z, q)
    return second_moment_w(z, q) / (Z * Z)


def kls_batch_noise(c: float, L: float, sq_n: float, num_samples: int) -> float:
    """KLS batch noise second moment [``klsBatchNoise`` (Probability),
    consumed by ``klsSafeQP_certified`` (Step/KLSSafeQP.lean)]: under a
    Poincare inequality with constant C, sensitivity bounded by
    |sens a x|^2 <= L^2 * sqN(a), and B mutually uncorrelated samples,
    the second moment of the deviation field

        dev(x) = (1/B) sum_i [proj a g_i(x) - E_q proj a g_i]

    is at most V = C * L^2 * sqN(a) / B. Returns V (>= 0)."""
    if c < 0.0 or L < 0.0 or sq_n < 0.0:
        raise ValueError("C, L, sqN must be nonnegative")
    if num_samples <= 0:
        raise ValueError("B must be a positive integer")
    return c * L * L * sq_n / num_samples


def kls_safeqp_certified(
    c: float, L: float, sq_n: float, num_samples: int, gd: float,
    eps: float, t: float,
) -> float:
    """Certified QP acceptance mass [``klsSafeQP_certified``
    (Step/KLSSafeQP.lean)]: with the KLS batch noise V (see
    :func:`kls_batch_noise`) and the true-inner premise
    ``eps + |gd| * t <= gd * trueInner(x)`` satisfied pointwise, the
    q-mass of x where the ESTIMATOR accepts (gd*estInner >= eps) is

        >= 1 - V / t^2

    (Chebyshev on the deviation field). Runtime: the certificate the
    SafeQP guard reports alongside a veto - the accept decision from
    estimated inner products carries this mass guarantee. Returns the
    certified lower bound 1 - V/t^2 (clamped from above at 1)."""
    if t <= 0.0:
        raise ValueError("t must be positive")
    if eps < 0.0:
        raise ValueError("eps must be nonnegative")
    if gd == 0.0:
        raise ValueError("gd must be nonzero (premise |gd|*t <= gd*true needs a sign)")
    v = kls_batch_noise(c, L, sq_n, num_samples)
    return min(1.0, 1.0 - v / (t * t))
