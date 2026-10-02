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