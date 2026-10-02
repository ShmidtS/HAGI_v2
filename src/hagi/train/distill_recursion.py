"""R108: recursive distillation has an entropy floor, and the fresh-data
mass is what puts it there.

``Hagi/Data/DistillRecursion.lean``. R107 closed the growth loop's last
free premise (``h_emp_frontier_scaling``) and, applied to this project's
measured numbers, answered ``CAPPED``: ``D/C = 5.53`` against a cone
boundary of ``alpha/gamma = 50``. The regime is bounded, so the next
question is the one a bounded regime raises -- what stops the bounded
quantity from collapsing to nothing? That is what R108 answers, for the
mechanism this project actually uses: each generation is distilled from
the previous one.

``entropy_mix_ge``
    ``H((1-nu)p + nu d) >= (1-nu)H(p) + nu H(d)`` -- concavity of
    entropy, proved through the JS identity. Mixing with a fresh-data
    fraction ``nu`` can only pull the entropy toward ``H(data)``, never
    below it.

``distill_entropy_recurrence``
    Solving ``x_{k+1} = (1-nu)x_k + nu h - delta`` exactly:

        H(p_T) >= (1-nu)^T H(p_0) + (1 - (1-nu)^T)(H(data) - delta/nu)

    The fixed point is ``H(data) - delta/nu`` and the solution is the
    geometric interpolation between ``H(p_0)`` and it.

``entropy_floor``
    The same thing as "floor minus an exponentially vanishing
    correction". The distance to the floor contracts by ``(1-nu) < 1``
    per generation.

``fresh_data_prevents_collapse``
    Started at or above the floor, the entropy never falls below it, at
    any horizon. At ``nu = 0`` the floor ``delta/nu`` diverges and the
    statement is VACUOUS -- collapse IS the ``nu = 0`` protocol, which
    the theorem says by making the denominator explicit rather than by
    assuming ``nu > 0``.

Two things the port makes operational rather than prose, because both
are decisions this growth loop has to make:

:func:`required_fresh_fraction` -- the mixing law gives the floor for
    free, and the floor is what a controller wants. This inverts it:
    for a target floor ``H(data) - eps`` and a per-step certified error
    ``delta``, the minimum ``nu`` is ``delta/eps``. A replay coefficient
    becomes arithmetic.

:func:`certified_step_is_not_held` -- the honest structural finding.
    The hypothesis ``hcert`` says every step's entropy is within
    ``delta`` of its fresh-mix target. This project does not distill:
    its generations are TRAINED on fresh data, and their entropies are
    whatever training produces. So the floor does not apply to it as
    stated, and :func:`collapse_risk` reports what it does apply to.
    The alternative is that the floor applies to the MERGED model, whose
    output distribution is a convex combination of the parents' --
    which is precisely the ``(1-nu)p + nu d`` form the theorem is about,
    at ``nu = 1/n_experts`` and ``d`` the fresh-data component.

One audit constant is refuted in the module, and the refutation is
executable: :func:`kl_does_not_bound_entropy` returns the counterexample
``m = (0.9, 0.1)``, ``q = (0.99, 0.01)``, where ``KL = 0.14`` but the
entropy loss is ``0.27``. Entropy is not Lipschitz in KL with a linear
constant, so "a ``delta``-KL step preserves entropy" is FALSE. The
per-step preservation the theorem needs is therefore the EXPLICIT
strengthened hypothesis ``hcert``, not a consequence of a ``delta``-KL
step.
"""

from __future__ import annotations

import math


def shannon_entropy(p: list[float]) -> float:
    """``H(p) = -sum_v p_v log p_v`` in nats, with ``0 log 0 = 0``.

    Args:
        p: a probability vector.

    Returns:
        The entropy in nats.

    Raises:
        ValueError: on an empty vector or any negative entry.
    """
    if not p:
        raise ValueError("entropy needs a non-empty distribution")
    h = 0.0
    for x in p:
        if x < 0.0:
            raise ValueError("probabilities must be non-negative")
        if x > 0.0:
            h -= x * math.log(x)
    return h


def fresh_mix(nu: float, p: list[float], d: list[float]) -> list[float]:
    """``(1-nu) p + nu d`` -- the mixture a distilling step trains toward.

    Args:
        nu: the fresh-data mass, in ``[0, 1]``.
        p: the parent's distribution.
        d: the fresh data's distribution.

    Returns:
        The mixture, entrywise.

    Raises:
        ValueError: on a shape mismatch or ``nu`` outside ``[0, 1]``.
    """
    if len(p) != len(d):
        raise ValueError("p and d must have the same support")
    if not 0.0 <= nu <= 1.0:
        raise ValueError("nu must be in [0, 1]")
    return [(1.0 - nu) * a + nu * b for a, b in zip(p, d)]


def entropy_mix_ge(nu: float, p: list[float], d: list[float]) -> tuple[float, float]:
    """``entropy_mix_ge``: concavity of entropy, as a measured inequality.

    ``H((1-nu)p + nu d) >= (1-nu)H(p) + nu H(d)``.

    Returns:
        ``(mixture entropy, convex combination)`` so a caller can check
        the inequality rather than trust the port.

    Raises:
        ValueError: as :func:`fresh_mix`.
    """
    mix = fresh_mix(nu, p, d)
    left = shannon_entropy(mix)
    right = (1.0 - nu) * shannon_entropy(p) + nu * shannon_entropy(d)
    return left, right


def distill_step_entropy(
    nu: float, delta: float, h_parent: float, h_data: float, h_next: float
) -> float:
    """``distill_step_entropy``: the one-step law.

        (1-nu) H(p_k) + nu H(data) - delta <= H(p_{k+1})

    i.e. fresh data pulls the floor up by ``nu (H(data) - H(p_k))`` and the
    certified step error pulls it down by at most ``delta``.

    Args:
        nu: the fresh-data mass.
        delta: the certified per-step entropy shortfall.
        h_parent: ``H(p_k)``.
        h_data: ``H(data)``.
        h_next: ``H(p_{k+1})``.

    Returns:
        The lower bound the next generation's entropy must meet.

    Raises:
        ValueError: on ``nu`` outside ``[0, 1]`` or a negative delta.
    """
    if not 0.0 <= nu <= 1.0:
        raise ValueError("nu must be in [0, 1]")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    return (1.0 - nu) * h_parent + nu * h_data - delta


def entropy_floor(nu: float, delta: float, h_data: float) -> float:
    """``H(data) - delta/nu`` -- the fixed point of the recurrence.

    ``entropy_floor``'s floor term. Diverges as ``nu -> 0``, which is the
    point: at no fresh data the floor is not a floor at all.

    Raises:
        ValueError: on non-positive ``nu`` or a negative delta.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive -- at nu=0 the floor diverges")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    return h_data - delta / nu


def distill_entropy_recurrence(
    nu: float, delta: float, h0: float, h_data: float, generations: int
) -> float:
    """``distill_entropy_recurrence``: the exact closed form.

        H(p_T) >= (1-nu)^T H(p_0) + (1 - (1-nu)^T)(H(data) - delta/nu)

    Args:
        nu: the fresh-data mass, strictly positive.
        delta: the certified per-step entropy shortfall.
        h0: ``H(p_0)``.
        h_data: ``H(data)``.
        generations: the horizon ``T``.

    Returns:
        The bound on ``H(p_T)``.

    Raises:
        ValueError: on non-positive ``nu``, a negative delta, or a
            negative horizon.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    if generations < 0:
        raise ValueError("generations must be non-negative")
    w = (1.0 - nu) ** generations
    floor = entropy_floor(nu, delta, h_data)
    return w * h0 + (1.0 - w) * floor


def fresh_data_prevents_collapse(
    nu: float, delta: float, h0: float, h_data: float, generations: int
) -> bool:
    """``fresh_data_prevents_collapse``: is the floor actually respected?

    Started at or above the floor, the entropy stays above it at every
    horizon. The check is two-sided on purpose: the theorem is a claim
    about a trajectory that must be ESTABLISHED, not a consequence of
    the algebra alone, so a caller supplies ``h0`` and gets told whether
    the initial condition holds.

    Raises:
        ValueError: as :func:`distill_entropy_recurrence`.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    if generations < 0:
        raise ValueError("generations must be non-negative")
    floor = entropy_floor(nu, delta, h_data)
    if h0 < floor:
        return False
    return distill_entropy_recurrence(nu, delta, h0, h_data, generations) >= floor - 1e-12


def required_fresh_fraction(delta: float, target_drop: float) -> float:
    """``nu >= delta/eps`` -- the replay coefficient as arithmetic.

    The mixture law hands the floor to a controller; this is the inverse,
    and it is the part that changes a decision: to keep the entropy
    within ``target_drop`` of ``H(data)``, the fresh-data mass must be at
    least ``delta/target_drop``. A mixing heuristic becomes a number that
    can be checked before training rather than swept afterwards.

    Args:
        delta: the certified per-step entropy shortfall.
        target_drop: the acceptable distance below ``H(data)``.

    Returns:
        The minimum ``nu``. Zero when ``delta`` is zero.

    Raises:
        ValueError: on a negative delta or a non-positive target drop.
    """
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    if target_drop <= 0.0:
        raise ValueError("target_drop must be positive")
    return delta / target_drop


def kl_does_not_bound_entropy() -> tuple[float, float, float]:
    """The audit's ``KL <= delta => H(q) >= H(m) - delta`` is FALSE.

    ``m = (0.9, 0.1)``, ``q = (0.99, 0.01)``: ``KL(m || q) = 0.14`` while
    the entropy loss is ``0.27``. Entropy is not Lipschitz in KL with a
    linear constant, so a ``delta``-KL step does NOT preserve entropy to
    ``delta``. The per-step preservation R108 needs is therefore the
    explicit strengthened hypothesis ``hcert``, not a consequence of
    step size -- which is the third time an audit premise has been
    refuted by a counterexample rather than argued.

    Returns:
        ``(KL, entropy loss, ratio)`` on that counterexample.
    """
    m = [0.9, 0.1]
    q = [0.99, 0.01]
    kl = sum(a * math.log(a / b) for a, b in zip(m, q))
    loss = shannon_entropy(m) - shannon_entropy(q)
    return kl, loss, loss / kl


def certified_step_is_not_held() -> bool:
    """This project's generations are TRAINED, not distilled.

    ``hcert`` says every step's entropy is within ``delta`` of its
    fresh-mix target. Training does not deliver that: the entropy after a
    gradient step is whatever the objective produced, and nothing in the
    loop checks it against the mixture. So R108's floor does not apply
    to the training path as stated.

    It does apply to the MERGED path, and that is the operative reading.
    A merge pools the experts' distributions by weights that sum to one,
    which is exactly ``(1-nu)p + nu d`` at ``nu = 1/n_experts`` when the
    pool is uniform -- so :func:`merge_nu` is the knob the floor speaks
    about, and it is set by the architecture rather than by training.

    Returns:
        True. The finding is structural, not a measurement, so it is
        reported as a constant rather than computed.
    """
    return True


def merge_nu(n_experts: int) -> float:
    """The effective fresh-data mass of a uniform merge: ``1/n_experts``.

    The analogue of ``nu`` for this project's pooling. A merge of ``n``
    experts weights each by ``1/n``, so the pool is the mixture
    ``(1 - nu)p_1 + ... + nu p_n`` with ``nu = 1/n`` on each -- and the
    floor R108 gives is ``H(data) - delta (1 - nu)/nu``, which DEEPENS as
    the expert count grows: merging more experts is what protects
    against collapse.

    Raises:
        ValueError: on a non-positive expert count.
    """
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    return 1.0 / n_experts


def collapse_risk(nu: float, delta: float, h_data: float,
                  h_current: float) -> tuple[float, float]:
    """How far the entropy is from the floor, and how deep that floor is.

    ``(distance_to_floor, depth_below_H_data)``. A controller wants both:
    the first says whether the floor binds now, the second says what it
    would cost to move it. ``depth_below_H_data = delta/nu`` is the
    number that makes ``nu`` a decision instead of a constant -- halving
    ``nu`` doubles it.

    Raises:
        ValueError: on non-positive ``nu`` or a negative delta.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    floor = entropy_floor(nu, delta, h_data)
    return h_current - floor, delta / nu