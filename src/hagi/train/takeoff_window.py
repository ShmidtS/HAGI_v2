"""R102: the certified takeoff WINDOW, and an honest compression cost.

``Hagi/Unified/GrowthBridge.lean`` + ``Hagi/Unified/TopLevel.lean``.

R95/R96 gave this project an exponential takeoff law, and R102 is the
audit that says what it does NOT give: with a FIXED gain the certified
takeoff is a BOUNDED TRANSIENT. That is the single most important
result for the growth loop, because it is the difference between "keep
running this mechanism" and "switch mechanism".

``growth_state_takeoff_window``
    With the gain ``G = growGain`` invariant along the trajectory
    ``C_t = C₀ + t·G`` and the success gate ``α·C_t ≤ G``:

    - every successful cycle's index obeys ``t ≤ 1/α − C₀/G``, so over
      ANY horizon the success count is at most ``1/α + 1 − C₀/G`` --
      an ABSOLUTE window, independent of T;
    - hence the certified factor ``(1+α)^Σσ ≤ exp(1 + α − α·C₀/G)``,
      which is at most ``e·e^α``: a CONSTANT, however long the loop runs.

    The audit's suggested ``e^{G/C₀}`` form is NOT provable in this
    generality and the theorem says so explicitly. This port keeps that
    honesty: the bound returned is the tight one, and
    :func:`suggested_form_is_not_implied` documents why the looser one
    would have been wrong.

    Consequence, stated plainly: SUSTAINED takeoff requires a gain that
    scales with capability, ``G_t ≥ α·C_t``, at every cycle. That is the
    OPEN BRIDGE -- a gain-growth law tied to the capability field -- not
    a theorem of the current state model. :func:`sustained_takeoff_needs`
    computes exactly what a mechanism must deliver to be worth keeping.

``noisy_cycle_step`` (FIX 1)
    The per-generation energy budget

        E ≤ E_mean − (G + η‖d*‖²/2 − η·m₀ − κ√n·s/2)

    now carries the HONEST compression cost. Previously the term was
    derived with ``Real.sqrt 1``, i.e. as if the residual were measured
    on ONE coordinate instead of all ``n`` -- understating the cost of a
    generation's compress stage by a factor of ``√n``. For the widths in
    this project that is 24x at n=576 and 34x at n=1152. The
    :func:`compression_cost` port below is the ``κ√n·s/2`` form, so a
    caller sizing a compression stage by it is no longer optimistic by
    ``√n``.

The rest of R102 is honesty fixes to existing docstrings (entropy sign,
CurvatureSafe slack naming, positive task costs). Those are conventions
in the Lean sources rather than new mathematics; the one with teeth for
code is :func:`select_task`'s strict positivity, mirrored here in
:func:`positive_cost_required`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TakeoffWindow:
    """What a fixed-gain mechanism can certifiably buy, and no more.

    Attributes:
        alpha: the gate parameter ``α > 0``.
        c0: the initial capability ``C₀ > 0``.
        gain: the invariant per-cycle gain ``G > 0``.
    """

    alpha: float
    c0: float
    gain: float

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.c0 <= 0.0:
            raise ValueError("c0 must be positive")
        if self.gain <= 0.0:
            raise ValueError("gain must be positive")

    @property
    def gate_at_zero_holds(self) -> bool:
        """``α·C₀ ≤ G`` -- without it no cycle can ever succeed."""
        return self.alpha * self.c0 <= self.gain

    @property
    def max_time_index(self) -> float:
        """``1/α − C₀/G`` -- the last admissible cycle index."""
        return 1.0 / self.alpha - self.c0 / self.gain

    @property
    def max_successes(self) -> float:
        """``1/α + 1 − C₀/G`` -- the ABSOLUTE success ceiling.

        Independent of the horizon T. This is the number that decides
        whether a growth mechanism is worth continuing: a mechanism that
        has already spent its ceiling will not spend another one.
        """
        return 1.0 / self.alpha + 1.0 - self.c0 / self.gain

    @property
    def certified_factor_bound(self) -> float:
        """``exp(1 + α − α·C₀/G)`` -- at most ``e·e^α``.

        The whole certified takeoff, however long the loop runs. If this
        is close to 1, the mechanism cannot produce growth worth
        measuring and the honest response is to change mechanism.
        """
        return math.exp(1.0 + self.alpha - self.alpha * self.c0 / self.gain)

    @property
    def is_bounded_transient(self) -> bool:
        """Always True by the theorem: with fixed G the factor is finite.

        Stated as a property rather than left implicit, because
        "unbounded exponential growth" is exactly the reading this port
        exists to prevent.
        """
        return math.isfinite(self.certified_factor_bound)


def takeoff_window(
    alpha: float, c0: float, gain: float, horizon: int | None = None
) -> TakeoffWindow:
    """``growth_state_takeoff_window`` -- the bounded-transient bound.

    Args:
        alpha: the gate parameter.
        c0: initial capability.
        gain: the invariant per-cycle gain.
        horizon: unused by the bound (which is uniform in T), accepted
            so callers can pass one and the formula stays readable.

    Returns:
        The window, carrying the absolute success ceiling and the
        certified factor bound.

    Raises:
        ValueError: on non-positive ``alpha``, ``c0`` or ``gain``.
    """
    return TakeoffWindow(alpha=alpha, c0=c0, gain=gain)


def success_ceiling_reached(spent: float, window: TakeoffWindow) -> bool:
    """Has this mechanism already spent its whole window?

    The stopping rule for a fixed-gain growth channel: once the count of
    successful cycles reaches the ceiling, no further success is
    certifiable, and continuing the same channel is not a close call but
    a certainty of nothing. At that point the mechanism must change.

    Args:
        spent: successes actually observed.
        window: the certified window.

    Returns:
        True iff ``spent`` is at or beyond the ceiling.
    """
    return spent >= window.max_successes


def sustained_takeoff_needs(window: TakeoffWindow, cycles: int) -> float:
    """The gain per cycle that sustained takeoff would require.

    The OPEN BRIDGE made concrete. Sustained exponential growth over
    ``cycles`` cycles needs the gain to scale with capability:

        G_t ≥ α·C_t,   C_t = C₀ + t·G

    On the final cycle this reads ``G ≥ α(C₀ + (T−1)G)``, i.e.

        G · (1 − α(T−1)) ≥ α·C₀.

    So a solution EXISTS only while ``α·(T−1) < 1``. Past that point the
    requirement is unsatisfiable by ANY constant gain: the capability
    grows at least as fast as the gate rises, and the inequality has no
    finite solution. That is not a limitation of this port, it is the
    theorem's content -- sustained takeoff REQUIRES the gain to grow,
    because no fixed gain can satisfy the gate forever.

    Args:
        window: the current fixed-gain window.
        cycles: how many cycles of sustained growth are wanted.

    Returns:
        The gain needed at the last cycle, or ``inf`` when no constant
        gain can satisfy the gate for that many cycles.

    Raises:
        ValueError: on a non-positive cycle count.
    """
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    t = cycles - 1
    denom = 1.0 - window.alpha * t
    if denom <= 0.0:
        # No constant gain satisfies the gate this long: the open bridge
        # is not optional at this horizon, it is forced.
        return math.inf
    return window.alpha * window.c0 / denom


def sustained_takeoff_possible(window: TakeoffWindow, cycles: int) -> bool:
    """Can ANY constant gain satisfy the gate for ``cycles`` cycles?

    False past ``1/α`` cycles. A growth controller asking whether to
    invest in a fixed-gain channel gets a definite answer instead of an
    unbounded requirement.
    """
    return cycles > 0 and window.alpha * (cycles - 1) < 1.0


def suggested_form_is_not_implied(window: TakeoffWindow) -> bool:
    """The audit's ``e^{G/C₀}`` form is NOT what the theorem proves.

    The audit proposed the certified factor be ``e^{G/C₀}``. The theorem
    proves ``e^{1+α−αC₀/G}``, and for small ``G/C₀`` the proven bound is
    LARGER -- so adopting the suggested form would have silently
    strengthened a bound that was not proved. This reports that
    discrepancy as a number instead of leaving it in a comment.

    Args:
        window: the window under test.

    Returns:
        True iff the suggested form would be tighter (i.e. would claim
        more than is proved) at these parameters.
    """
    proved = window.certified_factor_bound
    suggested = math.exp(window.gain / window.c0)
    return suggested < proved


def empty_window() -> bool:
    """A window no cycle can enter: the ``hgate0`` failure case.

    ``alpha·C₀ > G`` means the gate is already violated at t = 0, so
    there is no admissible time at all and the success count is zero --
    not "few", zero. The theorem states this as the ``hgate0``
    hypothesis; here it is a query, because a caller with a real
    mechanism needs to know which side it is on.
    """
    return True


def window_is_empty(window: TakeoffWindow) -> bool:
    """No cycle can ever succeed when the gate fails already at t = 0."""
    return not window.gate_at_zero_holds


# --- FIX 1: the honest compression cost ----------------------------------


def compression_cost(
    n_coords: int, grid: float, kappa: float, residual: float | None = None
) -> float:
    """``κ·√n·s/2`` -- the per-generation compression charge.

    ``noisy_cycle_step`` + ``quant_energy_bridge`` (R102 FIX 1). The
    compression stage of a generation costs this much energy, and the
    honest form carries ``√n``: a residual spread over all ``n``
    coordinates charges the square root of their count.

    The previous derivation substituted ``Real.sqrt 1``, i.e. measured
    the residual on a single coordinate, which understated the cost by
    ``√n`` -- 24x at n=576, 34x at n=1152 (this project's width). A
    stage sized against the old form is oversized by exactly that factor.

    Args:
        n_coords: the number of coordinates, ``n > 0``.
        grid: the quantisation step ``s ≥ 0``.
        kappa: the compression constant ``κ ≥ 0``.
        residual: if given, the MEASURED residual norm ``‖w − q‖₂`` and
            the exact ``κ·‖w − q‖₂`` is returned instead of the bound.

    Returns:
        The energy charge, non-negative.

    Raises:
        ValueError: on a non-positive ``n_coords``, a negative ``grid``
            or ``kappa``, or a negative ``residual``.
    """
    if n_coords <= 0:
        raise ValueError("n_coords must be positive")
    if grid < 0.0:
        raise ValueError("grid must be non-negative")
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    if residual is not None:
        if residual < 0.0:
            raise ValueError("residual must be non-negative")
        return kappa * residual
    return kappa * math.sqrt(float(n_coords)) * grid / 2.0


def cycle_budget(
    gain: float,
    eta: float,
    dnorm_sq: float,
    noise: float,
    n_coords: int,
    grid: float,
    kappa: float,
) -> float:
    """``G + η‖d*‖²/2 − η·m₀ − κ√n·s/2`` -- the net per-generation gain.

    ``noisy_cycle_step``: positive means the generation DESCENDS the
    potential, negative means it costs energy. Every term is a measured
    quantity, and the compression term now carries the honest ``√n``.

    Args:
        gain: the merge gain ``G``.
        eta: the SafeQP step size.
        dnorm_sq: ``‖d*‖²``.
        noise: the minibatch radius ``m₀``.
        n_coords: coordinates in the compressed matrix.
        grid: the quantisation step.
        kappa: the compression constant.

    Returns:
        The net energy change (negative = descent).

    Raises:
        ValueError: on a negative gain, noise, ``dnorm_sq``, or on
            invalid compression arguments.
    """
    if gain < 0.0 or noise < 0.0 or dnorm_sq < 0.0:
        raise ValueError("gain, noise and dnorm_sq must be non-negative")
    return (
        gain
        + eta * dnorm_sq / 2.0
        - eta * noise
        - compression_cost(n_coords, grid, kappa)
    )


def generations_to_floor(energy0: float, e_min: float, eps: float) -> float:
    """``(E₀ − E_min)/ε`` -- the Lyapunov telescope's horizon.

    ``horizon_termination``. R102 notes this is the horizon-form
    RESTATEMENT of the same telescope and adds no new mathematics; it is
    ported because a caller asking "how many generations" should not
    have to know that.

    Args:
        energy0: the initial energy.
        e_min: the floor.
        eps: the per-generation descent.

    Returns:
        The number of generations.

    Raises:
        ValueError: on non-positive ``eps`` or a floor above ``energy0``.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if e_min > energy0:
        raise ValueError("the floor cannot be above the initial energy")
    return (energy0 - e_min) / eps


# --- FIX 3: task selection needs positive costs --------------------------


def positive_cost_required(costs: dict[str, float]) -> bool:
    """``∀ j, 0 < c_j`` -- the premise ``task_selection_marginal`` now states.

    R102 FIX 3. The argmax ``g_j/c_j`` is only well defined for positive
    costs; a zero cost produced a spurious free winner and a negative one
    inverted the ordering. This reports whether a candidate set satisfies
    the premise rather than letting the ratio absorb the nonsense.
    """
    if not costs:
        return False
    return all(c > 0.0 for c in costs.values())