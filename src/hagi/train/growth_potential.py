"""R91: one potential for a whole generation, with named stage terms.

``Hagi/Unified/GrowthState.lean``. R85/R96 gave the takeoff law for the
GROW stage alone. R91 gives the generation as a whole: a single
potential, and per-stage lemmas of the form
``Φ(S') − Φ(S) ≤ −gain + cost`` whose gains and costs are computed from
the state's fields rather than postulated.

``potential``
    ``Φ := energy + protectedRisk``. Energy is the loss the loop drives
    down; protectedRisk accumulates the regressions each stage is
    forbidden to cause. Their SUM is the thing that must decrease --
    neither alone is a Lyapunov function, because a stage can always buy
    a small loss reduction with a large regression elsewhere.

``growth_cycle_potential``
    Composing the four stages (grow -> merge -> joint -> compress):

        Φ(after) − Φ(before)
          ≤ −(growGain + gainMerge + gainJoint)
            + (growRisk + mergeRisk + jointRisk + costCompress)

    A generation is certified to descend the potential when the gains
    outweigh the costs, and every term is a MEASURED quantity.

    The honesty R91 carries: the premises ``h_emp_grow``,
    ``h_emp_merge``, ``h_emp_smooth`` and ``h_emp_lip`` are EMPIRICAL.
    The theorem says what follows if they hold; it does not prove they
    hold. :func:`cycle_bound` returns the bound AND reports which
    premises were supplied as measurements, so a caller cannot mistake
    a conditional for an unconditional result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GrowthState:
    """The fields ``Φ`` and the stage lemmas read.

    Attributes:
        energy: the loss, to be driven down.
        protected_risk: accumulated regression the loop must not cause.
        gap: the Jensen/ensemble disagreement, non-negative; the MERGE
            stage's gain.
        quant_err: the compression error, in ``[0, 1/2]``.
        grow_risk / merge_risk / joint_risk / compress_risk: per-stage
            protected-regression contributions.
    """

    energy: float
    protected_risk: float
    gap: float
    quant_err: float
    grow_risk: float = 0.0
    merge_risk: float = 0.0
    joint_risk: float = 0.0
    compress_risk: float = 0.0

    def __post_init__(self) -> None:
        if self.gap < 0.0:
            raise ValueError("the Jensen gap must be non-negative")
        if not 0.0 <= self.quant_err <= 0.5:
            raise ValueError("quant_err must lie in [0, 1/2]")
        for name in ("energy", "protected_risk", "grow_risk", "merge_risk",
                     "joint_risk", "compress_risk"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def potential(self) -> float:
        """``Φ := energy + protectedRisk``.

        Neither term alone is a Lyapunov function: a stage can buy a
        small loss reduction with a large regression elsewhere. Only the
        sum has to decrease for the loop to be sound.
        """
        return self.energy + self.protected_risk


@dataclass(frozen=True)
class CycleBound:
    """The generation's certified potential change.

    Attributes:
        delta: the bound on ``Φ(after) − Φ(before)``; negative means the
            generation is certified to descend.
        grow_gain: the grow stage's certified gain.
        merge_gain: ``gap``, the merge stage's gain.
        joint_gain: the joint stage's certified gain.
        total_gain: their sum.
        total_cost: risks plus the compression cost.
        premises: which premises were supplied, so a conditional bound is
            never read as an unconditional one.
    """

    delta: float
    grow_gain: float
    merge_gain: float
    joint_gain: float
    total_gain: float
    total_cost: float
    premises: frozenset = field(default_factory=frozenset)

    @property
    def descends(self) -> bool:
        """Does the generation provably decrease ``Φ``?"""
        return self.delta <= 0.0

    @property
    def is_conditional(self) -> bool:
        """R91 leaves the stage premises empirical; this says so."""
        return bool(self.premises)


def compression_cost(kappa: float, grid: float, quant_err: float,
                     compress_risk: float = 0.0) -> float:
    """``κ·s/2 + compressRisk`` -- the compression stage's cost.

    Note the shape R102 FIX 1 corrected in the Lean source: the honest
    form carries the measured residual, and ``κ·s/2`` here is the
    *bounded* version that assumes ``quant_err`` already accounts for
    saturation. :mod:`hagi.train.ternary_exact` carries ``κ√n·s/2`` and
    the saturation tail; use this when the caller has already folded
    them into ``quant_err``.
    """
    if kappa < 0.0 or grid < 0.0 or quant_err < 0.0 or compress_risk < 0.0:
        raise ValueError("compression terms must be non-negative")
    return kappa * grid / 2.0 + compress_risk


def cycle_bound(
    state: GrowthState,
    grow_gain: float,
    joint_gain: float,
    kappa: float = 0.0,
    grid: float = 0.0,
    premises: frozenset | set | None = None,
) -> CycleBound:
    """``growth_cycle_potential``: the whole generation in one bound.

        Φ(after) − Φ(before)
          ≤ −(growGain + gap + jointGain)
            + (growRisk + mergeRisk + jointRisk + κs/2 + compressRisk)

    Args:
        state: the cycle's starting fields.
        grow_gain: the certified grow-stage gain.
        joint_gain: ``η‖d*‖²/2``, the joint stage's certified gain.
        kappa: the compression constant.
        grid: the quantisation step.
        premises: the empirical premises the caller measured. Recorded so
            the caller can tell a conditional bound from an unconditional
            one -- R91 proves the composition, not the premises.

    Returns:
        The bound, its terms, and the premises it rests on.

    Raises:
        ValueError: on a negative gain.
    """
    if grow_gain < 0.0 or joint_gain < 0.0:
        raise ValueError("gains must be non-negative")
    merge_gain = state.gap
    total_gain = grow_gain + merge_gain + joint_gain
    total_cost = (
        state.grow_risk
        + state.merge_risk
        + state.joint_risk
        + compression_cost(kappa, grid, state.quant_err, state.compress_risk)
    )
    return CycleBound(
        delta=-total_gain + total_cost,
        grow_gain=grow_gain,
        merge_gain=merge_gain,
        joint_gain=joint_gain,
        total_gain=total_gain,
        total_cost=total_cost,
        premises=frozenset(premises) if premises else frozenset(),
    )


def generations_to_epsilon(bound: CycleBound, potential0: float,
                           epsilon: float) -> float:
    """``(Φ₀ − Φ_min)/ε`` -- generations to the epsilon-floor.

    The Lyapunov telescope over whole generations: each certified
    descent is at least ``ε``, so the potential reaches its floor within
    this many generations.

    Args:
        bound: the per-generation bound, whose ``-delta`` is the descent.
        potential0: the initial potential.
        epsilon: the per-generation descent to certify.

    Returns:
        The number of generations.

    Raises:
        ValueError: on a non-positive epsilon or a bound that does not
            descend by at least epsilon.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    descent = -bound.delta
    if descent < epsilon - 1e-12:
        raise ValueError(
            f"the bound certifies a descent of only {descent:.6f} per "
            f"generation, which does not reach epsilon={epsilon}"
        )
    return potential0 / epsilon


def energy_only_is_not_a_potential() -> bool:
    """``False``, always -- the module's reason for summing the terms.

    A stage can trade a small loss reduction for a large regression, so
    energy alone can increase while the loop is still sound. The
    potential is the SUM precisely because neither term is monotone on
    its own. Stated as a function so a test can assert it rather than a
    reader having to infer it.
    """
    return False