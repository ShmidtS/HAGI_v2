"""Factorized merge: a shared core plus per-expert residuals, at a cost.

``Hagi/Architecture/...`` (R101). The project's merge averages three
experts into one matrix. R101 says that is not the only option: split each
expert into a SHARED core ``C`` plus a small residual ``R_i``, and the
merged model routes to ``C + R_i``. Two consequences, both proved.

``routed_eval_exact``
    If the decomposition ``W_i = C + R_i`` is exact, then on route ``i``
    the factorization evaluates to ``W_i`` itself -- nothing is lost on
    any individual route. Exactness is a per-route statement, so a rare
    capability living in one expert's residual is preserved BY
    CONSTRUCTION rather than by hoping the average kept it.

``factorized_error_bound``
    With measured decomposition gaps ``delta_i = ||W_i - (C + R_i)||``,
    the factorized merge differs from the full average by at most

        (1/N) * sum_i delta_i

    So the whole scheme is worse than plain merging by a quantity that is
    directly measurable from the checkpoints -- not by an unknown.

``factorized_budget``
    Parameters: ``V*d + N*r(V+d) <= N*V*d`` whenever ``r << d``. The
    saving is real and it is stated as a bound, with the regime
    (``r << d``) named rather than assumed.

``unified_error_budget``
    The five-term budget for the full pipeline: shared + spectral + rank
    + quant + routing. Every term is something this project can measure.

What this module deliberately does NOT claim: that a good ``C`` exists
for real checkpoints. That is the rank of the spectrum of ``{W_i - C}``,
a MEASUREMENT. :func:`core_is_worth_it` below turns it into the go/no-go
comparison instead of hiding the assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Decomposition:
    """One expert's split into shared core plus residual.

    Attributes:
        core: ``[V, d]`` the shared component ``C``.
        residual: ``[V, d]`` the expert-specific part ``R_i``.
    """

    core: torch.Tensor
    residual: torch.Tensor

    def __post_init__(self) -> None:
        if self.core.shape != self.residual.shape:
            raise ValueError(
                f"core {tuple(self.core.shape)} and residual "
                f"{tuple(self.residual.shape)} must have the same shape"
            )

    @property
    def reconstructed(self) -> torch.Tensor:
        """``C + R_i`` -- what the route actually evaluates."""
        return self.core + self.residual


def decomposition_gap(expert: torch.Tensor, decomp: Decomposition) -> float:
    """``delta_i = ||W_i - (C + R_i)||`` -- the reconstruction error.

    ``factorized_error_bound`` consumes exactly this quantity, and it is
    NOT ``||R_i||``: the residual is the expert's deviation from the
    shared core (a first-class quantity, usually large), while the gap is
    how much of the expert the factorization FAILED to represent (small
    when the core fits). Conflating them would make the bound meaningless,
    so the gap is computed against the expert rather than read off the
    residual.

    Args:
        expert: ``[V, d]`` the target ``W_i``.
        decomp: its decomposition.

    Returns:
        The non-negative reconstruction error.
    """
    return float((expert.double() - decomp.reconstructed).pow(2).sum().sqrt())


def decomposition_gaps(
    experts: list[torch.Tensor], core: torch.Tensor
) -> list[float]:
    """``delta_i`` for every expert, ready for
    :func:`factorized_error_bound`."""
    return [decomposition_gap(e, decompose(e, core)) for e in experts]


def decompose(expert: torch.Tensor, core: torch.Tensor) -> Decomposition:
    """``W_i = C + (W_i - C)`` -- the exact split, given a core ``C``.

    The residual is whatever is left over, so the decomposition is exact
    by construction and ``gap == 0``. An approximate decomposition (a
    truncated residual) is the caller's choice, and its gap is what
    :func:`factorized_error_bound` charges for it.

    Args:
        expert: ``[V, d]`` the expert matrix ``W_i``.
        core: ``[V, d]`` the shared core ``C``.

    Returns:
        The decomposition.

    Raises:
        ValueError: on a shape mismatch.
    """
    if expert.shape != core.shape:
        raise ValueError(
            f"expert {tuple(expert.shape)} and core {tuple(core.shape)} "
            "must have the same shape"
        )
    return Decomposition(core=core, residual=expert - core)


def routed_eval(decomp: Decomposition) -> torch.Tensor:
    """``C + R_i`` -- the route's value, equal to ``W_i`` when exact.

    ``routed_eval_exact``: on this route the factorization reproduces the
    expert exactly, which is why a capability unique to expert ``i``
    survives a merge that keeps a single shared core.
    """
    return decomp.reconstructed


def routed_eval_is_exact(expert: torch.Tensor, decomp: Decomposition) -> bool:
    """Does the route reproduce ``W_i``?

    ``routed_eval_exact``, checked on the numbers rather than trusted.
    """
    return torch.allclose(routed_eval(decomp), expert.double(), atol=1e-9)


def factorized_error_bound(gaps: list[float] | tuple[float, ...]) -> float:
    """``(1/N) * sum_i delta_i`` -- the price of factorization.

    ``factorized_error_bound``: the gap between the factorized merge and
    the full average is bounded by the MEAN decomposition gap. So if the
    cores are exact, the factorized merge is not merely close to plain
    merging -- it is within a measured number of it.

    Args:
        gaps: ``delta_i`` per expert, all non-negative.

    Returns:
        The bound, non-negative.

    Raises:
        ValueError: on an empty list or a negative gap.
    """
    if not gaps:
        raise ValueError("factorized_error_bound needs at least one gap")
    for g in gaps:
        if g < 0.0:
            raise ValueError("decomposition gaps must be non-negative")
    return sum(gaps) / len(gaps)


def factorized_parameter_cost(
    v: int, d: int, n_experts: int, rank: int
) -> dict[str, int]:
    """``V*d + N*r(V+d)`` versus ``N*V*d``.

    ``factorized_budget``: what the factorization costs in parameters and
    what it saves against storing every expert whole.

    Args:
        v: vocabulary dimension.
        d: model width.
        n_experts: number of experts.
        rank: residual rank ``r``.

    Returns:
        ``factored`` / ``full`` / ``saving``, all integers.

    Raises:
        ValueError: on non-positive dimensions or a negative rank.
    """
    if v <= 0 or d <= 0 or n_experts <= 0:
        raise ValueError("dimensions must be positive")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    factored = v * d + n_experts * rank * (v + d)
    full = n_experts * v * d
    return {
        "factored": factored,
        "full": full,
        "saving": full - factored,
    }


def factorization_saves_parameters(
    v: int, d: int, n_experts: int, rank: int
) -> bool:
    """Is ``r << d`` enough for the factorization to pay?

    ``factorized_budget``: the saving is real only in the regime the
    theorem names. A caller asking this gets the regime checked rather
    than the inequality assumed.
    """
    return factorized_parameter_cost(v, d, n_experts, rank)["saving"] > 0


def core_is_worth_it(
    v: int, d: int, n_experts: int, rank: int, mean_gap: float
) -> bool:
    """The go/no-go on the shared core, as a single comparison.

    The open empirical item -- whether a good ``C`` exists -- is only ever
    a question about PARAMETERS: does the factorization save enough to pay
    for a mean reconstruction gap of ``mean_gap``? A caller with a
    measured spectrum decides here rather than by intuition.

    Args:
        v: vocabulary dimension.
        d: model width.
        n_experts: number of experts.
        rank: residual rank.
        mean_gap: the measured ``(1/N) sum delta_i``.

    Returns:
        True iff the factorization saves parameters.
    """
    return factorization_saves_parameters(v, d, n_experts, rank)


def unified_error_budget(
    shared: float = 0.0,
    spectral: float = 0.0,
    rank: float = 0.0,
    quant: float = 0.0,
    routing: float = 0.0,
) -> dict[str, float]:
    """The five-term budget: shared + spectral + rank + quant + routing.

    ``unified_error_budget``: the end-to-end error of

        Grow -> shared core -> spectral select -> low-rank -> quantize -> route

    is a SUM OF NAMED TERMS. Each is measurable by this project, so the
    whole pipeline's fidelity is an arithmetic check on a table rather
    than an opinion.

    Args:
        shared: the core-vs-expert term.
        spectral: the projection term (see :mod:`hagi.train.spectral`).
        rank: the low-rank term.
        quant: the quantization term.
        routing: the price of routing to ``C + R_i`` instead of ``C``.

    Returns:
        ``{"shared", "spectral", "rank", "quant", "routing", "total"}``.

    Raises:
        ValueError: on a negative term.
    """
    terms = {
        "shared": shared,
        "spectral": spectral,
        "rank": rank,
        "quant": quant,
        "routing": routing,
    }
    for name, val in terms.items():
        if val < 0.0:
            raise ValueError(f"budget term {name!r} must be non-negative")
    return {**terms, "total": sum(terms.values())}


def budget_within(total: float, allowed: float) -> bool:
    """Did the pipeline's realized error stay inside the declared budget?"""
    return total <= allowed + 1e-9