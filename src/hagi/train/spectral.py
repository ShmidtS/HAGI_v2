"""Spectral selection: an exact error budget for Grow -> select -> quantize.

``Hagi/Spectral/SpectralProjector.lean`` (R100). Compression in this project
is not "quantize a matrix and hope". It is a three-stage chain -- spectral
selection, then low-rank, then quantization -- and R100's contribution is
that the error of the WHOLE chain decomposes into named terms, so the
trade between them is arithmetic done on paper before any GPU time.

``proj_residual_identity``
    For an orthogonal projector ``P`` (``P^2 = P``, self-adjoint),

        ||x - P x||^2 == ||x||^2 - ||P x||^2

    This is EXACT, not an estimate, and it is the statement that makes
    "what did we throw away" a measurement: the discarded energy is the
    difference of two norms of the same vector, not a residual proxy.

``spectral_tail_energy``
    In an orthonormal basis, selecting a subset ``S`` of coordinates
    leaves a residual equal to exactly the sum of squares of the DROPPED
    coefficients. The same law as top-k routing, lifted to projectors:
    what is not kept is exactly what is lost.

``three_stage_error_budget``
    ``||W - Q(A)|| <= ||W - P W|| + eps_rank + eps_quant``

    Every stage contributes its own named term and the bound is the sum,
    which is what makes the budget tradeable: more rank buys back tail
    energy, fewer bits pay for it in ``eps_quant``, and the choice is a
    comparison rather than a guess.

``spectral_safeQP_target`` + ``filtered_step_cost``
    A SafeQP step toward the FILTERED target ``P g`` keeps its guarantees,
    and the price of the filtering -- ``||g - P g||`` -- is carried
    explicitly as a term instead of being absorbed into the noise.

The one thing this module does NOT prove: that a good shared core ``C``
exists for real checkpoints (the rank of the spectrum of ``W_i - C``).
That is a MEASUREMENT on the checkpoints, not a theorem, and
:func:`required_rank` below states it as the measurement to make rather
than assuming the answer.
"""

from __future__ import annotations

import math

import torch


def projector_from_scores(scores: torch.Tensor, keep: int) -> torch.Tensor:
    """Diagonal ``[d, d]`` projector keeping the ``keep`` largest scores.

    The spectral projector in its diagonal form: ``P`` is zero everywhere
    except on the retained coordinates, so ``P^2 = P`` and ``P = P^T`` hold
    exactly by construction.

    Coordinates are ranked by MAGNITUDE, so this agrees with
    :func:`spectral_tail_energy` on the same coefficients. (For a real
    singular-value spectrum the two readings coincide, since singular
    values are non-negative; ranking by magnitude is what keeps them
    consistent for a signed saliency vector.) Ties are broken by index,
    so the projector is deterministic.

    Args:
        scores: ``[d]`` per-coordinate saliency (e.g. a singular-value
            spectrum, or per-coordinate energy).
        keep: how many coordinates to retain.

    Returns:
        ``[d, d]`` symmetric idempotent projector.

    Raises:
        ValueError: on non-positive ``keep``, ``keep > d``, or a
            non-finite score.
    """
    s = scores.double().flatten()
    d = int(s.numel())
    if d == 0:
        raise ValueError("projector_from_scores needs a non-empty score vector")
    if not bool(torch.isfinite(s).all()):
        raise ValueError("projector_from_scores needs finite scores")
    if keep <= 0 or keep > d:
        raise ValueError(f"keep must lie in (0, {d}], got {keep}")
    # topk indices, sorted, so ties resolve by coordinate order.
    idx = torch.topk(s.abs(), keep, largest=True, sorted=True).indices
    mask = torch.zeros(d, dtype=torch.float64)
    mask[idx] = 1.0
    return torch.diag(mask)


def residual_energy(x: torch.Tensor, p: torch.Tensor) -> float:
    """``||x - P x||^2`` -- the energy the projector discards.

    Computed directly from the definition rather than from
    ``||x||^2 - ||P x||^2``, so that
    :func:`proj_residual_identity` is a genuine check on the algebra and
    not a tautology.

    Args:
        x: ``[d]`` the vector being projected.
        p: ``[d, d]`` the projector.

    Returns:
        The squared residual norm, non-negative.
    """
    v = x.double().flatten()
    px = p.double() @ v
    return float(((v - px) ** 2).sum())


def proj_residual_identity(x: torch.Tensor, p: torch.Tensor) -> bool:
    """``||x - P x||^2 == ||x||^2 - ||P x||^2``, exactly.

    ``proj_residual_identity``. The identity is what licenses reading
    "energy kept" straight off ``||Px||^2``: without it, ``||x - Px||^2``
    and the lost norm could differ by a cross term and every downstream
    budget term would be an approximation.

    Returns True when the two sides agree to floating-point tolerance.
    """
    v = x.double().flatten()
    px = p.double() @ v
    lhs = float(((v - px) ** 2).sum())
    rhs = float((v**2).sum()) - float((px**2).sum())
    return math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-12)


def retained_energy(x: torch.Tensor, p: torch.Tensor) -> float:
    """``||P x||^2`` -- the energy the projector keeps."""
    v = x.double().flatten()
    px = p.double() @ v
    return float((px**2).sum())


def spectral_tail_energy(coeffs: torch.Tensor, keep: int) -> float:
    """Squared energy of the coordinates NOT selected.

    ``spectral_tail_energy``. In an orthonormal basis the residual is
    exactly the sum of squares of the dropped coefficients -- the top-k
    routing law, one level up. Selects the ``keep`` largest by magnitude.

    Args:
        coeffs: ``[d]`` coefficients in an ORTHONORMAL basis.
        keep: how many to retain.

    Returns:
        The dropped squared norm, in ``[0, ||coeffs||^2]``.

    Raises:
        ValueError: on an empty input or out-of-range ``keep``.
    """
    c = coeffs.double().flatten()
    d = int(c.numel())
    if d == 0:
        raise ValueError("spectral_tail_energy needs a non-empty vector")
    if keep <= 0 or keep > d:
        raise ValueError(f"keep must lie in (0, {d}], got {keep}")
    idx = torch.topk(c.abs(), keep, largest=True, sorted=False).indices
    mask = torch.ones(d, dtype=torch.float64)
    mask[idx] = 0.0
    return float((c * mask).pow(2).sum())


def rank_from_spectrum(
    values: torch.Tensor, total_energy: float, tolerance: float
) -> int:
    """Smallest rank whose retained energy reaches ``total - tolerance``.

    The operational form of :func:`required_rank`: given the measured
    singular-value spectrum, how many components may be discarded before
    the discarded energy exceeds ``tolerance``?

    Args:
        values: ``[d]`` singular values in NON-INCREASING order (the
            convention of ``torch.linalg.svdvals``).
        total_energy: ``||W||_F^2``; the spectrum must be consistent
            with it (``sum(values^2) <= total_energy``).
        tolerance: the maximum acceptable discarded energy.

    Returns:
        The rank in ``[1, d]``.

    Raises:
        ValueError: on an empty spectrum, a negative tolerance, or a
            tolerance larger than the available energy.
    """
    v = values.double().flatten()
    d = int(v.numel())
    if d == 0:
        raise ValueError("rank_from_spectrum needs a non-empty spectrum")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    sq = v.pow(2)
    if float(sq.sum()) > total_energy + 1e-9:
        raise ValueError(
            "spectrum energy exceeds total_energy; the two must describe "
            "the same matrix"
        )
    # At rank r the discarded energy is the sum of squares of values[r:],
    # i.e. the spectrum's own tail -- the same quantity
    # ``spectral_tail_energy`` measures. Go from r = 0 upward and return
    # the FIRST admissible rank: that is the smallest rank whose tail fits.
    tail = float(sq.sum())
    for r in range(0, d + 1):
        if tail <= tolerance + 1e-12:
            return max(1, r)
        if r < d:
            tail -= float(sq[r])
    return d


def three_stage_error_budget(
    w: torch.Tensor,
    project: torch.Tensor | None = None,
    eps_rank: float = 0.0,
    eps_quant: float = 0.0,
) -> dict[str, float]:
    """``||W - Q(A)|| <= ||W - P W|| + eps_rank + eps_quant``.

    ``three_stage_error_budget``: the whole chain's error is a SUM OF
    NAMED TERMS, so the three stages can be traded against each other
    before any of them runs.

    Args:
        w: ``[m, n]`` the original matrix.
        project: the spectral projector. It must act on the columns of
            ``W``, so it is ``[n, n]`` (or ``[m, m]`` only when ``m == n``).
            If ``None``, that stage contributes zero.
        eps_rank: the low-rank approximation error for the kept subspace.
        eps_quant: the quantization error.

    Returns:
        ``tail`` (the spectral term), ``rank``, ``quant`` and ``bound``
        (their sum), all non-negative.

    Raises:
        ValueError: on negative error terms, or a projector whose shape
            does not match the columns of ``W``.
    """
    if eps_rank < 0.0 or eps_quant < 0.0:
        raise ValueError("error terms must be non-negative")
    tail = 0.0
    if project is not None:
        p = project.double()
        wd = w.double()
        if int(p.shape[-1]) != int(wd.shape[-1]):
            raise ValueError(
                f"projector acts on {p.shape[-1]} coordinates but W has "
                f"{wd.shape[-1]} columns"
            )
        pw = wd @ p
        tail = float((wd - pw).pow(2).sum().sqrt())
    return {
        "tail": tail,
        "rank": float(eps_rank),
        "quant": float(eps_quant),
        "bound": tail + eps_rank + eps_quant,
    }


def budget_is_respected(measured: float, budget: dict[str, float]) -> bool:
    """Did the realized error stay inside the predicted budget?"""
    return measured <= budget["bound"] + 1e-9


def filtered_step_cost(g: torch.Tensor, p: torch.Tensor) -> float:
    """``||g - P g||`` -- the price of denoising the SafeQP target.

    ``filtered_step_cost``: moving the target from ``g`` to ``P g`` is not
    free. The displacement is the cost, and carrying it explicitly is what
    stops "we filtered, so it is better" from being an unfalsifiable claim.

    Args:
        g: ``[d]`` the gradient (or any direction).
        p: ``[d, d]`` the projector.

    Returns:
        The displacement norm.
    """
    v = g.double().flatten()
    pv = p.double() @ v
    return float((v - pv).pow(2).sum().sqrt())


def filtered_target(g: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``P g`` -- the spectrally cleaned SafeQP target."""
    return p.double() @ g.double().flatten()


def spectral_noise_reduction(
    noise: torch.Tensor, p: torch.Tensor
) -> tuple[float, float]:
    """``(||noise||, ||P noise||)`` -- does filtering actually cut noise?

    The empirical premise behind using ``P`` on a gradient: that ``P``
    removes noise faster than signal. This MEASURES both norms on a real
    (or simulated) noise vector instead of assuming the paper's claim,
    because that claim is an empiric of the source, not our theorem.

    Returns:
        ``(before, after)`` in ``[0, inf)``, with ``after <= before``
        guaranteed since ``P`` is a contraction.
    """
    n = noise.double().flatten()
    pn = p.double() @ n
    before = float(n.pow(2).sum().sqrt())
    after = float(pn.pow(2).sum().sqrt())
    return before, after


def is_orthogonal_projector(p: torch.Tensor) -> bool:
    """``P^2 == P`` and ``P == P^T`` -- the premise of every claim here."""
    q = p.double()
    idn = torch.eye(int(q.shape[0]), dtype=torch.float64)
    return bool(
        torch.allclose(q @ q, q, atol=1e-12)
        and torch.allclose(q, q.T, atol=1e-12)
    )