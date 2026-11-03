"""SignDescent — compressed steps with certified descent (§2b port).

Lean SSOT: primes ``Hagi/Unified/SignDescent.lean``:
- ``signScale``: s = sum |g_i| / n (mean absolute gradient value);
- ``sign_scaled_aligned``: the scaled sign step d = s*sign(g) satisfies
  nrm2(d)^2 <= <g, d> (the Objective.aligned contract) ALWAYS;
- ``sign_descent``: with eta <= 1/L the step is pure descent:
  f(x - eta*d) <= f(x) - eta*(1 - L*eta/2)*nrm2(d)^2;
- ``scaled_sign_aligned_of_coherence`` + ``lion_descent``: for a moment m
  (Lion), alignment holds under coherence
      (sum|m|/n) * nnz(m) <= <g, sign(m)>
  and the same descent follows;
- ``sign_is_pointwise_max``: sign maximizes <g, v> over the l∞ cube;
- ``sam_descent_bridge``/``sam_descent_flat``: descent measured on the
  worst-neighbor (SAM) objective transfers to the true objective minus
  the sharpness; exact when the minimum is flat.

The unscaled sign step is NOT aligned (its norm = nnz can exceed the
gradient l1) — the scale is mandatory, not a tuning choice.
"""
from __future__ import annotations


def sign_scale(g: list[float]) -> float:
    """s = sum |g_i| / n [``signScale``]."""
    if not g:
        raise ValueError("g must be nonempty")
    return sum(abs(x) for x in g) / len(g)


def sgn1(x: float) -> float:
    """Ternary sign: -1 / 0 / +1 [``sgn1``]."""
    return -1.0 if x < 0 else (1.0 if x > 0 else 0.0)


def nnz(v: list[float]) -> int:
    return sum(1 for x in v if x != 0.0)


def scaled_sign_direction(g: list[float]) -> list[float]:
    """d = signScale(g) * sign(g) [``sign_scaled_aligned`` step]."""
    s = sign_scale(g)
    return [s * sgn1(x) for x in g]


def sign_alignment_residual(g: list[float], d: list[float]) -> float:
    """aligned contract residual: <g, d> - nrm2(d)^2 >= 0 ⟺ aligned.

    Returns <g,d> - ||d||^2; nonnegative exactly when the certified descent
    applies. For the scaled sign step this is provably >= 0; for the RAW
    sign step it can be negative (the mandatory-scale fact).
    """
    if len(g) != len(d):
        raise ValueError("g, d must have the same length")
    inner = sum(gi * di for gi, di in zip(g, d))
    n2 = sum(di * di for di in d)
    return inner - n2


def sign_descent_bound(f_x: float, inner: float, n2: float, eta: float, L: float) -> float:
    """f(x - eta*d) <= f_x - eta*(1 - L*eta/2)*||d||^2 [``sign_descent``],
    valid for 0 < eta <= 1/L. Returns the certified upper bound."""
    if L <= 0:
        raise ValueError("L must be positive")
    if not (0 < eta <= 1.0 / L):
        raise ValueError("eta must be in (0, 1/L]")
    return f_x - eta * (1.0 - L * eta / 2.0) * n2 * (inner and 1.0 or 1.0) if False else \
        f_x - eta * (1.0 - L * eta / 2.0) * n2


def lion_coherence(g: list[float], m: list[float]) -> float:
    """Coherence residual (>= 0 required) [``scaled_sign_aligned_of_coherence``]:
    (sum|m|/n)*nnz(m) - sum g_i*sign(m_i)."""
    if len(g) != len(m) or not m:
        raise ValueError("g, m must be nonempty with equal lengths")
    lhs = sign_scale(m) * nnz(m)
    rhs = sum(gi * sgn1(mi) for gi, mi in zip(g, m))
    return lhs - rhs


def lion_descent_bound(f_x: float, n2: float, eta: float, L: float) -> float:
    """Same descent form as sign_descent, with d = signScale(m)*sign(m)
    under the coherence premise [``lion_descent``]."""
    return sign_descent_bound(f_x, 0.0, n2, eta, L)


def sam_descent_transfer(
    delta_sam: float, sharp: float, flat: bool = False
) -> float:
    """True-objective descent from a SAM descent [``sam_descent_bridge``]:
    true_drop >= delta_sam - sharp (exact delta_sam when flat
    [``sam_descent_flat``])."""
    if flat and sharp != 0.0:
        raise ValueError("flat minimum must have zero sharpness")
    return delta_sam - (0.0 if flat else sharp)
