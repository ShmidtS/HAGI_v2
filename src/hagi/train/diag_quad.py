"""DiagQuad — weighted (anisotropic) quadratic objective (port).

Lean SSOT: primes ``Hagi/Unified/DescentCoreDiag.lean`` (diagE,
diagGrad, diag_descent_eq, diag_descent_le, cMax, diagObjective).

The SECOND real instance of the descent Objective contract, after the
isotropic quadratic machine (L = 1): E(w) = 1/2 * sum_i c_i (w_i - a_i)^2
with diagonal curvature c_i >= 0. The smoothness constant is DERIVED,
not assumed: the exact identity

    E(x + d) = E(x) + <grad, d> + 1/2 * sum_i c_i d_i^2
             <= E(x) + <grad, d> + L/2 * ||d||^2,   L = max_i c_i

certifies the descent drop at any step eta <= 1/L (one step at eta = 1/c_i
per-coordinate hits the optimum exactly). Runtime consumers: SafeQP
eta windows on anisotropic curvature, layer_tuning per-layer gains.
"""
from __future__ import annotations


def diag_energy(c: list[float], a: list[float], x: list[float]) -> float:
    """E(w) = 1/2 * sum_i c_i * (w_i - a_i)^2 [``diagE``]."""
    if not (len(c) == len(a) == len(x)):
        raise ValueError("c, a, x must have the same length")
    return 0.5 * sum(ci * (xi - ai) ** 2 for ci, ai, xi in zip(c, a, x))


def diag_grad(c: list[float], a: list[float], x: list[float]) -> list[float]:
    """grad_i = c_i * (x_i - a_i) [``diagGrad``]."""
    if not (len(c) == len(a) == len(x)):
        raise ValueError("c, a, x must have the same length")
    return [ci * (xi - ai) for ci, ai, xi in zip(c, a, x)]


def diag_descent_expansion(
    c: list[float], a: list[float], x: list[float], d: list[float]
) -> tuple[float, float, float]:
    """Exact expansion [``diag_descent_eq``]: returns
    (E(x), <grad(x), d>, 1/2 * sum c_i d_i^2) with
    E(x + d) == E(x) + <grad, d> + quad exactly (identity, not bound).
    """
    if not (len(c) == len(a) == len(x) == len(d)):
        raise ValueError("c, a, x, d must have the same length")
    e0 = diag_energy(c, a, x)
    g = diag_grad(c, a, x)
    inner = sum(gi * di for gi, di in zip(g, d))
    quad = 0.5 * sum(ci * di * di for ci, di in zip(c, d))
    return e0, inner, quad


def c_max(c: list[float]) -> float:
    """max_i c_i [``cMax``]; nonempty, caller guarantees c_i >= 0."""
    if not c:
        raise ValueError("c must be nonempty")
    return max(c)


def diag_descent_le(
    c: list[float], a: list[float], x: list[float], d: list[float], L: float
) -> float:
    """Upper-quadratic model [``diag_descent_le``]:
    E(x + d) <= E(x) + <grad, d> + L/2*||d||^2, valid for any
    L >= max_i c_i (per-pointwise domination). Returns the bound.
    """
    e0, inner, _ = diag_descent_expansion(c, a, x, d)
    return e0 + inner + L / 2.0 * sum(di * di for di in d)


def diag_one_step(c: list[float], a: list[float], x: list[float]) -> list[float]:
    """Exact one-step optimum witness [``quad_one_step`` pattern]:
    x - eta * grad with eta_i = 1/c_i per coordinate lands exactly on
    the optimum a (E(a) = 0) whenever every c_i > 0."""
    if any(ci <= 0.0 for ci in c):
        raise ValueError("one-step witness needs strictly positive curvature")
    return [xi - (gi / ci) for xi, gi, ci in zip(x, diag_grad(c, a, x), c)]
