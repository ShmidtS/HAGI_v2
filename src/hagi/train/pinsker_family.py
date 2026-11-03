"""Pinsker family + TV duality + DV/Cramér — runtime ports (§18/§19).

Lean SSOT: primes ``Hagi/Probability/PinskerTV.lean`` and
``Hagi/Probability/ChernoffCGF.lean``:

Pinsker family (positive distributions on a finite alphabet):
- ``tv_le_pos_part_kl``: TV <= sum over {q<p} of p·ln(p/q) (the linear
  KL form — Pinsker-with-sqrt is the correct linear bound, TV <= KL is
  FALSE: counterexample 0.368 < 0.4);
- ``bhattacharyya_cauchy``: BC(p,q) = sum sqrt(p·q) <= 1 (Cauchy-Schwarz);
- ``one_minus_bc_le_half_kl``: 1 − BC <= KL/2;
- ``pinsker_sqrt_kl``: TV <= sqrt(KL) (via |p−q| = |√p−√q|(√p+√q) + CS);
- ``neg_log_bc_le_half_kl``: BC >= exp(−KL/2);
- ``tv_le_sqrt_one_minus_exp``: Bretagnolle–Huber TV <= sqrt(1 − e^{−KL})
  (tighter at large KL);
- ``tv_dual_abs``: |E_p f − E_q f| <= TV for any 0 <= f <= 1.

DV / Cramér vertical:
- ``kl_variational_lower``: KL >= E_p f − ln E_q e^f (Donsker–Varadhan);
- ``cramer_sup_le_kl``: sup over tilts bounds KL from above by CGF;
- ``subgauss_dev_kl``: (E−μ)²/2σ² <= KL (λ*-substitution);
- ``mean_shift_tv``: |E_p g − E_q g| <= (b−a)·TV.
"""
from __future__ import annotations

import math


def tv(p: list[float], q: list[float]) -> float:
    """Total variation 0.5·Σ|p_i − q_i|."""
    if len(p) != len(q):
        raise ValueError("p, q must have equal lengths")
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))


def kl(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    """KL(p || q) with positive-part convention for p_i = 0."""
    if len(p) != len(q):
        raise ValueError("p, q must have equal lengths")
    return sum(
        pi * math.log(pi / max(qi, eps)) for pi, qi in zip(p, q) if pi > 0.0
    )


def tv_le_pos_part_kl(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    """TV <= Σ_{q_i<p_i} p·ln(p/q) [``tv_le_pos_part_kl``]. Returns the
    bound (>= TV)."""
    pos = sum(
        pi * math.log(pi / max(qi, eps))
        for pi, qi in zip(p, q)
        if qi < pi and pi > 0.0
    )
    return pos


def bhattacharyya(p: list[float], q: list[float]) -> float:
    """BC(p,q) = Σ sqrt(p_i q_i) [``bhattacharyya_cauchy``: <= 1]."""
    if len(p) != len(q):
        raise ValueError("p, q must have equal lengths")
    return sum(math.sqrt(max(pi, 0.0) * max(qi, 0.0)) for pi, qi in zip(p, q))


def one_minus_bc_le_half_kl(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    """1 − BC <= KL/2 [``one_minus_bc_le_half_kl``]. Returns the bound."""
    return kl(p, q, eps) / 2.0


def pinsker_sqrt_kl(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    """TV <= sqrt(KL) [``pinsker_sqrt_kl``]. Returns the bound."""
    return math.sqrt(kl(p, q, eps))


def neg_log_bc_le_half_kl(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    """−ln BC <= KL/2 [``neg_log_bc_le_half_kl``]. Returns the bound."""
    max(bhattacharyya(p, q), eps)
    return kl(p, q, eps) / 2.0


def tv_le_sqrt_one_minus_exp(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    """Bretagnolle–Huber: TV <= sqrt(1 − e^{−KL})
    [``tv_le_sqrt_one_minus_exp``]. Returns the bound."""
    return math.sqrt(max(0.0, 1.0 - math.exp(-kl(p, q, eps))))


def tv_dual_abs(p: list[float], q: list[float], f: list[float]) -> float:
    """|E_p f − E_q f| for 0 <= f <= 1 [``tv_dual_abs``]. Returns the
    expectation gap (certified <= TV)."""
    if len(p) != len(q) or len(p) != len(f):
        raise ValueError("p, q, f must have equal lengths")
    if any(not (0.0 <= fi <= 1.0) for fi in f):
        raise ValueError("f must take values in [0, 1]")
    ep = sum(pi * fi for pi, fi in zip(p, f))
    eq = sum(qi * fi for qi, fi in zip(q, f))
    return abs(ep - eq)


def kl_variational_lower(p: list[float], q: list[float], f: list[float]) -> float:
    """Donsker–Varadhan lower bound: E_p f − ln E_q e^f <= KL
    [``kl_variational_lower`` family]. Returns the bound value."""
    if len(p) != len(q) or len(p) != len(f):
        raise ValueError("p, q, f must have equal lengths")
    ep = sum(pi * fi for pi, fi in zip(p, f))
    # log-sum-exp of ln(q_i) + f_i (stable shift)
    shifted = [math.log(max(qi, 1e-300)) + fi for qi, fi in zip(q, f)]
    m = max(shifted)
    ln_eq = m + math.log(sum(math.exp(x - m) for x in shifted))
    return ep - ln_eq


def subgauss_dev_kl(mean_shift: float, sigma: float) -> float:
    """(E−μ)²/2σ² <= KL [``subgauss_dev_kl``]. Returns the KL lower bound
    for a mean shift of ``mean_shift`` at scale ``sigma``."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (mean_shift * mean_shift) / (2.0 * sigma * sigma)


def mean_shift_tv_bound(b: float, a: float, tv_value: float) -> float:
    """|E_p g − E_q g| <= (b−a)·TV [``mean_shift_tv``]. Returns the bound."""
    if b < a:
        raise ValueError("b must be >= a")
    return (b - a) * tv_value


def cramer_sup_le_kl(cgf_sup: float) -> float:
    """sup_λ (λt − ln E e^{λX}) <= KL of the tilted family
    [``cramer_sup_le_kl``]. Identity pass-through: the sup of the CGF
    Legendre transform equals the KL of the exponentially tilted family
    with the same law."""
    return cgf_sup
