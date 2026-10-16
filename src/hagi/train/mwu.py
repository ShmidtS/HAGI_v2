"""MWU / hedge / Gibbs bridge port (R432-R443).

Ports ``C:\\primes\\Hagi\\Probability\\MWU.lean`` (read-only SSOT):
one-step and telescoped multiplicative-weights potentials, the
static regret sandwich, the Gibbs = hedge bridge at η = 1/T and
the double upper bound that yields the DERIVED optimal learning
rate (equalizing the two competing terms) — η is not a
hyperparameter but the solution of

    2η·⟨ℓ⟩_unif  vs  ln n + η·ℓ_min
    ⇒  η* = ln n / (2·⟨ℓ⟩_unif − ℓ_min)

Key identities (all machine-checked in Lean):

* ``mwu_potential`` Z(η,ℓ) = (1/n)·Σ e^{−ηℓ_v} — hedge potential;
* ``potential_sandwich`` — ηℓ_min ≤ −ln Z ≤ 2η⟨ℓ⟩ when ηℓ ≤ 1/2
  (`mwu_potential_squeeze` R435, via `mwu_potential_step` R433 and
  `mwu_potential_lower` R434);
* ``mwu_double_upper`` — −ln Z ≤ min(2η⟨ℓ⟩, ln n + ηℓ_min)
  (R443): two competing bounds in one theorem;
* ``eta_star`` — the equalizing learning rate (basis of optimal
  LR selection; classical ln n regret term from R437);
* ``gibbs_hedge`` — Gibbs free-energy statistic at temperature T
  IS the hedge potential at η = 1/T (`gibbs_hedge_squeeze` R441):
  cooling the thermostat = raising the learning rate;
* ``static_regret`` — multi-step (product/telescope) form for
  cumulative losses (`mwu_prod_exp` R438, `mwu_partition_sum`
  R439, `mwu_regret_static` R440).
"""

from __future__ import annotations

import math

__all__ = [
    "mwu_potential",
    "potential_sandwich",
    "mwu_double_upper",
    "eta_star",
    "gibbs_hedge",
    "static_regret",
]


def mwu_potential(losses, eta: float) -> float:
    """Z(η,ℓ) = (1/n)·Σ_v e^{−η·ℓ_v} — the hedge potential
    (`MWU.mwu_potential_step` and friends work on ln Z')."""
    n = len(losses)
    m = min(losses)
    # log-sum-exp for stability: ln Z = -ηm + ln((1/n)Σ e^{-η(ℓ-m)})
    lse = math.log(sum(math.exp(-eta * (l - m)) for l in losses) / n)
    return math.exp(-eta * m + lse)


def potential_sandwich(losses, eta: float) -> tuple[float, float, float]:
    """(lower, −ln Z, upper): ηℓ_min ≤ −ln Z ≤ 2η⟨ℓ⟩ when ηℓ ≤ 1/2 —
    `MWU.mwu_potential_squeeze` (R435) with `mwu_potential_lower`
    (R434). The uniform-start specialization is `mwu_uniform_squeeze`
    (R436); the classical ln n term is R437 (see
    :func:`mwu_double_upper`)."""
    n = len(losses)
    mean = sum(losses) / n
    l_min = min(losses)
    assert eta * max(losses) <= 0.5, "premise hsmall: eta*l <= 1/2"
    neg_ln_z = -math.log(mwu_potential(losses, eta))
    return eta * l_min, neg_ln_z, 2.0 * eta * mean


def mwu_double_upper(losses, eta: float) -> tuple[float, float, float]:
    """(bound1, −ln Z, bound2): −ln Z ≤ min(2η⟨ℓ⟩, ln n + ηℓ_min) —
    `MWU.mwu_double_upper` (R443). Returns both competing bounds
    around the statistic; equalizing them gives :func:`eta_star`."""
    n = len(losses)
    mean = sum(losses) / n
    l_min = min(losses)
    assert eta * max(losses) <= 0.5
    neg_ln_z = -math.log(mwu_potential(losses, eta))
    return 2.0 * eta * mean, neg_ln_z, math.log(n) + eta * l_min


def eta_star(losses) -> float:
    """Derived optimal learning rate: equalize the two R443 terms
    2η⟨ℓ⟩ = ln n + ηℓ_min  ⇒  η* = ln n/(2⟨ℓ⟩ − ℓ_min) — the LR is
    SOLVED, not tuned (R443 basis; requires 2⟨ℓ⟩ > ℓ_min, true
    whenever more than one expert has non-negligible loss)."""
    n = len(losses)
    mean = sum(losses) / n
    l_min = min(losses)
    denom = 2.0 * mean - l_min
    if denom <= 0.0:
        raise ValueError("degenerate losses: all mass on the minimizer")
    return math.log(n) / denom


def gibbs_hedge(energies, temp: float) -> tuple[float, float, float]:
    """Gibbs = hedge at η = 1/T — `MWU.gibbs_hedge_squeeze` (R441):
    (E_min/T, −ln((1/n)Σe^{−E/T}), 2⟨E⟩/T). Temperature IS the
    inverse learning rate; thermostat cooling = LR growth."""
    eta = 1.0 / temp
    return potential_sandwich(energies, eta)


def static_regret(loss_seqs, eta: float) -> tuple[float, float, float]:
    """Telescoped (multi-step) sandwich for cumulative losses —
    `MWU.mwu_regret_static` (R440) via the weight telescope
    `mwu_prod_exp` (R438) and the partition closed form
    `mwu_partition_sum` (R439): η·L_min ≤ −ln((1/n)Σ_v e^{−ηL_v})
    ≤ 2η·⟨L⟩ when η·L_v ≤ 1/2 for all v."""
    # cumulative losses per expert: loss_seqs[v] = sequence over t
    cum = [sum(seq) for seq in loss_seqs]
    return potential_sandwich(cum, eta)


if __name__ == "__main__":  # pragma: no cover
    ls = [0.2, 0.5, 0.9, 1.3]
    b1, z, b2 = mwu_double_upper(ls, 0.3)
    print(f"double upper: {b1:.4f} >= {z:.4f} <= {b2:.4f}, "
          f"eta*={eta_star(ls):.4f}")
