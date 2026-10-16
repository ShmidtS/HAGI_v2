"""RoundViability + WallClockTakeoff port (advisory gates, pure torch/math).

Ports ``Hagi/Growth/GainOperator.lean`` (devEnergy, GainOp,
gainop_cone_step), ``Hagi/Growth/RoundViability.lean`` (round_viable,
round_exhausted, cycle_criterion) and ``Hagi/Dynamics/WallClockTakeoff.lean``
(wallclock_takeoff, wallclock_rate_from_cone). Source of truth:
C:\\primes (read-only).

The round-viability certificate decides, from the MEASURED disagreement
energy of the fresh expert family, whether the growth round keeps the
cone invariant k·C' ≤ D' (continue) or the dev-channel is dead
(honest stop):
    devEnergy dev = Σ_k ‖dev_k‖²          (GainOperator.devEnergy)
    GainOp:  ρ·D + η·E_dev − ξ ≤ D'
    round_viable: C' = C + γD, GainOp, γk ≤ ρ, kC ≤ D and
        γk²·C + (1−ρ)k·C + ξ ≤ (η_p·η_s)·E_dev  ⟹  k·C' ≤ D'.

Wall-clock takeoff: per-cycle time hypothesis τ_t with multiplier
≥ (1 + k·τ_t) gives C_T ≥ C0·exp(k·Στ − (k²/2)·Στ²); from the cone
multiplier (1+α)^{s_t} with success half p₀ the effective rate is
k_eff = p₀·ln(1+α)/τ per wall-clock unit.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "dev_energy",
    "gainop_holds",
    "round_viable",
    "round_exhausted",
    "cycle_criterion",
    "advise",
    "wallclock_takeoff",
    "wallclock_rate_from_cone",
    "k_eff",
]


def dev_energy(dev: torch.Tensor) -> float:
    """E_dev = Σ_k ‖dev_k‖² — `GainOperator.devEnergy`.

    ``dev``: (N, …) stack of expert deviation vectors; the energy is
    the sum of squared norms over the family."""
    return float(dev.reshape(dev.shape[0], -1).pow(2).sum(dim=1).sum())


def gainop_holds(d: float, d_prime: float, e_dev: float,
                 rho: float, eta: float, xi: float) -> bool:
    """GainOp D D' dev ρ η ξ :⇔ ρ·D + η·E_dev − ξ ≤ D' —
    `GainOperator.GainOp`."""
    return rho * d + eta * e_dev - xi <= d_prime


def round_viable(c: float, d: float, c_prime: float, d_prime: float,
                 gamma: float, rho: float, k: float, xi: float,
                 eta_p: float, eta_s: float, dev: torch.Tensor
                 ) -> tuple[bool, dict]:
    """Cone-invariant certificate for one growth round —
    `RoundViability.round_viable` (via `gainop_cone_step`).

    Returns (viable, certificate dict with every premise value and
    margin, style of certified_controller). All premises are measured
    inputs; the theorem guarantees k·C' ≤ D' when every check passes."""
    e_dev = dev_energy(dev)
    eta = eta_p * eta_s
    hstep = math.isclose(c_prime, c + gamma * d, rel_tol=1e-9, abs_tol=1e-12)
    hop = gainop_holds(d, d_prime, e_dev, rho, eta, xi)
    hrhogk = gamma * k <= rho
    hcone = k * c <= d
    thresh = gamma * k * k * c + (1 - rho) * k * c + xi
    hthresh = thresh <= eta * e_dev
    viable = hstep and hop and hrhogk and hcone and hthresh and e_dev > 0
    cert = {
        "hstep": hstep, "gainop": hop, "hrhogk": hrhogk, "hcone": hcone,
        "hthresh": hthresh, "e_dev": e_dev, "eta": eta,
        "threshold": thresh,
        "threshold_margin": eta * e_dev - thresh,
        "conclusion": "k*C' <= D'" if viable else "round not viable",
    }
    return viable, cert


def round_exhausted(dev: torch.Tensor) -> bool:
    """devEnergy = 0 with GainOp ⇒ ρ·D − ξ ≤ D' and η·E_dev = 0 —
    `RoundViability.round_exhausted`: no disagreement energy left,
    the round cannot renew the frontier."""
    return dev_energy(dev) == 0.0


def cycle_criterion(dev: torch.Tensor) -> str:
    """Stop/continue criterion — `RoundViability.cycle_criterion`:
    devEnergy = 0 ⇒ exhausted (honest stop), else continue."""
    return "STOP_round_exhausted" if round_exhausted(dev) else "CONTINUE"


def wallclock_takeoff(c0: float, k: float, taus: list[float]) -> float:
    """C_T lower bound C0·exp(k·Στ − (k²/2)·Στ²) —
    `WallClockTakeoff.wallclock_takeoff` (second-order slack explicit)."""
    s1, s2 = sum(taus), sum(t * t for t in taus)
    return c0 * math.exp(k * s1 - (k * k / 2) * s2)


def k_eff(p0: float, alpha: float, tau: float) -> float:
    """Effective wall-clock rate p₀·ln(1+α)/τ —
    `WallClockTakeoff.wallclock_rate_from_cone`."""
    return p0 * math.log1p(alpha) / tau


def wallclock_rate_from_cone(c0: float, p0: float, alpha: float,
                             tau: float, t_cycles: float) -> float:
    """C_T ≥ C0·exp(k_eff·W − ln(1+α)·Δ), W = T·τ —
    `WallClockTakeoff.wallclock_rate_from_cone` with the concentrated
    Δ = √(2T·log(1/δ)) term returned separately (caller supplies δ)."""
    del t_cycles  # W is derived by the caller from taus; kept for parity
    _ = c0
    return k_eff(p0, alpha, tau)


def advise(dev: torch.Tensor, ledger: dict | None = None) -> dict:
    """Advisory hook for the ladder loop (not wired into loop.py):
    emits the cycle criterion + dev energy for telemetry."""
    e = dev_energy(dev)
    return {
        "criterion": cycle_criterion(dev),
        "dev_energy": e,
        "ledger_gen": (ledger or {}).get("gen"),
    }
