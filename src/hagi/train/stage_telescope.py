"""Certificate telescope across generations (System/GrowthState.lean,
``stage_certificate_telescope`` / ``termination_from_telescope``).

A generation emits a :class:`StageCertificate`: the potential Φ measured
before and after the cycle (runtime proxy: AVG exact CE on the common
protocol), the certified decrease, and the risk spend. The telescope
theorem says: if stage t is sound (after ≤ before − decrease +
riskSpend) and the chain is contiguous (before(t+1) = after(t)), then
after(T−1) ≤ before(0) − Σ decrease + Σ riskSpend — no hidden
identification between generations. Termination: with decrease ≥ ε > r ≥
riskSpend and a floor φmin, the ladder cannot run forever:
T ≤ (Φ₀ − φmin)/(ε − r).

The runtime module is deliberately pure — callers (real_cycle /
run_ladder) own measurement; this file owns the bookkeeping law.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

LEDGER_NAME = "stage_telescope.jsonl"


@dataclass(frozen=True)
class StageCertificate:
    """One generation's Φ bookkeeping entry (before/after/cert)."""

    generation: int
    before: float          # Φ at cycle start (AVG exact CE)
    after: float           # Φ at cycle end
    decrease: float        # certified decrease claimed by the stage
    risk_spend: float      # risk paid by the stage (≤ r for termination)
    n_eval: int            # eval batches behind before/after

    def sound(self) -> bool:
        """Stage soundness: after ≤ before − decrease + riskSpend."""
        return self.after <= self.before - self.decrease + self.risk_spend


def telescope_bound(
    certs: list[StageCertificate], *, contiguity_tol: float = 0.0
) -> float | None:
    """after(T−1) ≤ before(0) − Σ decrease + Σ riskSpend.

    Returns None when the chain is empty; raises ValueError when a stage
    is unsound or the chain is not contiguous (|before(t+1) − after(t)|
    > contiguity_tol) — a hidden identification the theorem forbids
    silently accepting. The Lean theorem needs exact equality; the
    runtime chain is measured with eval noise ε (Hoeffding, n samples),
    so callers pass contiguity_tol ≈ 2ε — the noise allowance is
    explicit, not hidden.
    """
    if not certs:
        return None
    ordered = sorted(certs, key=lambda c: c.generation)
    for c in ordered:
        if not c.sound():
            raise ValueError(f"unsound stage certificate at gen {c.generation}: {c}")
    for prev, nxt in zip(ordered, ordered[1:]):
        gap = abs(nxt.before - prev.after)
        if gap > contiguity_tol:
            raise ValueError(
                f"chain discontinuity gen {prev.generation}->{nxt.generation}: "
                f"|before-after| = {gap:.4f} > tol {contiguity_tol:.4f}"
            )
    total_dec = sum(c.decrease for c in ordered)
    total_risk = sum(c.risk_spend for c in ordered)
    return ordered[0].before - total_dec + total_risk


def termination_bound(
    certs: list[StageCertificate], phi_min: float, eps: float, r: float
) -> float | None:
    """termination_from_telescope: T ≤ (Φ₀ − φmin)/(ε − r).

    Preconditions (the theorem's): every decrease ≥ eps, every riskSpend
    ≤ r, eps > r, and phi_min ≤ after(T−1). Violations raise ValueError
    — the caller mis-declared the ladder, not the bookkeeping.
    """
    if not certs:
        return None
    if eps <= r:
        raise ValueError(f"eps ({eps}) must exceed r ({r})")
    for c in certs:
        if c.decrease < eps:
            raise ValueError(f"gen {c.generation} decrease {c.decrease} < eps {eps}")
        if c.risk_spend > r:
            raise ValueError(f"gen {c.generation} risk_spend {c.risk_spend} > r {r}")
    bound = telescope_bound(certs)
    assert bound is not None  # certs non-empty
    last_after = max(certs, key=lambda c: c.generation).after
    if phi_min > last_after:
        raise ValueError(
            f"floor violated: phi_min {phi_min} > last after {last_after}"
        )
    return (certs[0].before - phi_min) / (eps - r)


def append_certificate(root: Path, cert: StageCertificate) -> Path:
    """Append a certificate to the per-run ledger (one JSON per line)."""
    ledger = root / LEDGER_NAME
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(cert)) + "\n")
    return ledger


def load_ledger(root: Path) -> list[StageCertificate]:
    """Read the ledger; tolerate a missing file (empty chain)."""
    ledger = root / LEDGER_NAME
    if not ledger.exists():
        return []
    certs = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            certs.append(StageCertificate(**json.loads(line)))
    return certs


def certificate_from_ladder_entry(entry: dict) -> StageCertificate | None:
    """Build a StageCertificate from a growth_ladder.jsonl entry.

    Only ACCEPTED generations enter the chain (a rejected generation
    keeps the incumbent — no stage happened). before = incumbent_ce,
    after = the accepted candidate's CE (best of ft/cooldown/fiber as
    recorded), decrease = before − after (≥ accept margin by the §9
    gate), risk_spend = 0 (the certified A/B margin already prices the
    regression risk), n_eval = accept_n.
    """
    if not entry.get("accepted"):
        return None
    before = entry.get("incumbent_ce")
    after = entry.get("cooldown_ce") or entry.get("ft_ce")
    n = entry.get("accept_n") or 0
    if before is None or after is None:
        return None
    return StageCertificate(
        generation=int(entry["gen"]),
        before=float(before),
        after=float(after),
        decrease=max(0.0, float(before) - float(after)),
        risk_spend=0.0,
        n_eval=int(n),
    )


def rebuild_from_ladder(ladder_path: Path, root: Path) -> int:
    """Rebuild the telescope ledger from the accepted ladder history.

    Returns the number of certificates written. Idempotent: the ledger
    is rewritten from the (authoritative) ladder, not appended twice.
    """
    certs = []
    for line in ladder_path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            cert = certificate_from_ladder_entry(json.loads(line))
            if cert is not None:
                certs.append(cert)
    ledger = root / LEDGER_NAME
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("w", encoding="utf-8") as fh:
        for c in certs:
            fh.write(json.dumps(asdict(c)) + "\n")
    return len(certs)


def anytime_growth_budget(
    steps: int, band_mu: float, band_u: float
) -> float:
    """C_T <= C_0 + T*(mu + u) — the anytime growth budget [R548
    ``anytime_growth_budget`` (Foundations/GrowthDynamics.lean)].

    Composition of the telescope band with concentration: if each
    capability step is a freshwater band b_t <= mu + u (mu the mean
    per-step gain, u the anytime-concentration slack, e.g. the R525
    band sqrt(2 sigma^2 ln(T/alpha))), then after T steps capability
    NEVER exceeds the start plus T*(mu+u). The increment added to the
    telescope (returns T*(mu+u)) does NOT include C_0 — the caller
    adds its own measured start.
    Premises (the theorem's): C(t+1) = C(t) + G_t and G_t <= b_t,
    both measured (h_emp_) per step.
    """
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if band_u < 0.0:
        raise ValueError("band slack u must be nonnegative")
    return steps * (band_mu + band_u)


def geometric_band_ceiling(
    c0: float, b: float, sigma: float, steps: int
) -> tuple[float, float]:
    """Geometric decay band capability ceiling [R556
    ``geometric_band_ceiling`` (Foundations/GrowthDynamics.lean)].

    If per-step gain is covered by a geometrically decaying band
    G_t <= b*(1-sigma)^t (b >= 0, 0 < sigma <= 1 — measured decay of
    the growth axis), growth saturates:
    C_T <= C_0 + (b/sigma)*(1-(1-sigma)^T) <= C_0 + b/sigma.
    Returns (tight_T, asymptotic) — the T-exact and limit ceilings.
    Bridge to the saturation signature satC (R531): a measured
    decaying gain band certifies the saturation ceiling WITHOUT a
    separate capacity measurement.
    Premises (h_emp_): C(t+1) = C(t) + G_t and the band per step.
    """
    if b < 0.0:
        raise ValueError("b must be nonnegative")
    if not 0.0 < sigma <= 1.0:
        raise ValueError("sigma must lie in (0, 1]")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    tight = c0 + b * (1.0 - (1.0 - sigma) ** steps) / sigma
    return tight, c0 + b / sigma


def band_guaranteed_growth_floor(
    c0: float, b: float, sigma: float, steps: int
) -> float:
    """Lower-band guaranteed growth floor [R560 ``band_guaranteed_growth``
    (Foundations/GrowthDynamics.lean)].

    Mirror of ``geometric_band_ceiling``: if the per-step gain is
    SUPPORTED from below by a geometrically decaying band
    G_t >= b*(1-sigma)^t (b >= 0, 0 < sigma <= 1 — measured decay of
    the growth axis), then growth is CERTIFIED from below:
    C_T >= C_0 + (b/sigma)*(1-(1-sigma)^T).

    With R556 ``geometric_band_ceiling`` the band is now TWO-SIDED:
    an upper ceiling C_0 + b_up/sigma_up and a lower floor
    C_0 + (b_lo/sigma_lo)(1-(1-sigma_lo)^T) — the certified cascade
    closes in both directions (R555 upper, R560/R561 lower).
    Premises (h_emp_): C(t+1) = C(t) + G_t and the lower band per step.
    """
    if b < 0.0:
        raise ValueError("b must be nonnegative")
    if not 0.0 < sigma <= 1.0:
        raise ValueError("sigma must lie in (0, 1]")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    return c0 + b * (1.0 - (1.0 - sigma) ** steps) / sigma


def band_time_to_target(
    c_floor_after_T: float, c_star: float
) -> bool:
    """band_time_to_target (R560): if the guaranteed floor after T
    steps already reaches the target C* (hreach), the target IS
    reached: C* <= C_T. Composition of the floor with the thermostat
    target timer. Pure implication, so the runtime form takes the
    already-computed floor.
    """
    return c_floor_after_T >= c_star


def time_to_target_log(
    c0: float, c_star: float, b: float, sigma: float
) -> int:
    """Closed-form thermostat target timer [R567 ``time_to_target_log``].

    Under the lower band G_t >= b(1-sigma)^t (b >= 0, 0 < sigma < 1,
    C_0 <= C*, sigma*(C*-C_0) < b), the number of steps T satisfying
    T >= ln(1 - sigma(C*-C_0)/b)/ln(1-sigma) certifies C* <= C_T.
    Returns the minimal integer T. Requires STRICT sigma < 1 (the
    theorem degenerates at sigma = 1) and reachability (hc1).
    """
    if b <= 0.0:
        raise ValueError("b must be positive (reachability)")
    if not 0.0 < sigma < 1.0:
        raise ValueError("sigma must lie in (0, 1)")
    if c_star < c0:
        raise ValueError("target C* must be >= C0")
    if sigma * (c_star - c0) >= b:
        raise ValueError("unreachable: sigma*(C*-C0) >= b")
    return math.ceil(
        math.log(1.0 - sigma * (c_star - c0) / b) / math.log(1.0 - sigma)
    )


def risk_budget_termination(
    certs: list[StageCertificate], phi_min: float, eps: float, R: float
) -> float:
    """risk_budget_termination (Unified/TopLevel.lean, R529):
    T <= (Phi_0 - phi_min + R) / eps.

    Stronger and more honest than ``termination_bound``: risk spend
    enters the horizon numerator LINEARLY instead of shrinking the
    per-step margin. Premises (the theorem's): every decrease >= eps,
    the TOTAL risk spend <= R >= 0, phi_min <= after(T-1), and the chain
    is sound+contiguous (checked here, same tolerance policy as
    ``telescope_bound`` — exact by default). Violations raise
    ValueError: the caller mis-declared the ladder, not the law.
    """
    if not certs:
        raise ValueError("empty chain")
    if eps <= 0.0:
        raise ValueError(f"eps ({eps}) must be positive")
    if R < 0.0:
        raise ValueError(f"risk budget R ({R}) must be nonnegative")
    for c in certs:
        if c.decrease < eps:
            raise ValueError(
                f"gen {c.generation} decrease {c.decrease} < eps {eps}"
            )
    if sum(c.risk_spend for c in certs) > R:
        raise ValueError(
            f"total risk spend exceeds budget R = {R}"
        )
    # telescope_bound re-checks soundness and contiguity
    bound = telescope_bound(certs)
    assert bound is not None  # certs non-empty
    last_after = max(certs, key=lambda c: c.generation).after
    if phi_min > last_after:
        raise ValueError(
            f"floor violated: phi_min {phi_min} > last after {last_after}"
        )
    return (certs[0].before - phi_min + R) / eps
