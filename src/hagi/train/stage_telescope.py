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
