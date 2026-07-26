"""EXIT-chart novelty halt — convergence criterion for iterative refinement.

Rehabilitated from the V6-era ``exit_chart.py`` (deleted in the V8->V25 collapse)
and re-placed correctly. The original idea was sound: stop an iterative decoder
when its extrinsic updates stop carrying new information. Its failure was being
wired into a forward path; V28 uses it ONLY as a low-risk halt/anneal trigger
(never in the main LM path).

Convergence proxy (no GPU sync, differentiable-ish, cheap):
    novelty = ||ext_after|| / (||ext_before|| + eps)   in [0, inf)
    halt  <=>  novelty < tau

When extrinsic magnitude stops shrinking between iterations the representation
has converged and further refinement adds nothing (the EXIT chart flattens).
This is the operational form of the convergence-halt principle applied to
beta-annealing and off-path HEP refinement.
"""

from __future__ import annotations

import torch


class EXITChartHalt:
    """Norm-ratio novelty halt. Stateless; ``__call__``-driven.

    Args:
        threshold: novelty below this is "converged" (halt / freeze beta).
        min_steps: never halt before this many observations.
        window: require ``window`` consecutive sub-threshold observations
            before declaring halt (hysteresis; one noisy step won't fire it).
    """

    def __init__(self, threshold: float = 0.05, min_steps: int = 50, window: int = 5) -> None:
        if threshold <= 0.0:
            raise ValueError("EXITChartHalt.threshold must be positive")
        if min_steps < 1:
            raise ValueError("EXITChartHalt.min_steps must be >= 1")
        if window < 1:
            raise ValueError("EXITChartHalt.window must be >= 1")
        self.threshold = float(threshold)
        self.min_steps = int(min_steps)
        self.window = int(window)
        self._observations: list[float] = []
        self._halted = False

    @staticmethod
    def novelty(ext_before: torch.Tensor, ext_after: torch.Tensor, eps: float = 1e-8) -> float:
        """Ratio of extrinsic magnitudes after/before a refinement step.

        A small value means the iteration added little magnitude -> converged.
        Detaches both operands (this is a diagnostic, not a loss).
        """
        eb = ext_before.detach().float().reshape(-1)
        ea = ext_after.detach().float().reshape(-1)
        return float((ea.norm() / (eb.norm() + eps)).clamp(0.0, 1e6).item())

    def observe(self, novelty: float) -> bool:
        """Record one novelty observation; return True iff halt should fire.

        Halt is sticky: once True, stays True until :meth:`reset`.
        """
        self._observations.append(float(novelty))
        if self._halted:
            return True
        if len(self._observations) < self.min_steps:
            return False
        recent = self._observations[-self.window:]
        self._halted = all(v < self.threshold for v in recent)
        return self._halted

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def n_observations(self) -> int:
        return len(self._observations)

    def reset(self) -> None:
        """Clear all state (fresh anneal / refinement run)."""
        self._observations.clear()
        self._halted = False
