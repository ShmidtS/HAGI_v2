"""Analytic step size from measured curvature.

``Hagi/Unified/RecursiveGrowth.lean`` ``optimal_step_unconstrained``
(roadmap #1.3), in the spirit of the TorchLean Lyapunov controllers
(lean-dojo/TorchLean, ``NN/MLTheory/CROWN/Lyapunov`` — verified
landscape): with the smooth-descent bound

    E2 <= E1 - eta*<g,d> + L*||d||^2*eta^2/2

the guaranteed decrease ``g(eta) = eta*inner - L*dn2*eta^2/2`` is a
concave quadratic whose maximum over ALL eta is ANALYTIC:

    eta* = inner / (L*dn2)          value = inner^2 / (2*L*dn2)

``optimal_step_value`` pins the value; ``optimal_step_ge_recip`` gives
``eta* >= 1/L`` whenever ``inner >= dn2`` (the certified SafeQP regime),
so the safe clip ``min(1/L, eta*)`` is exactly ``1/L`` when the
direction is fully certified (``inner == dn2``).

This replaces ``lr`` as a hyperparameter class: the step is COMPUTED
from the measured gradient alignment and the measured curvature.

**Where L comes from.** ``L`` is the local Lipschitz constant of the
loss along the search direction. We estimate it by the standard
two-point probe on a probe batch: ``|f(theta + d) - f(theta) -
<g,d>| <= (L/2)*||d||^2``, i.e.

    L_hat = 2 * (|f(theta+d) - f(theta) - <g,d>|) / ||d||^2

clamped at >= ``floor`` because the descent certificate of
``safeQP_descent`` only needs an UPPER bound and L=0 would divide by
zero. The probe costs one extra forward per update and is amortized
over ``refresh`` steps, so its amortized cost is ``1/refresh``.

**Why this is safe to deploy.** The step is never larger than the
existing LR when the direction is certified and never larger than
``1/L`` otherwise -- both bounds are the ones the Lean theorems
actually certify. ``clip_to_lr`` keeps the historical behaviour as a
ceiling during the transition.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


def analytic_step(
    inner: float, dn2: float, smoothness: float
) -> float:
    """``eta* = <g,d>/(L*||d||^2)`` (optimal_step_unconstrained).

    Args:
        inner: ``<g, d>`` -- the alignment of the step direction with
            the gradient. Negative means the direction ascends; the
            caller must not step along it.
        dn2: ``||d||^2``, strictly positive.
        smoothness: the Lipschitz constant ``L``, strictly positive.

    Returns:
        The analytic step size maximizing the guaranteed descent.

    Raises:
        ValueError: on non-positive ``dn2`` or ``smoothness``.
    """
    if dn2 <= 0.0 or smoothness <= 0.0:
        raise ValueError("analytic_step needs positive dn2 and smoothness")
    return inner / (smoothness * dn2)


def guaranteed_descent(
    eta: float, inner: float, dn2: float, smoothness: float
) -> float:
    """``g(eta) = eta*inner - L*dn2*eta^2/2`` -- the certified decrease."""
    return eta * inner - 0.5 * smoothness * dn2 * eta * eta


class CurvatureProbe:
    """Measures the directional Lipschitz constant L for the optimizer.

    State: the estimate, refreshed every ``refresh`` steps and reused in
    between (a single probe forward per refresh, not per step). The
    estimate is deliberately conservative -- it keeps the running MAX
    so a single bad probe cannot inflate the step and trip the
    smoothness assumption.
    """

    def __init__(self, refresh: int = 50, floor: float = 1e-6,
                 max_cap: float = 1e6) -> None:
        self.refresh = max(1, int(refresh))
        self.floor = float(floor)
        self.max_cap = float(max_cap)
        self._l: float | None = None
        self._since = 0

    @property
    def smoothness(self) -> float | None:
        """The current L estimate, or None before the first probe."""
        return self._l

    def estimate(
        self,
        loss_fn: "callable",
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
    ) -> float:
        """Probe L along ``-g`` with a two-point quadratic model.

        Evaluates ``f(theta)`` (cached by the caller) and
        ``f(theta - alpha*g)``, then solves the smoothness identity for
        L. Displaces, measures, restores -- the model is left exactly
        as it was found.

        Args:
            loss_fn: zero-arg callable returning a scalar loss at the
                current parameters.
            params: the parameters being updated (tensors in-place).
            grads: their gradients, parallel to ``params``.

        Returns:
            The updated L estimate (kept in ``[floor, max_cap]``).
        """
        self._since += 1
        if self._l is not None and self._since < self.refresh:
            return self._l

        base = float(loss_fn())
        dn2 = sum(float(g.double().pow(2).sum()) for g in grads)
        if dn2 <= 0.0:
            self._since = 0
            return self._l if self._l is not None else self.floor

        inner = -dn2  # displacement is -g, so <g, -g> = -||g||^2
        alpha = self._probe_alpha(dn2)

        with torch.no_grad():
            backup = [p.detach().clone() for p in params]
            try:
                for p, g in zip(params, grads):
                    p.add_(g.reshape(-1)[: p.numel()].view_as(p), alpha=-alpha)
                moved = float(loss_fn())
            finally:
                for p, backup_p in zip(params, backup):
                    p.copy_(backup_p)
                del backup

        residual = abs(moved - base - inner)
        probe_l = 2.0 * residual / dn2
        probe_l = min(max(probe_l, self.floor), self.max_cap)
        # Running MAX: L must stay an upper bound for the descent
        # certificate to hold. Re-probing cannot lower it.
        self._l = probe_l if self._l is None else max(self._l, probe_l)
        self._since = 0
        logger.debug(
            "curvature probe: base=%.6f moved=%.6f ||g||^2=%.6e -> L=%.4e",
            base, moved, dn2, self._l,
        )
        return self._l

    def _probe_alpha(self, dn2: float) -> float:
        """Unit-ish displacement: normalize so the probe step is scale-free."""
        return 1.0 / math.sqrt(max(dn2, self.floor))


def select_step(
    inner: float,
    dn2: float,
    smoothness: float,
    base_lr: float | None = None,
    clip_to_lr: bool = True,
) -> float:
    """Analytic ``eta*`` with the certified ``1/L`` safety clip.

    ``optimal_step_ge_recip``: when ``inner >= dn2`` (certified SafeQP
    direction) ``eta* >= 1/L``, so the clip binds and we take
    ``1/L``. Otherwise ``eta* < 1/L`` already and the clip is inert.

    Args:
        inner: ``<g, d>``.
        dn2: ``||d||^2``.
        smoothness: ``L``.
        base_lr: when given and ``clip_to_lr``, the step never exceeds
            this -- the historical LR acts as a ceiling during the
            transition, so the change cannot make training more
            aggressive than the tuned baseline in one step.
        clip_to_lr: whether to apply the ``base_lr`` ceiling.

    Returns:
        The step size to apply.
    """
    if dn2 <= 0.0 or smoothness <= 0.0:
        raise ValueError("select_step needs positive dn2 and smoothness")
    eta = analytic_step(inner, dn2, smoothness)
    # Certified directions get exactly 1/L (optimal_step_ge_recip).
    if inner >= dn2:
        eta = 1.0 / smoothness
    elif eta > 1.0 / smoothness:
        eta = 1.0 / smoothness
    if clip_to_lr and base_lr is not None and base_lr > 0.0:
        eta = min(eta, base_lr)
    return max(eta, 0.0)
