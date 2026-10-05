"""Trust-region ball projection (MacroCycleV2.lean R175, kernel 2).

``Hagi/Unified/MacroCycleV2.lean``:

* ``trust_region_proj_norm`` -- for a step ``x`` overshooting the
  safety radius (``||x|| >= R > 0``), the rescaled step
  ``(R/||x||) * x`` has norm EXACTLY ``R``: it lands on the ball
  boundary.
* ``trust_region_nearest`` -- that rescaled point is the NEAREST
  point of the safety ball to ``x`` (the metric projection, reverse
  triangle inequality). No QP solver, no inner products.

This LEGALIZES what the loop already does: ``clip_grad_norm_`` is
exactly this rescaling, so replacing the exact SafeQP solve by the
clip on the non-critical weights loses nothing relative to the
ball-constraint semantics. The functions below are the tested,
cited form of that fact -- call sites use them where the projection
semantics (not just norm control) matter.
"""
from __future__ import annotations

import torch


def project_to_ball(x: torch.Tensor, radius: float) -> tuple[torch.Tensor, float]:
    """Metric projection of ``x`` onto the ball ``{y : ||y|| <= radius}``.

    In-ball steps pass through unchanged; overshooting steps are
    rescaled to the boundary. Returns ``(projected, pre_projection_norm)``
    so callers can log how much of the step the trust region ate.

    Raises:
        ValueError: on a non-positive radius.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    norm = float(torch.linalg.vector_norm(x.detach().float()))
    if norm <= radius:
        return x, norm
    return (radius / norm) * x, norm


def projection_is_nearest(x: torch.Tensor, y: torch.Tensor, radius: float) -> bool:
    """``trust_region_nearest`` as a runtime check.

    For any safe ``y`` (``||y|| <= radius``): ``||x - proj(x)|| <=
    ||x - y||``. Used in tests to pin the property the legalization
    claim rests on.
    """
    proj, _ = project_to_ball(x, radius)
    return float(torch.linalg.vector_norm(x - proj)) <= float(
        torch.linalg.vector_norm(x - y)
    )
