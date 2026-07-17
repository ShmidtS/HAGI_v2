"""Lorentz hyperboloid model for hyperbolic latent space.

Maps Euclidean hidden states to the Lorentz hyperboloid H^n = {x : <x,x>_L = -1, x_0 > 0},
where <x,y>_L = -x_0*y_0 + sum(x_i*y_i). The hyperboloid has exponential volume growth,
making it ideal for hierarchical/tree-like semantic structure of natural language.

Operations:
  exp_0(v): tangent vector at origin -> point on hyperboloid
  log_0(x): point on hyperboloid -> tangent vector at origin
  proj(x):  project arbitrary point to hyperboloid (renormalize x_0)
  dist(x,y): Lorentzian geodesic distance

Analogy with communication theory: the hyperboloid is the manifold of constant
negative curvature, analogous to a dispersive channel where high-frequency
components carry exponential information content. Sphere-packing on H^n is
the optimal code design for hierarchical distributions (Shannon-Hartley on
negatively-curved manifolds achieves higher capacity for tree sources).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _lorentz_dot(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Lorentzian inner product <x,y>_L = -x_0*y_0 + sum_{i>0} x_i*y_i.

    Args:
        x, y: [..., D+1] tensors on the ambient Minkowski space.
        keepdim: if True, keep the last dim as size 1.

    Returns:
        [...] scalar tensor per batch element.
    """
    # Flip the sign of the time component (first column), then sum all
    prod = x * y
    prod_t = -prod[..., :1]
    prod_s_sum = prod[..., 1:].sum(dim=-1, keepdim=True)
    result = prod_t + prod_s_sum
    if not keepdim:
        result = result.squeeze(-1)
    return result


def lorentz_exp_origin(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map at origin (north pole) o = (1, 0, ..., 0).

    exp_o(v) = (cosh(||v_s||), sinh(||v_s||) * v_s / ||v_s||)

    where v = (v_0, v_s) and v_0 must be 0 for tangent vectors at origin.

    Args:
        v: [..., D] Euclidean hidden state (tangent vector at origin).
            The time component is synthesized; v is treated as the spatial part.

    Returns:
        [..., D+1] tensor on the hyperboloid (Lorentz model).
    """
    v_s = v
    v_norm = v_s.norm(dim=-1, keepdim=True).clamp_min(eps)
    # Expand ambient dim by 1: prepend the time coordinate
    x_0 = torch.cosh(v_norm)
    x_s = torch.sinh(v_norm) * (v_s / v_norm)
    return torch.cat([x_0, x_s], dim=-1)


def lorentz_log_origin(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Logarithmic map at origin o = (1, 0, ..., 0).

    log_o(x) = arcosh(-<x,o>_L) * (x_s / ||x_s||)

    where x = (x_0, x_s). For origin o=(1,0,...,0), <x,o>_L = -x_0,
    so arcosh(-<x,o>_L) = arcosh(x_0) = arcosh(-<x,x>_L^{1/2}).

    Args:
        x: [..., D+1] tensor on the hyperboloid.

    Returns:
        [..., D] Euclidean tangent vector (spatial part only).
    """
    x_0 = x[..., :1]
    x_s = x[..., 1:]
    x_s_norm = x_s.norm(dim=-1, keepdim=True).clamp_min(eps)
    # arcosh(x_0) = log(x_0 + sqrt(x_0^2 - 1))
    # Stable: use clamp_min on the arg to avoid NaN
    inner = (x_0 + torch.sqrt(torch.clamp(x_0.pow(2) - 1.0, min=eps))).clamp_min(eps)
    dist = torch.log(inner)
    return dist * (x_s / x_s_norm)


def lorentz_proj(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project arbitrary ambient point onto the hyperboloid.

    Given x = (x_0, x_s), compute the unique point on H^n with the same
    spatial direction: x_0 := sqrt(1 + ||x_s||^2).

    Args:
        x: [..., D+1] ambient tensor (x_0 ignored, x_s used).

    Returns:
        [..., D+1] tensor on the hyperboloid.
    """
    x_s = x[..., 1:]
    x_s_norm_sq = x_s.pow(2).sum(dim=-1, keepdim=True)
    x_0_new = torch.sqrt(1.0 + x_s_norm_sq)
    return torch.cat([x_0_new, x_s], dim=-1)


def lorentz_dist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Geodesic distance on the Lorentz hyperboloid.

    d(x,y) = arcosh(-<x,y>_L)

    Args:
        x, y: [..., D+1] tensors on the hyperboloid.

    Returns:
        [...] scalar distance per batch element (last dim squeezed).
    """
    neg_dot = -_lorentz_dot(x, y, keepdim=True)
    # arcosh(z) = log(z + sqrt(z^2 - 1)), stable via clamp
    inner = (neg_dot + torch.sqrt(torch.clamp(neg_dot.pow(2) - 1.0, min=eps))).clamp_min(eps)
    return torch.log(inner).squeeze(-1)


class LorentzSphereNorm(nn.Module):
    """Map hidden states to the Lorentz hyperboloid (Minkowski sphere).

    This is a normalization layer analogous to RMSNorm, but instead of
    projecting to the Euclidean unit sphere, it projects to the Lorentz
    hyperboloid H^n: <x,x>_L = -1, x_0 > 0.

    Two modes:
      - 'exp' (default): apply exp_o(h * scale) for principled tangent mapping
      - 'proj': project (1, h) onto H^n via lorentz_proj (cheaper)

    The output has dim+1 ambient coordinates. Downstream layers should use
    log_o to return to Euclidean space before linear projections.

    Args:
        dim: Euclidean hidden dimension D. Output has dim+1.
        mode: 'exp' or 'proj'.
        eps: numerical stability.
    """

    def __init__(self, dim: int, mode: str = "exp", eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.eps = eps
        # Learnable scale on the tangent vector (like RMSNorm weight)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project h from R^D to the hyperboloid H^D ⊂ R^(D+1).

        Args:
            h: [..., D] Euclidean hidden state.

        Returns:
            [..., D+1] tensor on the hyperboloid.
        """
        scaled = h * self.scale
        if self.mode == "proj":
            # Prepend a time coordinate (any positive value; will be recomputed)
            ambient = torch.cat([torch.ones_like(scaled[..., :1]), scaled], dim=-1)
            return lorentz_proj(ambient, eps=self.eps)
        # 'exp' mode: scaled h is a tangent vector at origin
        return lorentz_exp_origin(scaled, eps=self.eps)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map hyperboloid point back to Euclidean tangent space.

        Args:
            x: [..., D+1] tensor on the hyperboloid.

        Returns:
            [..., D] Euclidean tangent vector (spatial part of log_o(x)).
        """
        return lorentz_log_origin(x, eps=self.eps) / (self.scale + self.eps)
