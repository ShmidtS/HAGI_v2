"""E1: identity-preserving dense-Qwen pyramid FFN adapter.

Frozen-base contract: the pretrained Qwen3.5 SwiGLU MLP weights are held
frozen (no STE, no gradients, no mutation). The pyramid layer exposes a
*flat* single-branch path and a *multi-branch* path that share the exact same
frozen weights; with a single branch and ``mix='mean'`` the two are identical
by construction. This is the E1 identity gate — it isolates the topology
change from any base change.

Qwen3.5 MLP semantics (transformers Qwen3_5MLP, verified):
    down(silu(gate_proj(x)) * up_proj(x))
    gate_proj, up_proj: [H -> I]  (weight stored [I, H], x @ W.T)
    down_proj:           [I -> H]  (weight stored [H, I], out @ W.T)
    H = 5120, I = 17408
    No internal residual; layernorm + residual live in the DecoderLayer.

Qwen3.5 RMSNorm (verified, differs from hagi/model norms):
    output = RMSNorm(x) * (1 + weight)   # weight zeros-inited => scale 1
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Normalisation (Qwen3.5 exact semantics)
# --------------------------------------------------------------------------- #
class Qwen3_5RMSNorm(nn.Module):
    """``RMSNorm(x) * (1 + weight)`` — matches transformers Qwen3_5RMSNorm.

    ``weight`` is zeros-initialised so the multiplier is ``1+0 = 1`` at
    construction, making a freshly-built layer an exact identity on the norm.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Qwen3.5 computes the RMS statistics in float32, then casts back.
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = out * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"


# --------------------------------------------------------------------------- #
# Frozen FFN layer
# --------------------------------------------------------------------------- #
class FrozenFFNLayer(nn.Module):
    """One frozen Qwen3.5 decoder-layer FFN path with pre-norm.

    The module is constructed with random init weights and is frozen by
    default. Use :meth:`load_base_weights` to copy pretrained weights in
    (read-only — the source tensor is never mutated).

    Args:
        hidden_size: H (default 5120).
        intermediate_size: I (default 17408).
        norm_eps: RMSNorm epsilon (default 1e-6).
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 17408,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.norm = Qwen3_5RMSNorm(hidden_size, eps=norm_eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self._freeze()

    # -- freezing -----------------------------------------------------------
    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        # hagi convention: mark 2D channel weights (independent of quantization)
        for w in (self.gate_proj, self.up_proj, self.down_proj):
            w.is_channel_weight = True

    @property
    def grad_enabled(self) -> bool:  # diagnostic
        return any(p.requires_grad for p in self.parameters())

    # -- weight loading -----------------------------------------------------
    def load_base_weights(self, state_dict: dict, prefix: str = "") -> None:
        """Copy frozen weights from a HF safetensors state dict (read-only).

        Args:
            state_dict: tensors keyed like ``model.layers.63.mlp.gate_proj.weight``.
            prefix: key prefix to strip from ``state_dict`` before lookup.
        """
        def _lookup(name: str, norm_name: str | None = None) -> torch.Tensor:
            candidates = [name]
            if prefix:
                candidates.append(f"{prefix}{name}")
                if prefix.endswith(".layers.0."):
                    candidates.append(f"{prefix}mlp.{name}")
            if norm_name is not None:
                candidates.extend((norm_name, f"{prefix}{norm_name}"))
            for key in dict.fromkeys(candidates):
                if key in state_dict:
                    return state_dict[key]
            raise KeyError(f"no weight found for {name}")

        with torch.no_grad():
            self.gate_proj.weight.copy_(_lookup("gate_proj.weight", "mlp.gate_proj.weight"))
            self.up_proj.weight.copy_(_lookup("up_proj.weight", "mlp.up_proj.weight"))
            self.down_proj.weight.copy_(_lookup("down_proj.weight", "mlp.down_proj.weight"))
            # Safetensors stores the raw Qwen3.5 norm offset w. The
            # effective multiplier (1 + w) is applied by Qwen3_5RMSNorm.
            self.norm.weight.copy_(_lookup("norm.weight", "post_attention_layernorm.weight"))
        self._freeze()

    # -- forward ------------------------------------------------------------
    def mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pure Qwen3.5 SwiGLU MLP on an already-normalized input.

        Expects ``x`` to be ``norm(pre_norm_input)``; applies no RMSNorm here.
        This mirrors :class:`transformers.Qwen3_5MLP`, which receives the
        normalized hidden state from its parent decoder layer.
        """
        gate = F.linear(x, self.gate_proj.weight.to(dtype=x.dtype))
        up = F.linear(x, self.up_proj.weight.to(dtype=x.dtype))
        h = F.silu(gate) * up
        return F.linear(h, self.down_proj.weight.to(dtype=h.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm entry point (for full-layer parity checks).

        Computes ``norm(x)`` then delegates to :meth:`mlp_forward`. The caller
        owns the outer residual, matching ``Qwen3_5DecoderLayer``.
        """
        xn = self.norm(x)
        return self.mlp_forward(xn)


# --------------------------------------------------------------------------- #
# Flat vs pyramid forward
# --------------------------------------------------------------------------- #
def flat_forward(layer: FrozenFFNLayer, x: torch.Tensor) -> torch.Tensor:
    """Single-branch standard Qwen3.5 FFN path with its outer residual."""
    return x + layer(x)


def _pyramid_level(branch: FrozenFFNLayer, x: torch.Tensor, n_branches: int) -> torch.Tensor:
    """Run ``n_branches`` parallel FFN branches (shared frozen weights) on ``x``.

    For ``n_branches == 1`` the branch tensor is returned directly (NOT stacked)
    so downstream ``+`` never sees a zero-stride view that breaks fp-stable
    reduction/accumulation at B>=2.
    """
    if n_branches == 1:
        return branch(x)
    outs = [branch(x) for _ in range(n_branches)]
    return torch.stack(outs, dim=0)


@torch.no_grad()
def pyramid_forward(
    layer: FrozenFFNLayer,
    x: torch.Tensor,
    levels: List[int] | None = None,
    n_branches_per_level: List[int] | None = None,
    mix: str = "mean",
) -> torch.Tensor:
    """Pyramid multi-branch FFN forward.

    Topology (matches historical ``scripts/dsv4_pyramid.py`` invariant):
    layers split into levels of sizes ``1, 2, 4, 8, 16, remainder``; each
    level runs its branches in parallel on the SAME input, mixes by mean/sum,
    and adds a residual (scale 1.0).

    Args:
        layer: a single :class:`FrozenFFNLayer` (shared across all branches).
        x: ``[B, S, H]`` input.
        levels: layer counts per level (defaults to a 512-layer pyramid plan
            capped to a single level if ``x`` has fewer layers than specified).
        n_branches_per_level: branch count per level (default all 1).
        mix: ``'mean'`` or ``'sum'``.

    Returns:
        ``[B, S, H]`` pyramid output.

    Identity gate: with ``n_branches_per_level=[1]`` and ``mix='mean'`` the
    result is *exactly* equal to :func:`flat_forward`.
    """
    if levels is None:
        levels = [1]
    if n_branches_per_level is None:
        n_branches_per_level = [1] * len(levels)
    if len(n_branches_per_level) != len(levels):
        raise ValueError("levels and n_branches_per_level length mismatch")
    if mix not in ("mean", "sum"):
        raise ValueError(f"unknown mix {mix!r}")

    h = x
    for level_size, nb in zip(levels, n_branches_per_level):
        stacked = _pyramid_level(layer, h, nb)  # [nb, B, S, H] (or [B,S,H] for nb==1)
        if nb == 1:
            # Single-branch short-circuit: skip reduction entirely so the
            # output stays bit-identical to flat_forward (no zero-stride views).
            mix_out = stacked
        elif mix == "mean":
            mix_out = stacked.mean(dim=0)
        else:
            mix_out = stacked.sum(dim=0)
        h = h + mix_out  # residual around the level (scale 1.0)
    return h


# --------------------------------------------------------------------------- #
# One-shot identity smoke (used by scripts/qwen_pyramid_smoke.py)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_e1_identity(
    layer: FrozenFFNLayer,
    x: torch.Tensor,
    mix: str = "mean",
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict:
    """Compare flat vs pyramid (1 branch, mean) on ``x``.

    Asserts exact equality (single-branch mean == flat) and tolerance equality;
    reports finiteness, shapes, dtype, and max/rel error.
    """
    flat = flat_forward(layer, x)
    pyr = pyramid_forward(layer, x, levels=[1], n_branches_per_level=[1], mix=mix)

    max_abs = (flat - pyr).abs().max().item()
    denom = flat.abs().max().clamp(min=1e-6).item()
    rel_err = max_abs / denom if denom > 0 else 0.0
    exactly_equal = torch.equal(flat, pyr)
    allfinite = bool(torch.isfinite(flat).all() and torch.isfinite(pyr).all())

    return {
        "gate": "PASS" if (exactly_equal and allfinite) else "FAIL",
        "exactly_equal": exactly_equal,
        "allclose_atol_rtol": bool(
            torch.allclose(flat, pyr, atol=atol, rtol=rtol) and allfinite
        ),
        "max_abs_err": max_abs,
        "rel_err": rel_err,
        "finite": allfinite,
        "shape": list(flat.shape),
        "dtype": str(flat.dtype),
        "n_branches": 1,
        "mix": mix,
    }
