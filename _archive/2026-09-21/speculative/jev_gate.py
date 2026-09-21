"""Feature-driven adapter update without full generation (JEV gate).

Replaces generation-based self-critique scoring with a cheap feature->adapter
delta mapping, following CCM-LoRA's budget-constrained routing (ACL 2026 #1329)
and the neuro-modulated delta-adapter pattern (ESANN 2026 / NeurIPS 2024).

The gate extracts three features from a single generation result:
  - confidence: exp(-mean_token_entropy), ∈ [0, 1]
  - entropy:    mean per-token entropy in nats (higher = more uncertain)
  - rep:        trigram repetition ratio, ∈ [0, 1]

It then computes a scalar delta for each pyramid adapter ``scale`` parameter:
  delta = lr * feature_weight * (confidence - confidence_center) * (1 - rep)

This is a Hebbian-style update: high-confidence, low-repetition output
pushes ``scale`` up (amplifying the adapter contribution); low-confidence or
repetitive output pushes it down. No gradient backward pass is needed — the
delta is applied directly to the parameter tensor.

The update is fully reversible via the existing `_snapshot_adapters` /
`_restore_adapters` machinery in `self_improve.py`, so a KL or quality
regression rolls it back transactionally.

Reference-first grounding:
  - DeltaNet (Schlag et al. 2021): delta-rule parameter updates are a direct
    mapping from features to parameter changes.
  - NDA / Neuromodulated Delta Adapters (ESANN 2026): rank-r fast-weight
    bottleneck updated by cheap online rules, no full backward pass.
  - Linear Attention TTT (NVIDIA 2025): test-time adapter updates driven by
    online feature statistics, not full-model retraining.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from hagi.model.adapters import BlockAdapter


@dataclass
class FeatureVector:
    """Compact feature descriptor of one generation result."""

    confidence: float
    entropy: float
    rep: float
    n_tokens: int

    def is_valid(self) -> bool:
        """True when all features are finite and in their nominal ranges."""
        return (
            math.isfinite(self.confidence)
            and math.isfinite(self.entropy)
            and math.isfinite(self.rep)
            and 0.0 <= self.confidence <= 1.0
            and self.entropy >= 0.0
            and 0.0 <= self.rep <= 1.0
            and self.n_tokens >= 0
        )


def extract_features(result: dict[str, Any]) -> FeatureVector:
    """Build a FeatureVector from a daemon generation result dict.

    Accepts either a raw result dict (from gen_one) or a dict that already
    contains ``entropy``/``rep``/``confidence``/``ntok`` keys.
    """
    conf = float(result.get("confidence", 0.0))
    ent = float(result.get("entropy", 0.0))
    rep = float(result.get("rep", 1.0))
    ntok = int(result.get("ntok", 0))
    return FeatureVector(confidence=conf, entropy=ent, rep=rep, n_tokens=ntok)


def feature_weight(fv: FeatureVector) -> float:
    """Routing weight in [0, 1]: how much the gate should move adapter scales.

    Low-confidence, high-entropy, or high-repetition output yields a weight
    near 1 (strong correction needed). High-confidence, low-entropy,
    non-repetitive output yields a weight near 0 (leave scales alone).
    """
    if not fv.is_valid():
        return 0.0
    confidence_term = 1.0 - fv.confidence
    entropy_term = min(1.0, fv.entropy / 4.0)  # 4 nats ≈ fully uncertain per token
    rep_term = fv.rep
    return max(0.0, min(1.0, 0.4 * confidence_term + 0.4 * entropy_term + 0.2 * rep_term))


def scale_deltas(
    model: torch.nn.Module,
    result: dict[str, Any],
    *,
    lr: float = 0.01,
    confidence_center: float = 0.5,
    max_abs_delta: float = 0.1,
) -> dict[int, float]:
    """Compute per-adapter scale deltas from features.

    Returns a dict mapping ``id(param)`` -> scalar delta. Only trainable
    ``PyramidAdapter.scale`` parameters are touched. ``TttLoraAdapter.lora_B``
    is excluded (it uses the RLS contour, not feature-gating).

    The delta is zero when features are invalid, keeping a no-op path.
    """
    fv = extract_features(result)
    if not fv.is_valid():
        return {}
    w = feature_weight(fv)
    if w == 0.0:
        return {}
    # Hebbian sign: push scale toward amplifying good outputs.
    delta = lr * w * (fv.confidence - confidence_center) * (1.0 - fv.rep)
    delta = max(-max_abs_delta, min(max_abs_delta, delta))

    deltas: dict[int, float] = {}
    for module in model.modules():
        if isinstance(module, BlockAdapter):
            pyramid = module.pyramid
            if pyramid is not None:
                scale = pyramid.scale
                if scale.requires_grad:
                    deltas[id(scale)] = delta
    return deltas


def apply_scale_deltas(
    model: torch.nn.Module,
    deltas: dict[int, float],
) -> int:
    """Apply feature-computed deltas to adapter scale parameters in-place.

    Returns the number of parameters updated. Caller is responsible for
    snapshotting/restoring via `_snapshot_adapters` / `_restore_adapters`
    or the HAGI transactional envelope.
    """
    applied = 0
    for module in model.modules():
        if isinstance(module, BlockAdapter):
            pyramid = module.pyramid
            if pyramid is None:
                continue
            scale = pyramid.scale
            delta = deltas.get(id(scale))
            if delta is None or not scale.requires_grad:
                continue
            with torch.no_grad():
                scale.clamp_(min=-10.0, max=10.0)
                scale.add_(float(delta))
            applied += 1
    return applied


def feature_adapt_step(
    model: torch.nn.Module,
    result: dict[str, Any],
    *,
    lr: float = 0.01,
    confidence_center: float = 0.5,
    max_abs_delta: float = 0.1,
) -> dict[str, Any]:
    """Full feature->adapter-delta step with summary.

    Extracts features, computes deltas, applies them to all trainable
    pyramid scales. Returns a summary dict for logging/verification.

    Safety:
      - Only modifies `PyramidAdapter.scale` (1 param per block).
      - `lr` and `max_abs_delta` bound the step.
      - Zero delta on invalid features.
    """
    fv = extract_features(result)
    deltas = scale_deltas(
        model,
        result,
        lr=lr,
        confidence_center=confidence_center,
        max_abs_delta=max_abs_delta,
    )
    n_updated = apply_scale_deltas(model, deltas)
    return {
        "features": fv.__dict__ if fv.is_valid() else None,
        "weight": feature_weight(fv) if fv.is_valid() else 0.0,
        "deltas": deltas if fv.is_valid() else {},
        "n_updated": n_updated,
        "n_adapters": sum(
            1
            for m in model.modules()
            if isinstance(m, BlockAdapter) and m.pyramid is not None
        ),
    }


__all__ = [
    "FeatureVector",
    "extract_features",
    "feature_weight",
    "scale_deltas",
    "apply_scale_deltas",
    "feature_adapt_step",
]
