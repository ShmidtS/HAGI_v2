"""Regression tests for the gen-34 rank-growth seam (nested lora basis).

Failure being guarded against (measured 2026-10-10, gen-34): the legacy
``_qr_orthonormal(hidden, r)`` basis depended on ``r`` itself, so the
r=32 and r=64 bases were NOT nested; zero-padding ``lora_B`` across
ranks silently loaded the contour into ANOTHER basis and the incumbent
evaluated 5.09 instead of 4.91 (a fictitious +0.179 ladder delta).

The fix has two halves, both tested here:
1. ``_nested_lora_basis`` — one full basis per seed, sliced per rank;
   r=32 is bit-compatible with the legacy draw, A(64)[:, :32] == A(32).
2. ``train/checkpoint.py`` pads ``lora_B`` with the r_new/r_old scaling
   compensation so the adapter's delta is EXACTLY equal after the load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.model.adapters import (  # noqa: E402
    _nested_lora_basis,
    _qr_orthonormal,
)


def test_rank32_is_bit_compatible_with_legacy_basis() -> None:
    g_new = torch.Generator().manual_seed(15101)
    nested = _nested_lora_basis(128, 32, g_new)
    g_old = torch.Generator().manual_seed(15101)
    legacy = _qr_orthonormal(128, 32, g_old)
    assert torch.allclose(nested, legacy, atol=1e-6)


def test_bases_are_nested_across_ranks() -> None:
    a32 = _nested_lora_basis(128, 32, torch.Generator().manual_seed(15101))
    a64 = _nested_lora_basis(128, 64, torch.Generator().manual_seed(15101))
    assert torch.allclose(a64[:, :32], a32, atol=1e-5)
    # The extension block itself stays orthonormal.
    gram = a64.T @ a64
    assert torch.allclose(gram, torch.eye(64), atol=1e-4)


def test_legacy_bases_were_not_nested() -> None:
    """Documents WHY the fix exists: the legacy draw was rank-dependent.

    If this ever fails (torch changes randn semantics), the gen-34
    incident rationale no longer applies — but the nested basis stays
    correct regardless.
    """
    a32 = _qr_orthonormal(128, 32, torch.Generator().manual_seed(15101))
    a64 = _qr_orthonormal(128, 64, torch.Generator().manual_seed(15101))
    assert not torch.allclose(a64[:, :32], a32, atol=1e-3)


def test_rank_growth_load_is_function_preserving() -> None:
    """End-to-end: a padded lora_B (with scaling compensation) yields
    the SAME adapter delta as the original — the property whose absence
    produced the fictitious gen-34 ladder delta."""
    # Contour in the r=32 basis.
    B32 = torch.randn(128, 32)
    r_new = 64
    alpha = 16.0
    s_old, s_new = alpha / 32, alpha / r_new
    a32 = _nested_lora_basis(128, 32, torch.Generator().manual_seed(15101))
    a64 = _nested_lora_basis(128, r_new, torch.Generator().manual_seed(15101))
    x = torch.randn(4, 128)
    delta_old = s_old * ((x @ a32) @ B32.T)

    # checkpoint.py padding rule (mirrored here):
    ratio = r_new / 32
    B64 = torch.zeros(128, r_new)
    B64[:, :32] = B32 * ratio
    delta_new = s_new * ((x @ a64) @ B64.T)
    assert torch.allclose(delta_old, delta_new, atol=1e-4)
