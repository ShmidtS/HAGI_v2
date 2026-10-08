"""Fixed-width latent merge: factorize -> latent-align -> root/contrast -> spectral compress.

Theory SSOT (docstrings are the spec):

* ``C:/primes/Hagi/Architecture/LatentAlign.lean`` (R242) — two experts may
  carry FUNCTIONALLY EQUIVALENT low-rank factorizations whose latent
  coordinates differ by an orthogonal transform (rotation / sign flip).
  Averaging raw latents then annihilates columns even though the expert
  MATRICES agree. The pipeline therefore aligns latents BEFORE averaging:
  ``rotated_factors_same_matrix`` (alignment is free — (U, V) -> (U Q, V Q)
  preserves U Vᵀ exactly), ``sign_flip_functional_equiv`` and
  ``naive_latent_merge_kills_flipped`` (the cancellation being avoided).

* ``C:/primes/Hagi/Architecture/ResidualSplit.lean`` (R243) — expert
  disagreement (R_expert) and compression error (R_quant) are DIFFERENT
  residuals. The root/contrast split keeps the disagreement as contrast
  fibers (PRESERVED, never compressed) while only the root is spectrally
  compressed (R_quant — the compressible residual). ``residual_gate`` /
  ``expert_residual_not_compressible`` are the decision rules.

* ``C:/primes/Hagi/Budget/FactorizedBPW.lean`` (R244) — the bits-per-weight
  arithmetic of the factorized ternary branch:
  ``b1 = (log2(3) * 2 d r + 16 * (2 d + r)) / d^2`` with the anchor
  d = 4096, r = 384 -> ~0.305 BPW, and ``bpw_lt_one_bound`` — growth by
  latent rank keeps the layer sub-1-BPW.

* ``C:/primes/ALGORITHMS.md`` §20 — ``fiberCross`` / ``fiberPythagoras``:
  align BEFORE merge so the energies add with no cross terms; the fiber of
  rank r costs r·d, not d² — growth by effective latent dimension, not
  width. The merged model is the SHARED WIDTH — no H -> 3H tripling
  anywhere (the old block-diagonal merge is declared wrong and replaced).

This module is pure torch, device-agnostic, and has no training side
effects: it consumes state dicts of weights and returns factors plus a
same-width root state dict. F3 mixing of the merged fixed-width body is
OUT OF SCOPE here (see :func:`apply_f3`); checkpoint I/O integration is a
later task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from hagi.train.spectral import rank_from_spectrum

LOG2_3 = math.log2(3.0)


def _spectral_rank(values: torch.Tensor, energy: float) -> int:
    """Smallest rank retaining an ``energy`` fraction of the spectrum.

    Reuses :func:`hagi.train.spectral.rank_from_spectrum` (the R100
    measurement helper): tolerance is the discarded tail
    ``(1 - energy) * ||W||_F^2``.
    """
    if not 0.0 < energy <= 1.0:
        raise ValueError(f"energy must lie in (0, 1], got {energy}")
    v = values.double().flatten()
    total = float(v.pow(2).sum())
    if total <= 0.0:
        # Degenerate (zero) delta: rank 1 of zero factors keeps shapes sane.
        return 1
    return rank_from_spectrum(v, total, (1.0 - energy) * total)


def factorize_delta(
    W: torch.Tensor, W_shared: torch.Tensor, energy: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin SVD of the expert disagreement ``ΔW = W - W_shared``.

    Returns ``(U, V, svals)`` with singular values folded into ``U``, so
    ``U @ V.T`` is the rank-``r`` approximation of ``ΔW`` and ``r`` is the
    smallest rank capturing an ``energy`` fraction of the spectral energy
    (R100 ``rank_from_spectrum`` on the measured spectrum).

    Args:
        W: expert weight ``[m, n]``.
        W_shared: shared (root) weight, same shape.
        energy: retained-energy fraction in ``(0, 1]``.

    Returns:
        ``(U [m, r], V [n, r], svals [r])`` in ``W``'s dtype; the SVD itself
        runs in float64 for numerical stability of the energy accounting.
    """
    if W.shape != W_shared.shape:
        raise ValueError(f"shape mismatch: {tuple(W.shape)} vs {tuple(W_shared.shape)}")
    if W.ndim != 2:
        raise ValueError("factorize_delta needs 2D weights")
    delta = (W.double() - W_shared.double())
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    r = _spectral_rank(S, energy)
    Ur = U[:, :r] * S[:r]
    Vr = Vh[:r, :].T.contiguous()
    return Ur.to(dtype=W.dtype), Vr.to(dtype=W.dtype), S[:r].to(dtype=W.dtype)


def _pad_columns(x: torch.Tensor, width: int) -> torch.Tensor:
    """Zero-pad the latent axis of ``x`` ``[d, r]`` up to ``width``."""
    if x.shape[1] >= width:
        return x
    pad = x.new_zeros(x.shape[0], width - x.shape[1])
    return torch.cat([x, pad], dim=1)


def latent_align(
    factors: list[tuple[torch.Tensor, torch.Tensor]], ref: int = 0
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Align every expert's latent coordinates to the ``ref`` expert's.

    For each ``i != ref``: orthogonal Procrustes
    ``Q_i = argmin_Q ||U_i Q - U_ref||_F`` over orthogonal ``Q`` — via the
    SVD ``U_refᵀ U_i = A S Bᵀ -> Q_i = A Bᵀ`` — applied to BOTH factors,
    ``(U_i, V_i) -> (U_i Q_i, V_i Q_i)``. By LatentAlign's
    ``rotated_factors_same_matrix`` this preserves ``U_i V_iᵀ`` EXACTLY:
    alignment is free in function, it only rotates the latent basis so that
    averaging latents no longer cancels ``sign_flip_functional_equiv``
    misalignments (``naive_latent_merge_kills_flipped`` is the failure this
    prevents).

    Rank mismatch is handled by zero-padding every factor to the shared
    maximum rank first (``fiberCross``: align BEFORE merge); the Procrustes
    SVD of the padded cross-Gram automatically aligns in the shared
    min-rank subspace while the padded zero columns carry no energy.

    Args:
        factors: list of ``(U_i [d, r_i], V_i [d, r_i])`` per expert,
            identical ``d``; ``r_i`` may differ.
        ref: index of the reference expert (returned unchanged, padded).

    Returns:
        Aligned factors, all padded to the common latent rank
        ``max_i r_i``.
    """
    if not factors:
        raise ValueError("latent_align needs at least one expert")
    if not 0 <= ref < len(factors):
        raise ValueError(f"ref index {ref} out of range for {len(factors)} experts")
    width = max(_factor_U(f).shape[1] for f in factors)
    padded = [
        (_pad_columns(_factor_U(f), width).double(), _pad_columns(_factor_V(f), width).double())
        for f in factors
    ]
    U_ref, _ = padded[ref]
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, (U, V) in enumerate(padded):
        if i != ref:
            M = U_ref.T @ U
            A, _, Bh = torch.linalg.svd(M)
            Q = A @ Bh
            U = U @ Q
            V = V @ Q
        out.append((U, V))
    return out


def root_contrast_merge(
    factors_aligned: list[tuple[torch.Tensor, torch.Tensor]],
    W_shared: torch.Tensor,
    keep_contrast: bool = True,
) -> dict:
    """Split aligned latents into a shared ROOT and per-expert CONTRAST fibers.

    The root is the mean of the ALIGNED latents; the root matrix is
    ``W_root = W_shared + mean_i(U'_i V'_iᵀ)``. The per-expert contrast
    fibers ``C_i = (U'_i - mean U', V'_i)`` are the R_expert residual of
    ResidualSplit: the expert DISAGREEMENT that must be PRESERVED, never
    compressed — compression gains on R_quant do not recover it
    (``expert_residual_not_compressible``). Only the root is compressed
    downstream (:func:`spectral_compress`), and a quantization-residual
    branch is justified iff ``η_r·E_r > C_r + P_quant``
    (``residual_gate``) — a later economic decision, not made here.

    Args:
        factors_aligned: output of :func:`latent_align` (common rank).
        W_shared: shared weight the experts were factorized against.
        keep_contrast: when False the contrast fibers are dropped (root-only
            merge); the reconstruction guarantee then no longer holds.

    Returns:
        Dict with ``root_matrix``, ``root_factors`` ``(mean U', mean V')``,
        ``contrast_fibers`` (list of ``{"dU", "dV", "V"}`` per expert),
        and ``contrast_matrices`` — ``U'_i V'_iᵀ - mean_i(U'_i V'_iᵀ)``
        per expert, so that
        ``root_matrix + contrast_matrices[i] == W_shared + U'_i V'_iᵀ``
        EXACTLY (ResidualSplit bookkeeping: the expert is root + fiber).
    """
    if not factors_aligned:
        raise ValueError("root_contrast_merge needs at least one expert")
    Ws = W_shared.double()
    Us = torch.stack([U.double() for U, _ in factors_aligned], dim=0)
    Vs = torch.stack([V.double() for _, V in factors_aligned], dim=0)
    mean_U = Us.mean(dim=0)
    mean_V = Vs.mean(dim=0)
    prods = torch.stack([Us[i] @ Vs[i].T for i in range(Us.shape[0])], dim=0)
    mean_prod = prods.mean(dim=0)
    root_matrix = Ws + mean_prod
    contrast_fibers: list[dict] = []
    contrast_matrices: list[torch.Tensor] = []
    for i in range(Us.shape[0]):
        dU = Us[i] - mean_U if keep_contrast else None
        dV = Vs[i] - mean_V if keep_contrast else None
        contrast_fibers.append(
            {
                "dU": None if dU is None else dU.to(W_shared.dtype),
                "dV": None if dV is None else dV.to(W_shared.dtype),
                "V": Vs[i].to(W_shared.dtype),
            }
        )
        contrast_matrices.append((prods[i] - mean_prod).to(W_shared.dtype))
    return {
        "root_matrix": root_matrix.to(W_shared.dtype),
        "root_factors": (
            mean_U.to(W_shared.dtype),
            mean_V.to(W_shared.dtype),
        ),
        "contrast_fibers": contrast_fibers,
        "contrast_matrices": contrast_matrices,
    }


def reconstruct_expert(merged: dict, i: int) -> torch.Tensor:
    """Exact per-expert reconstruction: root matrix + contrast fiber ``i``.

    ``W_i = W_shared + U'_i V'_iᵀ = root_matrix + contrast_matrices[i]``
    — the ResidualSplit identity: every expert is the shared root plus its
    (preserved) disagreement fiber, within the factorization error of the
    original :func:`factorize_delta` truncation.
    """
    n = len(merged["contrast_matrices"])
    if not 0 <= i < n:
        raise ValueError(f"expert index {i} out of range for {n} experts")
    return merged["root_matrix"] + merged["contrast_matrices"][i]


def spectral_compress(
    root_factors: tuple[torch.Tensor, torch.Tensor], energy: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate the root rank by spectral energy of the merged factors.

    This is the R_quant compression of ResidualSplit — the COMPRESSIBLE
    residual (the root), never the contrast fibers. Re-truncates the root
    product ``U Vᵀ`` to the smallest rank retaining an ``energy`` fraction,
    using the same R100 spectrum measurement as :func:`factorize_delta`.

    Returns:
        New ``(U_r, V_r)`` with singular values folded into ``U_r``.
    """
    U, V = root_factors
    M = U.double() @ V.double().T
    Qu, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = _spectral_rank(S, energy)
    Ur = (Qu[:, :r] * S[:r]).to(dtype=U.dtype)
    Vr = Vh[:r, :].T.contiguous().to(dtype=V.dtype)
    return Ur, Vr


def factorized_bpw(d_out: int, d_in: int, r: int) -> float:
    """Bits per original weight of ONE factorized ternary branch.

    FactorizedBPW (R244), generalized to a rectangular ``d_out x d_in``
    layer (the Lean ``bpw`` is the square case):

        b1 = (log2(3) * (d_out + d_in) * r + 16 * (d_out + d_in + r))
             / (d_out * d_in)

    Assumptions (documented, matching the Lean scheme): ternary factors at
    log2(3) bits per entry — ``d_out * r + d_in * r`` entries across U and
    V — plus FP16 scale vectors (h : d_out, g : d_in, ℓ : r) at 16 bits
    each. Anchor (NOT a theorem — arithmetic of the scheme): d = 4096,
    r = 384 gives b1 ≈ 0.305 BPW.
    """
    if d_out <= 0 or d_in <= 0 or r <= 0:
        raise ValueError("dimensions and rank must be positive")
    bits = LOG2_3 * (d_out + d_in) * r + 16.0 * (d_out + d_in + r)
    return bits / (d_out * d_in)


def bpw_lt_one_rank_bound(d: int) -> float:
    """``bpw_lt_one_bound`` (FactorizedBPW): the sub-1-BPW rank threshold.

    For a square d x d layer, ``b1 < 1`` iff
    ``r < (d^2 - 32 d) / (2 log2(3) d + 16)`` — the formal counterpart of
    "grow effective latent dimension, not width": rank linear in d keeps
    the layer below 1 BPW.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    return (d * d - 32.0 * d) / (2.0 * LOG2_3 * d + 16.0)


@dataclass
class MergedLatent:
    """Result of a fixed-width latent merge.

    The merged model is the SHARED WIDTH: ``root_state_dict`` has exactly
    the shapes of ``shared_state_dict`` — no H -> 3H tripling anywhere.
    The expert disagreement lives in ``contrast_fibers`` (R_expert,
    preserved), the root's own rank was chosen by spectral energy
    (R_quant, compressible).

    Attributes:
        root_state_dict: shared-shaped dict with 2D float weights replaced
            by their merged root matrices.
        root_factors: per-weight ``(U_root, V_root)`` of the compressed
            root (the factorized-ternary branch payload).
        contrast_fibers: per weight, a list (one entry per expert) of
            ``{"dU", "dV", "V"}`` fiber tensors.
        contrast_matrices: per weight, exact per-expert residuals so that
            ``root_matrix + contrast_matrices[i]`` reconstructs expert i.
        ranks: per weight, the compressed root rank.
        n_experts: number of merged experts.
    """

    root_state_dict: dict[str, torch.Tensor]
    root_factors: dict[str, tuple[torch.Tensor, torch.Tensor]]
    contrast_fibers: dict[str, list[dict]]
    contrast_matrices: dict[str, list[torch.Tensor]]
    ranks: dict[str, int]
    n_experts: int
    meta: dict = field(default_factory=dict)

    def bpw_report(self) -> dict:
        """FactorizedBPW accounting of the merged factorized branch.

        Per layer: ``b1 = (log2 3 · (d_out + d_in) · r + 16 · (d_out +
        d_in + r)) / (d_out · d_in)`` (ternary factors + FP16 scales;
        square layers reduce exactly to the Lean ``bpw``). Totals sum the
        bits and the original weights. Square layers also carry the
        ``bpw_lt_one_bound`` affordable rank threshold.
        """
        layers: dict[str, dict] = {}
        total_bits = 0.0
        total_weights = 0
        for key, (U, V) in self.root_factors.items():
            d_out, d_in = int(U.shape[0]), int(V.shape[0])
            r = int(U.shape[1])
            b1 = factorized_bpw(d_out, d_in, r)
            bits = b1 * d_out * d_in
            total_bits += bits
            total_weights += d_out * d_in
            entry = {
                "d_out": d_out,
                "d_in": d_in,
                "rank": r,
                "bpw": b1,
                "bits": bits,
            }
            if d_out == d_in:
                entry["bpw_lt_one_rank_bound"] = bpw_lt_one_rank_bound(d_out)
            layers[key] = entry
        return {
            "layers": layers,
            "total_bits": total_bits,
            "total_weights": total_weights,
            "total_bpw": (total_bits / total_weights) if total_weights else 0.0,
        }


def merge_fixed_width(
    expert_state_dicts: list[dict[str, torch.Tensor]],
    shared_state_dict: dict[str, torch.Tensor],
    energy: float = 0.9,
    root_energy: float = 0.95,
) -> MergedLatent:
    """Full fixed-width merge pipeline over flat state dicts.

    Per 2D float weight present in the shared dict and every expert:
    factorize each ``ΔW_i = W_i - W_shared`` (:func:`factorize_delta`,
    ``energy``), align latents (:func:`latent_align`), split root/contrast
    (:func:`root_contrast_merge`), and spectrally compress the root
    (:func:`spectral_compress`, ``root_energy``). Non-2D or non-float
    entries are copied from the shared dict verbatim (norms, biases);
    expert-only keys are ignored. The output root dict has EXACTLY the
    shared shapes — the headline property: the merged model is the shared
    width, NO 3x.

    Args:
        expert_state_dicts: list of expert state dicts (same keys/shapes
            as the shared dict for the merged weights).
        shared_state_dict: the shared (root-width) state dict.
        energy: factorization retained-energy fraction per expert delta.
        root_energy: root compression retained-energy fraction.

    Returns:
        :class:`MergedLatent`.
    """
    if not expert_state_dicts:
        raise ValueError("merge_fixed_width needs at least one expert")
    root_state: dict[str, torch.Tensor] = {}
    root_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    contrast_fibers: dict[str, list[dict]] = {}
    contrast_matrices: dict[str, list[torch.Tensor]] = {}
    ranks: dict[str, int] = {}
    for key, shared in shared_state_dict.items():
        mergeable = (
            shared.ndim == 2
            and shared.is_floating_point()
            and all(
                key in sd
                and sd[key].shape == shared.shape
                and sd[key].is_floating_point()
                for sd in expert_state_dicts
            )
        )
        if not mergeable:
            root_state[key] = shared
            continue
        factors = [
            factorize_delta(sd[key], shared, energy=energy)
            for sd in expert_state_dicts
        ]
        aligned = latent_align([(U, V) for U, V, _ in factors])
        merged = root_contrast_merge(aligned, shared)
        U_r, V_r = spectral_compress(merged["root_factors"], energy=root_energy)
        root_state[key] = merged["root_matrix"]
        root_factors[key] = (U_r, V_r)
        contrast_fibers[key] = merged["contrast_fibers"]
        contrast_matrices[key] = merged["contrast_matrices"]
        ranks[key] = int(U_r.shape[1])
    return MergedLatent(
        root_state_dict=root_state,
        root_factors=root_factors,
        contrast_fibers=contrast_fibers,
        contrast_matrices=contrast_matrices,
        ranks=ranks,
        n_experts=len(expert_state_dicts),
        meta={"energy": energy, "root_energy": root_energy},
    )


def apply_f3(model, **kwargs):  # pragma: no cover - later task
    """TODO: F3 mixing of the merged fixed-width body is a LATER task.

    The merged fixed-width body produced by :func:`merge_fixed_width` is
    consumable by the existing F3/mixer machinery (see
    ``hagi.model.merge`` for the transforms); wire the clean seam here
    once the checkpoint I/O integration lands.
    """
    raise NotImplementedError("F3 consumption of the merged body is a later task")


def _factor_U(f):
    """U of a factor item that may be (U, V) or (U, V, svals)."""
    return f[0]


def _factor_V(f):
    """V of a factor item that may be (U, V) or (U, V, svals)."""
    return f[1]
