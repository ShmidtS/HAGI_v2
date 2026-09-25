"""Block-diagonal expert merge: train-many-small, merge-into-big.

The hypothesis (information-theoretic): a large model trained from scratch
spends most of its early compute discovering *specialized* representations
(per-domain features) that are then composed. If we instead train N small
experts to saturation on N different corpora, each expert's hidden space is
already a well-formed, domain-tuned code. Concatenating those spaces
block-diagonally into one wide model gives the big model a head start: it
inherits N pre-formed specialized subspaces instead of having to discover them.

The merged model is built so that at step 0 it is *exactly* N independent
experts:

    W_Q = diag(W_Q^A, W_Q^B, ..., W_Q^N)   (and likewise for KV, out, FFN)

so the off-diagonal blocks are zero and each block reproduces its expert's
forward pass bit-for-bit. The only new parameters are a small number of
cross-block ``CrossMixer`` layers (identity-init, gain 0), which are the
connections that let the blocks interact. Short joint training then teaches
the blocks to compose — the mixer gain rises from 0 as the blocks learn to
exchange information.

Why block-diagonal rather than a dense re-init: a dense big model would
scramble the experts' learned geometry at step 0 (every weight is a linear
combination of unrelated experts), destroying the very representations we
merged to preserve. Block-diagonal preserves them exactly and adds only the
cheap mixer connections.

The mixer is placed *after* the expert stack, on the residual stream, so it
does not perturb the experts' internal dynamics — it only mixes their outputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from hagi.config import Config
from hagi.model.ffn import BranchScale, RecursiveBranchScale
from hagi.model.model import HAGI


def _hadamard_matrix(n: int) -> torch.Tensor:
    """Sylvester Hadamard matrix ``H_n`` of order ``n`` (power of two).

    ``H_1 = [1]`` and ``H_{2k} = [[H_k, H_k], [H_k, -H_k]]``. Orthogonal up to
    the ``sqrt(n)`` scale: ``H_n H_n^T = n I``, so ``H_n / sqrt(n)`` is
    orthonormal. The transform is a pure permutation of the expert axis — it
    adds no information, only re-mixes it (the user's core observation).
    """
    if n == 1:
        return torch.ones(1, 1)
    if n & (n - 1):
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    half = _hadamard_matrix(n // 2)
    top = torch.cat([half, half], dim=1)
    bot = torch.cat([half, -half], dim=1)
    return torch.cat([top, bot], dim=0)


# Cache of the orthonormal ``H_n / sqrt(n)`` per order, so the matmul path
# below never rebuilds the matrix. Keyed by ``(n, device, dtype)``.
_HADAMARD_CACHE: dict[tuple, torch.Tensor] = {}


def _hadamard_orthonormal(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the cached orthonormal ``H_n / sqrt(n)`` on ``device``/``dtype``.

    For ``n`` a power of two this is the exact Sylvester Hadamard transform
    (orthonormal, ``H H^T = I``). For arbitrary ``n`` (e.g. 18 experts) it
    pads to the next power of two ``m``, builds ``H_m``, takes the first ``n``
    rows, and orthonormalizes them via QR — yielding an ``n x n`` orthogonal
    matrix that reduces to the exact Hadamard whenever ``n`` is a power of
    two. This lets the Hadamard mixer be used for any expert count.
    """
    key = (n, device.type, device.index, dtype)
    hit = _HADAMARD_CACHE.get(key)
    if hit is not None:
        return hit
    if n & (n - 1):
        # General n: pad to next power of two, take first n rows, orthonormalize.
        # QR of the n x m matrix (n < m) yields an n x n orthogonal Q. QR is
        # done in float32 (geqrf has no bf16 CUDA kernel), then cast to dtype.
        m = 1
        while m < n:
            m <<= 1
        h = _hadamard_matrix(m)[:n].to(device=device, dtype=torch.float32)
        q, _ = torch.linalg.qr(h)
        h = q.to(dtype=dtype)  # n x n orthogonal
    else:
        h = (_hadamard_matrix(n) / math.sqrt(n)).to(device=device, dtype=dtype)
    if len(_HADAMARD_CACHE) > 16:
        _HADAMARD_CACHE.clear()
    _HADAMARD_CACHE[key] = h
    return h


# Cache of recursive (Kronecker) Hadamard matrices per group layout.
_RECURSIVE_CACHE: dict[tuple, torch.Tensor] = {}


def _hadamard_recursive_matrix(
    group_sizes: list[int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build the recursive/local Hadamard as a Kronecker product of per-level
    Hadamard matrices: ``H = H_{g_k} ⊗ ... ⊗ H_{g_1}`` where ``prod(g_i) = n``.

    This is the recursive tree (16→4→1): at each level a local Hadamard mixes
    the experts *within each group*, then the groups are grouped again. Each
    factor is orthonormal, so the product is orthonormal. Because the Sylvester
    Hadamard satisfies ``H_{ab} = H_a ⊗ H_b``, for uniform group sizes this
    equals the global ``H_n`` up to a channel permutation — the difference is
    the *ordering* of the sum/difference channels, which decides which
    combinations read as "large-scale shared" vs "small-scale differences"
    (the user's hierarchy idea). The mechanism is what matters: it is ready for
    a 16→4→1 growth pipeline where each level carries its own local mixer.
    """
    key = (tuple(group_sizes), device.type, device.index, dtype)
    hit = _RECURSIVE_CACHE.get(key)
    if hit is not None:
        return hit
    result = torch.ones(1, 1, device=device, dtype=dtype)
    for g in group_sizes:
        h = _hadamard_orthonormal(g, device, dtype)
        result = torch.kron(result, h)
    if len(_RECURSIVE_CACHE) > 16:
        _RECURSIVE_CACHE.clear()
    _RECURSIVE_CACHE[key] = result
    return result


def _dft3_pair(z: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate a complex pair ``(re, im)`` by ``omega**k`` (omega = e^{2pi i/3}).

    ``z`` is ``[..., 2]`` (last dim = re, im). Multiplication by omega is a
    rotation by 120 deg, by omega^2 a rotation by 240 deg. Pure real arithmetic
    (the RoPE trick) — no complex tensors needed on ROCm.
    """
    a = z[..., 0]
    b = z[..., 1]
    s = math.sqrt(3) / 2.0
    if k == 1:  # omega = cos120 + i sin120 = -1/2 + i sqrt(3)/2
        return torch.stack((-0.5 * a - s * b, s * a - 0.5 * b), dim=-1)
    # omega^2 = cos240 + i sin240 = -1/2 - i sqrt(3)/2
    return torch.stack((-0.5 * a + s * b, -s * a - 0.5 * b), dim=-1)


def _dft3_blocks3(x: torch.Tensor, bd: int) -> torch.Tensor:
    """Apply the complex DFT-3 ``F_3 ⊗ I_{bd/2}`` to ``[..., 3*bd]``.

    ``x`` is a concatenation of three expert blocks of width ``bd`` (must be
    even). Each pair of coordinates inside a block is treated as one complex
    number; the unitary ``F_3`` mixes the three experts on the complex axis and
    the result is mapped back to real pairs. The real representation of a
    unitary matrix is orthogonal, so this is an orthonormal transform — and
    unlike any real 3x3 it mixes all three experts with equal weight (no blind
    channel): every output pair sees every input pair with norm ``1/sqrt(3)``.
    """
    xb = x.reshape(x.shape[:-1] + (3, bd // 2, 2))
    z0, z1, z2 = xb[..., 0, :, :], xb[..., 1, :, :], xb[..., 2, :, :]
    s = 1.0 / math.sqrt(3)
    y0 = s * (z0 + z1 + z2)
    y1 = s * (z0 + _dft3_pair(z1, 1) + _dft3_pair(z2, 2))
    y2 = s * (z0 + _dft3_pair(z1, 2) + _dft3_pair(z2, 1))
    return torch.stack([y0, y1, y2], dim=-3).reshape(x.shape)


def _is_ternary_group(g: int) -> bool:
    """True if ``g`` is a power of three (3, 9, 27, ...)."""
    if g < 1:
        return False
    while g % 3 == 0:
        g //= 3
    return g == 1


def _dft3_apply(x: torch.Tensor, n_blocks: int, bd: int) -> torch.Tensor:
    """Apply the ternary DFT to ``[..., n_blocks*bd]``, ``n_blocks = 3^k``.

    Recursive Kronecker structure: ``F_{3^k} = F_3 ⊗ ... ⊗ F_3``, applied level
    by level exactly like the Hadamard butterfly. At each level every local
    triple of blocks (bd wide) is mixed by ``F_3``, then the triples are grouped
    into triples again. The product is orthonormal; this is the ternary analogue
    of ``H_{2^k}``.
    """
    if n_blocks == 1:
        return x
    if n_blocks == 3:
        return _dft3_blocks3(x, bd)
    # n_blocks = 9, 27, ...: first mix inside each local triple, then recurse
    # on the ``n_blocks/3`` super-groups whose effective block width grew to
    # ``3*bd`` (the Kronecker structure ``F_{3^k} = (F_3 ⊗ I) (I ⊗ F_3) ...``).
    xb = x.reshape(x.shape[:-1] + (n_blocks // 3, 3, bd))
    # Each local triple (3 blocks of width bd) becomes one contiguous row of
    # width 3*bd; _dft3_blocks3 mixes those three blocks per group.
    xc = xb.reshape(xb.shape[:-2] + (3 * bd,))
    yc = _dft3_blocks3(xc, bd)
    y2 = yc.reshape(x.shape)
    return _dft3_apply(y2, n_blocks // 3, 3 * bd)


def _dft3_apply_2d(weight: torch.Tensor, n_blocks: int, bd: int) -> torch.Tensor:
    """Right-multiply a 2D weight by the ternary DFT (head pre-rotation)."""
    if n_blocks == 1:
        return weight
    return _dft3_apply(weight, n_blocks, bd)


def _f3_real_column_matrix() -> torch.Tensor:
    """Return the canonical 6x6 real lift of the positive-exponent F3.

    Coordinates are branch-major and pair-interleaved.  This is the same
    matrix represented by :func:`_dft3_blocks3`; keeping one explicit golden
    definition makes orientation and provenance testable.
    """
    s = 1.0 / math.sqrt(3.0)
    b = math.sqrt(3.0) / 2.0
    return torch.tensor(
        [
            [s, 0.0, s, 0.0, s, 0.0],
            [0.0, s, 0.0, s, 0.0, s],
            [s, 0.0, -0.5 * s, -b * s, -0.5 * s, b * s],
            [0.0, s, b * s, -0.5 * s, -b * s, -0.5 * s],
            [s, 0.0, -0.5 * s, b * s, -0.5 * s, -b * s],
            [0.0, s, -b * s, -0.5 * s, b * s, -0.5 * s],
        ],
        dtype=torch.float64,
    )


_F3_C3_SHA256 = "07e2571f2bfd0ff907294851f9239e0bf0b858c91b47390e01b16838699b29ce"


def f3_real_column_matrix() -> torch.Tensor:
    """Public read-only copy of the canonical six-dimensional F3 lift."""
    return _f3_real_column_matrix().clone()


def f3_real_row_matrix() -> torch.Tensor:
    """Public copy of the row-action matrix ``R3 = C3.T``."""
    return f3_real_column_matrix().transpose(0, 1).contiguous()


def _f3_row_apply(x: torch.Tensor, block_dim: int) -> torch.Tensor:
    """Apply row-action ``R3`` to branch-major vectors of width ``3*block_dim``."""
    if block_dim < 1 or block_dim % 2:
        raise ValueError("block_dim must be a positive even integer")
    if x.shape[-1] != 3 * block_dim:
        raise ValueError(f"expected width {3 * block_dim}, got {x.shape[-1]}")
    # Branch-major [3, m, 2] -> pair-major [m, 3, 2], then row-action R3=C3.T.
    lead = x.shape[:-1]
    pair_major = x.reshape(lead + (3, block_dim // 2, 2)).movedim(-3, -2)
    pair_major = pair_major.reshape(lead + (block_dim // 2, 6))
    row_matrix = f3_real_column_matrix().transpose(0, 1).to(dtype=x.dtype)
    y = pair_major @ row_matrix
    y = y.reshape(lead + (block_dim // 2, 3, 2)).movedim(-2, -3)
    return y.reshape(x.shape)


def _f3_row_inverse_apply(x: torch.Tensor, block_dim: int) -> torch.Tensor:
    """Apply inverse row-action ``C3`` to branch-major vectors."""
    if block_dim < 1 or block_dim % 2:
        raise ValueError("block_dim must be a positive even integer")
    if x.shape[-1] != 3 * block_dim:
        raise ValueError(f"expected width {3 * block_dim}, got {x.shape[-1]}")
    lead = x.shape[:-1]
    pair_major = x.reshape(lead + (3, block_dim // 2, 2)).movedim(-3, -2)
    pair_major = pair_major.reshape(lead + (block_dim // 2, 6))
    row_matrix = f3_real_column_matrix().to(dtype=x.dtype)
    y = pair_major @ row_matrix
    y = y.reshape(lead + (block_dim // 2, 3, 2)).movedim(-2, -3)
    return y.reshape(x.shape)


_PARENT_PRESERVING_TERNARY_Q = (
    (2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0),
    (2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0),
    (-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
)
_PARENT_PRESERVING_TERNARY_SHA256 = (
    "421b8bede29cdcade694a181d1ec104b2aa4aaacaa010b08782b1fa57cb5def4"
)  # provenance pin; the live value comes from parent_preserving_ternary_digest()


def parent_preserving_ternary_matrix() -> torch.Tensor:
    """Return the canonical outer ternary lift ``Q(pi/2)`` for three parents.

    The matrix is orthogonal and fixes ``(1, 1, 1)``. Consequently, lifting
    three identical parent copies is the identity on their diagonal. This is
    a separate transform from :class:`TernaryF3Tree`, whose staged complex
    transform deliberately aggregates repeated branches instead.
    """
    return torch.tensor(_PARENT_PRESERVING_TERNARY_Q, dtype=torch.float64)


def parent_preserving_ternary_digest() -> str:
    """Return the digest of the canonical parent-preserving lift.

    The digest is computed from the matrix itself rather than repeated as a
    literal, so a changed matrix cannot keep a matching pinned constant.
    """
    matrix = parent_preserving_ternary_matrix()
    raw = b"".join(
        struct.pack("<d", float(value)) for row in matrix.tolist() for value in row
    )
    return hashlib.sha256(raw).hexdigest()


class ParentPreservingTernaryLift:
    """Apply one outer orthogonal lift to three parent branches per channel."""

    schema_version = 1
    coordinate_layout = "branch_major_outer_ternary"

    def __init__(self) -> None:
        self.transform_digest = parent_preserving_ternary_digest()
        if self.transform_digest != _PARENT_PRESERVING_TERNARY_SHA256:
            raise ValueError("parent-preserving lift digest does not match the pin")

    @staticmethod
    def _matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if dtype not in {torch.float32, torch.float64, torch.bfloat16, torch.float16}:
            raise TypeError("parent-preserving lift requires a floating dtype")
        return parent_preserving_ternary_matrix().to(device=device, dtype=dtype)

    def apply_row(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] < 1 or x.shape[-1] % 3:
            raise ValueError("expected a non-empty width divisible by three")
        if not x.is_floating_point():
            raise TypeError("parent-preserving lift requires floating input")
        matrix = self._matrix(x.device, x.dtype)
        width = x.shape[-1]
        branches = x.reshape(x.shape[:-1] + (3, width // 3))
        return torch.matmul(matrix, branches).reshape_as(x)

    def apply_row_inverse(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] < 1 or x.shape[-1] % 3:
            raise ValueError("expected a non-empty width divisible by three")
        if not x.is_floating_point():
            raise TypeError("parent-preserving lift requires floating input")
        matrix = self._matrix(x.device, x.dtype)
        width = x.shape[-1]
        branches = x.reshape(x.shape[:-1] + (3, width // 3))
        return torch.matmul(matrix.transpose(0, 1), branches).reshape_as(x)

    def inverse_row_matrix(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        return parent_preserving_ternary_matrix().to(dtype=dtype).transpose(0, 1).contiguous()


class TernaryF3Tree(nn.Module):
    """Parameter-free staged row-action for ``F3**depth``.

    A level first applies F3 to each local triple of current blocks, then the
    next level treats each resulting super-block as one branch.  Thus the
    action is the hierarchical recurrence, not a guessed flat reshape.
    """

    def __init__(self, depth: int, leaf_hidden: int) -> None:
        super().__init__()
        if type(depth) is not int or depth < 0:
            raise ValueError("depth must be a nonnegative integer")
        if type(leaf_hidden) is not int or leaf_hidden < 1 or leaf_hidden % 2:
            raise ValueError("leaf_hidden must be a positive even integer")
        self.depth = depth
        self.leaf_hidden = leaf_hidden
        self.n_leaves = 3**depth
        self.coordinate_layout = "branch_major_pair_interleaved"
        digest_input = f"{self.coordinate_layout}:{depth}:{leaf_hidden}:{_F3_C3_SHA256}"
        self.transform_digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    def _check_width(self, x: torch.Tensor) -> None:
        expected = self.n_leaves * self.leaf_hidden
        if x.shape[-1] != expected:
            raise ValueError(f"expected hidden width {expected}, got {x.shape[-1]}")

    def apply_row(self, x: torch.Tensor) -> torch.Tensor:
        self._check_width(x)
        if self.depth == 0:
            return x
        lead = x.shape[:-1]
        n_blocks = self.n_leaves
        block = self.leaf_hidden
        current = x
        while n_blocks > 1:
            groups = n_blocks // 3
            current = current.reshape(lead + (groups, 3, block))
            current = _f3_row_apply(current.reshape(lead + (groups, 3 * block)), block)
            current = current.reshape(lead + (groups, 3 * block))
            block *= 3
            n_blocks //= 3
        return current.reshape(lead + (self.n_leaves * self.leaf_hidden,))

    def apply_row_inverse(self, x: torch.Tensor) -> torch.Tensor:
        self._check_width(x)
        if self.depth == 0:
            return x
        lead = x.shape[:-1]
        # Undo the last staged level first.  At each inverse level the
        # current stream is laid out as ``groups`` super-blocks, each made of
        # three equal branches.
        block = self.leaf_hidden * (3 ** (self.depth - 1))
        groups = 1
        current = x
        while groups < self.n_leaves:
            current = current.reshape(lead + (groups, 3 * block))
            current = _f3_row_inverse_apply(current, block)
            block //= 3
            groups *= 3
        return current.reshape(lead + (self.n_leaves * self.leaf_hidden,))

    def row_matrix(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Build the small reference matrix; intended for tests/diagnostics."""
        width = self.n_leaves * self.leaf_hidden
        return self.apply_row(torch.eye(width, dtype=dtype))

    def inverse_row_matrix(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        width = self.n_leaves * self.leaf_hidden
        return self.apply_row_inverse(torch.eye(width, dtype=dtype))

    def extra_repr(self) -> str:
        return f"depth={self.depth}, leaf_hidden={self.leaf_hidden}"


def hadamard_blocks(x: torch.Tensor, n_blocks: int, group_sizes: list[int] | None = None) -> torch.Tensor:
    """Apply ``(H_n / sqrt(n)) ⊗ I_H`` to the last axis of ``x``.

    ``x`` is ``[..., n_blocks * block_dim]`` (a concatenated expert stream).
    The transform mixes the ``n_blocks`` expert blocks while leaving each
    block's internal ``block_dim`` untouched — the ``I_H`` factor. It is the
    fixed, parameter-free cross-expert communication: after it, every output
    block is a normalized sum/difference of all expert blocks, so the merged
    model is no longer N independent subspaces at step 0.

    For small ``n_blocks`` (the typical merge, N ≤ 64) this is a single
    batched matmul against the cached ``H_n / sqrt(n)`` — measured ~9x faster
    than the butterfly on the Radeon 8060S (0.81 vs 7.76 ms at N=16, H=2048,
    B·T=8192), because the tiny ``[n, n]`` matrix rides the tensor cores and
    the reshape/stack butterfly is launch-bound. For very large ``n_blocks``
    (N > 64) the butterfly's ``O(NH log N)`` beats the ``O(N^2 H)`` matmul and
    is used instead. The ``1/sqrt(n_blocks)`` keeps the output norm equal to
    the input norm (the transform is orthonormal).
    """
    if n_blocks == 1:
        return x
    bd = x.shape[-1] // n_blocks
    if group_sizes is not None:
        # Ternary groups (all powers of three): use the complex DFT-3, the
        # orthonormal ternary analogue of the Hadamard. It mixes triples with
        # equal weight (no blind channel) and is applied level by level.
        if all(_is_ternary_group(g) for g in group_sizes) and bd % 2 == 0:
            return _dft3_apply(x, n_blocks, bd)
        # Recursive/local Hadamard: Kronecker product of per-level Hadamards.
        h = _hadamard_recursive_matrix(group_sizes, x.device, x.dtype)
        xb = x.reshape(x.shape[:-1] + (n_blocks, bd)).transpose(-1, -2)
        return torch.matmul(xb, h.t()).transpose(-1, -2).reshape(x.shape)
    if _is_ternary_group(n_blocks) and bd % 2 == 0:
        # Flat ternary: F_3 (or F_9, ...) directly.
        return _dft3_apply(x, n_blocks, bd)
    if n_blocks <= 64:
        h = _hadamard_orthonormal(n_blocks, x.device, x.dtype)
        # x: [..., n, bd] -> transpose to [..., bd, n] for the matmul, then back.
        xb = x.reshape(x.shape[:-1] + (n_blocks, bd)).transpose(-1, -2)
        return torch.matmul(xb, h.t()).transpose(-1, -2).reshape(x.shape)
    # Butterfly fallback for very large N (O(NH log N) < O(N^2 H)). Only valid
    # for power-of-two N; for non-power-of-two N > 64 we still use the matmul.
    if n_blocks & (n_blocks - 1):
        h = _hadamard_orthonormal(n_blocks, x.device, x.dtype)
        xb = x.reshape(x.shape[:-1] + (n_blocks, bd)).transpose(-1, -2)
        return torch.matmul(xb, h.t()).transpose(-1, -2).reshape(x.shape)
    shape = x.shape[:-1] + (n_blocks, bd)
    xb = x.reshape(shape)
    h = xb
    step = 1
    while step < n_blocks:
        h = h.reshape(x.shape[:-1] + (n_blocks // (2 * step), 2, step, bd))
        a = h[..., 0, :, :]
        b = h[..., 1, :, :]
        h = torch.stack((a + b, a - b), dim=-3)
        step *= 2
    h = h.reshape(x.shape[:-1] + (n_blocks, bd))
    return (h / math.sqrt(n_blocks)).reshape(x.shape)


def hadamard_apply_2d(
    weight: torch.Tensor, n_blocks: int, group_sizes: list[int] | None = None
) -> torch.Tensor:
    """Right-multiply a 2D weight by ``(H_n / sqrt(n)) ⊗ I_H``.

    Used to keep the merged head consistent with a Hadamard mixer: the head
    projection is ``hidden @ weight.T``, so if the mixer rotates the hidden
    stream by ``Q = (H_n/sqrt(n)) ⊗ I_H``, the head must be ``weight @ Q^T``
    (i.e. ``weight`` right-multiplied by ``Q``) for the step-0 logits to be
    unchanged. ``weight`` is ``[out, n_blocks * block_dim]``.
    """
    if n_blocks == 1:
        return weight
    bd = weight.shape[1] // n_blocks
    if group_sizes is not None:
        if all(_is_ternary_group(g) for g in group_sizes) and bd % 2 == 0:
            return _dft3_apply_2d(weight, n_blocks, bd)
        h = _hadamard_recursive_matrix(group_sizes, weight.device, weight.dtype)
        wb = weight.reshape(weight.shape[0], n_blocks, bd)
        return torch.matmul(wb.transpose(1, 2), h.t()).transpose(1, 2).reshape(weight.shape)
    if _is_ternary_group(n_blocks) and bd % 2 == 0:
        return _dft3_apply_2d(weight, n_blocks, bd)
    if n_blocks <= 64 or (n_blocks & (n_blocks - 1)):
        h = _hadamard_orthonormal(n_blocks, weight.device, weight.dtype)
        wb = weight.reshape(weight.shape[0], n_blocks, bd)
        # weight @ (H ⊗ I) : [out, n, bd] -> transpose to [out, bd, n] for the
        # matmul over the block axis, then back.
        return torch.matmul(wb.transpose(1, 2), h.t()).transpose(1, 2).reshape(weight.shape)
    wb = weight.reshape(weight.shape[0], n_blocks, bd)
    h = wb
    step = 1
    while step < n_blocks:
        h = h.reshape(weight.shape[0], n_blocks // (2 * step), 2, step, bd)
        a = h[:, :, 0, :, :]
        b = h[:, :, 1, :, :]
        h = torch.stack((a + b, a - b), dim=2)
        step *= 2
    h = h.reshape(weight.shape[0], n_blocks, bd)
    return (h / math.sqrt(n_blocks)).reshape(weight.shape)


class CrossMixer(nn.Module):
    """A cross-block mixing layer on the residual stream.

    ``y = x + gain * down(silu(gate(x)) * up(x))`` with ``gain`` initialized to
    ``mixer_init_scale`` (0 by default). At gain 0 the mixer is the identity, so
    a merged model with zero-init mixers is exactly N independent experts. The
    gain is a single learnable scalar (kept in fp32) that rises as joint
    training teaches the blocks to interact; it is the "how much do the blocks
    talk to each other" knob.

    The mixer is a full-width (H = N * expert_hidden) SwiGLU, so it can mix
    across all blocks. Its ``down`` projection is scaled by ``residual_scale``
    like any other residual branch.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        norm_eps: float = 1e-5,
        residual_scale: float = 1.0,
        mixer_init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        nn.init.normal_(self.gate.weight, std=hidden_size**-0.5)
        nn.init.normal_(self.up.weight, std=hidden_size**-0.5)
        nn.init.normal_(self.down.weight, std=residual_scale / intermediate_size**0.5)
        self.branch_scale = BranchScale(residual_scale)
        self.keep_fp32 = True
        self.gain = nn.Parameter(torch.tensor(float(mixer_init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out = self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h))
        return x + self.branch_scale(out) * self.gain.to(x.dtype)


class HadamardMixer(nn.Module):
    """Fixed fast-Hadamard cross-expert mixing plus a small low-rank residual.

    ``y = H·norm(x) + gain * down(silu(gate(x)) * up(x))``

    The fixed part ``H = (H_n / sqrt(n)) ⊗ I_H`` is the fast Hadamard transform
    over the expert axis. It is the user's key idea: instead of a block-
    diagonal merge that keeps the experts independent and a full-width mixer
    that must *discover* all cross-expert communication from scratch, the
    Hadamard gives every block a normalized sum/difference view of all the
    others at step 0 — for O(NH log N) FLOPs and zero parameters. If the
    experts share a common component ``s`` and differ by ``a, b``, the
    transform separates ``sum`` (shared) from ``difference`` (specialization)
    channels automatically.

    The learned part is deliberately small: ``gate/up`` are ``H x rank`` and
    ``down`` is ``rank x H``, so its FLOPs are ``O(H * rank)`` against the
    full-width SwiGLU's ``O(H^2)``. The gain starts at ``mixer_init_scale``
    (0 by default), so at step 0 the mixer is exactly the fixed Hadamard —
    the base mixing already exists, and training only corrects it.

    The Hadamard is orthonormal (``H H^T = I``), so it preserves the residual-
    stream norm and adds no information — it only re-mixes what is already
    there. The learned residual is what lets the blocks go beyond the fixed
    sum/difference geometry.
    """

    def __init__(
        self,
        hidden_size: int,
        n_blocks: int,
        rank: int = 64,
        norm_eps: float = 1e-5,
        residual_scale: float = 1.0,
        mixer_init_scale: float = 0.0,
        group_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % n_blocks:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by n_blocks {n_blocks}")
        self.n_blocks = int(n_blocks)
        self.rank = int(rank)
        self.group_sizes = list(group_sizes) if group_sizes else None
        if self.group_sizes is not None:
            prod = 1
            for g in self.group_sizes:
                # Each level mixes a local group; the fixed orthonormal
                # transform is the Hadamard for powers of two and the complex
                # DFT-3 for powers of three (ternary merge).
                if g < 1 or (not _is_ternary_group(g) and (g & (g - 1))):
                    raise ValueError(
                        f"group_sizes must be powers of two or powers of three, got {self.group_sizes}"
                    )
                prod *= g
            if prod != self.n_blocks:
                raise ValueError(
                    f"group_sizes {self.group_sizes} product {prod} != n_blocks {self.n_blocks}"
                )
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.gate = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(hidden_size, rank, bias=False)
        self.down = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.gate.weight, std=hidden_size**-0.5)
        nn.init.normal_(self.up.weight, std=hidden_size**-0.5)
        nn.init.normal_(self.down.weight, std=residual_scale / rank**0.5)
        self.branch_scale = BranchScale(residual_scale)
        self.keep_fp32 = True
        self.gain = nn.Parameter(torch.tensor(float(mixer_init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The fixed Hadamard is applied to the residual stream ``x`` itself
        # (not to ``norm(x)``), so at ``gain=0`` the mixer is exactly
        # ``y = Q·x``. Combined with the head pre-rotation ``WQ`` in
        # :func:`merge_experts`, the step-0 logits are identical to the
        # block-diagonal merge — the Hadamard re-mixes the stream and the head
        # is pre-rotated to match, so the merged model is not degraded before
        # joint training. The learned low-rank residual operates on the
        # normalized stream and is gated by ``gain`` (0 by default).
        mixed = hadamard_blocks(x, self.n_blocks, self.group_sizes)
        h = self.norm(x)
        out = self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h))
        return mixed + self.branch_scale(out) * self.gain.to(x.dtype)


class MergedHAGI(HAGI):
    """A HAGI whose body is N block-diagonally merged experts.

    The encoder, out-norm and head are the *merged* (wide) versions: the
    codebook is ``[V, N*H_i]`` and the head projection is ``[V, N*H_i]``. The
    body blocks are the block-diagonal concatenation of the experts' blocks,
    followed by ``n_mixers`` cross-block mixers.

    The merged model is constructed by :func:`merge_experts` from N expert
    checkpoints. When no checkpoints are given, the current model's weights are
    replicated N times (for machinery smoke tests).
    """

    def __init__(
        self,
        cfg: Config,
        n_mixers: int = 1,
        mixer_init_scale: float = 0.0,
    ) -> None:
        if str(getattr(cfg.merge, "mixer_type", "swiglu")) == "ternary_f3":
            raise ValueError(
                "ternary_f3 requires RecursiveF3HAGI; legacy MergedHAGI refuses this mode"
            )
        # Ternary quantization does not commute with block-diagonal merging:
        # ternarize normalizes each row by its absmean, and a merged row
        # includes the zero off-diagonal blocks, which changes the scale and
        # breaks the exact-expert equivalence. The merged body therefore uses
        # plain fp16 linear layers (block-diagonal weights applied exactly);
        # ternary can be re-enabled during joint training if desired.
        m = cfg.model
        _saved_ternary = m.ternary.enabled
        m.ternary.enabled = False
        try:
            super().__init__(cfg)
        finally:
            m.ternary.enabled = _saved_ternary
        h = m.hidden_size
        n = cfg.merge.n_experts
        if h % n != 0:
            raise ValueError(f"hidden_size {h} must be divisible by n_experts {n}")
        self.n_experts = n
        self.expert_hidden = h // n

        # Replace the wide RMSNorms with block-wise RMSNorms so each expert's
        # block is normalized independently (a plain wide RMSNorm would
        # normalize the whole concatenated stream at once, which is not the
        # same as per-expert normalization).
        from hagi.model.norms import BlockRMSNorm

        for block in self.blocks:
            block.attn.attn_norm = BlockRMSNorm(n, self.expert_hidden, m.norm_eps)
            block.mixer.norm = BlockRMSNorm(n, self.expert_hidden, m.norm_eps)
        self.out_norm = BlockRMSNorm(n, self.expert_hidden, m.norm_eps)

        # Re-attach opt-in adapters to the final merged blocks. The base HAGI
        # constructor already ran `_attach_adapters`, but MergedHAGI replaces
        # `self.blocks` (and swaps each block's mixer.norm) after that call, so
        # the adapters built on the pre-merge blocks are stale. Re-run the
        # attachment on the final blocks so each merged Block keeps its adapter
        # wired to its (possibly swapped) mixer. For the common
        # disabled-adapter default this is a cheap no-op (the loop body never
        # runs).
        self._attach_adapters(h)

        # Cross-block mixers on the residual stream.
        inter = max(64, int(2.0 * h))
        residual_scale = (2.0 * m.num_layers * max(1, int(m.loop_depth))) ** -0.5
        mixer_type = str(getattr(cfg.merge, "mixer_type", "swiglu"))
        mixer_rank = int(getattr(cfg.merge, "mixer_rank", 64))
        group_sizes = getattr(cfg.merge, "mixer_hadamard_groups", None)
        if group_sizes is not None:
            group_sizes = [int(g) for g in group_sizes]
        if mixer_type == "hadamard":
            self.mixers = nn.ModuleList(
                [
                    HadamardMixer(
                        h,
                        n,
                        rank=mixer_rank,
                        norm_eps=m.norm_eps,
                        residual_scale=residual_scale,
                        mixer_init_scale=mixer_init_scale,
                        group_sizes=group_sizes,
                    )
                    for _ in range(n_mixers)
                ]
            )
        else:
            self.mixers = nn.ModuleList(
                [
                    CrossMixer(h, inter, m.norm_eps, residual_scale, mixer_init_scale)
                    for _ in range(n_mixers)
                ]
            )

    def _apply_mixers(self, h: torch.Tensor) -> torch.Tensor:
        """Run the cross-block mixers on the normalized residual stream.

        The mixers run *after* ``out_norm`` (the hook is called from
        :meth:`HAGI.forward` right after the output norm). This ordering is
        what makes the Hadamard mixer's head pre-rotation exact: the head sees
        ``mixer(out_norm(h))``, and at ``gain=0`` the Hadamard mixer is
        ``Q·x``, so the pre-rotated head ``WQ`` reproduces the block-diagonal
        logits bit-for-bit.
        """
        for mixer in self.mixers:
            h = mixer(h)
        return h


def _ternarize_block(weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Ternarize a 2D weight per output row (BitNet b1.58), matching the
    experts' BitLinear forward. Returns the effective quantized weight.

    Row means are accumulated in float64 so finite FP32/BF16/FP16 masters do
    not overflow merely because the temporary reduction is wider. If even
    that finite input cannot produce a finite result, fail before returning a
    corrupted effective-sparse tensor.
    """
    if weight.ndim != 2:
        raise ValueError("ternarized block must be a 2D tensor")
    if not torch.isfinite(weight).all():
        raise ValueError("cannot ternarize a non-finite block")
    work = weight.detach().to(dtype=torch.float64)
    scale = work.abs().mean(dim=1, keepdim=True).clamp_min(float(eps))
    effective = (work / scale).clamp(-1.0, 1.0).round() * scale
    if not torch.isfinite(scale).all() or not torch.isfinite(effective).all():
        raise ValueError("ternarization overflowed despite a finite input block")
    result = effective.to(dtype=weight.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("ternarization produced a non-finite effective block")
    return result


def _block_diag(blocks: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate 2D weight blocks into a block-diagonal matrix.

    ``blocks`` are ``[out_i, in_i]``; the result is ``[sum out_i, sum in_i]``
    with the blocks on the diagonal and zeros elsewhere.
    """
    out_total = sum(b.shape[0] for b in blocks)
    in_total = sum(b.shape[1] for b in blocks)
    result = torch.zeros(out_total, in_total, dtype=blocks[0].dtype, device=blocks[0].device)
    row = 0
    col = 0
    for b in blocks:
        result[row : row + b.shape[0], col : col + b.shape[1]] = b
        row += b.shape[0]
        col += b.shape[1]
    return result


def _merge_2d(weights: list[torch.Tensor], block_diag: bool) -> torch.Tensor:
    """Merge a list of same-shaped 2D weights.

    ``block_diag=True``: block-diagonal concatenation (for hidden-mixing
    matrices, where each expert's weight acts on its own subspace).
    ``block_diag=False``: row-wise concatenation (for codebooks / head
    projections, where the output rows are the vocabulary and the input is the
    full hidden space).
    """
    if block_diag:
        return _block_diag(weights)
    return torch.cat(weights, dim=1)


def _merge_1d(weights: list[torch.Tensor]) -> torch.Tensor:
    """Merge 1D gains by concatenation (norms, per-expert gains)."""
    return torch.cat(weights, dim=0)


def _recursive_tree_norm_weights(
    child_weights: list[torch.Tensor], target_depth: int
) -> torch.Tensor:
    """Assemble ``[3, 3**(depth-1), leaf]`` tree-norm gains."""
    if len(child_weights) != 3:
        raise ValueError("recursive norm assembly requires exactly three children")
    if target_depth < 1:
        raise ValueError("recursive target depth must be positive")
    normalized: list[torch.Tensor] = []
    for weight in child_weights:
        if weight.ndim == 1:
            weight = weight.reshape(1, 1, -1)
        elif weight.ndim == 3:
            if tuple(weight.shape[:1]) != (3,):
                raise ValueError("recursive child norm must have shape [3, inner, leaf]")
        else:
            raise ValueError("recursive child norm must have shape [h] or [3, inner, h]")
        if weight.shape[-1] < 1:
            raise ValueError("recursive child norm leaf dimension must be positive")
        # A source depth-g norm contributes 3**g leaves to each new
        # top-level child.  Flatten its source hierarchy, then keep the three
        # target children as the first axis of the result.
        normalized.append(weight.reshape(1, -1, weight.shape[-1]))
    shapes = {tuple(weight.shape) for weight in normalized}
    if len(shapes) != 1:
        raise ValueError(f"recursive child norm shapes differ: {sorted(shapes)}")
    return torch.cat(normalized, dim=0)


def _recursive_prepare_blocks(
    blocks: list[torch.Tensor], eps: float, source: str
) -> list[torch.Tensor]:
    if source == "ternary_master":
        return [_ternarize_block(block, eps) for block in blocks]
    if source == "effective_sparse":
        return blocks
    raise ValueError("expert_weight_source must be ternary_master or effective_sparse")


class CrossParentPreservingTernaryTree(nn.Module):
    """Cross-parent mixing step of a recursive merge, applied as Q(pi/2).

    Why this exists. The recursive merge has two distinct transforms and only
    one of them is a cross-parent step:

    * the *inner* step builds a parent from its own three children. That is
      :class:`TernaryF3Tree`, and it is correct exactly as it stands, so this
      class does not touch or replace it;
    * the *outer* step mixes the three parent streams of a recursive merge.
      Done with the staged F3 tree it maps a duplicated triple ``(x, x, x)`` to
      ``(sqrt(3) x, 0, 0)`` per complex coordinate, so a self-merge of three
      identical parents is not function-preserving.

    This class is the outer step only. It applies the already-validated
    orthogonal parent-preserving lift :class:`ParentPreservingTernaryLift`
    (Q(pi/2)) to the flat three-parent stream, which fixes the all-ones
    vector and therefore leaves a duplicated self-merge unchanged, while
    remaining an orthogonal, invertible, capacity-preserving mixer of three
    distinct parents.

    Accepted limitation (not a claim): unlike the staged tree, this performs a
    single independent rotation per channel group, so it does not also mix the
    ``3**depth - 1`` internal sub-block groups. Mixing capacity is not measured
    to be sufficient; that is a separate experiment.

    The surface mirrors :class:`TernaryF3Tree` (row action, inverse, reference
    matrix, geometry-bound transform digest) so the model can hold either as
    its ``target_tree``. The transform digest is bound to the geometry, so a
    checkpoint written under one cross-parent transform cannot be silently
    replayed under the other.
    """

    mode_name = "parent_preserving"

    def __init__(self, depth: int, leaf_hidden: int) -> None:
        super().__init__()
        if type(depth) is not int or depth < 1:
            raise ValueError("depth must be a positive integer")
        if type(leaf_hidden) is not int or leaf_hidden < 1:
            raise ValueError("leaf_hidden must be a positive integer")
        self.depth = depth
        self.leaf_hidden = leaf_hidden
        self.n_leaves = 3**depth
        self.coordinate_layout = ParentPreservingTernaryLift.coordinate_layout
        digest_input = (
            f"{self.coordinate_layout}:{depth}:{leaf_hidden}"
            f":{parent_preserving_ternary_digest()}"
        )
        self.transform_digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        self._lift = ParentPreservingTernaryLift()

    def _check_width(self, x: torch.Tensor) -> None:
        expected = self.n_leaves * self.leaf_hidden
        if x.shape[-1] != expected:
            raise ValueError(f"expected hidden width {expected}, got {x.shape[-1]}")

    def apply_row(self, x: torch.Tensor) -> torch.Tensor:
        self._check_width(x)
        if not x.is_floating_point():
            raise TypeError("cross-parent parent-preserving transform requires float input")
        return self._lift.apply_row(x)

    def apply_row_inverse(self, x: torch.Tensor) -> torch.Tensor:
        self._check_width(x)
        if not x.is_floating_point():
            raise TypeError("cross-parent parent-preserving transform requires float input")
        return self._lift.apply_row_inverse(x)

    def row_matrix(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        width = self.n_leaves * self.leaf_hidden
        return self.apply_row(torch.eye(width, dtype=dtype))

    def inverse_row_matrix(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        width = self.n_leaves * self.leaf_hidden
        return self.apply_row_inverse(torch.eye(width, dtype=dtype))

    def extra_repr(self) -> str:
        return f"depth={self.depth}, leaf_hidden={self.leaf_hidden}"


#: Every coordinate layout a recursive candidate may declare. ``coordinate_layout``
#: is a NAME for how a vector is laid out, never a geometry-dependent string, so
#: the set stays small and stable; unknown names fail closed downstream. The
#: orchestrator imports this from here (merge does not import the orchestrator),
#: so both transforms are accepted by one check.
KNOWN_COORDINATE_LAYOUTS: frozenset[str] = frozenset(
    {
        "branch_major_pair_interleaved",  # TernaryF3Tree (instance attribute)
        ParentPreservingTernaryLift.coordinate_layout,
    }
)

#: Names of the cross-parent transforms a recursive merge may use. The legacy
#: staged F3 tree stays the default so existing behaviour remains reachable
#: unchanged; the parent-preserving lift must always be selected by name.
_CROSS_PARENT_TRANSFORMS = ("f3_tree", "parent_preserving")

#: Public alias for the set of cross-parent transform names.
CROSS_PARENT_TRANSFORMS = _CROSS_PARENT_TRANSFORMS


def _cross_parent_mode_from_config(cfg: Config) -> str:
    """Read the cross-parent transform name out of the merge config.

    The config is the only place a mode is inferred rather than passed by
    name; :meth:`RecursiveF3HAGI.from_state_dict` and the ``__init__`` seam
    both write the selected name back into the model config so the saved
    checkpoint and the shipped transform can never disagree.
    """
    mode = getattr(cfg.merge, "ternary_lift_mode", "f3_tree")
    if mode not in _CROSS_PARENT_TRANSFORMS:
        raise ValueError(f"unknown merge.ternary_lift_mode: {mode!r}")
    return mode


def _build_cross_parent_tree(
    cross_parent_transform: str, cfg: Config
) -> TernaryF3Tree | CrossParentPreservingTernaryTree:
    """Instantiate the named cross-parent transform for ``cfg``'s geometry.

    ``TernaryF3Tree`` remains the transform that builds a parent from its
    three children and is used unchanged; this function only picks the
    transform that mixes the three *parent* streams.
    """
    if type(cross_parent_transform) is not str:
        raise TypeError("cross_parent_transform must be a string")
    depth = cfg.merge.ternary_depth
    leaf_hidden = _recursive_leaf_hidden(cfg)
    if cross_parent_transform == "f3_tree":
        return TernaryF3Tree(depth, leaf_hidden)
    if cross_parent_transform == "parent_preserving":
        return CrossParentPreservingTernaryTree(depth, leaf_hidden)
    raise ValueError(
        "cross_parent_transform must be one of "
        f"{sorted(_CROSS_PARENT_TRANSFORMS)}, got {cross_parent_transform!r}"
    )


def _recursive_leaf_hidden(cfg: Config) -> int:
    return cfg.merge.expert_hidden // (3 ** (cfg.merge.ternary_depth - 1))


def state_key_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Return a stable digest of tensor names, shapes, and dtypes.

    The digest identifies a checkpoint's state schema without serializing
    parameter values. It is used to bind recursive candidate provenance.
    """
    entries = []
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError("state values must be tensors")
        entries.append({"key": key, "shape": list(value.shape), "dtype": str(value.dtype)})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RecursiveF3HAGI(HAGI):
    """Opt-in three-parent model assembled with the fixed hierarchical F3."""

    _PROVENANCE_KEYS = (
        "recursive_f3_marker",
        "recursive_f3_child_config_fingerprint",
        "recursive_f3_transform_digest",
        "recursive_f3_logit_scale_source",
        "recursive_f3_logit_scale_target",
        "recursive_f3_cross_parent_transform",
    )
    _PROVENANCE_MAGIC = "HAGI_RECURSIVE_F3_V1"

    @staticmethod
    def _text_buffer(value: str) -> torch.Tensor:
        return torch.tensor(list(value.encode("utf-8")), dtype=torch.uint8)

    @staticmethod
    def _decode_text_buffer(value: torch.Tensor, name: str) -> str:
        raw = bytes(value.detach().cpu().tolist())
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"recursive provenance {name} is not valid UTF-8") from exc

    @classmethod
    def is_recursive_state(cls, state: Mapping[str, torch.Tensor]) -> bool:
        return cls._PROVENANCE_KEYS[0] in state

    def _register_provenance_buffers(
        self,
        child_config_fingerprint: str,
        transform_digest: str,
        logit_scale_source: float,
        logit_scale_target: float,
        cross_parent_transform: str,
    ) -> None:
        self.register_buffer(
            "recursive_f3_marker", self._text_buffer(self._PROVENANCE_MAGIC),
            persistent=True,
        )
        self.register_buffer(
            "recursive_f3_child_config_fingerprint",
            self._text_buffer(child_config_fingerprint),
            persistent=True,
        )
        self.register_buffer(
            "recursive_f3_transform_digest",
            self._text_buffer(transform_digest),
            persistent=True,
        )
        self.register_buffer(
            "recursive_f3_logit_scale_source",
            torch.tensor(logit_scale_source, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "recursive_f3_logit_scale_target",
            torch.tensor(logit_scale_target, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "recursive_f3_cross_parent_transform",
            self._text_buffer(cross_parent_transform),
            persistent=True,
        )
        self.child_config_fingerprint = child_config_fingerprint
        self.transform_digest = transform_digest
        self.cross_parent_transform = cross_parent_transform
        self.logit_scale_source = logit_scale_source
        self.logit_scale_target = logit_scale_target

    def _validate_provenance_buffers(self) -> None:
        marker = self._decode_text_buffer(self.recursive_f3_marker, "marker")
        if marker != self._PROVENANCE_MAGIC:
            raise ValueError("recursive F3 provenance marker is invalid")
        for name in (
            "child_config_fingerprint",
            "transform_digest",
        ):
            value = self._decode_text_buffer(getattr(self, f"recursive_f3_{name}"), name)
            if not value:
                raise ValueError(f"recursive provenance {name} is empty")
        cross_parent = self._decode_text_buffer(
            self.recursive_f3_cross_parent_transform, "cross_parent_transform"
        )
        if cross_parent not in _CROSS_PARENT_TRANSFORMS:
            raise ValueError(
                f"recursive provenance cross_parent_transform is unknown: {cross_parent!r}"
            )
        for name in ("logit_scale_source", "logit_scale_target"):
            value = getattr(self, f"recursive_f3_{name}")
            if value.numel() != 1 or not torch.isfinite(value).all() or value.item() <= 0:
                raise ValueError(f"recursive provenance {name} is invalid")
        self.child_config_fingerprint = self._decode_text_buffer(
            self.recursive_f3_child_config_fingerprint, "child_config_fingerprint"
        )
        self.transform_digest = self._decode_text_buffer(
            self.recursive_f3_transform_digest, "transform_digest"
        )
        self.cross_parent_transform = cross_parent
        self.logit_scale_source = float(self.recursive_f3_logit_scale_source.item())
        self.logit_scale_target = float(self.recursive_f3_logit_scale_target.item())

    @classmethod
    def from_state_dict(
        cls,
        cfg: Config,
        state: Mapping[str, torch.Tensor],
        device: str | torch.device = "cpu",
        cross_parent_transform: str | None = None,
    ) -> RecursiveF3HAGI:
        """Rebuild a recursive model without replaying its three children.

        ``cross_parent_transform`` names the transform that mixed the three
        parent streams when the checkpoint was written. When ``None`` the name
        is read from ``cfg.merge.ternary_lift_mode``; pass it explicitly to
        select the transform without mutating the config. A checkpoint whose
        recorded transform differs from the requested one is rejected rather
        than silently replayed.
        """
        if not cls.is_recursive_state(state):
            raise ValueError("state_dict has no RecursiveF3HAGI provenance")
        if cfg.merge.mixer_type != "ternary_f3":
            raise ValueError("recursive state requires merge.mixer_type='ternary_f3'")
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        HAGI.__init__(model, cfg)
        from hagi.model.norms import BlockTreeNorm

        parent_depth = cfg.merge.ternary_depth - 1
        leaf_hidden = cfg.merge.expert_hidden // (3**parent_depth)
        if leaf_hidden < 1 or leaf_hidden * (3**parent_depth) != cfg.merge.expert_hidden:
            raise ValueError("recursive checkpoint has invalid leaf geometry")
        for block in model.blocks:
            block.attn.attn_norm = BlockTreeNorm(
                cfg.merge.ternary_depth, leaf_hidden, cfg.model.norm_eps
            )
            block.mixer.norm = BlockTreeNorm(
                cfg.merge.ternary_depth, leaf_hidden, cfg.model.norm_eps
            )
            count = 3**cfg.merge.ternary_depth
            block.attn.branch_scale = RecursiveBranchScale(
                count, block.attn.branch_scale.residual_scale
            )
            block.mixer.mixer.branch_scale = RecursiveBranchScale(
                count, block.mixer.mixer.branch_scale.residual_scale
            )
        model.out_norm = BlockTreeNorm(
            cfg.merge.ternary_depth, leaf_hidden, cfg.model.norm_eps
        )
        model.parent_tree = TernaryF3Tree(parent_depth, leaf_hidden)
        selected_transform = (
            _cross_parent_mode_from_config(cfg)
            if cross_parent_transform is None
            else cross_parent_transform
        )
        model.target_tree = _build_cross_parent_tree(selected_transform, cfg)
        model.parent_depth = parent_depth
        model.ternary_depth = cfg.merge.ternary_depth
        model.leaf_hidden = leaf_hidden
        model.expert_weight_source = cfg.merge.expert_weight_source
        model.lift_mode = selected_transform
        model.cross_parent_transform = selected_transform
        required = set(cls._PROVENANCE_KEYS)
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"recursive state_dict missing provenance: {missing}")
        model.register_buffer(
            "recursive_f3_marker", state["recursive_f3_marker"].detach().clone(), persistent=True
        )
        model.register_buffer(
            "recursive_f3_child_config_fingerprint",
            state["recursive_f3_child_config_fingerprint"].detach().clone(),
            persistent=True,
        )
        model.register_buffer(
            "recursive_f3_transform_digest",
            state["recursive_f3_transform_digest"].detach().clone(),
            persistent=True,
        )
        model.register_buffer(
            "recursive_f3_logit_scale_source",
            state["recursive_f3_logit_scale_source"].detach().clone(),
            persistent=True,
        )
        model.register_buffer(
            "recursive_f3_logit_scale_target",
            state["recursive_f3_logit_scale_target"].detach().clone(),
            persistent=True,
        )
        model.register_buffer(
            "recursive_f3_cross_parent_transform",
            state["recursive_f3_cross_parent_transform"].detach().clone(),
            persistent=True,
        )
        model._validate_provenance_buffers()
        # Fail closed on a replay across cross-parent transforms. The recorded
        # name is checked first so the error names the actual transform swap;
        # the digest check then catches any change inside a named transform
        # (its geometry or its matrix), because the digest is bound to both.
        if model.cross_parent_transform != selected_transform:
            raise ValueError(
                "recursive checkpoint was written with cross-parent transform "
                f"{model.cross_parent_transform!r}, but {selected_transform!r} was "
                "requested; a checkpoint cannot be replayed under the other "
                "cross-parent transform"
            )
        if model.transform_digest != model.target_tree.transform_digest:
            raise ValueError(
                "recursive checkpoint transform digest does not match the "
                f"configured merge.ternary_lift_mode={model.lift_mode!r} tree"
            )
        model.load_state_dict(state, strict=True)
        return model.to(device)

    def __init__(
        self,
        cfg: Config,
        child_states: list[Mapping[str, torch.Tensor]],
        *,
        child_configs: Sequence[Config],
        parent_depth: int | None = None,
        expert_weight_source: str | None = None,
        cross_parent_transform: str | None = None,
        drop_expert_mixers: bool = False,
    ) -> None:
        """Assemble the recursive parent from three child states.

        ``cross_parent_transform`` names the transform that mixes the three
        parent streams: ``"f3_tree"`` (the legacy staged transform, unchanged)
        or ``"parent_preserving"`` (the orthogonal Q(pi/2) cross-parent lift,
        which leaves a self-merge of three identical parents unchanged). When
        ``None`` the name is read from ``cfg.merge.ternary_lift_mode`` so
        existing behaviour stays reachable; the new transform is never a silent
        default. The choice is recorded in the persistent provenance buffers.
        """
        if drop_expert_mixers:
            raise ValueError("drop_expert_mixers=True is forbidden in recursive F3")
        if cfg.merge.mixer_type != "ternary_f3":
            raise ValueError("RecursiveF3HAGI requires merge.mixer_type='ternary_f3'")
        if len(child_states) != 3:
            raise ValueError("RecursiveF3HAGI requires exactly three child states")
        if len(child_configs) != 3:
            raise ValueError("child_configs must contain exactly three values")
        if not all(isinstance(child_cfg, Config) for child_cfg in child_configs):
            raise ValueError("child_configs must contain Config values")
        from hagi.train.checkpoint import config_to_dict

        canonical_config_json = json.dumps(
            config_to_dict(child_configs[0]),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        canonical_child_config = child_configs[0]
        for child_cfg in child_configs[1:]:
            child_json = json.dumps(
                config_to_dict(child_cfg),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            if child_json != canonical_config_json:
                raise ValueError("all recursive children must use one canonical Config")
        if not all(isinstance(state, Mapping) for state in child_states):
            raise ValueError("child states must be mappings of tensor names")
        selected_source = (
            cfg.merge.expert_weight_source
            if expert_weight_source is None
            else expert_weight_source
        )
        if type(selected_source) is not str or selected_source not in {
            "ternary_master",
            "effective_sparse",
        }:
            raise ValueError(
                "expert_weight_source must be 'ternary_master' or 'effective_sparse'"
            )
        target_depth = cfg.merge.ternary_depth
        if type(target_depth) is not int or target_depth < 1:
            raise ValueError("recursive target depth must be a positive integer")
        actual_parent_depth = target_depth - 1 if parent_depth is None else parent_depth
        if type(actual_parent_depth) is not int or actual_parent_depth < 0:
            raise ValueError("parent_depth must be a nonnegative integer")
        if actual_parent_depth != target_depth - 1:
            raise ValueError("parent_depth must equal target ternary_depth - 1")

        first = child_states[0]
        provenance_keys = set(self._PROVENANCE_KEYS)
        keys = set(first) - provenance_keys
        if not keys:
            raise ValueError("child state is empty")
        for state in child_states:
            forbidden = sorted(
                key
                for key in state
                if key.startswith("mixers.")
                or key.startswith("cortex.")
                or key.startswith("decision_head.")
                or ".adapters." in key
                or ".ttt_lora." in key
            )
            if forbidden:
                raise ValueError(
                    f"recursive child contains forbidden state key: {forbidden}"
                )
        for state in child_states[1:]:
            if set(state) - provenance_keys != keys:
                raise ValueError("recursive child state key sets must match")
        reference_dtype: torch.dtype | None = None
        reference_device: torch.device | None = None
        for key in sorted(keys):
            tensors = [state[key] for state in child_states]
            if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
                raise ValueError(f"child state value is not a tensor: {key}")
            if any(tuple(tensor.shape) != tuple(tensors[0].shape) for tensor in tensors[1:]):
                raise ValueError(f"recursive child shape mismatch on {key}")
            if any(tensor.dtype != tensors[0].dtype for tensor in tensors[1:]):
                raise ValueError(f"recursive child dtype mismatch on {key}")
            if any(tensor.device != tensors[0].device for tensor in tensors[1:]):
                raise ValueError(f"recursive child device mismatch on {key}")
            if reference_dtype is None:
                reference_dtype = tensors[0].dtype
                reference_device = tensors[0].device
            elif tensors[0].dtype != reference_dtype or tensors[0].device != reference_device:
                raise ValueError("recursive child state must use one dtype and device")
            if tensors[0].is_floating_point() and not all(
                torch.isfinite(tensor).all() for tensor in tensors
            ):
                raise ValueError(f"recursive child contains non-finite tensor: {key}")

        child_config_fingerprint = hashlib.sha256(
            canonical_config_json.encode("utf-8")
        ).hexdigest()
        child_m = canonical_child_config.model
        if child_m.ternary.enabled:
            raise ValueError("recursive child config must use an effective-sparse plain body")
        if child_m.embedding.tie_lm_head:
            raise ValueError("recursive child config requires tie_lm_head=False")
        if child_m.embedding.conv_kernel != 1:
            raise ValueError("recursive child config requires embedding.conv_kernel=1")
        if child_m.adapters.enabled or child_m.cortex.enabled or child_m.decision.enabled:
            raise ValueError("recursive child config must disable adaptive child state")
        if child_m.adapters.ttt_lora.enabled:
            raise ValueError("recursive child config must disable TTT-LoRA")
        if child_m.head.unigram_prior:
            raise ValueError("recursive child config must disable unigram prior")
        if child_m.loop_depth != 2:
            raise ValueError("recursive child config requires loop_depth=2")

        model_cfg = copy.deepcopy(cfg)
        model_cfg.merge.expert_weight_source = selected_source
        from hagi.config import validate_config

        # The explicitly requested transform wins over the config's mode, and
        # the config is rewritten BEFORE validation so the saved checkpoint
        # names the transform that actually ships instead of disagreeing with
        # the persistent provenance.
        selected_transform = (
            _cross_parent_mode_from_config(model_cfg)
            if cross_parent_transform is None
            else cross_parent_transform
        )
        if selected_transform not in _CROSS_PARENT_TRANSFORMS:
            raise ValueError(
                "cross_parent_transform must be one of "
                f"{sorted(_CROSS_PARENT_TRANSFORMS)}, got {selected_transform!r}"
            )
        model_cfg.merge.ternary_lift_mode = selected_transform
        validate_config(model_cfg)
        parent_hidden = model_cfg.merge.expert_hidden
        m = model_cfg.model
        if child_m.vocab_size != m.vocab_size:
            raise ValueError("recursive child vocabulary size mismatch")
        if child_m.hidden_size != parent_hidden:
            raise ValueError("recursive child hidden size mismatch")
        if child_m.attention.num_query_heads * 3 != m.attention.num_query_heads:
            raise ValueError("recursive child query-head geometry mismatch")
        if child_m.attention.num_kv_heads * 3 != m.attention.num_kv_heads:
            raise ValueError("recursive child KV-head geometry mismatch")
        if child_m.attention.head_dim != m.attention.head_dim:
            raise ValueError("recursive child attention head_dim mismatch")
        if child_m.ffn.intermediate_size * 3 != m.ffn.intermediate_size:
            raise ValueError("recursive child FFN geometry mismatch")
        if child_m.num_layers != m.num_layers:
            raise ValueError("recursive child layer count mismatch")
        if model_cfg.model.hidden_size != 3 * parent_hidden:
            raise ValueError("recursive target hidden_size must be 3*expert_hidden")
        if model_cfg.model.attention.head_dim % 2:
            raise ValueError("recursive F3 requires an even head_dim")
        leaf_hidden = parent_hidden // (3**actual_parent_depth)
        if leaf_hidden < 1 or leaf_hidden * (3**actual_parent_depth) != parent_hidden:
            raise ValueError("parent hidden width is not divisible by 3**parent_depth")
        if leaf_hidden % 2:
            raise ValueError("recursive leaf hidden width must be even")

        # The recursive body stores effective sparse weights in plain Linear
        # modules. Keep the runtime config truthful after construction instead
        # of restoring a ternary flag that no longer describes these modules.
        model_cfg.model.ternary.enabled = False
        super().__init__(model_cfg)

        from hagi.model.norms import BlockTreeNorm

        for block in self.blocks:
            block.attn.attn_norm = BlockTreeNorm(
                target_depth, leaf_hidden, model_cfg.model.norm_eps
            )
            block.mixer.norm = BlockTreeNorm(
                target_depth, leaf_hidden, model_cfg.model.norm_eps
            )
            child_leaf_count = 3**target_depth
            block.attn.branch_scale = RecursiveBranchScale(
                child_leaf_count, block.attn.branch_scale.residual_scale
            )
            block.mixer.mixer.branch_scale = RecursiveBranchScale(
                child_leaf_count, block.mixer.mixer.branch_scale.residual_scale
            )
        self.out_norm = BlockTreeNorm(target_depth, leaf_hidden, model_cfg.model.norm_eps)
        # The target model may enable the fresh candidate pyramid. HAGI has
        # already attached it; replacing norms does not change the mixer hook.

        self.parent_tree = TernaryF3Tree(actual_parent_depth, leaf_hidden)
        self.target_tree = _build_cross_parent_tree(selected_transform, model_cfg)
        self.parent_depth = actual_parent_depth
        self.ternary_depth = target_depth
        self.leaf_hidden = leaf_hidden
        self.expert_weight_source = selected_source
        self.lift_mode = selected_transform
        self.cross_parent_transform = selected_transform
        self.child_config_fingerprint = child_config_fingerprint
        self.logit_scale_source: float | None = None
        self.logit_scale_target: float | None = None
        self.transform_digest = self.target_tree.transform_digest
        self._register_provenance_buffers(
            child_config_fingerprint,
            self.transform_digest,
            1.0,
            1.0,
            selected_transform,
        )

        target_state = self.state_dict()
        target_keys = set(target_state)
        forbidden_target = sorted(
            key for key in target_keys if key.startswith("mixers.") or key.startswith("cortex.")
            or key.startswith("decision_head.")
        )
        if forbidden_target:
            raise ValueError(f"recursive target contains forbidden state keys: {forbidden_target}")
        assembled: dict[str, torch.Tensor] = dict(target_state)
        child_keys = keys
        missing = sorted(
            key
            for key in child_keys
            if key not in target_keys and ".adapters." not in key
        )
        if missing:
            raise ValueError(f"recursive child state keys are not constructible: {missing[:8]}")
        allowed_target_missing = sorted(
            key
            for key in target_keys
            if key not in child_keys
            and ".adapters." not in key
            and key not in provenance_keys
        )
        if allowed_target_missing:
            raise ValueError(f"recursive target state keys missing from children: {allowed_target_missing[:8]}")

        q_heads = m.attention.num_query_heads // 3
        kv_heads = m.attention.num_kv_heads // 3
        if m.attention.num_query_heads % 3 or m.attention.num_kv_heads % 3:
            raise ValueError("recursive attention head counts must be divisible by three")
        q_out = q_heads * m.attention.head_dim
        kv_out = kv_heads * m.attention.head_dim

        def validated_scalar(key: str) -> list[torch.Tensor]:
            values = [state[key] for state in child_states]
            if not values[0].is_floating_point():
                raise ValueError(f"recursive child scalar must be floating point: {key}")
            if any(not torch.isfinite(value).all() for value in values):
                raise ValueError(f"recursive child scalar is non-finite: {key}")
            # Validate the original signed tensor before abs/square operations.
            if any((value <= 0).any() for value in values):
                raise ValueError(f"recursive child scalar must be positive: {key}")
            return values

        # ``head_scale_divisor`` is the receiver gain of the chosen transform.
        # Legacy F3 aggregates the three children, so its declared receiver is
        # ``sum(s_i) / sqrt(3)``. The parent-preserving lift fixes the repeated
        # branch, so its exact equivalent composition is ``sum(s_i) / 3``:
        # ``(Q [h,h,h]) . (W_stack Q) = 3 (h.W)`` and the gain cancels it.
        if self.cross_parent_transform == "parent_preserving":
            head_scale_divisor = 3.0
        else:
            head_scale_divisor = math.sqrt(3.0)

        def head_logit_scale(key: str) -> torch.Tensor:
            values = validated_scalar(key)
            source = torch.stack(
                [value.detach().to(dtype=torch.float64).sum() for value in values]
            ).sum()
            if not torch.isfinite(source).all() or source.item() <= 0:
                raise ValueError("recursive head logit_scale sum is invalid: {key}")
            result = (source / head_scale_divisor).to(dtype=values[0].dtype)
            if result.numel() != 1 or not torch.isfinite(result).all() or result.item() <= 0:
                raise ValueError(
                    f"recursive head logit_scale is invalid after child-dtype cast: {key}"
                )
            return result

        def branch_scale(key: str) -> torch.Tensor:
            values = validated_scalar(key)
            if values[0].ndim == 0:
                leaves = [value.reshape(1) for value in values]
            elif values[0].ndim == 1:
                leaves = [value.reshape(-1) for value in values]
            else:
                raise ValueError(f"recursive branch scale must be scalar or vector: {key}")
            if any(leaf.shape != leaves[0].shape for leaf in leaves[1:]):
                raise ValueError(f"recursive branch scale shape mismatch: {key}")
            expected_child = 3**actual_parent_depth
            if tuple(leaves[0].shape) != (expected_child,):
                raise ValueError(
                    f"recursive child branch scale must have shape ({expected_child},): {key}"
                )
            return torch.cat(leaves, dim=0)

        if "head.logit_scale" not in keys:
            raise ValueError("recursive assembly requires head.logit_scale")
        head_scale_values = validated_scalar("head.logit_scale")
        head_scale_sum = sum(
            float(value.detach().to(dtype=torch.float64).sum())
            for value in head_scale_values
        )
        if not math.isfinite(head_scale_sum) or head_scale_sum <= 0:
            raise ValueError("recursive head logit_scale sum is invalid")

        for key in sorted(child_keys):
            if ".adapters." in key:
                continue
            values = [state[key] for state in child_states]
            if key == "encoder.embedding.weight":
                assembled[key] = torch.cat(values, dim=1)
            elif key == "head.projection.weight":
                # Oracle-approved receiver: the only receiver scale is
                # ``s_target = sum(s_i) / sqrt(3)``.  The projection itself is
                # C @ Q, where C concatenates gain-normalized child blocks.
                normalized_children = [
                    value.to(dtype=torch.float64)
                    * (
                        float(head_scale_values[index].detach().sum())
                        / head_scale_sum
                    )
                    for index, value in enumerate(values)
                ]
                base_target = torch.cat(normalized_children, dim=1)
                assembled[key] = self.target_tree.apply_row(base_target).to(
                    dtype=values[0].dtype
                )
            elif key == "head.logit_scale":
                value = head_logit_scale(key)
                self.logit_scale_source = head_scale_divisor * float(value.item())
                self.logit_scale_target = float(value.item())
                self.recursive_f3_logit_scale_source.copy_(
                    torch.tensor(self.logit_scale_source, dtype=torch.float64)
                )
                self.recursive_f3_logit_scale_target.copy_(
                    torch.tensor(self.logit_scale_target, dtype=torch.float64)
                )
                assembled[key] = value
            elif key == "head.log_prior":
                # Prior is a child-independent receiver offset; preserve the
                # first canonical value rather than inventing a new scale.
                assembled[key] = values[0]
            elif key == "out_norm.weight" or key.endswith(
                (".attn.attn_norm.weight", ".mixer.norm.weight")
            ):
                assembled[key] = _recursive_tree_norm_weights(values, target_depth)
            elif key.endswith(".attn.q_norm.weight") or key.endswith(".attn.k_norm.weight"):
                is_q = key.endswith("q_norm.weight")
                if values[0].ndim == 1:
                    repeat = q_heads if is_q else kv_heads
                    blocks = [value.unsqueeze(0).repeat(repeat, 1) for value in values]
                else:
                    blocks = values
                assembled[key] = torch.cat(blocks, dim=0)
            elif key.endswith(".attn.sink_bias"):
                assembled[key] = torch.cat(values, dim=1)
            elif key.endswith(".attn.qkv_proj.weight"):
                prepared = _recursive_prepare_blocks(values, m.ternary.eps, selected_source)
                expected = q_out + 2 * kv_out
                if any(value.shape != (expected, parent_hidden) for value in prepared):
                    raise ValueError("recursive child fused QKV geometry is inconsistent")
                q_blocks = [value[:q_out] for value in prepared]
                kv_blocks = [value[q_out:] for value in prepared]
                k_blocks = [value[:kv_out] for value in kv_blocks]
                v_blocks = [value[kv_out:] for value in kv_blocks]
                assembled[key] = torch.cat(
                    [_block_diag(q_blocks), _block_diag(k_blocks), _block_diag(v_blocks)],
                    dim=0,
                )
            elif key.endswith((".out_proj.weight", ".mixer.gate.weight", ".mixer.up.weight", ".mixer.down.weight")):
                prepared = _recursive_prepare_blocks(values, m.ternary.eps, selected_source)
                assembled[key] = _block_diag(prepared)
            elif key.endswith(".branch_scale.scale"):
                assembled[key] = branch_scale(key)
            elif values[0].ndim == 0:
                raise ValueError(f"unsupported recursive scalar state key: {key}")
            else:
                raise ValueError(f"unsupported recursive state key: {key}")

        self.load_state_dict(assembled, strict=True)
        if self.logit_scale_source is None or self.logit_scale_target is None:
            raise ValueError("recursive assembly requires head.logit_scale")

    def _apply_mixers(self, h: torch.Tensor) -> torch.Tensor:
        return self.target_tree.apply_row(h)


def merge_recursive_f3(
    cfg: Config,
    child_states: list[Mapping[str, torch.Tensor]],
    *,
    child_configs: Sequence[Config],
    parent_depth: int | None = None,
    expert_weight_source: str | None = None,
    cross_parent_transform: str | None = None,
    drop_expert_mixers: bool = False,
) -> RecursiveF3HAGI:
    """Public construction seam for the opt-in recursive ternary F3 model.

    ``cross_parent_transform`` is passed through by name; see
    :class:`RecursiveF3HAGI` for the accepted values.
    """
    return RecursiveF3HAGI(
        cfg,
        child_states,
        child_configs=child_configs,
        parent_depth=parent_depth,
        expert_weight_source=expert_weight_source,
        cross_parent_transform=cross_parent_transform,
        drop_expert_mixers=drop_expert_mixers,
    )


def build_model_from_payload(
    cfg: Config,
    state: Mapping[str, torch.Tensor],
    *,
    n_mixers: int = 1,
    mixer_init_scale: float = 0.0,
    device: str | torch.device = "cpu",
    cross_parent_transform: str | None = None,
) -> HAGI | MergedHAGI | RecursiveF3HAGI:
    """Build the checkpoint's model class without trusting config alone."""
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("model state must map names to tensors")
    recursive = RecursiveF3HAGI.is_recursive_state(state)
    if recursive:
        return RecursiveF3HAGI.from_state_dict(
            cfg, state, device=device, cross_parent_transform=cross_parent_transform
        )
    if cfg.merge.mixer_type == "ternary_f3":
        raise ValueError("ternary_f3 model state has no recursive provenance")
    if cfg.merge.enabled:
        model = MergedHAGI(cfg, n_mixers=n_mixers, mixer_init_scale=mixer_init_scale)
    else:
        model = HAGI(cfg)
    # A freshly constructed model carries random weights. Without this load the
    # caller would score an unrelated model and every measurement taken from it
    # would be noise. Validation stays delegated to the strict state load.
    model.load_state_dict(state, strict=True)
    return model.to(device)


def merge_experts(
    cfg: Config,
    expert_states: list[dict],
    n_mixers: int = 1,
    mixer_init_scale: float = 0.0,
    drop_expert_mixers: bool = False,
    expert_weight_source: str | None = None,
) -> MergedHAGI:
    """Build a merged model from N expert state dicts.

    Args:
        cfg: config with ``model.hidden_size = N * expert_hidden`` and
            ``merge.n_experts = N``.
        expert_states: list of N expert ``state_dict`` mappings (the ``model``
            payload of each checkpoint), in block order.
        n_mixers: number of cross-block mixers to add.
        mixer_init_scale: initial mixer gain (0 = exact independent experts).
        drop_expert_mixers: when True, ignore any ``mixers.*`` keys in the
            expert states (used for hierarchical merging: the level-1 experts
            are themselves merged models with trained mixers, which must be
            dropped and replaced by a fresh level-2 mixer).

    Returns:
        A :class:`MergedHAGI` with the experts' weights block-diagonally
        concatenated and zero-init mixers.
    """
    if str(getattr(cfg.merge, "mixer_type", "swiglu")) == "ternary_f3":
        raise ValueError(
            "merge_experts does not implement ternary_f3; use merge_recursive_f3"
        )
    selected_weight_source = (
        cfg.merge.expert_weight_source if expert_weight_source is None else expert_weight_source
    )
    if type(selected_weight_source) is not str or selected_weight_source not in {
        "ternary_master",
        "effective_sparse",
    }:
        raise ValueError(
            "expert_weight_source must be 'ternary_master' or 'effective_sparse'"
        )
    model_cfg = copy.deepcopy(cfg)
    model_cfg.merge.expert_weight_source = selected_weight_source

    n = model_cfg.merge.n_experts
    if len(expert_states) != n:
        raise ValueError(f"expected {n} expert states, got {len(expert_states)}")
    m = model_cfg.model
    h = m.hidden_size
    if h % n != 0:
        raise ValueError(f"hidden_size {h} must be divisible by n_experts {n}")

    model = MergedHAGI(
        model_cfg,
        n_mixers=n_mixers,
        mixer_init_scale=mixer_init_scale,
    )
    sd = model.state_dict()

    def _is_level_local_state(key: str) -> bool:
        return (
            key.startswith("cortex.")
            or key.startswith("decision_head.")
            or ".adapters." in key
        )

    def _is_dropped_mixer_state(key: str) -> bool:
        return drop_expert_mixers and key.startswith("mixers.")

    # Group expert tensors by key. Each expert has the same key set.
    keys = list(expert_states[0].keys())
    for k in keys:
        if _is_level_local_state(k) or _is_dropped_mixer_state(k):
            # Model-global cortex and per-block adaptive contours are not
            # expert-width tensors. The merged level starts with fresh
            # zero-init adaptive state, just like its fresh cross-expert mixer.
            continue
        for st in expert_states:
            if k not in st:
                raise ValueError(f"expert state missing key {k!r}")
            if tuple(st[k].shape) != tuple(expert_states[0][k].shape):
                raise ValueError(f"expert shape mismatch on {k!r}: {tuple(st[k].shape)}")

    for k in keys:
        if k not in sd:
            continue
        if _is_level_local_state(k) or _is_dropped_mixer_state(k):
            # Drop the expert-level contour and replace it with the fresh
            # merged-level module initialized by MergedHAGI.
            continue
        target = sd[k]
        blocks = [st[k] for st in expert_states]
        # Buffers that are shared across the whole model (not per-expert) keep
        # the merged model's own value: the unigram log-prior is a function of
        # the vocabulary, not of any single expert.
        if k.endswith("log_prior"):
            continue
        if target.ndim == 4 and k.endswith("sink_bias"):
            # Learnable attention-sink bias is [1, n_heads, 1, sink_len]; each
            # expert contributes its own heads, so concatenate along the head
            # axis (dim=1). This mirrors the per-head QK gains handling below.
            merged = torch.cat(blocks, dim=1)
        elif target.ndim == 2 and (k.endswith("q_norm.weight") or k.endswith("k_norm.weight")):
            # Per-head QK gains: each expert's gain applies to its own heads.
            # The merged model has per_head_qk=True, so the target is
            # [n_heads, head_dim]. Each expert has q_per_exp (or kv_per_exp)
            # heads, so repeat each expert's gain that many times. For a
            # hierarchical merge the experts are themselves merged models with
            # 2D per-head gains [n_heads_exp, head_dim]; concatenate those
            # along the head axis instead of repeating.
            if blocks[0].ndim == 2:
                merged = torch.cat(blocks, dim=0)
            else:
                q_per_exp = m.attention.num_query_heads // n
                kv_per_exp = m.attention.num_kv_heads // n
                rep = q_per_exp if k.endswith("q_norm.weight") else kv_per_exp
                merged = torch.cat([b.unsqueeze(0).repeat(rep, 1) for b in blocks], dim=0)
        elif target.ndim == 2 and (
            k.endswith("attn_norm.weight")
            or k.endswith("mixer.norm.weight")
            or k.endswith("out_norm.weight")
        ):
            # Block-wise RMSNorm gains: [n_blocks, block_dim]. Stack the
            # experts' [block_dim] gains along the block axis. For a
            # hierarchical merge the experts are themselves merged models with
            # 2D block norms [n_blocks, block_dim]; each expert's blocks must be
            # flattened into a single contiguous block (its own hidden width)
            # and stacked along the block axis, so the merged model's
            # [n_experts, expert_hidden] norm applies per expert.
            if blocks[0].ndim == 2:
                merged = torch.stack([b.reshape(-1) for b in blocks], dim=0)
            else:
                merged = torch.stack(blocks, dim=0)
        elif target.ndim == 2:
            # Hidden-mixing matrices (qkv/out/gate/up/down) merge block-diagonal;
            # codebooks and head projections merge row-wise (concat over input).
            if k.endswith((".weight",)) and (
                "qkv_proj" in k or "out_proj" in k or "mixer.gate" in k or "mixer.up" in k or "mixer.down" in k
            ):
                # The experts' BitLinear layers ternarize their weights at
                # forward time. The merged body uses plain fp16 linear layers,
                # so to reproduce the experts exactly we ternarize each expert
                # block *individually* (per-block absmean, matching the expert's
                # own per-row normalization) before the block-diagonal merge.
                if selected_weight_source == "ternary_master":
                    tern_blocks = [
                        _ternarize_block(b, m.ternary.eps) for b in blocks
                    ]
                else:
                    tern_blocks = blocks
                if "qkv_proj" in k:
                    # The fused QKV projection has a q block on top and a kv
                    # block below. Each expert contributes its own q and kv
                    # blocks; they must be merged separately so all q blocks
                    # stay on top and all kv blocks below (the merged model's
                    # layout), not interleaved along the diagonal.
                    q_per_exp = m.attention.num_query_heads // n
                    q_out = q_per_exp * m.attention.head_dim
                    q_blocks = [b[:q_out] for b in tern_blocks]
                    kv_blocks = [b[q_out:] for b in tern_blocks]
                    # Each expert's kv block is laid out as [k_all, v_all]
                    # (2 * n_kv_exp * head_dim). The merged model views the
                    # whole kv region as [B, T, 2, n_kv, head_dim], i.e. all
                    # k heads first, then all v heads. So we must collect all
                    # k blocks, then all v blocks -- not interleave per expert.
                    n_kv_exp = m.attention.num_kv_heads // n
                    hd = m.attention.head_dim
                    k_blocks = [b[: n_kv_exp * hd] for b in kv_blocks]
                    v_blocks = [b[n_kv_exp * hd :] for b in kv_blocks]
                    merged = torch.cat(
                        [
                            _block_diag(q_blocks),
                            _block_diag(k_blocks),
                            _block_diag(v_blocks),
                        ],
                        dim=0,
                    )
                else:
                    merged = _merge_2d(tern_blocks, block_diag=True)
            else:
                merged = _merge_2d(blocks, block_diag=False)
        elif target.ndim == 1:
            # Per-hidden-dim gains (attn_norm, mixer.norm, out_norm) concatenate
            # so each expert's norm applies to its own block.
            merged = _merge_1d(blocks)
        else:
            # Scalars (branch_scale, logit_scale): take the first expert's value.
            # logit_scale must be divided by sqrt(N): the merged head projection
            # is the column-concatenation of the experts' codebooks, so
            # ``hidden @ weight.T`` sums N contributions of roughly equal
            # magnitude. Without the 1/sqrt(N) the logits are ~Nx sharper and
            # the output distribution collapses (too confident), which is why a
            # freshly merged model scores poorly (AVG~9) before joint training.
            # This mirrors grow.py's ``_fill_head`` (ref_scale / sqrt(N)).
            if k.endswith("logit_scale"):
                merged = blocks[0] / math.sqrt(n)
            else:
                merged = blocks[0]
        if tuple(merged.shape) != tuple(target.shape):
            raise ValueError(
                f"merged shape {tuple(merged.shape)} != target {tuple(target.shape)} for {k!r}"
            )
        sd[k] = merged

    # With a Hadamard mixer the hidden stream is rotated by
    # ``Q = (H_n/sqrt(n)) ⊗ I_H`` at the mixer. The head projection is
    # ``hidden @ weight.T``, so to keep the step-0 logits identical to the
    # block-diagonal merge (each expert's head acting on its own block) the
    # merged head weight must be right-multiplied by ``Q`` (symmetric). This
    # makes the Hadamard mixer's fixed mixing *consistent* with the head at
    # step 0: the mixer re-mixes the stream, and the head is pre-rotated to
    # match, so the merged model is not degraded before joint training.
    #
    # Both the tied codebook (``encoder.embedding.weight``) and the untied
    # projection (``head.projection.weight``) are ``[V, H]`` and both feed the
    # logits, so both must be rotated. ``head.log_prior`` is a function of the
    # vocabulary, not of the hidden space, and is left untouched.
    if str(getattr(cfg.merge, "mixer_type", "swiglu")) == "hadamard" and n > 1:
        group_sizes = getattr(cfg.merge, "mixer_hadamard_groups", None)
        if group_sizes is not None:
            group_sizes = [int(g) for g in group_sizes]
        # Only the *output* head must be pre-rotated. The input embedding
        # (``encoder.embedding.weight``) feeds the first block, not the head,
        # so it must NOT be rotated — rotating it would change the input
        # representation and break the block-diagonal equivalence. When
        # ``tie_lm_head`` is true the embedding doubles as the head, so it is
        # rotated then; otherwise only ``head.projection.weight`` is.
        if getattr(cfg.model, "tie_lm_head", False):
            if "encoder.embedding.weight" in sd:
                sd["encoder.embedding.weight"] = hadamard_apply_2d(
                    sd["encoder.embedding.weight"], n, group_sizes
                )
        if "head.projection.weight" in sd:
            sd["head.projection.weight"] = hadamard_apply_2d(
                sd["head.projection.weight"], n, group_sizes
            )

    # Zero-init the mixer gains (already 0 from constructor) and keep the
    # merged model's own out_norm / head as-is (they are the merged versions).
    model.load_state_dict(sd, strict=True)
    return model
