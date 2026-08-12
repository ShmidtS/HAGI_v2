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

import math

import torch
from torch import nn

from hagi.config import Config
from hagi.model.ffn import BranchScale
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
    experts' BitLinear forward. Returns the effective quantized weight."""
    scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    return (weight / scale).clamp(-1.0, 1.0).round() * scale


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


def merge_experts(
    cfg: Config,
    expert_states: list[dict],
    n_mixers: int = 1,
    mixer_init_scale: float = 0.0,
    drop_expert_mixers: bool = False,
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
    n = cfg.merge.n_experts
    if len(expert_states) != n:
        raise ValueError(f"expected {n} expert states, got {len(expert_states)}")
    m = cfg.model
    h = m.hidden_size
    if h % n != 0:
        raise ValueError(f"hidden_size {h} must be divisible by n_experts {n}")

    model = MergedHAGI(cfg, n_mixers=n_mixers, mixer_init_scale=mixer_init_scale)
    sd = model.state_dict()

    # Group expert tensors by key. Each expert has the same key set.
    keys = list(expert_states[0].keys())
    for k in keys:
        if drop_expert_mixers and k.startswith("mixers."):
            # The experts' own mixers are dropped and replaced by a fresh
            # level-2 mixer, so their geometry may differ across experts (e.g.
            # a SwiGLU-trained expert vs a Hadamard-trained one). Skip the
            # shape check for them.
            continue
        for st in expert_states:
            if k not in st:
                raise ValueError(f"expert state missing key {k!r}")
            if tuple(st[k].shape) != tuple(expert_states[0][k].shape):
                raise ValueError(f"expert shape mismatch on {k!r}: {tuple(st[k].shape)}")

    for k in keys:
        if k not in sd:
            continue
        if drop_expert_mixers and k.startswith("mixers."):
            # Drop the experts' own mixers; the fresh level-2 mixer (zero-init
            # from the constructor) replaces them.
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
                tern_blocks = [
                    _ternarize_block(b, m.ternary.eps) for b in blocks
                ]
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
