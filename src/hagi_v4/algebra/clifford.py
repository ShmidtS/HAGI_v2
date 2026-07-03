"""Clifford algebra Cl(3,0,0) geometric product.

Cl(3,0,0): three orthonormal basis vectors e1, e2, e3, each squaring to +1.
8 basis blades indexed by 3-bit bitmask (bit i set => e_{i+1} present):

    0b000 = 1            (grade 0, scalar)
    0b001 = e1           (grade 1)
    0b010 = e2           (grade 1)
    0b100 = e3           (grade 1)
    0b011 = e1 e2        (grade 2, bivector)
    0b101 = e1 e3        (grade 2, bivector)
    0b110 = e2 e3        (grade 2, bivector)
    0b111 = e1 e2 e3     (grade 3, trivector / pseudoscalar)

The geometric product of two basis blades a, b (bitmasks):
    result_blade = a XOR b
    sign         = (-1)^(reordering transpositions)   [metric is all +1]

V4 cleanup: identical algebra to V3, no sandwich, clean docstrings.
"""

from __future__ import annotations

import torch

BLADE_COUNT = 8
METRIC = [1, 1, 1]

GRADE = [0, 1, 1, 2, 1, 2, 2, 3]
GRADE_DIMS = (64, 96, 96, 64, 256)
GRADE_SLICES = [(0, 64), (64, 160), (160, 256), (256, 320), (320, 576)]

REVERSE_SIGNS = [1, 1, 1, -1, 1, -1, -1, -1]

_G02_INDICES = [i for i in range(BLADE_COUNT) if GRADE[i] in (0, 2)]


def _reordering_sign(a: int, b: int) -> int:
    """Sign from reordering the product of two basis blades into canonical order.

    Counts transpositions needed to sort the concatenated basis vectors.
    Metric is Euclidean (+1) so shared indices contribute no extra sign.
    """
    a >>= 1
    swaps = 0
    while a:
        swaps += bin(a & b).count("1")
        a >>= 1
    return -1 if (swaps & 1) else 1


def _build_product_table() -> tuple[torch.Tensor, torch.Tensor]:
    """Build the Cl(3,0,0) Cayley table.

    Returns:
        out_index: [8, 8] long tensor, out_index[a, b] = resulting blade index.
        sign:      [8, 8] float tensor, sign[a, b] = +1 or -1.
    """
    out_index = torch.zeros(BLADE_COUNT, BLADE_COUNT, dtype=torch.long)
    sign = torch.zeros(BLADE_COUNT, BLADE_COUNT, dtype=torch.float32)
    for a in range(BLADE_COUNT):
        for b in range(BLADE_COUNT):
            out_index[a, b] = a ^ b
            sign[a, b] = float(_reordering_sign(a, b))
    return out_index, sign


_OUT_INDEX, _SIGN = _build_product_table()

_STRUCT = torch.zeros(BLADE_COUNT, BLADE_COUNT, BLADE_COUNT, dtype=torch.float32)
for _a in range(BLADE_COUNT):
    for _b in range(BLADE_COUNT):
        _STRUCT[_a, _b, int(_OUT_INDEX[_a, _b])] = _SIGN[_a, _b]

_STRUCT_TRITON = _STRUCT.permute(2, 0, 1).contiguous()

_SELF_PROD_G02_STRUCT = _STRUCT_TRITON[_G02_INDICES].contiguous()

_SCATTER_G02 = torch.zeros(BLADE_COUNT, len(_G02_INDICES), dtype=torch.float32)
for _i, _idx in enumerate(_G02_INDICES):
    _SCATTER_G02[_idx, _i] = 1.0

_STRUCT_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_SELF_PROD_G02_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_SCATTER_G02_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_GRADE_MASK_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
_REVERSE_SIGNS_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _prime_caches() -> None:
    """Pre-fill caches for the default CUDA device to avoid recompilation."""
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda", 0)
    for dt in (torch.bfloat16, torch.float32, torch.float16):
        _STRUCT_CACHE[(dev, dt)] = _STRUCT_TRITON.to(device=dev, dtype=dt)
        _SELF_PROD_G02_CACHE[(dev, dt)] = _SELF_PROD_G02_STRUCT.to(device=dev, dtype=dt)
        _SCATTER_G02_CACHE[(dev, dt)] = _SCATTER_G02.to(device=dev, dtype=dt)
        _REVERSE_SIGNS_CACHE[(dev, dt)] = torch.tensor(REVERSE_SIGNS, dtype=dt, device=dev)
        for grade in range(4):
            _GRADE_MASK_CACHE[(dev, dt, grade)] = torch.tensor(
                [1.0 if GRADE[i] == grade else 0.0 for i in range(BLADE_COUNT)],
                dtype=dt,
                device=dev,
            )


_prime_caches()


def _get_struct(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _STRUCT_CACHE:
        _STRUCT_CACHE[key] = _STRUCT_TRITON.to(device=device, dtype=dtype)
    return _STRUCT_CACHE[key]


def _get_self_prod_struct(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _SELF_PROD_G02_CACHE:
        _SELF_PROD_G02_CACHE[key] = _SELF_PROD_G02_STRUCT.to(device=device, dtype=dtype)
    return _SELF_PROD_G02_CACHE[key]


def _get_scatter(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _SCATTER_G02_CACHE:
        _SCATTER_G02_CACHE[key] = _SCATTER_G02.to(device=device, dtype=dtype)
    return _SCATTER_G02_CACHE[key]


def _get_grade_mask(device: torch.device, dtype: torch.dtype, grade: int) -> torch.Tensor:
    key = (device, dtype, grade)
    if key not in _GRADE_MASK_CACHE:
        _GRADE_MASK_CACHE[key] = torch.tensor(
            [1.0 if GRADE[i] == grade else 0.0 for i in range(BLADE_COUNT)],
            dtype=dtype,
            device=device,
        )
    return _GRADE_MASK_CACHE[key]


def _get_reverse_signs(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _REVERSE_SIGNS_CACHE:
        _REVERSE_SIGNS_CACHE[key] = torch.tensor(REVERSE_SIGNS, dtype=dtype, device=device)
    return _REVERSE_SIGNS_CACHE[key]


def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geometric product of two batched multivectors.

    Args:
        a: [..., 8] multivector coefficients.
        b: [..., 8] multivector coefficients.

    Returns:
        [..., 8] product coefficients.
    """
    struct = _get_struct(a.device, a.dtype)
    return torch.einsum("cab,...a,...b->...c", struct, a, b)


def geometric_product_self_g02(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Grade-0 and grade-2 projections of the self-product x*x in one fused step.

    Computes only the 4 needed output blades (grade 0: blade 0; grade 2:
    blades 3,5,6) instead of all 8, cutting self-product compute ~50%.

    Args:
        x: [..., 8] multivector coefficients.

    Returns:
        (g0, g2) as full 8-blade tensors. g0 has only blade 0 non-zero;
        g2 has only blades 3,5,6 non-zero.
    """
    struct = _get_self_prod_struct(x.device, x.dtype)
    scatter = _get_scatter(x.device, x.dtype)
    reduced = torch.einsum("rab,...a,...b->...r", struct, x, x)
    full = torch.einsum("cr,...r->...c", scatter, reduced)
    g0_mask = _get_grade_mask(x.device, x.dtype, 0)
    g2_mask = _get_grade_mask(x.device, x.dtype, 2)
    return full * g0_mask, full * g2_mask


def grade_projection(x: torch.Tensor, grade: int) -> torch.Tensor:
    """Zero out all blades not of the given grade. Returns [..., 8]."""
    mask = _get_grade_mask(x.device, x.dtype, grade)
    return x * mask


def reverse_mv(x: torch.Tensor) -> torch.Tensor:
    """Clifford reverse: sign (-1)^(k(k-1)/2) per grade k. Returns [..., 8]."""
    signs = _get_reverse_signs(x.device, x.dtype)
    return x * signs
