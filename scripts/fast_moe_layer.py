"""Drop-in fast MoE layer for the release loader/TTT decode path.

fast_layer_moe(z, ids, wts, bank, lut, h, y, out) -> out (+= weighted sums):
  z   [D] f32 pre-rotated token
  ids [T] int32 top-k expert ids (GPU)
  wts [T] f32 router weights
  bank: dict of w13t/s13/b13/w2a/s2h tensors on device
Returns acc [D] f32 = sum_t wts[t] * expert(ids[t])(z).
"""
import sys

import torch

sys.path.insert(0, "scripts")
import triton_bank_kernels as tbk
from ternary_bank_kernels import k_h13t


def fast_layer_moe(z, ids, wts, bank, lut, h, y, out):
    T = ids.shape[0]
    D, I, GS = 4096, 2048, 128
    out.zero_()
    k_h13t[(T, I // 64)](z, ids, bank["w13t"], bank["s13"], bank["b13"], lut, h,
                         D=D, I=I, GS=GS, BI=64, num_warps=2)
    tbk.k_yb[(T, D // 512)](h, ids, wts, bank["w2a"], bank["s2h"], y, out,
                            D=D, I=I, GS=GS, BD=512, BK=256, num_warps=8)
    return out
