"""Ternary route-indirect bank kernel for the RELEASE bank layout.

k_h13t: one launch per layer per token for ALL top-k experts. Reads w13t
(5 trits/byte base-3) via LUT gather, per-g128 scales, soft_lim, silu*,
writes h [T, I]. Pair with k_yb (int4 W2, scripts/triton_bank_kernels.py).

Measured (gfx1151, layer10, T=6): h13 940 us + y 280 us = 1.2 ms/layer
vs ~45 ms per-expert bf16-unpack -> ~35x. Full-model projection ~52 ms/token
MoE-bound (~19 tok/s ceiling; measured end-to-end lower).

Pitfall (cost a debug session): byte index of trit k is (k0+ks)//5 - NOT
k0//5 + ks//5 (k0 % 5 != 0). Integer div decode is slow on gfx1151;
LUT gather (v*5+slot -> int8 trit) is the fast path.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _soft_lim(x, LIM: tl.constexpr, KNEE: tl.constexpr):
    th = KNEE * LIM
    tail = LIM - th
    ax = tl.abs(x)
    y_abs = tl.where(ax <= th, ax, th + tail * libdevice.tanh((ax - th) / tail))
    return y_abs * tl.where(x >= 0, 1.0, -1.0)


@triton.jit
def k_h13t(
    z_ptr,        # [D] f32 pre-rotated token
    ids_ptr,      # [T] int32 expert ids (GPU)
    w13_ptr,      # u8 [256, 2I, T3] ternary 5 trits/byte base-3
    s13_ptr,      # f32 [256, 2I, D/GS]
    b13_ptr,      # f32 [256, 2I]
    lut_ptr,      # i8 [256, 5] trit LUT
    h_ptr,        # [T, I] f32 out
    D: tl.constexpr, I: tl.constexpr, GS: tl.constexpr, BI: tl.constexpr,
):
    t = tl.program_id(0)
    ib = tl.program_id(1)
    e = tl.load(ids_ptr + t)
    T3: tl.constexpr = (D + 4) // 5
    NG: tl.constexpr = D // GS
    offs_i = tl.arange(0, BI) + ib * BI
    acc_g = tl.zeros((BI,), dtype=tl.float32)
    acc_u = tl.zeros((BI,), dtype=tl.float32)
    base13 = e * (2 * I * T3)
    ks = tl.arange(0, GS)
    for g in range(0, NG):
        kb = g * GS + ks
        byte = kb // 5
        sloti = (kb % 5).to(tl.int32)
        zf = tl.load(z_ptr + kb)
        b1 = tl.load(w13_ptr + base13 + offs_i[:, None] * T3 + byte[None, :]).to(tl.int32)
        b3 = tl.load(w13_ptr + base13 + (I + offs_i[:, None]) * T3 + byte[None, :]).to(tl.int32)
        t1 = tl.load(lut_ptr + b1 * 5 + sloti[None, :]).to(tl.float32)
        t3 = tl.load(lut_ptr + b3 * 5 + sloti[None, :]).to(tl.float32)
        sg1 = tl.load(s13_ptr + e * (2 * I * NG) + offs_i * NG + g).to(tl.float32)
        sg3 = tl.load(s13_ptr + e * (2 * I * NG) + (I + offs_i) * NG + g).to(tl.float32)
        # emulate the validated path exactly: W = (trits*scale).bf16, z.bf16,
        # products/accumulation fp32 (bf16 GEMM semantics)
        w1 = (t1 * sg1[:, None]).to(tl.bfloat16).to(tl.float32)
        w3 = (t3 * sg3[:, None]).to(tl.bfloat16).to(tl.float32)
        zb = zf.to(tl.bfloat16).to(tl.float32)
        acc_g += tl.sum(w1 * zb[None, :], axis=1)
        acc_u += tl.sum(w3 * zb[None, :], axis=1)
    bv1 = tl.load(b13_ptr + e * (2 * I) + offs_i).to(tl.bfloat16).to(tl.float32)
    bv3 = tl.load(b13_ptr + e * (2 * I) + I + offs_i).to(tl.bfloat16).to(tl.float32)
    # mirror torch bf16 op boundaries: g/u are bf16 tensors in the reference
    gv = _soft_lim((acc_g + bv1).to(tl.bfloat16).to(tl.float32), 10.0, 0.75)
    uv = _soft_lim((acc_u + bv3).to(tl.bfloat16).to(tl.float32), 10.0, 0.75)
    gv = gv.to(tl.bfloat16).to(tl.float32)
    uv = uv.to(tl.bfloat16).to(tl.float32)
    h = (gv * tl.sigmoid(gv)).to(tl.bfloat16).to(tl.float32) * uv
    h = h.to(tl.bfloat16).to(tl.float32)
    tl.store(h_ptr + t * I + offs_i, h)


def trit_lut(device="cuda") -> torch.Tensor:
    """uint8 value -> 5 base-3 trits {-1,0,1} little-end, [256,5] int8."""
    return torch.tensor([[((v // 3 ** j) % 3) - 1 for j in range(5)]
                         for v in range(256)], dtype=torch.int8,
                        device=device).contiguous()


@triton.jit
def k_h13t6(
    z_ptr, ids_ptr,
    w13_0, w13_1, w13_2, w13_3, w13_4, w13_5,
    s13_0, s13_1, s13_2, s13_3, s13_4, s13_5,
    b13_0, b13_1, b13_2, b13_3, b13_4, b13_5,
    lut_ptr, h_ptr,
    D: tl.constexpr, I: tl.constexpr, GS: tl.constexpr, BI: tl.constexpr,
):
    """TOP_K=6 unrolled: reads each expert's packed tensors directly (no
    stacking). ids_ptr values select 0..5 -> pointer via static branches."""
    t = tl.program_id(0)
    ib = tl.program_id(1)
    e = tl.load(ids_ptr + t)  # 0..5
    T3: tl.constexpr = (D + 4) // 5
    NG: tl.constexpr = D // GS
    offs_i = tl.arange(0, BI) + ib * BI
    acc_g = tl.zeros((BI,), dtype=tl.float32)
    acc_u = tl.zeros((BI,), dtype=tl.float32)
    ks = tl.arange(0, GS)
    sel = (e == 0).to(tl.int32) + 2 * (e == 1).to(tl.int32) + 3 * (e == 2).to(tl.int32) + 4 * (e == 3).to(tl.int32) + 5 * (e == 4).to(tl.int32)
    for g in range(0, NG):
        kb = g * GS + ks
        byte = kb // 5
        sloti = (kb % 5).to(tl.int32)
        zf = tl.load(z_ptr + kb)
        # select pointers (compile-time branch on runtime value is not allowed;
        # instead compute offsets relative to each tensor and use tl.where on
        # a per-program basis via scalar checks)
        # Triton supports scalar if on tensor-derived scalar? No. Use nested
        # masks over 6 loads with flag = (e == j): wasteful x6 loads.
        # Practical approach: launch grid (1, tiles) per expert t already
        # separates t -> pass per-expert args by launching 6 small grids is
        # back to 6 launches. Instead: accept x6 masked loads (cheap u8 loads).
        v1a = tl.load(w13_0 + offs_i[:, None] * T3 + byte[None, :]).to(tl.int32)
        v1b = tl.load(w13_1 + offs_i[:, None] * T3 + byte[None, :]).to(tl.int32)
        v1c = tl.load(w13_2 + offs_i[:, None] * T3 + byte[None, :]).to(tl.int32)
        v1d = tl.load(w13_3 + offs_i[:, None] * T3 + byte[None, :]).to(tl.int32)
        v1e = tl.load(w13_4 + offs_i[:, None] * T3 + byte[None, :]).to(tl.int32)
        v1f = tl.load(w13_5 + offs_i[:, None] * T3 + byte[None, :]).to(tl.int32)
        v1 = tl.where(sel == 0, v1a, tl.where(sel == 1, v1b, tl.where(sel == 2, v1c, tl.where(sel == 3, v1d, tl.where(sel == 4, v1e, v1f)))))
        # NOTE: x6 load amplification is bandwidth x6 on W13; W13 slice is only
        # 1.7MB x6 = 10MB per layer per token - acceptable vs launch savings.
        sg1 = tl.load(s13_0 + offs_i * NG + g)
        sg1 = tl.where(sel == 0, sg1, tl.load(s13_1 + offs_i * NG + g))
        sg1 = tl.where(sel == 1, sg1, tl.load(s13_2 + offs_i * NG + g))
        sg1 = tl.where(sel == 2, sg1, tl.load(s13_3 + offs_i * NG + g))
        sg1 = tl.where(sel == 3, sg1, tl.load(s13_4 + offs_i * NG + g))
        sg1 = tl.where(sel == 4, sg1, tl.load(s13_5 + offs_i * NG + g))
        t1 = tl.load(lut_ptr + v1 * 5 + sloti[None, :]).to(tl.float32)
        zf_b = zf.to(tl.bfloat16).to(tl.float32)
        w1 = (t1 * sg1.to(tl.float32)[:, None]).to(tl.bfloat16).to(tl.float32)
        acc_g += tl.sum(w1 * zf_b[None, :], axis=1)
        # U branch analogous with second half rows: handled by caller passing
        # w13 already split; here we also load U halves:
        # (kept symmetrical below via s13/b13 index offset NG*I)
        sg3 = tl.load(s13_0 + (I + offs_i) * NG + g)
        sg3 = tl.where(sel == 0, sg3, tl.load(s13_1 + (I + offs_i) * NG + g))
        sg3 = tl.where(sel == 1, sg3, tl.load(s13_2 + (I + offs_i) * NG + g))
        sg3 = tl.where(sel == 2, sg3, tl.load(s13_3 + (I + offs_i) * NG + g))
        sg3 = tl.where(sel == 3, sg3, tl.load(s13_4 + (I + offs_i) * NG + g))
        sg3 = tl.where(sel == 4, sg3, tl.load(s13_5 + (I + offs_i) * NG + g))
        # w3 lives in the same w13 tensor rows I..2I:
        v3 = tl.load(w13_0 + (I + offs_i[:, None]) * T3 + byte[None, :]).to(tl.int32)
        v3 = tl.where(sel == 0, v3, tl.load(w13_1 + (I + offs_i[:, None]) * T3 + byte[None, :]).to(tl.int32))
        v3 = tl.where(sel == 1, v3, tl.load(w13_2 + (I + offs_i[:, None]) * T3 + byte[None, :]).to(tl.int32))
        v3 = tl.where(sel == 2, v3, tl.load(w13_3 + (I + offs_i[:, None]) * T3 + byte[None, :]).to(tl.int32))
        v3 = tl.where(sel == 3, v3, tl.load(w13_4 + (I + offs_i[:, None]) * T3 + byte[None, :]).to(tl.int32))
        v3 = tl.where(sel == 4, v3, tl.load(w13_5 + (I + offs_i[:, None]) * T3 + byte[None, :]).to(tl.int32))
        t3 = tl.load(lut_ptr + v3 * 5 + sloti[None, :]).to(tl.float32)
        w3 = (t3 * sg3.to(tl.float32)[:, None]).to(tl.bfloat16).to(tl.float32)
        acc_u += tl.sum(w3 * zf_b[None, :], axis=1)
    # biases
    bb0 = tl.load(b13_0 + offs_i); bb1 = tl.load(b13_1 + offs_i); bb2 = tl.load(b13_2 + offs_i)
    bb3 = tl.load(b13_3 + offs_i); bb4 = tl.load(b13_4 + offs_i); bb5 = tl.load(b13_5 + offs_i)
    bg = tl.where(sel == 0, bb0, tl.where(sel == 1, bb1, tl.where(sel == 2, bb2, tl.where(sel == 3, bb3, tl.where(sel == 4, bb4, bb5)))))
    bu = tl.where(sel == 0, tl.load(b13_0 + I + offs_i), tl.where(sel == 1, tl.load(b13_1 + I + offs_i), tl.where(sel == 2, tl.load(b13_2 + I + offs_i), tl.where(sel == 3, tl.load(b13_3 + I + offs_i), tl.where(sel == 4, tl.load(b13_4 + I + offs_i), tl.load(b13_5 + I + offs_i))))))
    gv = _soft_lim((acc_g + bg.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32), 10.0, 0.75)
    uv = _soft_lim((acc_u + bu.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32), 10.0, 0.75)
    gv = gv.to(tl.bfloat16).to(tl.float32)
    uv = uv.to(tl.bfloat16).to(tl.float32)
    h = (gv * tl.sigmoid(gv)).to(tl.bfloat16).to(tl.float32) * uv
    tl.store(h_ptr + t * I + offs_i, h.to(tl.bfloat16).to(tl.float32))
