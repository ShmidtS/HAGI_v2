"""Route-indirect Triton kernels reading the CONTIGUOUS expert banks.

One launch per layer per token for ALL top-k experts: the kernel reads the
expert id from a GPU tensor (no host sync), computes bank offsets, and
dequantizes inside the K-loop. Requires scripts/build_bank.py layout:
  w13a u8 [256, 2I, D/8], s13 f32 [256, 2I], b13 f32 [256, 2I]
  w2a  u8 [256, D, I/2],  s2  f16 [256, D, I/GS]

Pitfalls honored (measured on gfx1151 / triton 3.5.1+rocm7.2):
  - uint8 >> 4 is ARITHMETIC -> widen to uint16 first
  - BK <= GS in k_y (one group scale per K-block)
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
def k_h13b(
    z_ptr,        # [D] fp32 pre-rotated token
    ids_ptr,      # [T] int32 top-k expert ids ON GPU
    w13_ptr,      # u8 [256, 2I, D/8]
    s13_ptr,      # f32 [256, 2I]
    b13_ptr,      # f32 [256, 2I]
    h_ptr,        # [T, I] fp32 out
    D: tl.constexpr, I: tl.constexpr, BI: tl.constexpr, BK: tl.constexpr,
):
    t = tl.program_id(0)
    ib = tl.program_id(1)
    e = tl.load(ids_ptr + t)  # expert id - stays on GPU, no host sync
    offs_i = ib * BI + tl.arange(0, BI)
    acc_g = tl.zeros((BI,), dtype=tl.float32)
    acc_u = tl.zeros((BI,), dtype=tl.float32)
    base13 = e * (2 * I * (D // 8)) + offs_i[:, None] * (D // 8)
    base13u = base13 + I * (D // 8)
    cols = tl.arange(0, BK // 8)[None, :]
    for k0 in range(0, D, BK):
        offs_k = k0 + tl.arange(0, BK)
        z = tl.load(z_ptr + offs_k).to(tl.float32)
        raw1 = tl.load(w13_ptr + base13 + (k0 // 8) + cols)
        bit1 = ((raw1[:, :, None] >> tl.arange(0, 8)[None, None, :]) & 1).to(tl.float32)
        w1 = bit1.reshape(BI, BK) * 2.0 - 1.0
        raw3 = tl.load(w13_ptr + base13u + (k0 // 8) + cols)
        bit3 = ((raw3[:, :, None] >> tl.arange(0, 8)[None, None, :]) & 1).to(tl.float32)
        w3 = bit3.reshape(BI, BK) * 2.0 - 1.0
        acc_g += tl.sum(z[None, :] * w1, axis=1)
        acc_u += tl.sum(z[None, :] * w3, axis=1)
    s1 = tl.load(s13_ptr + e * (2 * I) + offs_i)
    s3 = tl.load(s13_ptr + e * (2 * I) + I + offs_i)
    b1 = tl.load(b13_ptr + e * (2 * I) + offs_i)
    b3 = tl.load(b13_ptr + e * (2 * I) + I + offs_i)
    g = _soft_lim(acc_g * s1 + b1, 10.0, 0.8)
    u = _soft_lim(acc_u * s3 + b3, 10.0, 0.8)
    h = g * tl.sigmoid(g) * u
    tl.store(h_ptr + t * I + offs_i, h)


@triton.jit
def k_yb(
    h_ptr,        # [T, I] fp32
    ids_ptr,      # [T] int32
    w_ptr,        # [T] fp32 router weights
    w2_ptr,       # u8 [256, D, I/2]
    s2_ptr,       # f16 [256, D, I/GS]
    y_ptr,        # [T, D] fp32 scratch (ALWAYS written)
    out_ptr,      # [D] fp32 accumulator: out += w[t] * y[t]
    D: tl.constexpr, I: tl.constexpr, GS: tl.constexpr,
    BD: tl.constexpr, BK: tl.constexpr,
):
    t = tl.program_id(0)
    ob = tl.program_id(1)
    e = tl.load(ids_ptr + t)
    wscalar = tl.load(w_ptr + t).to(tl.float32)
    offs_o = ob * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), dtype=tl.float32)
    base2 = e * (D * (I // 2)) + offs_o[:, None] * (I // 2)
    cols = tl.arange(0, BK // 2)[None, :]
    for k0 in range(0, I, BK):
        offs_k = k0 + tl.arange(0, BK)
        hv = tl.load(h_ptr + t * I + offs_k).to(tl.float32)
        raw = tl.load(w2_ptr + base2 + (k0 // 2) + cols).to(tl.uint16)
        lo = (raw & 15).to(tl.float32) - 8.0
        hi = (raw >> 4).to(tl.float32) - 8.0
        wq = tl.join(lo, hi).reshape(BD, BK)
        sg = tl.load(s2_ptr + e * (D * (I // GS)) + offs_o * (I // GS) + (k0 // GS)).to(tl.float32)
        acc += tl.sum(hv[None, :] * (wq * sg[:, None]), axis=1)
    tl.store(y_ptr + t * D + offs_o, acc)
    tl.atomic_add(out_ptr + offs_o, acc * wscalar)
