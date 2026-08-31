"""Triton kernels for i1i4 expert decode (WSL + ROCm gfx1151).

Reads the PACKED checkpoint tensors DIRECTLY (per-expert tensors already
resident on GPU via I4X_PACKED - no unpack to bf16, no LRU dequant cache):

  k_h13:  h = silu(soft_lim(z@W1q^T+b1)) * soft_lim(z@W3q^T+b3)
          W1/W3 packed BINARY uint8 [I, D/8] (1 bit/sign, LSB-first per byte),
          per-row fp32 scales s1/s3 [I], fp32 biases b1/b3 [I].
          soft_lim: identity below knee*lim, tanh rolloff to lim (refit-exact).
  k_y:    y = h @ (int4 W2)^T; optionally ACCUMULATES w*y into out (router
          weighted sum fused). W2 packed int4 uint8 [D, I/2] (+8 offset,
          lo nibble = even column), per-(row, g128) fp32 scales [D, I/GS].
          BK <= GS (one group scale per K-block; measured pitfall) and
          uint8>>4 is ARITHMETIC on this stack -> widen to uint16 first.

Lineage: scripts/proto_triton_i1i4.py (verified 0.000% rel err).
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
def k_h13(
    z_ptr,                     # [D] fp32 pre-rotated input (one token)
    w1_ptr, s1_ptr, b1_ptr,    # bin [I, D/8] u8 | fp32 [I] | fp32 [I]
    w3_ptr, s3_ptr, b3_ptr,
    h_ptr,                     # [I] fp32 out
    D: tl.constexpr, I: tl.constexpr, BI: tl.constexpr, BK: tl.constexpr,
):
    ib = tl.program_id(0)
    offs_i = ib * BI + tl.arange(0, BI)
    acc_g = tl.zeros((BI,), dtype=tl.float32)
    acc_u = tl.zeros((BI,), dtype=tl.float32)
    s1 = tl.load(s1_ptr + offs_i)
    s3 = tl.load(s3_ptr + offs_i)
    for k0 in range(0, D, BK):
        offs_k = k0 + tl.arange(0, BK)
        z = tl.load(z_ptr + offs_k).to(tl.float32)
        cols = tl.arange(0, BK // 8)[None, :]
        base1 = offs_i[:, None] * (D // 8) + (k0 // 8)
        raw1 = tl.load(w1_ptr + base1 + cols)
        bit1 = ((raw1[:, :, None] >> tl.arange(0, 8)[None, None, :]) & 1).to(tl.float32)
        w1 = bit1.reshape(BI, BK) * 2.0 - 1.0
        acc_g += tl.sum(z[None, :] * (w1 * s1[:, None]), axis=1)
        base3 = offs_i[:, None] * (D // 8) + (k0 // 8)
        raw3 = tl.load(w3_ptr + base3 + cols)
        bit3 = ((raw3[:, :, None] >> tl.arange(0, 8)[None, None, :]) & 1).to(tl.float32)
        w3 = bit3.reshape(BI, BK) * 2.0 - 1.0
        acc_u += tl.sum(z[None, :] * (w3 * s3[:, None]), axis=1)
    b1 = tl.load(b1_ptr + offs_i)
    b3 = tl.load(b3_ptr + offs_i)
    g = _soft_lim(acc_g + b1, 10.0, 0.8)
    u = _soft_lim(acc_u + b3, 10.0, 0.8)
    h = g * tl.sigmoid(g) * u
    tl.store(h_ptr + offs_i, h)


@triton.jit
def k_y(
    h_ptr,        # [I] fp32
    w2_ptr,       # int4 [D, I/2] u8
    s2_ptr,       # fp32 [D, I/GS]
    y_ptr,        # [D] fp32 scratch (ALWAYS written; pass h_ptr if unused)
    out_ptr,      # [D] fp32 accumulator: out += w_scalar * y
    w_scalar,     # router weight (fp32)
    D: tl.constexpr, I: tl.constexpr, GS: tl.constexpr,
    BD: tl.constexpr, BK: tl.constexpr,
):
    ob = tl.program_id(0)
    offs_o = ob * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), dtype=tl.float32)
    cols = tl.arange(0, BK // 2)[None, :]
    for k0 in range(0, I, BK):
        offs_k = k0 + tl.arange(0, BK)
        hv = tl.load(h_ptr + offs_k).to(tl.float32)
        base = offs_o[:, None] * (I // 2) + (k0 // 2)
        raw = tl.load(w2_ptr + base + cols).to(tl.uint16)
        lo = (raw & 15).to(tl.float32) - 8.0
        hi = (raw >> 4).to(tl.float32) - 8.0
        wq = tl.join(lo, hi).reshape(BD, BK)
        sg = tl.load(s2_ptr + offs_o * (I // GS) + (k0 // GS)).to(tl.float32)
        acc += tl.sum(hv[None, :] * (wq * sg[:, None]), axis=1)
    tl.store(y_ptr + offs_o, acc)
    prev = tl.load(out_ptr + offs_o)
    tl.store(out_ptr + offs_o, prev + acc * w_scalar)
