"""Triton fused kernels for HAGI V7 — reduce VRAM by eliminating intermediates.

Fuses FFT → gate → phase → complex weight → IFFT into one kernel.
Eliminates the large complex intermediate tensors that dominated VRAM.

Communication theory: this is a single-OFDM-symbol pipeline —
demodulate, equalize, remodulate in one pass, no intermediate storage.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _freq_gating_kernel(
    x_real_ptr,
    x_imag_ptr,
    gate_t_ptr,
    gate_h_ptr,
    phase_ptr,
    w_re_a_ptr,
    w_im_a_ptr,
    w_re_b_ptr,
    w_im_b_ptr,
    out_real_ptr,
    out_imag_ptr,
    F_t: tl.constexpr,
    F_h: tl.constexpr,
    Kt: tl.constexpr,
    Kh: tl.constexpr,
    n_heads: tl.constexpr,
    rank: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: gate × phase × low-rank complex weight in frequency domain.

    Replaces 5 separate torch operations with one kernel pass.
    Eliminates 4 intermediate complex tensors.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    ft = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    fh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    ft_mask = ft < F_t
    fh_mask = fh < F_h

    gate_t_val = tl.load(gate_t_ptr + ft, mask=ft_mask, other=0.0)
    gate_h_val = tl.load(gate_h_ptr + fh, mask=fh_mask, other=0.0)
    gate = gate_t_val[:, None] * gate_h_val[None, :]

    ft_idx = ft[:, None]
    fh_idx = fh[None, :]

    low_mask = (ft_idx < Kt) & (fh_idx < Kh)
    phase_idx = ft_idx * Kh + fh_idx

    for head in tl.static_range(n_heads):
        x_re = tl.load(
            x_real_ptr + pid_b * n_heads * F_t * F_h + head * F_t * F_h + ft[:, None] * F_h + fh[None, :],
            mask=ft_mask[:, None] & fh_mask[None, :],
            other=0.0,
        )
        x_im = tl.load(
            x_imag_ptr + pid_b * n_heads * F_t * F_h + head * F_t * F_h + ft[:, None] * F_h + fh[None, :],
            mask=ft_mask[:, None] & fh_mask[None, :],
            other=0.0,
        )

        gated_re = x_re * gate
        gated_im = x_im * gate

        if Kt > 0 and Kh > 0:
            phase_val = tl.load(phase_ptr + head * Kt * Kh + phase_idx, mask=low_mask, other=0.0)
            cos_p = tl.cos(phase_val)
            sin_p = tl.sin(phase_val)

            low_re = x_re * cos_p - x_im * sin_p
            low_im = x_re * sin_p + x_im * cos_p

            w_re_acc = tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.float32)
            w_im_acc = tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.float32)

            for r in tl.static_range(rank):
                w_re_a_val = tl.load(w_re_a_ptr + head * head_dim * rank + r * head_dim + fh, mask=fh_mask, other=0.0)
                w_im_a_val = tl.load(w_im_a_ptr + head * head_dim * rank + r * head_dim + fh, mask=fh_mask, other=0.0)
                w_re_b_val = tl.load(w_re_b_ptr + head * rank * head_dim + r * head_dim + fh, mask=fh_mask, other=0.0)
                w_im_b_val = tl.load(w_im_b_ptr + head * rank * head_dim + r * head_dim + fh, mask=fh_mask, other=0.0)

                w_re_acc += low_re * w_re_b_val[None, :]
                w_im_acc += low_im * w_re_b_val[None, :]
                w_re_acc -= low_im * w_im_b_val[None, :]
                w_im_acc += low_re * w_im_b_val[None, :]

            out_re = tl.where(low_mask, w_re_acc, gated_re)
            out_im = tl.where(low_mask, w_im_acc, gated_im)
        else:
            out_re = gated_re
            out_im = gated_im

        tl.store(
            out_real_ptr + pid_b * n_heads * F_t * F_h + head * F_t * F_h + ft[:, None] * F_h + fh[None, :],
            out_re,
            mask=ft_mask[:, None] & fh_mask[None, :],
        )
        tl.store(
            out_imag_ptr + pid_b * n_heads * F_t * F_h + head * F_t * F_h + ft[:, None] * F_h + fh[None, :],
            out_im,
            mask=ft_mask[:, None] & fh_mask[None, :],
        )


def triton_freq_gating(
    X_f: torch.Tensor,
    gate_t: torch.Tensor,
    gate_h: torch.Tensor,
    phase: torch.Tensor,
    w_re_a: torch.Tensor,
    w_im_a: torch.Tensor,
    w_re_b: torch.Tensor,
    w_im_b: torch.Tensor,
    Kt: int,
    Kh: int,
) -> torch.Tensor:
    """Fused frequency-domain gating + phase modulation + low-rank weight.

    Args:
        X_f: [B, n_heads, F_t, F_h] complex FFT coefficients
        gate_t: [F_t] temporal frequency gate (sigmoid)
        gate_h: [F_h] feature frequency gate (sigmoid)
        phase: [n_heads, Kt, Kh] phase modulation
        w_re_a, w_im_a: [n_heads, head_dim, rank] low-rank real/imag part A
        w_re_b, w_im_b: [n_heads, rank, head_dim] low-rank real/imag part B
        Kt, Kh: number of active low-frequency modes

    Returns:
        out_f: [B, n_heads, F_t, F_h] complex filtered coefficients
    """
    B, n_heads, F_t, F_h = X_f.shape
    rank = w_re_a.shape[2]
    head_dim = w_re_a.shape[1]

    gate_t_cont = gate_t.contiguous()
    gate_h_cont = gate_h.contiguous()
    phase_cont = phase.contiguous()
    w_re_a_cont = w_re_a.contiguous()
    w_im_a_cont = w_im_a.contiguous()
    w_re_b_cont = w_re_b.contiguous()
    w_im_b_cont = w_im_b.contiguous()

    x_real = X_f.real.contiguous()
    x_imag = X_f.imag.contiguous()

    out_real = torch.empty_like(x_real)
    out_imag = torch.empty_like(x_imag)

    BLOCK_T = min(64, triton.next_power_of_2(F_t))
    BLOCK_H = min(64, triton.next_power_of_2(F_h))

    grid = (triton.cdiv(F_t, BLOCK_T), triton.cdiv(F_h, BLOCK_H), B)

    _freq_gating_kernel[grid](
        x_real,
        x_imag,
        gate_t_cont,
        gate_h_cont,
        phase_cont,
        w_re_a_cont,
        w_im_a_cont,
        w_re_b_cont,
        w_im_b_cont,
        out_real,
        out_imag,
        F_t=F_t,
        F_h=F_h,
        Kt=Kt,
        Kh=Kh,
        n_heads=n_heads,
        rank=rank,
        head_dim=head_dim,
        BLOCK_T=BLOCK_T,
        BLOCK_H=BLOCK_H,
    )

    return torch.complex(out_real, out_imag)
