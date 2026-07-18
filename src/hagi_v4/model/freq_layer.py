"""2D FFT phase-frequency coding — replaces attention (V7).

Information theory techniques for memory efficiency:
  rFFT2: exploit Hermitian symmetry — 50% fewer frequency bins
  Factored projection: low-rank Linear (CDMA spreading code)
  Factored FFN: SVD-style compression of weight matrix
  Shared complex weights: one LDPC decoder reused across layers

Communication theory:
  2D rFFT = OFDM demodulation (Hermitian-symmetric, real signal)
  Factored proj = CDMA spreading (low-rank channel)
  Complex weight = MIMO channel equalizer (low-rank, rank=16)
  Phase modulation = PSK
  Soft frequency gating = adaptive modulation (5G AMC)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.norms import RMSNorm


class FactoredLinear(nn.Module):
    """Low-rank factored linear: Linear(in, r) -> Linear(r, out).

    CDMA analog: spreading code (in->r) + despreading (r->out).
    Params: in*r + r*out instead of in*out. 2-5x reduction.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False) -> None:
        super().__init__()
        self.compress = nn.Linear(in_features, rank, bias=bias)
        self.expand = nn.Linear(rank, out_features, bias=bias)
        c_std = 1.0 / (in_features**0.5)
        e_std = 1.0 / (rank**0.5)
        nn.init.normal_(self.compress.weight, std=c_std)
        nn.init.normal_(self.expand.weight, std=e_std)
        if bias:
            nn.init.zeros_(self.compress.bias)
            nn.init.zeros_(self.expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.compress(x))


class FactoredSwiGLU(nn.Module):
    """Factored SwiGLU FFN: H->r->2*intermediate, intermediate->r->H.

    SVD analog: keep top-r singular values of the FFN weight matrix.
    Params: H*r + r*2*int + int*r + r*H instead of H*2*int + int*H.
    3-5x reduction with r = H/4.
    """

    def __init__(self, hidden_size: int, intermediate: int, rank: int) -> None:
        super().__init__()
        self.up = FactoredLinear(hidden_size, intermediate * 2, rank)
        self.down = FactoredLinear(intermediate, hidden_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def _interp_modes(coef: torch.Tensor, target_len: int) -> torch.Tensor:
    """Expand mode coefficients to target length via linear interpolation.

    Models a smooth frequency-selective response from a small number of
    mode coefficients (analogous to OFDM channel taps → frequency response
    via DFT). This replaces per-position learnable parameters of length
    T_max (which overfit and introduce positional noise) with a compact
    rank-n_modes representation that generalizes across sequence lengths.

    Args:
        coef: [n_modes] learnable coefficients.
        target_len: F_t or F_h (actual number of frequency bins).

    Returns:
        [target_len] interpolated values.
    """
    n = coef.shape[0]
    if target_len <= n:
        return coef[:target_len]
    dst = torch.arange(target_len, dtype=torch.float32, device=coef.device)
    scale = (n - 1) / max(target_len - 1, 1)
    dst_scaled = dst * scale
    idx_lo = dst_scaled.floor().long().clamp(max=n - 1)
    idx_hi = dst_scaled.ceil().long().clamp(max=n - 1)
    frac = (dst_scaled - idx_lo.float()).to(coef.dtype)
    return coef[idx_lo] * (1.0 - frac) + coef[idx_hi] * frac


def frequency_derivative_multiplier(
    n: int,
    dim_is_real: bool,
    alpha: torch.Tensor,
    device: torch.device | str = "cpu",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute fractional frequency-derivative multiplier (1j*2*pi*f)^alpha.

    Multiplying an FFT coefficient by this is a fractional derivative of order
    alpha in the original (time/feature) domain: alpha=0 is identity, alpha=1
    is the first derivative (high-pass edge detector).

    Args:
        n: length of the original axis (before FFT).
        dim_is_real: True for the rFFT axis (Hermitian, returns n//2+1 bins),
            False for the full-FFT axis (returns n bins).
        alpha: derivative order in [0, 1] as a SCALAR TENSOR (0-dim). Passed as
            tensor (not float) to preserve gradient flow for learnable alpha.
        device: torch device for the result.
        eps: clamp applied to |f| to avoid 0**alpha singularity at DC.

    Returns:
        Complex tensor of shape [n] (full) or [n//2+1] (real).
    """
    if dim_is_real:
        freqs = torch.fft.rfftfreq(n, d=1.0, device=device)
    else:
        freqs = torch.fft.fftfreq(n, d=1.0, device=device)
    freqs = freqs.float()
    mag = freqs.abs().clamp_min(eps)
    sign = freqs.sign()
    # signed frequency with DC epsilon: preserves sign, avoids 0
    f_safe = sign * mag
    base = 1j * 2.0 * torch.pi * f_safe
    # alpha is a scalar tensor; power op keeps gradient w.r.t. alpha
    result = base**alpha
    # At DC (f=0 originally), force exact 0 when alpha > 0 (derivative of constant = 0).
    dc_mask = freqs.abs() < eps
    result = torch.where(dc_mask, torch.zeros_like(result), result)
    return result


class FreqCoding2D(nn.Module):
    """2D rFFT phase-frequency coding layer.

    Symmetry optimizations:
      rFFT2: Hermitian symmetry (50% fewer freq bins)
      No projection when H = n_heads*head_dim (FFT IS the projection)
      Low-rank shared complex weights (MIMO channel, rank=16)
      Soft frequency gating (adaptive modulation)

    In OFDM, the FFT/IFFT IS the subcarrier mapping — no learned
    projection needed. Cross-subcarrier mixing comes from the
    equalizer (complex weight) and FFN (time-domain processing).
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 8,
        head_dim: int = 72,
        n_modes_t: int = 16,
        n_modes_h: int = 12,
        T_max: int = 4096,
        rank: int = 16,
        proj_rank: int = 144,
        shared_weights: tuple | None = None,
        shared_phase: nn.Parameter | None = None,
        shared_phase_dT: nn.Parameter | None = None,
        shared_phase_dH: nn.Parameter | None = None,
        norm_eps: float = 1e-6,
        use_derivative: bool = True,
        share_branch_weights: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_modes_t = n_modes_t
        self.n_modes_h = n_modes_h
        self.use_derivative = use_derivative
        self.share_branch_weights = share_branch_weights
        freq_dim = n_heads * head_dim

        self.norm = RMSNorm(hidden_size, eps=norm_eps)

        if freq_dim == hidden_size:
            self.proj_in = None
            self.proj_out = None
        else:
            self.proj_in = FactoredLinear(hidden_size, freq_dim, proj_rank)
            self.proj_out = FactoredLinear(freq_dim, hidden_size, proj_rank)

        if shared_weights is not None:
            self.w_re_a = shared_weights[0]
            self.w_im_a = shared_weights[1]
            self.w_re_b = shared_weights[2]
            self.w_im_b = shared_weights[3]
        else:
            self.w_re_a = nn.Parameter(torch.zeros(n_heads, head_dim, rank))
            self.w_im_a = nn.Parameter(torch.zeros(n_heads, head_dim, rank))
            self.w_re_b = nn.Parameter(torch.zeros(n_heads, rank, head_dim))
            self.w_im_b = nn.Parameter(torch.zeros(n_heads, rank, head_dim))
            w_std = 1.0 / (head_dim**0.5)
            nn.init.normal_(self.w_re_a, std=w_std)
            nn.init.normal_(self.w_im_a, std=w_std)
            nn.init.normal_(self.w_re_b, std=w_std)
            nn.init.normal_(self.w_im_b, std=w_std)

        self.freq_gate_t = nn.Parameter(torch.zeros(n_modes_t))
        self.freq_gate_h = nn.Parameter(torch.zeros(n_modes_h))

        # V11: non-zero init for channel response (MIMO equalizer). The V10
        # checkpoint showed channel_response_h near dead (max 0.011) because
        # zero init left the exp(1j*0)=1 identity with a weak gradient. Small
        # random phase perturbation gives the equalizer a non-trivial starting
        # point and a stronger gradient signal.
        self.channel_response_t = nn.Parameter(torch.randn(n_modes_t) * 0.1)
        self.channel_response_h = nn.Parameter(torch.randn(n_modes_h) * 0.1)

        if shared_phase is not None:
            self.phase = shared_phase
        else:
            self.phase = nn.Parameter(torch.zeros(n_heads, n_modes_t, n_modes_h))
            nn.init.normal_(self.phase, std=1.0 / (n_modes_t**0.5))

        # --- Derivative branch params (only when use_derivative) ---
        if use_derivative and not share_branch_weights:
            # 12-tuple shared_weights convention: 4 main + 4 dT + 4 dH.
            if shared_weights is not None and len(shared_weights) == 12:
                self.w_re_a_dT = shared_weights[4]
                self.w_im_a_dT = shared_weights[5]
                self.w_re_b_dT = shared_weights[6]
                self.w_im_b_dT = shared_weights[7]
                self.w_re_a_dH = shared_weights[8]
                self.w_im_a_dH = shared_weights[9]
                self.w_re_b_dH = shared_weights[10]
                self.w_im_b_dH = shared_weights[11]
            else:
                self.w_re_a_dT = nn.Parameter(torch.zeros(n_heads, head_dim, rank))
                self.w_im_a_dT = nn.Parameter(torch.zeros(n_heads, head_dim, rank))
                self.w_re_b_dT = nn.Parameter(torch.zeros(n_heads, rank, head_dim))
                self.w_im_b_dT = nn.Parameter(torch.zeros(n_heads, rank, head_dim))
                self.w_re_a_dH = nn.Parameter(torch.zeros(n_heads, head_dim, rank))
                self.w_im_a_dH = nn.Parameter(torch.zeros(n_heads, head_dim, rank))
                self.w_re_b_dH = nn.Parameter(torch.zeros(n_heads, rank, head_dim))
                self.w_im_b_dH = nn.Parameter(torch.zeros(n_heads, rank, head_dim))
                w_std_d = 1.0 / (head_dim**0.5)
                for p in (
                    self.w_re_a_dT,
                    self.w_im_a_dT,
                    self.w_re_b_dT,
                    self.w_im_b_dT,
                    self.w_re_a_dH,
                    self.w_im_a_dH,
                    self.w_re_b_dH,
                    self.w_im_b_dH,
                ):
                    nn.init.normal_(p, std=w_std_d)

            if shared_phase_dT is not None:
                self.phase_dT = shared_phase_dT
            else:
                self.phase_dT = nn.Parameter(torch.zeros(n_heads, n_modes_t, n_modes_h))
                nn.init.normal_(self.phase_dT, std=1.0 / (n_modes_t**0.5))
            if shared_phase_dH is not None:
                self.phase_dH = shared_phase_dH
            else:
                self.phase_dH = nn.Parameter(torch.zeros(n_heads, n_modes_t, n_modes_h))
                nn.init.normal_(self.phase_dH, std=1.0 / (n_modes_t**0.5))

        # Fractional orders + branch gates + magnitude safety (always present when use_derivative)
        if use_derivative:
            # V10: init raw_alpha with small random perturbation around 0.0
            # (sigmoid(0)=0.5) instead of the tautological 0.5 that left the
            # parameter stuck at its init value in the V9 checkpoint. Small
            # random init breaks symmetry between layers and gives the
            # optimizer a non-saturated gradient signal.
            self.raw_alpha_t = nn.Parameter(torch.randn(1) * 0.1)
            self.raw_alpha_h = nn.Parameter(torch.randn(1) * 0.1)
            # Branch gates: small random init so branches differentiate early.
            self.branch_gate_main = nn.Parameter(torch.randn(1) * 0.1)
            self.branch_gate_dT = nn.Parameter(torch.randn(1) * 0.1)
            self.branch_gate_dH = nn.Parameter(torch.randn(1) * 0.1)
            self.deriv_norm_t = nn.Parameter(torch.ones(1))
            self.deriv_norm_h = nn.Parameter(torch.ones(1))

        self._w_cache: torch.Tensor | None = None
        self._phase_cache: torch.Tensor | None = None
        self._mult_dT_cache: torch.Tensor | None = None
        self._mult_dH_cache: torch.Tensor | None = None
        self._w_cache_dT: torch.Tensor | None = None
        self._w_cache_dH: torch.Tensor | None = None
        self._phase_cache_dT: torch.Tensor | None = None
        self._phase_cache_dH: torch.Tensor | None = None
        self._cache_key_T: int | None = None

    def reset_cache(self) -> None:
        """Invalidate eval caches (complex weights, phase, derivative multipliers).

        Caches are populated lazily on first eval forward and never cleared,
        which produces stale data when seq_len/n_modes change between calls.
        Call on train/eval mode toggle.
        """
        self._w_cache = None
        self._phase_cache = None
        self._mult_dT_cache = None
        self._mult_dH_cache = None
        self._w_cache_dT = None
        self._w_cache_dH = None
        self._phase_cache_dT = None
        self._phase_cache_dH = None
        self._cache_key_T = None

    def forward(
        self,
        x: torch.Tensor,
        cached_w: torch.Tensor | None = None,
        cached_phase: torch.Tensor | None = None,
        cached_w_dT: torch.Tensor | None = None,
        cached_w_dH: torch.Tensor | None = None,
        cached_phase_dT: torch.Tensor | None = None,
        cached_phase_dH: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, H = x.shape
        orig_dtype = x.dtype
        x_n = self.norm(x)

        if self.proj_in is not None:
            h = self.proj_in(x_n).view(B, T, self.n_heads, self.head_dim)
        else:
            h = x_n.view(B, T, self.n_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3).contiguous()

        X_f = torch.fft.rfft2(h.float())

        F_t = X_f.shape[2]
        F_h = X_f.shape[3]
        Kt = min(self.n_modes_t, F_t)
        Kh = min(self.n_modes_h, F_h)

        ch_t_raw = _interp_modes(self.channel_response_t, F_t)
        ch_h_raw = _interp_modes(self.channel_response_h, F_h)
        ch_t = torch.exp(1j * ch_t_raw.float())
        ch_h = torch.exp(1j * ch_h_raw.float())
        ch_2d = ch_t.unsqueeze(1) * ch_h.unsqueeze(0)
        X_f = X_f * ch_2d.unsqueeze(0)

        gate_t = torch.sigmoid(_interp_modes(self.freq_gate_t, F_t).float())
        gate_h = torch.sigmoid(_interp_modes(self.freq_gate_h, F_h).float())
        gate_2d = gate_t.unsqueeze(1) * gate_h.unsqueeze(0)

        # --- Main branch (existing path) ---
        low_main = X_f[:, :, :Kt, :Kh]
        if cached_phase is not None:
            phase = cached_phase
        elif (
            not self.training
            and self._phase_cache is not None
            and self._phase_cache.shape[1] >= Kt
            and self._phase_cache.shape[2] >= Kh
        ):
            phase = self._phase_cache[:, :Kt, :Kh]
        else:
            phase = torch.exp(1j * self.phase[:, :Kt, :Kh].float())
            if not self.training:
                self._phase_cache = phase
        low_main = low_main * phase.unsqueeze(0)

        if cached_w is not None:
            w = cached_w
        elif not self.training and self._w_cache is not None:
            w = self._w_cache
        else:
            w_re = self.w_re_a.float() @ self.w_re_b.float()
            w_im = self.w_im_a.float() @ self.w_im_b.float()
            w = torch.complex(w_re, w_im)
            if not self.training:
                self._w_cache = w
        low_main = low_main @ w[:, :Kh, :F_h]

        if not self.use_derivative:
            # Single-branch short-circuit (post-Phase-1 behavior, no compression).
            out_f = torch.empty_like(X_f)
            out_f[:, :, :Kt, :] = low_main
            if Kt < F_t:
                out_f[:, :, Kt:, :] = X_f[:, :, Kt:, :] * gate_2d[Kt:, :].unsqueeze(0).unsqueeze(0)
            x_out = torch.fft.irfft2(out_f, s=(T, self.head_dim)).to(orig_dtype)
            x_out = x_out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
            if self.proj_out is not None:
                return self.proj_out(x_out)
            return x_out

        # --- Derivative branches ---
        # Recompute derivative multipliers if seq_len changed or cache empty.
        # NOTE: alpha must stay a tensor to preserve gradient flow to raw_alpha_*.
        cache_key_T = T
        need_recompute = self._mult_dT_cache is None or self._cache_key_T != cache_key_T or self.training
        if need_recompute:
            alpha_t = torch.sigmoid(self.raw_alpha_t)
            alpha_h = torch.sigmoid(self.raw_alpha_h)
            mult_dT = frequency_derivative_multiplier(
                n=T,
                dim_is_real=False,
                alpha=alpha_t,
                device=X_f.device,
            )
            mult_dH = frequency_derivative_multiplier(
                n=self.head_dim,
                dim_is_real=True,
                alpha=alpha_h,
                device=X_f.device,
            )
            mult_dT = mult_dT * self.deriv_norm_t.float()
            mult_dH = mult_dH * self.deriv_norm_h.float()
            if not self.training:
                self._mult_dT_cache = mult_dT
                self._mult_dH_cache = mult_dH
                self._cache_key_T = cache_key_T
        else:
            mult_dT = self._mult_dT_cache
            mult_dH = self._mult_dH_cache

        # Apply derivative multipliers (broadcast over heads and the other axis).
        X_dT = X_f * mult_dT.to(X_f.dtype).view(1, 1, F_t, 1)
        X_dH = X_f * mult_dH.to(X_f.dtype).view(1, 1, 1, F_h)

        if self.share_branch_weights:
            w_dT = w
            w_dH = w
            phase_dT = phase
            phase_dH = phase
        else:
            if cached_w_dT is not None:
                w_dT = cached_w_dT
            elif not self.training and self._w_cache_dT is not None:
                w_dT = self._w_cache_dT
            else:
                w_re_dT = self.w_re_a_dT.float() @ self.w_re_b_dT.float()
                w_im_dT = self.w_im_a_dT.float() @ self.w_im_b_dT.float()
                w_dT = torch.complex(w_re_dT, w_im_dT)
                if not self.training:
                    self._w_cache_dT = w_dT

            if cached_w_dH is not None:
                w_dH = cached_w_dH
            elif not self.training and self._w_cache_dH is not None:
                w_dH = self._w_cache_dH
            else:
                w_re_dH = self.w_re_a_dH.float() @ self.w_re_b_dH.float()
                w_im_dH = self.w_im_a_dH.float() @ self.w_im_b_dH.float()
                w_dH = torch.complex(w_re_dH, w_im_dH)
                if not self.training:
                    self._w_cache_dH = w_dH

            if cached_phase_dT is not None:
                phase_dT = cached_phase_dT
            elif not self.training and self._phase_cache_dT is not None:
                phase_dT = self._phase_cache_dT[:, :Kt, :Kh]
            else:
                phase_dT = torch.exp(1j * self.phase_dT[:, :Kt, :Kh].float())
                if not self.training:
                    self._phase_cache_dT = phase_dT

            if cached_phase_dH is not None:
                phase_dH = cached_phase_dH
            elif not self.training and self._phase_cache_dH is not None:
                phase_dH = self._phase_cache_dH[:, :Kt, :Kh]
            else:
                phase_dH = torch.exp(1j * self.phase_dH[:, :Kt, :Kh].float())
                if not self.training:
                    self._phase_cache_dH = phase_dH

        low_dT = X_dT[:, :, :Kt, :Kh] * phase_dT.unsqueeze(0)
        low_dT = low_dT @ w_dT[:, :Kh, :F_h]

        low_dH = X_dH[:, :, :Kt, :Kh] * phase_dH.unsqueeze(0)
        low_dH = low_dH @ w_dH[:, :Kh, :F_h]

        # Combine branches as ADDITIVE residual: main branch is primary,
        # derivative branches contribute high-pass detail. This ensures
        # gradients flow to derivative params even when the main branch
        # dominates the final output (DC-symmetry bypass).
        g_main = torch.sigmoid(self.branch_gate_main.float())
        g_dT = torch.sigmoid(self.branch_gate_dT.float())
        g_dH = torch.sigmoid(self.branch_gate_dH.float())
        # main is scaled by g_main; derivative branches ADD residual scaled
        # by their gates. This guarantees the derivative contribution is
        # non-zero in the final output (not washed out by main).
        combined_low = g_main * low_main + g_dT * (low_dT - low_main.detach()) + g_dH * (low_dH - low_main.detach())

        out_f = torch.empty_like(X_f)
        out_f[:, :, :Kt, :] = combined_low
        if Kt < F_t:
            out_f[:, :, Kt:, :] = X_f[:, :, Kt:, :] * gate_2d[Kt:, :].unsqueeze(0).unsqueeze(0)

        x_out = torch.fft.irfft2(out_f, s=(T, self.head_dim)).to(orig_dtype)
        x_out = x_out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)

        if self.proj_out is not None:
            return self.proj_out(x_out)
        return x_out


class FreqBlock(nn.Module):
    """Frequency-domain block — drop-in replacement for TransformerBlock.

    V7: uses factored FFN (SVD compression) for 3x param reduction.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 8,
        head_dim: int = 72,
        n_modes_t: int = 16,
        n_modes_h: int = 12,
        ffn_intermediate: int = 384,
        T_max: int = 4096,
        rank: int = 16,
        proj_rank: int = 144,
        ffn_rank: int | None = None,
        shared_weights: tuple | None = None,
        shared_phase: nn.Parameter | None = None,
        shared_phase_dT: nn.Parameter | None = None,
        shared_phase_dH: nn.Parameter | None = None,
        shared_ffn: nn.Module | None = None,
        norm_eps: float = 1e-6,
        use_derivative: bool = True,
        share_branch_weights: bool = False,
    ) -> None:
        super().__init__()
        if ffn_rank is None:
            ffn_rank = max(32, hidden_size // 4)

        self.freq = FreqCoding2D(
            hidden_size,
            n_heads,
            head_dim,
            n_modes_t,
            n_modes_h,
            T_max=T_max,
            rank=rank,
            proj_rank=proj_rank,
            shared_weights=shared_weights,
            shared_phase=shared_phase,
            shared_phase_dT=shared_phase_dT,
            shared_phase_dH=shared_phase_dH,
            norm_eps=norm_eps,
            use_derivative=use_derivative,
            share_branch_weights=share_branch_weights,
        )
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        if shared_ffn is not None:
            self.ffn = shared_ffn
        else:
            self.ffn = FactoredSwiGLU(hidden_size, ffn_intermediate, ffn_rank)

        self.freq_scale = nn.Parameter(torch.ones(hidden_size))
        self.ffn_scale = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        cached_w: torch.Tensor | None = None,
        cached_phase: torch.Tensor | None = None,
        cached_w_dT: torch.Tensor | None = None,
        cached_w_dH: torch.Tensor | None = None,
        cached_phase_dT: torch.Tensor | None = None,
        cached_phase_dH: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.freq_scale * self.freq(
            x,
            cached_w=cached_w,
            cached_phase=cached_phase,
            cached_w_dT=cached_w_dT,
            cached_w_dH=cached_w_dH,
            cached_phase_dT=cached_phase_dT,
            cached_phase_dH=cached_phase_dH,
        )
        x = x + self.ffn_scale * self.ffn(self.ffn_norm(x))
        return x

    def reset_cache(self) -> None:
        self.freq.reset_cache()
