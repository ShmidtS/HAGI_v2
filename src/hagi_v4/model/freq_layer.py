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
        nn.init.normal_(self.compress.weight, std=0.02)
        nn.init.normal_(self.expand.weight, std=0.02)
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
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_modes_t = n_modes_t
        self.n_modes_h = n_modes_h
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
            nn.init.normal_(self.w_re_a, std=0.02)
            nn.init.normal_(self.w_im_a, std=0.02)
            nn.init.normal_(self.w_re_b, std=0.02)
            nn.init.normal_(self.w_im_b, std=0.02)

        self.freq_gate_t = nn.Parameter(torch.zeros(T_max))
        self.freq_gate_h = nn.Parameter(torch.zeros(head_dim))
        nn.init.normal_(self.freq_gate_t[:n_modes_t], mean=2.0, std=0.5)
        nn.init.normal_(self.freq_gate_t[n_modes_t:], mean=-2.0, std=0.5)
        nn.init.normal_(self.freq_gate_h[:n_modes_h], mean=2.0, std=0.5)
        nn.init.normal_(self.freq_gate_h[n_modes_h:], mean=-2.0, std=0.5)

        self.channel_response_t = nn.Parameter(torch.zeros(T_max))
        self.channel_response_h = nn.Parameter(torch.zeros(head_dim))
        nn.init.normal_(self.channel_response_t[:n_modes_t], mean=0.0, std=0.02)
        nn.init.normal_(self.channel_response_h[:n_modes_h], mean=0.0, std=0.02)

        if shared_phase is not None:
            self.phase = shared_phase
        else:
            self.phase = nn.Parameter(torch.zeros(n_heads, n_modes_t, n_modes_h))
            nn.init.normal_(self.phase, std=0.1)

        self._w_cache: torch.Tensor | None = None
        self._phase_cache: torch.Tensor | None = None

    def reset_cache(self) -> None:
        """Invalidate eval caches (complex weight, phase).

        Caches are populated lazily on first eval forward and never cleared,
        which produces stale data when seq_len/n_modes change between calls.
        Call on train/eval mode toggle.
        """
        self._w_cache = None
        self._phase_cache = None

    def forward(
        self,
        x: torch.Tensor,
        cached_w: torch.Tensor | None = None,
        cached_phase: torch.Tensor | None = None,
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

        ch_t = torch.exp(1j * self.channel_response_t[:F_t].float())
        ch_h = torch.exp(1j * self.channel_response_h[:F_h].float())
        ch_2d = ch_t.unsqueeze(1) * ch_h.unsqueeze(0)
        X_f = X_f * ch_2d.unsqueeze(0)

        gate_t = torch.sigmoid(self.freq_gate_t[:F_t].float())
        gate_h = torch.sigmoid(self.freq_gate_h[:F_h].float())

        gate_2d = gate_t.unsqueeze(1) * gate_h.unsqueeze(0)
        low = X_f[:, :, :Kt, :Kh]
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
        low = low * phase.unsqueeze(0)
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
        low = low @ w[:, :Kh, :F_h]
        out_f = torch.empty_like(X_f)
        out_f[:, :, :Kt, :] = low
        if Kt < F_t:
            out_f[:, :, Kt:, :] = X_f[:, :, Kt:, :] * gate_2d[Kt:, :].unsqueeze(0).unsqueeze(0)

        mag = out_f.abs().float()
        scale = 1.0 / (1.0 + mag / 10.0)
        out_f = out_f * scale.to(out_f.dtype)

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
        shared_ffn: nn.Module | None = None,
        norm_eps: float = 1e-6,
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
            norm_eps=norm_eps,
        )
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        if shared_ffn is not None:
            self.ffn = shared_ffn
        else:
            self.ffn = FactoredSwiGLU(hidden_size, ffn_intermediate, ffn_rank)

    def forward(
        self,
        x: torch.Tensor,
        cached_w: torch.Tensor | None = None,
        cached_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.freq(x, cached_w=cached_w, cached_phase=cached_phase)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def reset_cache(self) -> None:
        self.freq.reset_cache()
