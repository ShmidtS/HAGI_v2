"""2D FFT phase-frequency coding — replaces attention (V7).

2D FFT over (T, head_dim) per head = OFDM with spatial multiplexing.
Frequency decomposition in BOTH temporal AND feature dimensions.

Communication theory:
  2D FFT = 2D OFDM (time + space subcarriers)
  Sparse K_t x K_h modes = 2D frequency allocation (water-filling)
  Complex weight per head = MIMO channel equalizer
  Phase modulation = QAM/PSK
  2D IFFT = 2D OFDM synthesis

Complexity: O(n_heads * T * head_dim * log(T*head_dim))
vs attention O(T^2 * H). ~15-20x cheaper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.norms import RMSNorm


class FreqCoding2D(nn.Module):
    """2D FFT phase-frequency coding layer.

    Replaces GroupedQueryAttention. No QKV, no softmax, no RoPE.
    2D FFT naturally encodes position (time freq) and feature (space freq).
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 8,
        head_dim: int = 72,
        n_modes_t: int = 16,
        n_modes_h: int = 12,
        T_max: int = 4096,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_modes_t = n_modes_t
        self.n_modes_h = n_modes_h

        self.proj_in = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.proj_out = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size, eps=norm_eps)

        self.w_re = nn.Parameter(torch.zeros(n_heads, head_dim, head_dim))
        self.w_im = nn.Parameter(torch.zeros(n_heads, head_dim, head_dim))
        nn.init.normal_(self.w_re, std=0.02)
        nn.init.normal_(self.w_im, std=0.02)

        self.freq_gate_t = nn.Parameter(torch.zeros(T_max))
        self.freq_gate_h = nn.Parameter(torch.zeros(head_dim))
        nn.init.normal_(self.freq_gate_t[:n_modes_t], mean=2.0, std=0.5)
        nn.init.normal_(self.freq_gate_t[n_modes_t:], mean=-2.0, std=0.5)
        nn.init.normal_(self.freq_gate_h[:n_modes_h], mean=2.0, std=0.5)
        nn.init.normal_(self.freq_gate_h[n_modes_h:], mean=-2.0, std=0.5)

        self.phase = nn.Parameter(torch.zeros(n_heads, n_modes_t, n_modes_h))
        nn.init.normal_(self.phase, std=0.1)

        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, H = x.shape
        orig_dtype = x.dtype
        x_n = self.norm(x)

        h = self.proj_in(x_n).view(B, T, self.n_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3).contiguous()

        X_f = torch.fft.fft2(h.float())

        F_t = T
        F_h = self.head_dim
        gate_t = torch.sigmoid(self.freq_gate_t[:F_t].float())
        gate_h = torch.sigmoid(self.freq_gate_h[:F_h].float())
        gate_2d = gate_t.unsqueeze(1) * gate_h.unsqueeze(0)

        Kt = min(self.n_modes_t, F_t)
        Kh = min(self.n_modes_h, self.head_dim)

        low = X_f[:, :, :Kt, :Kh]

        phase = torch.exp(1j * self.phase[:, :Kt, :Kh].float())
        low = low * phase.unsqueeze(0)

        w = torch.complex(self.w_re.float(), self.w_im.float())
        low = low @ w[:, :Kh, :]

        out_f = X_f * gate_2d.unsqueeze(0).unsqueeze(0)
        out_f[:, :, :Kt, :] = low

        x_out = torch.fft.ifft2(out_f).real.to(orig_dtype)
        x_out = x_out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)

        return self.proj_out(x_out)


class FreqBlock(nn.Module):
    """Frequency-domain block — drop-in replacement for TransformerBlock.

    Compatible with HRM: blk(h, cos, sin) -> (h, aux, rp).
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
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.freq = FreqCoding2D(
            hidden_size,
            n_heads,
            head_dim,
            n_modes_t,
            n_modes_h,
            T_max=T_max,
            norm_eps=norm_eps,
        )
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_up = nn.Linear(hidden_size, ffn_intermediate * 2, bias=False)
        self.ffn_down = nn.Linear(ffn_intermediate, hidden_size, bias=False)
        nn.init.normal_(self.ffn_up.weight, std=0.02)
        nn.init.normal_(self.ffn_down.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        all_outputs: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.freq(x, cos, sin)

        h = self.ffn_norm(x)
        up, gate = self.ffn_up(h).chunk(2, dim=-1)
        x = x + self.ffn_down(F.silu(gate) * up)

        return x, torch.tensor(0.0, device=x.device), x.new_zeros((0,))
