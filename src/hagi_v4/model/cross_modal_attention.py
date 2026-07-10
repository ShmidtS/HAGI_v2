"""Cross-modal frequency mixing — MIMO space-time coding (V7.1).

Replaces V7's attention-based cross-modal with frequency-domain
cross-spectrum mixing. Each modality's frequency representation
is combined via cross-spectrum (MIMO channel estimation analog).

5G NR MIMO: signals from multiple antennas are combined through
a channel matrix. In HAGI, modalities = antennas, cross-spectrum =
channel estimation, gated residual = adaptive MIMO receiver.

Information theory:
  - Self-modality FreqBlock = intra-modality equalizer (existing channel)
  - Cross-modal freq mix = inter-modality equalizer (MIMO channel)
  - Cross-spectrum = X_text_f * conj(X_other_f) = cross-correlation
  - Gated cross-modal = adaptive MIMO (start SISO, gradually enable MIMO)
  - Diversity order = num_modalities (3x outage reduction)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.model.norms import RMSNorm


class CrossModalFreqMix(nn.Module):
    """Cross-modal frequency-domain mixing via cross-spectrum.

    Computes cross-spectrum between modality streams in frequency domain,
    applies learnable complex weight, and blends via gated residual.

    5G analog: MIMO channel estimation — cross-spectrum between antennas
    gives the channel response, which is then equalized.
    """

    def __init__(
        self, hidden_size: int, n_heads: int = 8, head_dim: int = 72, gate_init: float = 0.0, norm_eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        freq_dim = n_heads * head_dim

        self.norm = RMSNorm(hidden_size, eps=norm_eps)

        if freq_dim == hidden_size:
            self.proj_in: nn.Module | None = None
            self.proj_out: nn.Module | None = None
        else:
            self.proj_in = nn.Linear(hidden_size, freq_dim, bias=False)
            self.proj_out = nn.Linear(freq_dim, hidden_size, bias=False)

        rank = 16
        self.w_re = nn.Parameter(torch.zeros(n_heads, head_dim, rank))
        self.w_im = nn.Parameter(torch.zeros(n_heads, head_dim, rank))
        nn.init.normal_(self.w_re, std=0.02)
        nn.init.normal_(self.w_im, std=0.02)

        self.gate = nn.Parameter(torch.tensor(gate_init))

    def _split_by_modality(self, h: torch.Tensor, modality_ids: torch.Tensor, mod_id: int) -> torch.Tensor | None:
        mask = modality_ids == mod_id
        if not mask.any():
            return None
        return h[mask].unsqueeze(0)

    def _cross_spectrum(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        B, T, H = h_a.shape
        h_a_n = self.norm(h_a)
        h_b_n = self.norm(h_b)

        if self.proj_in is not None:
            a = self.proj_in(h_a_n).view(B, T, self.n_heads, self.head_dim)
            b = self.proj_in(h_b_n).view(B, T, self.n_heads, self.head_dim)
        else:
            a = h_a_n.view(B, T, self.n_heads, self.head_dim)
            b = h_b_n.view(B, T, self.n_heads, self.head_dim)

        a = a.permute(0, 2, 1, 3).contiguous()
        b = b.permute(0, 2, 1, 3).contiguous()

        A_f = torch.fft.rfft2(a.float())
        B_f = torch.fft.rfft2(b.float())

        T_a = A_f.shape[2]
        H_a = A_f.shape[3]
        Kt = min(16, T_a)
        Kh = min(12, H_a)

        cross = A_f[:, :, :Kt, :Kh] * B_f[:, :, :Kt, :Kh].conj()

        w_re = self.w_re.float() @ self.w_re.t().float()
        w_im = self.w_im.float() @ self.w_im.t().float()
        w = torch.complex(w_re, w_im)

        cross = cross @ w[:, :Kh, :H_a]

        out_f = torch.zeros_like(A_f)
        out_f[:, :, :Kt, :] = cross

        x_out = torch.fft.irfft2(out_f, s=(T, self.head_dim)).to(h_a.dtype)
        x_out = x_out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)

        if self.proj_out is not None:
            return self.proj_out(x_out)
        return x_out

    def forward(
        self,
        h: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
        num_modalities: int = 3,
    ) -> torch.Tensor:
        if modality_ids is None:
            return h

        B, T, H = h.shape
        gate = torch.sigmoid(self.gate)
        cross_residual = h.new_zeros(B, T, H)

        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                h_i = self._split_by_modality(h, modality_ids, i)
                h_j = self._split_by_modality(h, modality_ids, j)
                if h_i is None or h_j is None:
                    continue

                n_min = min(h_i.shape[1], h_j.shape[1])
                cross_i = self._cross_spectrum(h_i[:, :n_min], h_j[:, :n_min])
                cross_j = self._cross_spectrum(h_j[:, :n_min], h_i[:, :n_min])

                mask_i = modality_ids == i
                mask_j = modality_ids == j
                for b in range(B):
                    idx_i = torch.where(mask_i[b])[0]
                    idx_j = torch.where(mask_j[b])[0]
                    n_i = min(idx_i.shape[0], n_min)
                    n_j = min(idx_j.shape[0], n_min)
                    if n_i > 0:
                        cross_residual[b, idx_i[:n_i]] += cross_i[0, :n_i]
                    if n_j > 0:
                        cross_residual[b, idx_j[:n_j]] += cross_j[0, :n_j]

        return h + gate * cross_residual
