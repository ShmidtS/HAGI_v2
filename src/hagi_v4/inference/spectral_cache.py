"""Spectral inference cache — OFDM cyclic prefix analog for FreqBlock.

Traditional KV cache stores K,V matrices for attention layers. HAGI uses
2D FFT (global transform), so K,V cache doesn't apply. Instead:

5G NR OFDM analog:
  - Cyclic prefix = context window from previous OFDM symbols
  - Channel state = Kalman P covariance (persists across blocks)
  - Equalizer taps = MSA/HARQ ring buffer (already persists)

Cache components:
  1. Spectral context: last W hidden states per FreqBlock layer boundary
     → sliding window FFT: O((W+block)*log(W+block)) vs O(T*log(T))
  2. Kalman state: P [B, T_new, C] from last turbo iteration of prev block
     → new block starts with cached P instead of ones(C)
  3. MSA slots: ring buffer persists across forward passes (existing)

Memory: O(W * H * num_boundaries) vs KV cache O(T * H * num_layers).
For W=128, T=512: 4x reduction. For T=4096: 32x reduction.
"""

from __future__ import annotations

import torch


class SpectralCache:
    """Inference cache for FreqBlock-based block-parallel generation.

    Stores spectral context (hidden states at layer boundaries) and
    Kalman state across generation blocks. MSA ring buffer persists
    independently via MSAModule.clear() (not called during cached inference).
    """

    def __init__(self, context_window: int = 128) -> None:
        self.context_window = context_window
        self._context: list[torch.Tensor | None] = []
        self._kalman_p: torch.Tensor | None = None
        self._total_len: int = 0

    @property
    def context_len(self) -> int:
        return min(self._total_len, self.context_window)

    def get_context(self, layer_idx: int) -> torch.Tensor | None:
        if layer_idx >= len(self._context):
            return None
        return self._context[layer_idx]

    def update_context(self, layer_idx: int, h: torch.Tensor) -> None:
        while len(self._context) <= layer_idx:
            self._context.append(None)
        W = self.context_window
        if h.shape[1] <= W:
            self._context[layer_idx] = h
        else:
            self._context[layer_idx] = h[:, -W:, :].detach()

    def get_kalman_p(self) -> torch.Tensor | None:
        return self._kalman_p

    def update_kalman_p(self, p: torch.Tensor) -> None:
        if p.dim() <= 1:
            self._kalman_p = p.detach()
        else:
            W = self.context_window
            if p.shape[1] <= W:
                self._kalman_p = p.detach()
            else:
                self._kalman_p = p[:, -W:, :].detach()

    def reset(self) -> None:
        self._context.clear()
        self._kalman_p = None
        self._total_len = 0

    def get_cached_prefix(self, layer_idx: int, h_new: torch.Tensor) -> torch.Tensor:
        ctx = self.get_context(layer_idx)
        if ctx is None or ctx.shape[0] != h_new.shape[0]:
            return h_new
        return torch.cat([ctx, h_new], dim=1)
