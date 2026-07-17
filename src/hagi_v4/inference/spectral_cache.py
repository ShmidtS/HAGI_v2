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
    3. MSA feedback slots: serialized in DecodeState across forward passes

Memory: O(W * H * num_boundaries) vs KV cache O(T * H * num_layers).
For W=128, T=512: 4x reduction. For T=4096: 32x reduction.
"""

from __future__ import annotations

import torch

from hagi_v4.model.codec_contracts import DecodeState


class SpectralCache:
    """Inference cache for FreqBlock-based block-parallel generation.

    Stores the pre-FreqBlock spectral boundary and decoder state across
    generation blocks.
    """

    def __init__(self, context_window: int = 128) -> None:
        self.context_window = context_window
        self._context: list[torch.Tensor | None] = []
        self._context_boundary: list[torch.Tensor | None] = []
        self._decode_state = DecodeState(cache_active=True)
        self._decode_state_boundary = DecodeState(cache_active=True)
        self._total_len: int = 0

    @property
    def context_len(self) -> int:
        return min(self._total_len, self.context_window)

    def get_context(self, layer_idx: int) -> torch.Tensor | None:
        if layer_idx >= len(self._context):
            return None
        return self._context[layer_idx]

    def update_context(self, layer_idx: int, h: torch.Tensor, new_tokens: int | None = None) -> None:
        while len(self._context) <= layer_idx:
            self._context.append(None)
            self._context_boundary.append(None)
        if layer_idx == 0:
            self._context_boundary = [ctx.detach().clone() if ctx is not None else None for ctx in self._context]
            self._decode_state_boundary = self._copy_decode_state(self._decode_state)
        W = self.context_window
        if h.shape[1] <= W:
            self._context[layer_idx] = h
        else:
            self._context[layer_idx] = h[:, -W:, :].detach()
        if layer_idx == 0:
            self._total_len += h.shape[1] if new_tokens is None else new_tokens

    def get_kalman_p(self) -> torch.Tensor | None:
        return self._decode_state.kalman_p

    def update_kalman_p(self, p: torch.Tensor) -> None:
        # deprecated in V5 — LearnedUncertainty has no persistent state.
        # Kept for API compatibility with existing callers; intentionally a no-op.
        return

    def to_decode_state(self) -> DecodeState:
        state = self._decode_state
        return DecodeState(state.kalman_p, state.msa_feedback, state.iteration, cache_active=True)

    def update_decode_state(self, state: DecodeState) -> None:
        self._decode_state = DecodeState(
            kalman_p=None,
            msa_feedback=state.msa_feedback.detach() if state.msa_feedback is not None else None,
            iteration=state.iteration,
            cache_active=True,
        )
        if state.kalman_p is not None:
            self.update_kalman_p(state.kalman_p)

    def rollback(self, block_len: int) -> None:
        """Restore the boundary before the latest speculative token block."""
        if block_len < 0:
            raise ValueError("block_len must be non-negative")
        for layer_idx, context in enumerate(self._context):
            if context is None:
                continue
            boundary = self._context_boundary[layer_idx] if layer_idx < len(self._context_boundary) else None
            self._context[layer_idx] = boundary
        self._total_len = max(0, self._total_len - block_len)
        self._decode_state = self._copy_decode_state(self._decode_state_boundary)

    def reset(self) -> None:
        self._context.clear()
        self._context_boundary.clear()
        self._decode_state = DecodeState(cache_active=True)
        self._decode_state_boundary = DecodeState(cache_active=True)
        self._total_len = 0

    def get_cached_prefix(self, layer_idx: int, h_new: torch.Tensor) -> torch.Tensor:
        ctx = self.get_context(layer_idx)
        if ctx is None or ctx.shape[0] != h_new.shape[0]:
            return h_new
        return torch.cat([ctx, h_new], dim=1)

    @staticmethod
    def _copy_decode_state(state: DecodeState) -> DecodeState:
        return DecodeState(
            kalman_p=state.kalman_p.detach().clone() if state.kalman_p is not None else None,
            msa_feedback=state.msa_feedback.detach().clone() if state.msa_feedback is not None else None,
            iteration=state.iteration,
            cache_active=True,
        )
