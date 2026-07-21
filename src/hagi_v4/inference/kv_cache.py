# V21 DEFER: not wired in V21 forward path. Available for V22+ integration.
# See docs/ARCHITECTURE.md for integration roadmap.

"""KV Cache for efficient block-parallel generation.

Shannon analogy: In convolutional codes, the decoder maintains state
across decoding steps — each new bit uses the decoder state from
previous bits without recomputing from scratch.

V5's generation recomputes the ENTIRE sequence for each block:
  Block 1: forward(prompt + 16 masks) -> 16 tokens
  Block 2: forward(prompt + 16 tokens + 16 masks) -> 16 tokens (recomputes all!)
  Block 3: forward(prompt + 32 tokens + 16 masks) -> 16 tokens (recomputes all!)

This is O(T^2) in sequence length — quadratic waste.

V6's KV cache stores Key/Value tensors for frozen (already-generated)
positions, so only the new block needs computation:
  Block 1: forward(prompt + 16 masks) -> 16 tokens, cache KV for prompt
  Block 2: forward(16 new masks, cached KV) -> 16 tokens, append KV
  Block 3: forward(16 new masks, cached KV) -> 16 tokens, append KV

This is O(T) in sequence length — linear, like incremental decoding.

The cache stores compressed KV from MSA (latent representation),
not raw attention KV, keeping memory footprint small.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVCacheEntry:
    """Cached key-value pair for a sequence position."""

    k: torch.Tensor  # [n_kv_heads, head_dim] or [compress_dim]
    v: torch.Tensor  # [n_kv_heads, head_dim] or [compress_dim]
    pos: int


class KVCache:
    """Ring buffer KV cache for efficient generation.

    Stores compressed KV representations for frozen positions.
    When generating block N, only new block positions are computed;
    frozen positions are retrieved from cache.
    """

    def __init__(
        self,
        max_seq_len: int = 4096,
        compress_dim: int = 128,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.max_seq_len = max_seq_len
        self.compress_dim = compress_dim
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self._k_cache: torch.Tensor | None = None
        self._v_cache: torch.Tensor | None = None
        self._write_pos: int = 0
        self._cached_len: int = 0

    def allocate(self, n_kv_heads: int, head_dim: int) -> None:
        """Pre-allocate cache tensors."""
        self._k_cache = torch.zeros(
            self.max_seq_len,
            n_kv_heads,
            head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._v_cache = torch.zeros(
            self.max_seq_len,
            n_kv_heads,
            head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._write_pos = 0
        self._cached_len = 0

    def write(self, k: torch.Tensor, v: torch.Tensor, n_positions: int) -> None:
        """Write KV for n_positions starting at current write position.

        k: [n_positions, n_kv_heads, head_dim]
        v: [n_positions, n_kv_heads, head_dim]
        """
        if self._k_cache is None:
            return
        pos = self._write_pos
        end = pos + n_positions
        if end > self.max_seq_len:
            end = self.max_seq_len
            n_positions = end - pos

        self._k_cache[pos:end] = k[:n_positions].to(self.dtype)
        self._v_cache[pos:end] = v[:n_positions].to(self.dtype)
        self._write_pos = end
        self._cached_len = max(self._cached_len, end)

    def read(self, start: int = 0, end: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Read cached KV for positions [start, end).

        Returns:
            k: [cached_len, n_kv_heads, head_dim]
            v: [cached_len, n_kv_heads, head_dim]
        """
        if self._k_cache is None:
            return torch.empty(0), torch.empty(0)
        if end is None:
            end = self._cached_len
        return self._k_cache[start:end], self._v_cache[start:end]

    @property
    def cached_len(self) -> int:
        return self._cached_len

    def clear(self) -> None:
        """Reset cache for new generation."""
        self._write_pos = 0
        self._cached_len = 0
        if self._k_cache is not None:
            self._k_cache.zero_()
            self._v_cache.zero_()

    def trim(self, n: int) -> None:
        """Trim cache to first n positions (for rollback on bad speculation)."""
        self._cached_len = min(self._cached_len, n)
        self._write_pos = self._cached_len


class MSACache:
    """Cache for MSA slot registry across generation blocks.

    Instead of clearing MSA between blocks, persist it so each new
    block can use side information from all previous blocks.
    This implements Slepian-Wolf coding across generation blocks.
    """

    def __init__(self):
        self._persisted_keys: torch.Tensor | None = None
        self._persisted_kv: torch.Tensor | None = None
        self._persisted: bool = False

    def save(self, msa_registry) -> None:
        """Save MSA registry state."""
        self._persisted_keys = msa_registry.slot_keys.clone()
        self._persisted_kv = msa_registry.slot_kv.clone()
        self._persisted = True

    def restore(self, msa_registry) -> None:
        """Restore MSA registry state."""
        if not self._persisted:
            return
        msa_registry.slot_keys.copy_(self._persisted_keys)
        msa_registry.slot_kv.copy_(self._persisted_kv)
        msa_registry.write_ptr.zero_()
        msa_registry.num_written.fill_(self._persisted_keys.shape[0])

    def clear(self) -> None:
        self._persisted_keys = None
        self._persisted_kv = None
        self._persisted = False
