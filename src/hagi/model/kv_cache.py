"""KV-cache for incremental (O(T)) autoregressive decoding.

Convolutional-code analogy: a convolutional decoder maintains state across
decoding steps — each new bit reuses prior decoder state instead of
recomputing from scratch. The KV-cache is that state for the transformer:
once a position is processed its (key, value) are frozen and reused by every
later position, so generation is O(T) rather than O(T^2).

Each :class:`Attention` layer owns one cache. ``model.allocate_for_cache`` /
``model.reset_cache`` drive allocation per layer from the top level.
"""

from __future__ import annotations

import torch


class KVCache:
    """Preallocated ring-free KV store, one per attention layer.

    The cache is preallocated to ``max_seq_len`` and grown monotonically by
    :meth:`update`. Read returns the frozen prefix + newly appended entries.

    Args:
        max_seq_len: maximum total sequence length (prompt + generated).
        n_kv_heads: number of KV heads (GQA).
        head_dim: per-head dimension.
        dtype, device: storage dtype/device (bf16 on CUDA).
    """

    def __init__(
        self,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self._k: torch.Tensor | None = None
        self._v: torch.Tensor | None = None
        self._length = 0

    @property
    def length(self) -> int:
        """Number of positions currently cached."""
        return self._length

    def reset(self) -> None:
        """Drop all cached entries (does not free the preallocation)."""
        self._length = 0

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Append ``new_k``/``new_v`` for ``T_new`` new positions.

        Args:
            new_k, new_v: ``[B, n_kv_heads, T_new, head_dim]``.
        """
        if self._k is None:
            b = new_k.shape[0]
            self._k = torch.zeros((b, self.n_kv_heads, self.max_seq_len, self.head_dim), dtype=self.dtype, device=self.device)
            self._v = torch.zeros_like(self._k)
        t_new = new_k.shape[2]
        end = self._length + t_new
        if end > self.max_seq_len:
            raise ValueError(f"KV-cache overflow: {end} > max_seq_len {self.max_seq_len}")
        self._k[:, :, self._length:end, :] = new_k.to(self.dtype)
        self._v[:, :, self._length:end, :] = new_v.to(self.dtype)
        self._length = end

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached ``(k, v)``: ``[B, n_kv_heads, length, head_dim]``."""
        if self._k is None or self._length == 0:
            raise RuntimeError("KV-cache is empty — call update() before get()")
        return self._k[:, :, : self._length, :], self._v[:, :, : self._length, :]
