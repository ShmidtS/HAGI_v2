"""KV-cache — the receiver's state register.

A convolutional decoder keeps state across symbol times instead of re-deriving it
from the whole history; the KV-cache is the same idea for attention. Once a
position's key and value are computed they are frozen, so generating token ``t``
costs one position of work instead of ``t``. That is the difference between O(T)
and O(T^2) for a T-token completion.

The buffer is preallocated to ``max_seq_len``: growth by concatenation would
reallocate and copy the entire cache on every token, which is the same quadratic
cost the cache exists to remove.
"""

from __future__ import annotations

import torch


class KVCache:
    """Preallocated per-layer key/value store.

    Args:
        max_seq_len: capacity (prompt + generated).
        n_kv_heads: GQA key/value head count.
        head_dim: per-head width.
        dtype: storage dtype (match the model's compute dtype).
        device: storage device.
    """

    def __init__(
        self,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.max_seq_len = int(max_seq_len)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = device
        self._k: torch.Tensor | None = None
        self._v: torch.Tensor | None = None
        self._length = 0

    @property
    def length(self) -> int:
        """Number of cached positions."""
        return self._length

    def reset(self) -> None:
        """Forget all cached positions, keeping the allocation."""
        self._length = 0

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Append ``[B, n_kv_heads, T_new, head_dim]`` keys and values.

        Raises:
            ValueError: on shape mismatch or capacity overflow.
        """
        if new_k.shape != new_v.shape:
            raise ValueError(f"key/value shape mismatch: {tuple(new_k.shape)} vs {tuple(new_v.shape)}")
        if new_k.ndim != 4 or new_k.shape[1] != self.n_kv_heads or new_k.shape[3] != self.head_dim:
            raise ValueError(
                f"expected [B, {self.n_kv_heads}, T, {self.head_dim}], got {tuple(new_k.shape)}"
            )
        if self._k is None or self._k.shape[0] != new_k.shape[0]:
            self._k = torch.zeros(
                (new_k.shape[0], self.n_kv_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self._v = torch.zeros_like(self._k)
            self._length = 0
        end = self._length + new_k.shape[2]
        if end > self.max_seq_len:
            raise ValueError(f"KV-cache overflow: {end} > max_seq_len {self.max_seq_len}")
        self._k[:, :, self._length : end] = new_k.to(self.dtype)
        self._v[:, :, self._length : end] = new_v.to(self.dtype)
        self._length = end

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached ``(k, v)`` prefix as views.

        Raises:
            RuntimeError: if nothing has been cached yet.
        """
        if self._k is None or self._length == 0:
            raise RuntimeError("KV-cache is empty; call update() first")
        return self._k[:, :, : self._length], self._v[:, :, : self._length]
