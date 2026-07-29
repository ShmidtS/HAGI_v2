"""LatentMemoryBank — CTM-inspired FIFO latent-state memory with cross-attention.

Stores a FIFO buffer of recent latent states z (from the IB bottleneck) and
cross-attends the current block hidden state h_ctx over the bank. This gives
each layer access to a short compressed memory of past channel states without
the O(T^2) cost of full self-attention over history.

At ``bank_size=0`` the module is a no-op (backward compatible).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentMemoryBank(nn.Module):
    """FIFO latent bank with lightweight cross-attention readout.

    Args:
        dim: hidden size of the main block output H (not bottleneck C!).
        latent_dim: size of the IB latent state C.
        bank_size: max number of recent z states to store (FIFO). 0 = disabled.
        num_heads: cross-attention heads.
        head_dim: dimension per head.
    """

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        bank_size: int = 16,
        num_heads: int = 4,
        head_dim: int = 32,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.bank_size = bank_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        inner = num_heads * head_dim
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(latent_dim, inner, bias=False)
        self.v_proj = nn.Linear(latent_dim, inner, bias=False)
        self.out_proj = nn.Linear(inner, dim, bias=False)

        # FIFO bank — allocated lazily on first forward for device/dtype.
        self.register_buffer("_bank", None, persistent=False)
        self.register_buffer("_bank_fill", torch.tensor(0), persistent=False)

    def reset(self) -> None:
        """Clear the memory bank (e.g. new dialogue / eval restart)."""
        self._bank = None
        self._bank_fill = torch.tensor(0)

    def _ensure_bank(self, z: torch.Tensor) -> torch.Tensor:
        """Lazily allocate or reallocate the bank buffer on the correct device/dtype."""
        B = z.shape[0]
        need_alloc = (
            self._bank is None
            or self._bank.shape[0] != B
            or self._bank.shape[1] != self.bank_size
            or self._bank.device != z.device
        )
        if need_alloc:
            self._bank = torch.zeros(
                B, self.bank_size, self.latent_dim,
                dtype=z.dtype, device=z.device,
            )
            self._bank_fill = torch.tensor(0)
        return self._bank

    @staticmethod
    def _reshape_for_attn(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        """[B, T, inner] -> [B, num_heads, T, head_dim]."""
        B, T, _ = x.shape
        return x.view(B, T, num_heads, head_dim).transpose(1, 2).contiguous()

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Cross-attend h over the latent memory bank.

        Args:
            h: ``[B, T, H]`` current block hidden state.
            z: ``[B, T, C]`` current IB latent state.

        Returns:
            ``h + cross_attn_output`` of shape ``[B, T, H]``.

        The bank is built functionally — old states are detached (no gradient
        through time) and concatenated with the current z so gradients flow
        only through the current step's k_proj/v_proj path.
        """
        B, T, _ = h.shape
        z = z.to(dtype=h.dtype)

        # Build bank: concat old (detached) + new z, keep last bank_size.
        if self._bank is not None and self._bank.shape[0] == B:
            old_bank = self._bank.detach()
            old_fill = int(self._bank_fill.item())
        else:
            old_bank = z.new_zeros(B, 0, self.latent_dim)
            old_fill = 0

        combined = torch.cat([old_bank, z], dim=1)               # [B, old_fill+T, C]
        if combined.shape[1] < self.bank_size:
            pad = combined.new_zeros(B, self.bank_size - combined.shape[1], self.latent_dim)
            combined = torch.cat([pad, combined], dim=1)          # left-pad to bank_size
        bank = combined[:, -self.bank_size:, :]                  # [B, bank_size, C]
        fill = min(self.bank_size, old_fill + T)

        # Store detached for next forward — no gradient through time.
        self._bank = bank.detach()
        self._bank_fill = torch.tensor(fill)

        # Mask: only attend over rightmost filled slots (left-padded with zeros).
        attn_mask = torch.arange(self.bank_size, device=h.device) >= (self.bank_size - fill)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, bank_size]

        # Cross-attention: Q from h, K/V from bank.
        q = self._reshape_for_attn(self.q_proj(h), self.num_heads, self.head_dim)
        k = self._reshape_for_attn(self.k_proj(bank), self.num_heads, self.head_dim)
        v = self._reshape_for_attn(self.v_proj(bank), self.num_heads, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)

        return h + out
