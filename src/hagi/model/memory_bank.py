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
        """
        B, T, _ = h.shape
        bank = self._ensure_bank(z)

        # Push z into FIFO bank: append new, drop oldest.
        fill = int(self._bank_fill.item())
        if T >= self.bank_size:
            # The new batch alone fills/exceeds the bank — just take the last bank_size.
            bank[:, :] = z[:, -self.bank_size:, :]
            fill = self.bank_size
        else:
            room = self.bank_size - fill
            take = min(room, T)
            if take > 0:
                bank[:, fill:fill + take, :] = z[:, :take, :]
                fill += take
            if fill > self.bank_size:
                fill = self.bank_size
        self._bank_fill = torch.tensor(fill)

        # Mask: only attend over filled slots (fill may be < bank_size early in seq).
        attn_mask = torch.arange(self.bank_size, device=h.device).unsqueeze(0) < fill
        # attn_mask: [1, bank_size] -> broadcast to [B, num_heads, T_q, bank_size]
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 1, bank_size]

        # Cross-attention: Q from h, K/V from bank.
        q = self._reshape_for_attn(self.q_proj(h), self.num_heads, self.head_dim)
        # q: [B, num_heads, T, head_dim]
        k = self._reshape_for_attn(self.k_proj(bank), self.num_heads, self.head_dim)
        # k: [B, num_heads, bank_size, head_dim]
        v = self._reshape_for_attn(self.v_proj(bank), self.num_heads, self.head_dim)
        # v: [B, num_heads, bank_size, head_dim]

        # scaled_dot_product_attention: Q[*, heads, T_q, d] over K/V[*, heads, T_kv, d]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        # out: [B, num_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, inner]
        out = self.out_proj(out)

        return h + out
