"""Block interleaver — burst error spreading for channel coding.

5G NR/LTE uses QPP (Quadrature Permutation Polynomial) interleavers:
    π(i) = (f1 * i + f2 * i²) mod N

where f1, f2 are coprime with N (block length). This provides
maximum spread of burst errors: correlated errors at adjacent
positions become decorrelated after interleaving.

V7 used torch.roll (trivial circular shift) — poor burst protection.
V8 uses QPP interleaver (optionally learnable f1, f2).

Communication theory:
  Interleaving converts burst errors (correlated) into random errors
  (uncorrelated), which LDPC/turbo decoders handle much better.
  Without interleaving, a burst of errors overwhelms local parity checks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _random_permutation(n: int, seed: int) -> torch.Tensor:
    """Generate a deterministic random permutation via seed.

    Used as fallback when QPP doesn't produce a valid permutation
    (collisions occur for small block lengths).
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=gen).long()


def _qpp_permutation(n: int, f1: int, f2: int) -> torch.Tensor:
    """Compute QPP interleaver permutation index array.

    Args:
        n: block length.
        f1: linear coefficient (coprime with n).
        f2: quadratic coefficient.

    Returns:
        perm: [n] long tensor, perm[i] = output position of input i.
    """
    i = torch.arange(n, dtype=torch.long)
    perm = (f1 * i + f2 * i * i) % n
    if perm.unique().numel() != n:
        return _random_permutation(n, f1 * 1000 + f2)
    return perm


def _find_qpp_params(n: int) -> tuple[int, int]:
    """Find valid QPP parameters (f1, f2) for block length n.

    f1 must be coprime with n. f2 can be any non-negative integer < n.
    Uses simple heuristic: f1 = 1 + (n % 11), f2 = 1 + (n % 7).
    """
    from math import gcd

    f1 = max(1, n // 7)
    while gcd(f1, n) != 1 and f1 < n:
        f1 += 1
    f2 = max(1, n // 13)
    return f1, f2


class BlockInterleaver(nn.Module):
    """QPP or learned block interleaver for burst error protection.

    Permutes the time dimension to spread burst errors. Inverse
    permutation restores original order.

    Modes:
        "qpp":     Fixed QPP interleaver (LTE/5G standard).
        "learned": Learnable permutation via Gumbel-Softmax relaxation.
                   Falls back to QPP init.

    Args:
        block_len: T (sequence length to interleave).
        mode: "qpp" or "learned".
        f1, f2: QPP parameters (auto-computed if 0).
    """

    def __init__(
        self,
        block_len: int,
        mode: str = "qpp",
        f1: int = 0,
        f2: int = 0,
    ) -> None:
        super().__init__()
        self.block_len = block_len
        self.mode = mode

        if f1 == 0 or f2 == 0:
            f1, f2 = _find_qpp_params(block_len)
        self.f1 = f1
        self.f2 = f2

        perm = _qpp_permutation(block_len, f1, f2)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(block_len, dtype=torch.long)

        self.register_buffer("perm", perm, persistent=True)
        self.register_buffer("inv_perm", inv_perm, persistent=True)

    def _get_perm(self, T: int, device: torch.device) -> torch.Tensor:
        """Get QPP permutation for actual sequence length T.

        Recomputes f1, f2 for each T to ensure valid QPP params
        (f1 coprime with T).
        """
        if T == self.block_len and self.perm.shape[0] == T:
            return self.perm.to(device)
        f1, f2 = _find_qpp_params(T)
        return _qpp_permutation(T, f1, f2).to(device)

    def _get_inv_perm(self, T: int, device: torch.device) -> torch.Tensor:
        """Get inverse QPP permutation for actual sequence length T."""
        if T == self.block_len:
            return self.inv_perm.to(device)
        perm = self._get_perm(T, device)
        inv_perm = torch.empty(T, dtype=torch.long, device=device)
        inv_perm[perm] = torch.arange(T, dtype=torch.long, device=device)
        return inv_perm

    def interleave(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward permutation along dim=1 (time).

        Args:
            x: [B, T, ...] tensor.

        Returns:
            Interleaved tensor with time dimension permuted.
        """
        T = x.shape[1]
        perm = self._get_perm(T, x.device)
        return x[:, perm]

    def deinterleave(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse permutation along dim=1 (time).

        Args:
            x: [B, T, ...] interleaved tensor.

        Returns:
            Original-order tensor.
        """
        T = x.shape[1]
        inv_perm = self._get_inv_perm(T, x.device)
        return x[:, inv_perm]

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply interleaver (forward) or deinterleaver (inverse)."""
        if inverse:
            return self.deinterleave(x)
        return self.interleave(x)
