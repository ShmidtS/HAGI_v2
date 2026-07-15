"""Sparse parity encoder/checker — LDPC-style channel coding.

V8 key innovation: parity is generated BEFORE the channel (true
Source-Channel Separation), not inside the decoder.

Structure:
  SparseParityEncoder: systematic → parity (channel encoding)
  SparseParityChecker: systematic → residual (parity check at decoder)

Both share the same sparse connectivity pattern (Tanner graph):

  Variable nodes: C (systematic dimensions)
  Check nodes:    M (parity dimensions)
  Edges:          ~edges_per_check per check node (sparse)

5G NR analog: LDPC code with base graph, sparse parity-check matrix.
Complexity: O(M × edges_per_check) — linear in C, not quadratic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from hagi_v4.model.norms import RMSNorm


def _build_sparse_mask(n_checks: int, n_vars: int, edges_per_check: int, generator: torch.Generator) -> torch.Tensor:
    """Build a sparse bipartite connectivity mask [n_checks, n_vars].

    Each check node connects to exactly edges_per_check variable nodes.
    Uses a generator for deterministic, reproducible sparsity patterns.

    Ensures each variable node has at least one connection (no orphan vars).
    """
    mask = torch.zeros(n_checks, n_vars, dtype=torch.float32)
    for check_idx in range(n_checks):
        indices = torch.randperm(n_vars, generator=generator)[:edges_per_check]
        mask[check_idx, indices] = 1.0

    coverage = mask.sum(dim=0)
    for var_idx in range(n_vars):
        if coverage[var_idx] == 0:
            check_idx = int(torch.randint(0, n_checks, (1,), generator=generator).item())
            mask[check_idx, var_idx] = 1.0

    return mask


class SparseParityEncoder(nn.Module):
    """LDPC-style sparse parity generator (channel encoder).

    Takes systematic bits [B, T, C] and produces parity bits [B, T, M].
    Sparse connectivity: each check node reads from edges_per_check var nodes.

    The sparse weight matrix is initialized small (near-identity residual)
    so the model starts with weak parity and learns optimal weights.

    Args:
        n_vars: C (systematic dimension, number of variable nodes)
        n_checks: M (parity dimension, number of check nodes)
        edges_per_check: sparsity (edges per check node, typically 3-6)
        seed: deterministic sparsity pattern
    """

    def __init__(
        self,
        n_vars: int,
        n_checks: int,
        edges_per_check: int = 4,
        seed: int = 42,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.edges_per_check = edges_per_check

        gen = torch.Generator().manual_seed(seed)
        mask = _build_sparse_mask(n_checks, n_vars, edges_per_check, gen)
        self.register_buffer("sparse_mask", mask, persistent=True)

        self.parity_weights = nn.Parameter(torch.zeros(n_checks, n_vars))
        nn.init.normal_(self.parity_weights, mean=0.0, std=1.0 / math.sqrt(max(edges_per_check, 1)))

        self.norm = RMSNorm(n_checks, eps=norm_eps)

    @property
    def masked_weights(self) -> torch.Tensor:
        """Sparse weights applied: element-wise mask × learnable weights."""
        return self.parity_weights * self.sparse_mask

    def forward(self, systematic: torch.Tensor) -> torch.Tensor:
        """Generate parity bits from systematic bits.

        Args:
            systematic: [B, T, C] systematic information bits.

        Returns:
            parity: [B, T, M] parity/redundancy bits.
        """
        w = self.masked_weights
        parity = torch.einsum("mc,btc->btm", w, systematic)
        return self.norm(parity)


class SparseParityChecker(nn.Module):
    """LDPC-style sparse parity checker (decoder side).

    Computes the parity-check residual: how much the current systematic
    estimate deviates from satisfying the parity equations.

    In LDPC decoding, the check node computes:
        residual = parity_received - parity_computed_from_estimate

    If the estimate is correct, residual → 0. Non-zero residual drives
    belief propagation corrections.

    Shares the same sparse mask as the encoder (transposed connectivity).

    Args:
        n_vars: C (systematic dimension)
        n_checks: M (parity dimension)
        edges_per_check: sparsity
        seed: must match the encoder's seed for consistent graph
    """

    def __init__(
        self,
        n_vars: int,
        n_checks: int,
        edges_per_check: int = 4,
        seed: int = 42,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.edges_per_check = edges_per_check

        gen = torch.Generator().manual_seed(seed)
        mask = _build_sparse_mask(n_checks, n_vars, edges_per_check, gen)
        self.register_buffer("sparse_mask", mask, persistent=True)

        self.check_weights = nn.Parameter(torch.zeros(n_checks, n_vars))
        nn.init.normal_(self.check_weights, mean=0.0, std=1.0 / math.sqrt(max(edges_per_check, 1)))

        self.norm = RMSNorm(n_checks, eps=norm_eps)

    @property
    def masked_weights(self) -> torch.Tensor:
        return self.check_weights * self.sparse_mask

    def forward(
        self,
        systematic: torch.Tensor,
        parity_received: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute parity-check residual.

        Args:
            systematic: [B, T, C] current systematic estimate.
            parity_received: [B, T, M] parity from encoder (during training).
                             None during inference (pure consistency check).

        Returns:
            residual: [B, T, M] parity-check residual (extrinsic info).
            parity_computed: [B, T, M] parity computed from current estimate.
        """
        w = self.masked_weights
        parity_computed = torch.einsum("mc,btc->btm", w, systematic)
        parity_computed = self.norm(parity_computed)

        if parity_received is not None:
            residual = parity_received - parity_computed
        else:
            residual = parity_computed

        return residual, parity_computed
