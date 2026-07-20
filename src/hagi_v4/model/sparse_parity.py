"""Sparse parity encoder/checker — LDPC-style channel coding.

V8 key innovation: parity is generated BEFORE the channel (true
Source-Channel Separation), not inside the decoder.

V13: fixed Tanner graph (parity_base buffer) + learnable per-check
edge_log_scale. Prevents channel-code collapse (par 0.14→0.01) by
keeping H geometry fixed while still allowing amplitude adaptation.

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

    V13 fixed-graph design:
      - sparse_mask: fixed connectivity buffer
      - parity_base: fixed random edge weights (buffer, non-trainable)
      - edge_log_scale: learnable per-check amplitude [n_checks]
      - masked_weights = parity_base * sparse_mask * exp(edge_log_scale)

    Args:
        n_vars: C (systematic dimension, number of variable nodes)
        n_checks: M (parity dimension, number of check nodes)
        edges_per_check: sparsity (edges per check node, typically 3-6)
        seed: deterministic sparsity pattern
        freeze_base: if True (default V13), parity_base is a buffer
    """

    def __init__(
        self,
        n_vars: int,
        n_checks: int,
        edges_per_check: int = 4,
        seed: int = 42,
        norm_eps: float = 1e-6,
        freeze_base: bool = True,
    ) -> None:
        del norm_eps
        super().__init__()
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.edges_per_check = edges_per_check
        self.freeze_base = freeze_base

        gen = torch.Generator().manual_seed(seed)
        mask = _build_sparse_mask(n_checks, n_vars, edges_per_check, gen)
        self.register_buffer("sparse_mask", mask, persistent=True)

        base = torch.zeros(n_checks, n_vars)
        nn.init.normal_(base, mean=0.0, std=1.0 / math.sqrt(max(edges_per_check, 1)))
        base = base * mask
        if freeze_base:
            self.register_buffer("parity_base", base, persistent=True)
        else:
            self.parity_base = nn.Parameter(base)

        self.edge_log_scale = nn.Parameter(torch.zeros(n_checks))

    @property
    def parity_weights(self) -> nn.Parameter:
        """Compatibility: expose edge_log_scale as the learnable shared param.

        Checker shares edge_log_scale + parity_base buffer + mask.
        """
        return self.edge_log_scale

    @property
    def masked_weights(self) -> torch.Tensor:
        """Sparse weights: fixed base × mask × learnable per-check scale."""
        scale = torch.exp(self.edge_log_scale).unsqueeze(-1)
        return self.parity_base * self.sparse_mask * scale

    def forward(self, systematic: torch.Tensor) -> torch.Tensor:
        """Generate parity bits from systematic bits.

        Args:
            systematic: [B, T, C] systematic information bits.

        Returns:
            parity: [B, T, M] parity/redundancy bits.
        """
        w = self.masked_weights
        parity = torch.einsum("mc,btc->btm", w, systematic)
        return parity / math.sqrt(max(self.edges_per_check, 1))


class SparseParityChecker(nn.Module):
    """LDPC-style sparse parity checker (decoder side).

    Computes the parity-check residual: how much the current systematic
    estimate deviates from satisfying the parity equations.

    SHARED H graph: same mask + parity_base + edge_log_scale as encoder.

    Args:
        n_vars: C (systematic dimension)
        n_checks: M (parity dimension)
        edges_per_check: sparsity
        seed: must match the encoder's seed for consistent graph
        shared_edge_log_scale: encoder's edge_log_scale Parameter
        shared_mask: encoder's sparse_mask buffer
        shared_parity_base: encoder's parity_base buffer
        freeze_base: standalone init behaviour when not sharing
    """

    def __init__(
        self,
        n_vars: int,
        n_checks: int,
        edges_per_check: int = 4,
        seed: int = 42,
        norm_eps: float = 1e-6,
        shared_weights: nn.Parameter | None = None,
        shared_mask: torch.Tensor | None = None,
        shared_norm: nn.Module | None = None,
        shared_edge_log_scale: nn.Parameter | None = None,
        shared_parity_base: torch.Tensor | None = None,
        freeze_base: bool = True,
    ) -> None:
        del shared_norm
        del norm_eps
        super().__init__()
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.edges_per_check = edges_per_check
        self.freeze_base = freeze_base

        if shared_mask is not None:
            self.register_buffer("sparse_mask", shared_mask, persistent=True)
        else:
            gen = torch.Generator().manual_seed(seed)
            mask = _build_sparse_mask(n_checks, n_vars, edges_per_check, gen)
            self.register_buffer("sparse_mask", mask, persistent=True)

        if shared_parity_base is not None:
            self.register_buffer("parity_base", shared_parity_base, persistent=True)
        else:
            base = torch.zeros(n_checks, n_vars)
            nn.init.normal_(base, mean=0.0, std=1.0 / math.sqrt(max(edges_per_check, 1)))
            base = base * self.sparse_mask
            if freeze_base:
                self.register_buffer("parity_base", base, persistent=True)
            else:
                self.parity_base = nn.Parameter(base)

        # Prefer explicit edge_log_scale; fall back to shared_weights alias.
        scale_param = shared_edge_log_scale if shared_edge_log_scale is not None else shared_weights
        if scale_param is not None:
            self.edge_log_scale = scale_param
        else:
            self.edge_log_scale = nn.Parameter(torch.zeros(n_checks))

    @property
    def parity_weights(self) -> nn.Parameter:
        """Compatibility alias for shared learnable scales."""
        return self.edge_log_scale

    @property
    def masked_weights(self) -> torch.Tensor:
        scale = torch.exp(self.edge_log_scale).unsqueeze(-1)
        return self.parity_base * self.sparse_mask * scale

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
        parity_computed = parity_computed / math.sqrt(max(self.edges_per_check, 1))

        if parity_received is not None:
            residual = parity_received - parity_computed
        else:
            residual = parity_computed

        return residual, parity_computed
