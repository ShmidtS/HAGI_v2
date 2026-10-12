"""Low-rank table adapters (Element.lean residualLeaf, round 22).

The measured mechanism (round 19): leaves from a shared pretrained
start have tables E_i = E0 + dE_common + eps_i, and the deltas are
65% common. The synthesis' element redesign: keep the shared base
FROZEN and train only low-rank per-expert deltas,

    W_i = W0 + A_i @ B_i^T,   A_i in R^{m x r}, B_i in R^{n x r}

which for V=32768, H=128, N experts compresses N*V*H parameters into
V*H + N*r*(V+H) (N=3, r=8: 2.52x; N=9, r=8: 5.75x).

residualLeaf_zero (Lean): rank-0 deltas are definitionally the base,
so the step-0 lift is safe -- the leaf starts EXACTLY at the prior.
"""
from __future__ import annotations

import torch
from torch import nn

from hagi.model.adaptive import AdaptiveComponent


class TableLoRA(AdaptiveComponent):
    """A low-rank residual on a frozen table W0: W0 + A @ B^T.

    ``A`` is a frozen orthonormal buffer (QR init, matching the
    TttLoraAdapter convention); ``B`` is the only trainable tensor,
    zero-init so the delta is exactly zero at construction
    (residualLeaf_zero: step-0 == the prior, bit-for-bit).

    The base is registered as a buffer (not a parameter): it stays
    byte-frozen through optimizer steps and the state_dict carries
    it once.
    """

    def __init__(self, base_weight: torch.Tensor, rank: int) -> None:
        super().__init__()
        if base_weight.ndim != 2:
            raise ValueError(f"base must be 2D, got {base_weight.ndim}D")
        if type(rank) is not int or rank < 1:
            raise ValueError(f"rank must be a positive int, got {rank!r}")
        m, n = base_weight.shape
        self.rank = rank
        self.register_buffer("base", base_weight.detach().clone())
        # frozen orthonormal features A: [m, r] via QR of a random probe
        g = torch.Generator().manual_seed(0)
        probe = torch.randn(m, min(rank, m), generator=g)
        a, _ = torch.linalg.qr(probe)
        self.register_buffer("A", a[:, :rank].to(base_weight.dtype))
        # trainable delta coefficients: [rank, n], zero-init
        self.B = nn.Parameter(torch.zeros(rank, n, dtype=base_weight.dtype))

    def effective_weight(self) -> torch.Tensor:
        """W0 + A @ B: the full effective table (for forward paths
        that need the materialized matrix)."""
        return self.base + (self.A @ self.B)

    def extra_repr(self) -> str:
        m, n = self.base.shape
        r = self.rank
        return f"{m}x{n}, rank={r}, delta_params={r*(m+n)}, base_params={m*n}"


def lora_compression(n_experts: int, vocab: int, hidden: int, rank: int) -> float:
    """The synthesis' compression ratio P_LR / P_ind for the tables.

    Independent: N*V*H. Shared base + rank-r children:
    V*H + N*r*(V+H). Returns the ratio (< 1 = compression).
    """
    if n_experts < 1 or rank < 1:
        raise ValueError("need n_experts >= 1 and rank >= 1")
    return (vocab * hidden + n_experts * rank * (vocab + hidden)) / (
        n_experts * vocab * hidden)


def recenter(deltas: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Synthesis §7 step 4-6: pull the common delta into the parent.

    Given the children's delta matrices, compute mean_delta and
    return (mean_delta, [delta_i - mean_delta]) so that
    sum of the recentred deltas is exactly zero -- the parent absorbs
    the common component, the children keep only their
    specialization.
    """
    stacked = torch.stack([d.reshape(-1) for d in deltas])
    mean_delta = stacked.mean(0)
    recentred = [d - mean_delta.view_as(d) for d in deltas]
    return mean_delta, recentred
