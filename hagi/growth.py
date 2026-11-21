"""Residual-block growth (primes C450/C504 GrowTrunkBlock).

A block is y = x + gamma * h(x). gamma = 0 makes the block the
IDENTITY function exactly (benchResidual_zero) — appending it to a
trunk preserves the trunk function and every objective over the
trunk output (grow_block_obj_invariant). Utility arrives only from
the certified training of h afterwards (measured canonical drop).

The honest contract: admission of a new block is NEUTRAL at step 0
(0 <= 0, function_preserving_admissible); no Lipschitz claim while
gamma = 0; the block carries no CertStepData semantics — it is a
pure function arrow, matched to the contract form by
bench_bridge_to_C450 (same function, different carrier).
"""
from __future__ import annotations

from typing import Callable, Sequence


class ResidualBlock:
    """y = x + gamma * h(x); gamma=0 -> exact identity."""

    def __init__(self, h: Callable, gamma: float = 0.0):
        self.h = h
        self.gamma = float(gamma)

    def __call__(self, x):
        if self.gamma == 0.0:
            return x                      # exact identity (benchResidual_zero)
        return x + self.gamma * self.h(x)

    def set_gamma(self, gamma: float) -> None:
        self.gamma = float(gamma)


def grow_trunk(trunk: Sequence[Callable],
               h: Callable) -> list[Callable]:
    """Append a residual-zero block: trunk function preserved EXACTLY
    (grow_block_bench_sound). Returns the new trunk."""
    return list(trunk) + [ResidualBlock(h, gamma=0.0)]


def run_trunk(trunk: Sequence[Callable], x):
    """Compose blocks left-to-right (benchTrunk foldl form)."""
    for b in trunk:
        x = b(x)
    return x


def obj_invariant_check(trunk_old: Sequence[Callable],
                        trunk_new: Sequence[Callable],
                        xs, obj: Callable) -> bool:
    """grow_block_obj_invariant check: any objective over trunk
    output is unchanged by the append (exact float equality here —
    the Fraction-exact form lives in the bench)."""
    return all(obj(run_trunk(trunk_old, x)) == obj(run_trunk(trunk_new, x))
               for x in xs)
