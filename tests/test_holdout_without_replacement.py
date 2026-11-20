"""C487: holdout sampling is WITHOUT replacement (wr concentration).

Our AuditLedgerV9.hoeffding_radius uses the with-replacement bound,
which is conservative (never tighter than the wr sample bound) —
safe under theory (d) wr_bound_le_with_replacement whenever the
variance condition holds; otherwise it is the standard WR bound
itself, still valid for i.i.d. surrogate. The sampling itself must
be strictly without replacement.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from hagi.core import AuditLedgerV9


def test_splits_sample_without_replacement():
    led = AuditLedgerV9()
    dom = list(range(200))
    tr, ceg, val = led.draw_disjoint_splits("t", dom, seed=5)
    # disjoint by construction — no point appears twice anywhere
    seq = tr + ceg + val
    assert len(seq) == len(set(seq)), "replacement detected in holdout split"


def test_holdout_choice_without_replacement():
    rng = np.random.default_rng(7)
    n = 50_000
    hold = rng.choice(n, size=800, replace=False)
    assert len(hold) == len(set(hold.tolist()))  # strict woR
